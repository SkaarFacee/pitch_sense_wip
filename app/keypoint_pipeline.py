"""KeypointPipeline — End-to-end pipeline: keypoint→homography→player projection→team colors→ball→segmentation→5 output videos."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from constants import (
    PLAYER_CONF, SEG_CONF, CANVAS_W, CANVAS_H, PITCH_LENGTH, PITCH_WIDTH,
    BALL_CONF, BALL_TRAIL_LENGTH, BALL_BBOX_COLOR, BALL_DOT_COLOR,
)
from keypoint_service import KeypointHomographyComputer
from player_service import PlayerDetector
from ball_service import BallDetector
from segmentation import Segmentor
from pitch import PitchArtist
from director import Director
from team_analyzer import TeamColorAnalyzer


class KeypointPipeline:
    def __init__(self, keypoint_model_path: str, player_model_path: str, seg_model_path: str,
                 ball_model_path: str = "", flip_projection_x: bool = False, enable_team_colors: bool = True):
        self.keypoint_computer = KeypointHomographyComputer(keypoint_model_path)
        self.player_detector = PlayerDetector(player_model_path)
        self.segmentor = Segmentor(seg_model_path)
        self.team_analyzer = TeamColorAnalyzer() if enable_team_colors else None
        self.ball_detector = BallDetector(ball_model_path, conf=BALL_CONF) if ball_model_path else None
        self.last_H = None
        self.pitch_artist = PitchArtist()
        self.flip_projection_x = flip_projection_x
        self.ball_trajectory = []

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0):
        frame_h, frame_w = frame.shape[:2]
        processed_segments = []
        seg_overlay_frame = frame.copy()

        # 1. Segmentation
        seg_op = self.segmentor.model.predict(frame, conf=SEG_CONF, verbose=False)[0]
        if seg_op is not None and getattr(seg_op, 'masks', None) is not None:
            processed_segments = self.segmentor.extract(seg_op, frame_w, last_side=None)
            seg_overlay_frame = self._create_seg_overlay(frame, seg_op, processed_segments)

        # 2. Keypoint → Homography
        H, H_info = self.keypoint_computer.compute_homography(frame, last_H=self.last_H)
        if H is not None:
            self.last_H = H

        # 3a. Player detection & projection
        formatted = self.player_detector.track_players(frame, conf=PLAYER_CONF)
        if formatted is not None:
            player_xyxy, player_conf, _, track_ids = formatted
        else:
            player_xyxy = np.empty((0, 4), dtype=np.float32)
            player_conf = np.empty((0,), dtype=np.float32)
            track_ids = np.empty((0,), dtype=np.int32)
        player_pitch_pts = np.empty((0, 2), dtype=np.float32)
        if H is not None and len(player_xyxy) > 0:
            player_pitch_pts = self.player_detector.project_points(player_xyxy, H)
            if self.flip_projection_x and len(player_pitch_pts) > 0:
                player_pitch_pts[:, 0] = PITCH_LENGTH - player_pitch_pts[:, 0]

        # 3b. Team colors
        team_info = None
        if self.team_analyzer is not None and len(player_xyxy) > 0:
            team_info = self.team_analyzer.assign_team_colors(frame, player_xyxy, player_conf, track_ids=track_ids)

        # 3c. Ball detection
        ball_xyxy = np.empty((0, 4), dtype=np.float32)
        ball_conf = np.empty((0,), dtype=np.float32)
        ball_pitch_pt = None
        if self.ball_detector is not None:
            ball_xyxy, ball_conf = self.ball_detector.detect_ball(frame)
            if len(ball_xyxy) > 0 and H is not None:
                ball_pitch_pt = self.ball_detector.project_ball_to_pitch(ball_xyxy, H, flip_x=self.flip_projection_x, pitch_length=PITCH_LENGTH)
                self.ball_trajectory.append(ball_pitch_pt.copy())
                if len(self.ball_trajectory) > BALL_TRAIL_LENGTH:
                    self.ball_trajectory.pop(0)

        # 4. Pitch canvas (top-down)
        pitch_canvas = self.pitch_artist.draw_pitch_base()
        if len(self.ball_trajectory) > 1:
            pitch_canvas = self.pitch_artist.draw_ball_trajectory(pitch_canvas, self.ball_trajectory, max_trail=BALL_TRAIL_LENGTH)
        if len(player_pitch_pts) > 0:
            mask = ((player_pitch_pts[:, 0] >= -5) & (player_pitch_pts[:, 0] <= PITCH_LENGTH + 5)
                    & (player_pitch_pts[:, 1] >= -5) & (player_pitch_pts[:, 1] <= PITCH_WIDTH + 5))
            valid_pts = player_pitch_pts[mask]
            if len(valid_pts) > 0:
                colors = None
                if team_info is not None:
                    c = team_info['team_colors']
                    colors = [c[i] for i in range(len(c)) if mask[i]] if len(c) == len(mask) else None
                pitch_canvas = self.pitch_artist.draw_players_on_pitch(pitch_canvas, valid_pts, colors=colors, default_color=(0, 0, 255))
            if team_info is not None:
                pitch_canvas = self.pitch_artist.draw_team_legend(pitch_canvas, team_info['team1_bgr'], team_info['team2_bgr'])
        if ball_pitch_pt is not None:
            pitch_canvas = self.pitch_artist.draw_ball_on_pitch(pitch_canvas, ball_pitch_pt, ball_color=BALL_DOT_COLOR)

        # 5. Annotated frames
        used_kpts = H_info.get('used_keypoints', [])
        annotated_frame = self._draw_keypoints_on_frame(frame, used_kpts)
        if team_info is not None and len(player_xyxy) > 0:
            annotated_frame = self._draw_team_bboxes(annotated_frame, player_xyxy, team_info['team_colors'], player_conf=player_conf)
        if len(ball_xyxy) > 0:
            annotated_frame = self._draw_ball_bbox(annotated_frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR)
        deep_analysis_frame = self._draw_keypoints_on_frame(seg_overlay_frame, used_kpts) if used_kpts else seg_overlay_frame
        if team_info is not None and len(player_xyxy) > 0:
            deep_analysis_frame = self._draw_team_bboxes(deep_analysis_frame, player_xyxy, team_info['team_colors'], player_conf=player_conf)
        if len(ball_xyxy) > 0:
            deep_analysis_frame = self._draw_ball_bbox(deep_analysis_frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR)

        return {'H': H, 'H_info': H_info, 'player_xyxy': player_xyxy, 'player_conf': player_conf,
                'track_ids': track_ids, 'player_pitch_pts': player_pitch_pts, 'keypoints_used': used_kpts,
                'seg_result': seg_op, 'processed_segments': processed_segments, 'pitch_canvas': pitch_canvas,
                'annotated_frame': annotated_frame, 'deep_analysis_frame': deep_analysis_frame, 'team_info': team_info,
                'ball_xyxy': ball_xyxy, 'ball_conf': ball_conf, 'ball_pitch_pt': ball_pitch_pt,
                'ball_trajectory': list(self.ball_trajectory)}

    # Drawing helpers
    _KPT_CONNECTIONS = [
        (0, 16), (0, 9), (16, 25), (9, 25),
        (1, 2), (3, 4), (1, 3), (2, 4),
        (5, 6), (7, 8), (5, 7), (6, 8),
        (17, 18), (19, 20), (17, 19), (18, 20),
        (21, 22), (23, 24), (21, 23), (22, 24),
        (11, 12), (13, 14),
    ]

    def _draw_keypoints_on_frame(self, frame, used_keypoints, radius=6):
        out = frame.copy()
        kpt_pos = {kp['kpt_id']: (int(kp['image_pt'][0]), int(kp['image_pt'][1])) for kp in used_keypoints}
        for s, e in self._KPT_CONNECTIONS:
            if s in kpt_pos and e in kpt_pos:
                cv2.line(out, kpt_pos[s], kpt_pos[e], (0, 255, 255), 2, cv2.LINE_AA)
        for kp in used_keypoints:
            x, y = int(kp['image_pt'][0]), int(kp['image_pt'][1])
            cv2.circle(out, (x, y), radius, (0, 255, 255), -1)
            cv2.circle(out, (x, y), radius + 2, (255, 255, 0), 2)
            cv2.putText(out, f"{kp['kpt_id']}:{kp['name'][:15]}", (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _create_seg_overlay(self, frame, seg_op, processed_segments):
        out = frame.copy()
        overlay = frame.copy()
        color_map = {'18Yard': (255, 0, 0), '18Yard Circle': (0, 255, 0), '5Yard': (0, 0, 255),
                     'Half Central Circle': (255, 255, 0), 'Half Field': (255, 0, 255)} #BGR format
        for seg in processed_segments:
            contour = seg.get('image_contour')
            if contour is not None:
                cv2.drawContours(overlay, [contour], -1, color_map.get(seg['class_name'], (128, 128, 128)), -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out) #adds that transparency
        for seg in processed_segments:
            bbox = seg.get('image_bbox')
            if bbox is not None:
                cx, cy = float(bbox[:, 0].mean()), float(bbox[:, 1].mean())
                cv2.putText(out, f"{seg['class_name']} {seg['confidence']:.2f}", (int(cx) - 30, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    def _draw_team_bboxes(self, frame, player_xyxy, team_colors, player_conf=None):
        out = frame.copy()
        h, w = frame.shape[:2]
        for i in range(min(len(player_xyxy), len(team_colors))):
            x1, y1, x2, y2 = max(0, int(player_xyxy[i][0])), max(0, int(player_xyxy[i][1])), min(w - 1, int(player_xyxy[i][2])), min(h - 1, int(player_xyxy[i][3]))
            color = team_colors[i]
            ov = out.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, 0.25, out, 0.75, 0, out)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if player_conf is not None and i < len(player_conf):
                label = f"{player_conf[i]:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                tx, ty = (x1 + x2 - tw) // 2, y1 - 5
                if ty - th < 0:
                    ty = y1 + th + 2
                cv2.putText(out, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _draw_ball_bbox(self, frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR):
        if len(ball_xyxy) == 0:
            return frame
        out = frame.copy()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, int(ball_xyxy[0, 0])), max(0, int(ball_xyxy[0, 1])), min(w - 1, int(ball_xyxy[0, 2])), min(h - 1, int(ball_xyxy[0, 3]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        conf = ball_conf[0] if len(ball_conf) > 0 else 0.0
        label = f"Ball {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = x1, y1 - 5
        if ty - th < 0:
            ty = y2 + th + 5
        cv2.rectangle(out, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    def process_video(self, source_video_path: str, output_dir: str = "output", fps: float = 30.0,
                      start_frame: int = 0, max_frames: Optional[int] = None, process_every_n: int = 1):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths = {k: output_path / v for k, v in [("pitch", "full_pitch_debug_map.mp4"), ("annotated", "annotated_video.mp4"),
                                                   ("deep", "deep_analysis.mp4"), ("draft", "final_draft.mp4"),
                                                   ("keypoint", "keypoint_annotations.mp4")]}
        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source_video_path}")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_w, frame_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pw = Director.make_video_writer(paths["pitch"], actual_fps, (CANVAS_W, CANVAS_H))
        aw = Director.make_video_writer(paths["annotated"], actual_fps, (frame_w, frame_h))
        dw = Director.make_video_writer(paths["deep"], actual_fps, (frame_w, frame_h))
        fw = Director.make_video_writer(paths["draft"], actual_fps, (frame_w, frame_h))
        kw = Director.make_video_writer(paths["keypoint"], actual_fps, (frame_w, frame_h))
        if self.team_analyzer is not None:
            self.team_analyzer.reset()
        frame_idx = processed_count = 0
        self.ball_trajectory = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if max_frames is not None and processed_count >= max_frames:
                    break
                if frame_idx % process_every_n != 0:
                    frame_idx += 1
                    continue
                result = self.process_frame(frame, frame_idx)
                processed_count += 1
                if result['pitch_canvas'] is not None:
                    pw.write(result['pitch_canvas'])
                if result['annotated_frame'] is not None:
                    aw.write(result['annotated_frame'])
                if result['deep_analysis_frame'] is not None:
                    dw.write(result['deep_analysis_frame'])
                kpts = result.get('keypoints_used', [])
                kw.write(self._draw_keypoints_on_frame(frame.copy(), kpts))
                if result['pitch_canvas'] is not None:
                    pip_w, pip_h = frame_w // 4, int((frame_w // 4) * CANVAS_H / CANVAS_W)
                    pip = cv2.resize(result['pitch_canvas'], (pip_w, pip_h))
                    ox, oy = frame_w - pip_w - 15, frame_h - pip_h - 15
                    draft = frame.copy()
                    ov = draft.copy()
                    cv2.rectangle(ov, (ox - 5, oy - 5), (ox + pip_w + 5, oy + pip_h + 5), (0, 0, 0), -1)
                    cv2.addWeighted(ov, 0.4, draft, 0.6, 0, draft)
                    draft[oy:oy + pip_h, ox:ox + pip_w] = pip
                    cv2.rectangle(draft, (ox - 2, oy - 2), (ox + pip_w + 2, oy + pip_h + 2), (255, 255, 255), 2)
                    fw.write(draft)
                yield result
                frame_idx += 1
        finally:
            cap.release()
            for w in [pw, aw, dw, fw, kw]:
                w.release()
