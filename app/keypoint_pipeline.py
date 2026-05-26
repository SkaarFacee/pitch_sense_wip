"""
KeypointPipeline — End-to-end pipeline orchestrator.

Pipeline flow per frame:
    Keypoint detection → Homography matrix
    Player detection → Bottom-center projection via H
    Team color analysis → Team segregation via K-means on jersey colors
    Segmentation → Deep analysis overlay (on original frame + pitch canvas)

Outputs:
    - Annotated original frame (keypoints + seg masks + team-colored bboxes)
    - Top-down pitch canvas (players projected + team-colored dots + legend)
    - Deep analysis frame (seg overlay + team colors)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from constants import (
    PLAYER_CONF,
    SEG_CONF,
    REUSE_LAST_HOMOGRAPHY,
    CANVAS_W,
    CANVAS_H,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from keypoint_service import KeypointHomographyComputer, PitchKeypointMapper
from player_service import PlayerDetector
from segmentation import Segmentor, BboxManipulor, GeometryManipulor
from seg_helpers import CanvasMapper
from pitch import PitchArtist
from seg_plural import BestSegmentPicker
from director import Director
from team_analyzer import TeamColorAnalyzer


class KeypointPipeline:
    """
    Orchestrates the full keypoint → homography → player projection →
    team color analysis → segmentation analysis pipeline for each video frame.
    """

    def __init__(
        self,
        keypoint_model_path: str,
        player_model_path: str,
        seg_model_path: str,
        flip_projection_x: bool = False,
        enable_team_colors: bool = True,
    ):
        """
        Args:
            keypoint_model_path: Path to YOLO-Pose keypoint model.
            player_model_path: Path to YOLO player detection model.
            seg_model_path: Path to YOLO segmentation model.
            flip_projection_x: If True, mirror the x-axis of projected player
                               positions (PITCH_LENGTH - x). Fixes left-right
                               flip caused by camera being on opposite side.
            enable_team_colors: If True, run team color analysis per frame.
        """
        # Models
        self.keypoint_computer = KeypointHomographyComputer(keypoint_model_path)
        self.player_detector = PlayerDetector(player_model_path)
        self.segmentor = Segmentor(seg_model_path)
        self.team_analyzer = TeamColorAnalyzer() if enable_team_colors else None

        # State
        self.last_H = None
        self.last_H_info = None
        self.pitch_artist = PitchArtist()
        self.flip_projection_x = flip_projection_x

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0):
        """
        Run the full pipeline on a single frame.

        Args:
            frame: BGR image (H, W, 3).
            frame_idx: Frame index for logging.

        Returns:
            dict with keys:
                'H'               : 3x3 homography or None
                'H_info'          : dict from KeypointHomographyComputer
                'player_xyxy'     : player bboxes in image coords (N,4)
                'player_conf'     : player confidences (N,)
                'track_ids'       : (N,) int — ByteTrack track IDs per detection
                'player_pitch_pts': player bottom-center in pitch coords (M,2)
                'keypoints_used'  : list of used keypoint dicts
                'seg_result'      : raw YOLO seg result
                'processed_segments': list from Segmentor.extract()
                'pitch_canvas'    : np.ndarray — top-down pitch with players
                'annotated_frame' : np.ndarray — original frame with overlays
                'deep_analysis_frame': np.ndarray — frame with seg mask overlay
        """
        frame_h, frame_w = frame.shape[:2]

        # --------------------------------------------------------------
        # 1. Segmentation — run FIRST to validate keypoints
        # --------------------------------------------------------------
        processed_segments = []
        seg_overlay_frame = frame.copy()

        seg_output = self.segmentor.model.predict(
            frame, conf=SEG_CONF, verbose=False
        )
        seg_op = seg_output[0]

        if seg_op is not None and getattr(seg_op, 'masks', None) is not None:
            processed_segments = self.segmentor.extract(
                seg_op, frame_w, last_side=None
            )
            # Build deep analysis frame: original + seg mask overlay
            seg_overlay_frame = self._create_seg_overlay(
                frame, seg_op, processed_segments
            )

        # --------------------------------------------------------------
        # 2. Keypoint → Homography (with segmentation validation)
        # --------------------------------------------------------------
        H, H_info = self.keypoint_computer.compute_homography(
            frame, last_H=self.last_H, processed_segments=processed_segments
        )

        if H is not None:
            self.last_H = H
            self.last_H_info = H_info

        # --------------------------------------------------------------
        # 3. Player detection & projection (with ByteTrack tracking)
        # --------------------------------------------------------------
        # Use model.track() with persist=True for cross-frame track IDs
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
            # Fix left-right flip if camera is on opposite side
            if self.flip_projection_x and len(player_pitch_pts) > 0:
                player_pitch_pts[:, 0] = PITCH_LENGTH - player_pitch_pts[:, 0]

        # --------------------------------------------------------------
        # 3b. Team color analysis (with per-track majority voting)
        # --------------------------------------------------------------
        team_info = None
        if self.team_analyzer is not None and len(player_xyxy) > 0:
            team_info = self.team_analyzer.assign_team_colors(
                frame, player_xyxy, player_conf, track_ids=track_ids
            )

        # --------------------------------------------------------------
        # 4. Pitch canvas (top-down view)
        # --------------------------------------------------------------
        pitch_canvas = self.pitch_artist.draw_pitch_base()
        if len(player_pitch_pts) > 0:
            # Filter points that are within pitch bounds
            valid_mask = (
                (player_pitch_pts[:, 0] >= -5)
                & (player_pitch_pts[:, 0] <= PITCH_LENGTH + 5)
                & (player_pitch_pts[:, 1] >= -5)
                & (player_pitch_pts[:, 1] <= PITCH_WIDTH + 5)
            )
            valid_pts = player_pitch_pts[valid_mask]
            if len(valid_pts) > 0:
                # Get per-player team colors for valid points only
                colors = None
                if team_info is not None:
                    colors = team_info['team_colors']
                    # Filter colors to match valid_mask
                    colors = [colors[i] for i in range(len(colors)) if valid_mask[i]] if len(colors) == len(valid_mask) else None

                pitch_canvas = self.pitch_artist.draw_players_on_pitch(
                    pitch_canvas, valid_pts, colors=colors, default_color=(0, 0, 255)
                )

            # Add team legend
            if team_info is not None:
                pitch_canvas = self.pitch_artist.draw_team_legend(
                    pitch_canvas,
                    team_info['team1_bgr'],
                    team_info['team2_bgr'],
                    team1_label="Team 1",
                    team2_label="Team 2",
                )

        # --------------------------------------------------------------
        # 5. Annotated frames with keypoints + team-colored bboxes
        # --------------------------------------------------------------
        used_kpts = H_info.get('used_keypoints', [])
        annotated_frame = self._draw_keypoints_on_frame(frame, used_kpts)
        # Overlay team-colored player bounding boxes with confidence scores
        if team_info is not None and len(player_xyxy) > 0:
            annotated_frame = self._draw_team_bboxes(
                annotated_frame, player_xyxy, team_info['team_colors'],
                player_conf=player_conf,
            )

        # Deep analysis frame with seg overlay + keypoints + team bboxes
        if used_kpts:
            deep_analysis_frame = self._draw_keypoints_on_frame(
                seg_overlay_frame, used_kpts
            )
        else:
            deep_analysis_frame = seg_overlay_frame
        if team_info is not None and len(player_xyxy) > 0:
            deep_analysis_frame = self._draw_team_bboxes(
                deep_analysis_frame, player_xyxy, team_info['team_colors'],
                player_conf=player_conf,
            )

        return {
            'H': H,
            'H_info': H_info,
            'player_xyxy': player_xyxy,
            'player_conf': player_conf,
            'track_ids': track_ids,
            'player_pitch_pts': player_pitch_pts,
            'keypoints_used': used_kpts,
            'seg_result': seg_op,
            'processed_segments': processed_segments,
            'pitch_canvas': pitch_canvas,
            'annotated_frame': annotated_frame,
            'deep_analysis_frame': deep_analysis_frame,
            'team_info': team_info,
        }

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    # Keypoint skeleton connections (from reference Soccer_Analysis repo)
    _KPT_CONNECTIONS = [
        # Field boundary
        (0, 16),   # top-left → top-right (top sideline)
        (0, 9),    # top-left → bottom-left (left sideline)
        (16, 25),  # top-right → bottom-right (right sideline)
        (9, 25),   # bottom-left → bottom-right (bottom sideline)
        # Left penalty area
        (1, 2),    # top edge
        (3, 4),    # bottom edge
        (1, 3),    # outer vertical (sideline side)
        (2, 4),    # inner vertical (center side)
        # Left goal area
        (5, 6),    # top edge
        (7, 8),    # bottom edge
        (5, 7),    # outer vertical
        (6, 8),    # inner vertical
        # Right penalty area
        (17, 18),  # top edge
        (19, 20),  # bottom edge
        (17, 19),  # outer vertical (sideline side)
        (18, 20),  # inner vertical (center side)
        # Right goal area
        (21, 22),  # top edge
        (23, 24),  # bottom edge
        (21, 23),  # outer vertical
        (22, 24),  # inner vertical
        # Center line & circle
        (11, 12),  # center line
        (13, 14),  # center circle vertical
    ]

    def _draw_keypoints_on_frame(self, frame, used_keypoints, radius=6):
        """
        Draw used keypoints on the frame with labels and skeleton connections.
        Also builds a lookup for fast skeleton drawing.
        """
        out = frame.copy()

        # Build dict: kpt_id → (x, y) for used keypoints
        kpt_positions = {}
        for kp in used_keypoints:
            kid = kp['kpt_id']
            kpt_positions[kid] = (int(kp['image_pt'][0]), int(kp['image_pt'][1]))

        # Draw skeleton connections between keypoints
        for start_id, end_id in self._KPT_CONNECTIONS:
            if start_id in kpt_positions and end_id in kpt_positions:
                cv2.line(out, kpt_positions[start_id], kpt_positions[end_id],
                         (0, 255, 255), 2, cv2.LINE_AA)

        # Draw keypoint circles + labels
        for kp in used_keypoints:
            x, y = int(kp['image_pt'][0]), int(kp['image_pt'][1])
            # Draw circle
            cv2.circle(out, (x, y), radius, (0, 255, 255), -1)
            cv2.circle(out, (x, y), radius + 2, (255, 255, 0), 2)
            # Label
            label = f"{kp['kpt_id']}:{kp['name'][:15]}"
            cv2.putText(
                out, label, (x + 10, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out

    def _create_seg_overlay(self, frame, seg_op, processed_segments):
        """Create a frame with segmentation masks overlaid."""
        # We use a simple approach: draw filled polygons for each segment
        out = frame.copy()
        overlay = frame.copy()

        for segment in processed_segments:
            contour = segment.get('image_contour', None)
            if contour is None:
                continue
            # Draw contour fill (semi-transparent)
            color_map = {
                '18Yard': (255, 0, 0),
                '18Yard Circle': (0, 255, 0),
                '5Yard': (0, 0, 255),
                'Half Central Circle': (255, 255, 0),
                'Half Field': (255, 0, 255),
            }
            color = color_map.get(segment['class_name'], (128, 128, 128))
            cv2.drawContours(overlay, [contour], -1, color, -1)

        # Blend
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # Add labels
        for segment in processed_segments:
            bbox = segment.get('image_bbox', None)
            if bbox is not None:
                cx = float(bbox[:, 0].mean())
                cy = float(bbox[:, 1].mean())
                label = f"{segment['class_name']} {segment['confidence']:.2f}"
                cv2.putText(
                    out, label, (int(cx) - 30, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
                )

        return out

    def _draw_team_bboxes(
        self,
        frame: np.ndarray,
        player_xyxy: np.ndarray,
        team_colors: list,
        player_conf: np.ndarray = None,
    ) -> np.ndarray:
        """
        Draw team-colored bounding boxes around detected players, with
        confidence scores displayed above each bbox.

        Args:
            frame: BGR image (H, W, 3).
            player_xyxy: (N, 4) array of bboxes in [x1, y1, x2, y2].
            team_colors: List of N BGR tuples, one per player.
            player_conf: (N,) array of confidence scores for each player.
                         If None, no confidence text is drawn.

        Returns:
            Frame with bounding boxes and confidence labels drawn.
        """
        out = frame.copy()
        n = min(len(player_xyxy), len(team_colors))
        for i in range(n):
            x1, y1, x2, y2 = map(int, player_xyxy[i])
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)

            color = team_colors[i]
            # Draw filled rectangle with low opacity for the bbox
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
            # Draw border
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Draw confidence score above the bounding box
            if player_conf is not None and i < len(player_conf):
                conf = player_conf[i]
                label = f"{conf:.2f}"
                # Choose text color: bright white with a dark outline for readability
                text_color = (255, 255, 255)
                outline_color = (0, 0, 0)
                font_scale = 0.5
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                # Position: centered above the bbox top edge
                tx = (x1 + x2 - tw) // 2
                ty = y1 - 5  # 5px above the bbox

                # If the text would go above the frame, place it just inside the bbox top
                if ty - th < 0:
                    ty = y1 + th + 2

                # Draw text outline for readability
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(
                        out, label, (tx + dx, ty + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness, cv2.LINE_AA,
                    )
                # Draw the actual text
                cv2.putText(
                    out, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA,
                )

        return out

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    def process_video(
        self,
        source_video_path: str,
        output_dir: str = "output",
        fps: float = 30.0,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        process_every_n: int = 1,
    ):
        """
        Process an entire video and produce output videos.

        Outputs:
            - full_pitch_debug_map.mp4: Top-down pitch view
            - annotated_video.mp4: Original frame with keypoints + team bboxes
            - deep_analysis.mp4: Original frame with segmentation overlay
            - final_draft.mp4: Original frame with pitch debug map overlayed (PIP)

        Args:
            source_video_path: Path to input video.
            output_dir: Directory for output files.
            fps: Output video FPS.
            start_frame: Frame index to start from.
            max_frames: Maximum frames to process (None = all).
            process_every_n: Process every Nth frame.

        Yields:
            dict results from process_frame() for each processed frame.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        full_pitch_path = output_path / "full_pitch_debug_map.mp4"
        annotated_path = output_path / "annotated_video.mp4"
        deep_analysis_path = output_path / "deep_analysis.mp4"
        final_draft_path = output_path / "final_draft.mp4"

        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source_video_path}")

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {actual_fps:.1f}FPS, {frame_w}x{frame_h}, {total_frames} frames")
        print(f"Processing from frame {start_frame}, max_frames={max_frames}")

        # Video writers
        pitch_writer = Director.make_video_writer(
            full_pitch_path, actual_fps, (CANVAS_W, CANVAS_H)
        )
        annotated_writer = Director.make_video_writer(
            annotated_path, actual_fps, (frame_w, frame_h)
        )
        deep_writer = Director.make_video_writer(
            deep_analysis_path, actual_fps, (frame_w, frame_h)
        )
        final_draft_writer = Director.make_video_writer(
            final_draft_path, actual_fps, (frame_w, frame_h)
        )

        frame_idx = 0
        processed_count = 0

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

                # Write outputs
                if result['pitch_canvas'] is not None:
                    pitch_writer.write(result['pitch_canvas'])
                if result['annotated_frame'] is not None:
                    annotated_writer.write(result['annotated_frame'])
                if result['deep_analysis_frame'] is not None:
                    deep_writer.write(result['deep_analysis_frame'])

                # Build final_draft frame: overlay pitch canvas onto original frame
                if result['pitch_canvas'] is not None:
                    # Make a copy of the raw (non-annotated) original frame
                    final_draft_frame = frame.copy()
                    # Resize pitch canvas to ~25% of frame width, place bottom-right
                    pip_width = frame_w // 4
                    pip_height = int(pip_width * CANVAS_H / CANVAS_W)
                    pip_canvas = cv2.resize(result['pitch_canvas'], (pip_width, pip_height))
                    # Overlay with a semi-transparent border background
                    x_offset = frame_w - pip_width - 15
                    y_offset = frame_h - pip_height - 15
                    # Dark semi-transparent background behind the PIP
                    overlay = final_draft_frame.copy()
                    cv2.rectangle(overlay,
                        (x_offset - 5, y_offset - 5),
                        (x_offset + pip_width + 5, y_offset + pip_height + 5),
                        (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.4, final_draft_frame, 0.6, 0, final_draft_frame)
                    # Paste the resized pitch canvas
                    final_draft_frame[y_offset:y_offset + pip_height, x_offset:x_offset + pip_width] = pip_canvas
                    # Add a white border
                    cv2.rectangle(final_draft_frame,
                        (x_offset - 2, y_offset - 2),
                        (x_offset + pip_width + 2, y_offset + pip_height + 2),
                        (255, 255, 255), 2)
                    final_draft_writer.write(final_draft_frame)

                # Log every 30 frames
                if processed_count % 30 == 0:
                    mode = result['H_info'].get('mode', '?')
                    n_kpts = len(result['H_info'].get('used_keypoints', []))
                    n_players = len(result['player_pitch_pts'])
                    print(
                        f"  Frame {frame_idx}: H={mode}, "
                        f"kpts={n_kpts}, players={n_players}"
                    )

                yield result
                frame_idx += 1

        finally:
            cap.release()
            pitch_writer.release()
            annotated_writer.release()
            deep_writer.release()
            final_draft_writer.release()
            print(f"\nDone. Processed {processed_count} frames.")
            print(f"  Pitch canvas:  {full_pitch_path}")
            print(f"  Annotated:     {annotated_path}")
            print(f"  Deep analysis: {deep_analysis_path}")
            print(f"  Final draft:   {final_draft_path}")