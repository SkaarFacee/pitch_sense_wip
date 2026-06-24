"""KeypointPipeline — End-to-end pipeline: keypoint→homography→player projection→team colors→ball→segmentation→5 output videos."""
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from constants import (
    PLAYER_CONF, SEG_CONF, CANVAS_W, CANVAS_H, PITCH_LENGTH, PITCH_WIDTH,
    CENTER_X, CENTER_Y,
    BALL_CONF, BALL_TRAIL_LENGTH, BALL_BBOX_COLOR, BALL_DOT_COLOR,
    TEAM0, TEAM1, NO_TEAM, ROLE_UNKNOWN, ROLE_OUTFIELD, ROLE_GK, ROLE_REF,
)

# How many frames the "PASS DETECTED" banner stays on screen after a pass.
PASS_FLASH_FRAMES = 30

# Minimum pitch-distance between confirmed stable owners. This is kept low
# enough for short kickoff/build-up passes; flicker is handled by the owner
# stability gates below rather than by discarding all short passes.
MIN_PASS_DISTANCE_M = 1.5

# Minimum ball travel accumulated while ownership is being transferred.
# The older single-frame displacement gate was too sensitive to bbox and
# homography jitter; pass validation now measures travel across the whole
# release/receiver window instead.
MIN_PASS_BALL_TRAVEL_M = 1.5

# Ownership must be stable before it can start or receive a pass. These gates
# suppress one-frame bbox-overlap flicker in crowded frames.
MIN_PASSER_STABLE_FRAMES = 3
MIN_RECEIVER_STABLE_FRAMES = 2

# Bounds for retaining an in-flight candidate through short no-overlap or
# missing-ball gaps without comparing against stale ownership later.
MAX_PASS_TRANSIT_FRAMES = 45
MAX_PASS_BALL_MISSING_FRAMES = 15
PASS_COOLDOWN_FRAMES = 12

# Defensive-line drawing: smoothing factor for the running estimate of
# each team's mean pitch-X. A small alpha makes the "which goal does this
# team defend" decision stable across frame-to-frame jitter so the two
# defensive lines don't flip sides when players momentarily cross the
# halfway line.
TEAM_X_EMA_ALPHA = 0.04
from keypoint_service import KeypointHomographyComputer
from player_service import PlayerDetector
from ball_service import BallDetector
from segmentation import Segmentor
from pitch import PitchArtist
from director import Director
from team_analyzer import TeamColorAnalyzer
from team_stabilizer import TeamCalibration, TeamSequenceStabilizer, validate_frame_team_infos


class KeypointPipeline:
    def __init__(self, keypoint_model_path: str, player_model_path: str, seg_model_path: str,
                 ball_model_path: str = "", flip_projection_x: bool = False,
                 flip_projection_y: bool = False, enable_team_colors: bool = True):
        self.keypoint_computer = KeypointHomographyComputer(keypoint_model_path)
        self.player_detector = PlayerDetector(player_model_path)
        self.segmentor = Segmentor(seg_model_path)
        self.team_analyzer = TeamColorAnalyzer() if enable_team_colors else None
        self.ball_detector = BallDetector(ball_model_path, conf=BALL_CONF) if ball_model_path else None
        self.last_H = None
        self.pitch_artist = PitchArtist()
        self.flip_projection_x = flip_projection_x
        self.flip_projection_y = flip_projection_y
        self.ball_trajectory = []
        self._reset_pass_state()
        # Running EMA of each team's mean pitch-X. Used ONLY to decide
        # which goal each team defends for the defensive-line overlay —
        # far more stable than a per-frame mean, which flickers as players
        # cross the halfway line.
        self._team_x_ema: dict[int, Optional[float]] = {0: None, 1: None}

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0):
        frame_h, frame_w = frame.shape[:2]
        processed_segments = []
        seg_overlay_frame = frame.copy()

        # 1. Segmentation
        seg_op = self.segmentor.model.predict(frame, conf=SEG_CONF, verbose=False)[0]
        if seg_op is not None and getattr(seg_op, 'masks', None) is not None:
            processed_segments = self.segmentor.extract(seg_op, frame_w, last_side=None)
            seg_overlay_frame = self._create_seg_overlay(frame, seg_op, processed_segments)

        # 1b. Build a binary pitch mask from the segmentation contours so
        # off-pitch detections (referees in the crowd, false positives in
        # the stands, etc.) can be filtered before they're used for
        # possession, passing, or display. Falls back to None when no
        # segmentation is available — callers treat None as "no filter".
        pitch_mask = self._build_pitch_mask(processed_segments, frame_h, frame_w)

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

        # 3a.1 Segmentation-based filtering — drop any player detection
        # whose lower-torso anchor (closer to the feet, where the player
        # actually touches the pitch) falls OUTSIDE the segmented pitch
        # mask. This prevents false positives in the stands, advertising
        # boards, or background from contaminating possession / passing /
        # defensive-line analytics and the on-screen overlay.
        if pitch_mask is not None and len(player_xyxy) > 0:
            keep = self._filter_bboxes_by_mask(
                player_xyxy, pitch_mask, anchor="lower_torso",
            )
            if len(keep) < len(player_xyxy):
                orig_n = len(player_xyxy)
                player_xyxy = player_xyxy[keep]
                if len(player_conf) == orig_n:
                    player_conf = player_conf[keep]
                else:
                    player_conf = player_conf[:0]
                if len(track_ids) == orig_n:
                    track_ids = track_ids[keep]
                else:
                    track_ids = track_ids[:0]

        player_pitch_pts = np.empty((0, 2), dtype=np.float32)
        if H is not None and len(player_xyxy) > 0:
            player_pitch_pts = self.player_detector.project_points(player_xyxy, H)
            if self.flip_projection_x and len(player_pitch_pts) > 0:
                player_pitch_pts[:, 0] = PITCH_LENGTH - player_pitch_pts[:, 0]
            if self.flip_projection_y and len(player_pitch_pts) > 0:
                player_pitch_pts[:, 1] = PITCH_WIDTH - player_pitch_pts[:, 1]

        # 3b. Team colors
        team_info = None
        if self.team_analyzer is not None and len(player_xyxy) > 0:
            team_info = self.team_analyzer.assign_team_colors(
                frame, player_xyxy, player_conf, track_ids=track_ids, H=H,
                player_pitch_pts=player_pitch_pts if len(player_pitch_pts) > 0 else None,
            )

        # 3c. Ball detection
        ball_xyxy = np.empty((0, 4), dtype=np.float32)
        ball_conf = np.empty((0,), dtype=np.float32)
        ball_pitch_pt = None
        if self.ball_detector is not None:
            ball_xyxy, ball_conf = self.ball_detector.detect_ball(frame)

            # 3c.1 Segmentation-based filtering — discard any ball detection
            # whose center is outside the segmented pitch mask.
            if pitch_mask is not None and len(ball_xyxy) > 0:
                if not self._bbox_center_in_mask(ball_xyxy[0], pitch_mask):
                    ball_xyxy = np.empty((0, 4), dtype=np.float32)
                    ball_conf = np.empty((0,), dtype=np.float32)

            if len(ball_xyxy) > 0 and H is not None:
                ball_pitch_pt = self.ball_detector.project_ball_to_pitch(
                    ball_xyxy, H,
                    flip_x=self.flip_projection_x,
                    flip_y=self.flip_projection_y,
                    pitch_length=PITCH_LENGTH,
                    pitch_width=PITCH_WIDTH,
                )
                self.ball_trajectory.append(ball_pitch_pt.copy())
                if len(self.ball_trajectory) > BALL_TRAIL_LENGTH:
                    self.ball_trajectory.pop(0)

        # 4. Pitch canvas (top-down)
        pitch_canvas = self.pitch_artist.draw_pitch_base()
        if processed_segments:
            pitch_canvas = self.pitch_artist.draw_seg_zones(pitch_canvas, processed_segments, alpha=0.25)
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

        # 4b. Pass detection. Keep online processing on the same stable
        # ownership state machine used by the two-pass video renderer.
        tids_arr = np.asarray(track_ids) if len(track_ids) > 0 else np.empty((0,), dtype=np.int32)
        team_ids_arr = (np.asarray(team_info['team_ids'])
                        if (team_info is not None
                            and 'team_ids' in team_info
                            and len(team_info['team_ids']) > 0)
                        else np.empty((0,), dtype=np.int32))
        role_ids_arr = (np.asarray(team_info.get('role_ids', []), dtype=np.int32)
                        if (team_info is not None
                            and len(team_info.get('role_ids', [])) > 0)
                        else np.full(len(team_ids_arr), ROLE_UNKNOWN, dtype=np.int32))
        pass_event = self._update_pass_state(
            ball_xyxy=ball_xyxy,
            player_xyxy=player_xyxy,
            tids_arr=tids_arr,
            owner_ids_arr=tids_arr,
            team_ids_arr=team_ids_arr,
            player_pitch_pts=player_pitch_pts,
            ball_pitch_pt=ball_pitch_pt,
        )

        # 5. Annotated frames
        used_kpts = H_info.get('used_keypoints', [])
        annotated_frame = self._draw_keypoints_on_frame(frame, used_kpts)
        if team_info is not None and len(player_xyxy) > 0:
            annotated_frame = self._draw_team_bboxes(annotated_frame, player_xyxy,
                                                    team_info['team_colors'],
                                                    player_conf=player_conf,
                                                    team_ids=team_info['team_ids'],
                                                    role_ids=team_info.get('role_ids'))
        if len(ball_xyxy) > 0:
            annotated_frame = self._draw_ball_bbox(annotated_frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR)
        deep_analysis_frame = self._draw_keypoints_on_frame(seg_overlay_frame, used_kpts) if used_kpts else seg_overlay_frame
        if team_info is not None and len(player_xyxy) > 0:
            deep_analysis_frame = self._draw_team_bboxes(deep_analysis_frame, player_xyxy,
                                                       team_info['team_colors'],
                                                       player_conf=player_conf,
                                                       team_ids=team_info['team_ids'],
                                                       role_ids=team_info.get('role_ids'),
                                                       track_ids=track_ids)
        if len(ball_xyxy) > 0:
            deep_analysis_frame = self._draw_ball_bbox(deep_analysis_frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR)
        # Defensive-line overlay: project each team's deepest outfield
        # player back into image coords with H⁻¹ and draw a horizontal
        # line at that depth so the viewer can see the defensive line
        # shift over the match.
        if team_info is not None and H is not None and len(player_pitch_pts) > 0:
            deep_analysis_frame = self._draw_defensive_lines(
                deep_analysis_frame, H, tids_arr, team_ids_arr, player_pitch_pts,
                team_info.get("team1_bgr", (255, 0, 0)),
                team_info.get("team2_bgr", (0, 0, 255)),
                role_ids_arr=role_ids_arr,
            )
        # Pass flash overlay — drawn LAST so it sits on top of every other
        # annotation. Only on the Deep Analysis video (per the request).
        if self._pass_flash_counter > 0:
            deep_analysis_frame = self._draw_pass_flash(
                deep_analysis_frame, self._last_pass_info, self._pass_flash_counter,
                PASS_FLASH_FRAMES,
            )

        return {'H': H, 'H_info': H_info, 'player_xyxy': player_xyxy, 'player_conf': player_conf,
                'track_ids': track_ids, 'player_pitch_pts': player_pitch_pts, 'keypoints_used': used_kpts,
                'seg_result': seg_op, 'processed_segments': processed_segments, 'pitch_canvas': pitch_canvas,
                'annotated_frame': annotated_frame, 'deep_analysis_frame': deep_analysis_frame, 'team_info': team_info,
                'ball_xyxy': ball_xyxy, 'ball_conf': ball_conf, 'ball_pitch_pt': ball_pitch_pt,
                'ball_trajectory': list(self.ball_trajectory), 'pass_event': pass_event,
                'pitch_mask': pitch_mask}

    @staticmethod
    def _find_ball_owner_by_bbox_overlap(ball_xyxy: np.ndarray,
                                         player_xyxy: np.ndarray
                                         ) -> tuple:
        """Find the player whose bounding box overlaps the ball bounding box.

        Per spec: a pass is a transition where the ball bbox overlaps a
        player bbox. Ball-ownership for the current frame is therefore
        resolved by BBOX overlap (NOT by nearest pitch-space distance).

        Strategy (preference order):
          1. Player bbox that contains the ball-bbox center → strongest.
          2. Otherwise, the player bbox with the largest pixel-area
             intersection with the ball bbox (rare; happens when the ball
             is partially behind a player).

        Args:
            ball_xyxy: (1, 4) or (N, 4) ball bbox(es) from the ball model.
            player_xyxy: (M, 4) player bboxes from the player model.

        Returns:
            (idx, score) — the player index into ``player_xyxy`` whose
            bbox best matches, and the overlap score. ``(None, 0.0)`` is
            returned when the arrays are empty or no bbox overlap exists.
        """
        if len(ball_xyxy) == 0 or len(player_xyxy) == 0:
            return None, 0.0
        bx1 = float(ball_xyxy[0, 0]); by1 = float(ball_xyxy[0, 1])
        bx2 = float(ball_xyxy[0, 2]); by2 = float(ball_xyxy[0, 3])
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        best_idx: Optional[int] = None
        best_score = 0.0
        for i in range(len(player_xyxy)):
            px1 = float(player_xyxy[i, 0]); py1 = float(player_xyxy[i, 1])
            px2 = float(player_xyxy[i, 2]); py2 = float(player_xyxy[i, 3])
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                return i, float("inf")
            ix1 = max(bx1, px1); iy1 = max(by1, py1)
            ix2 = min(bx2, px2); iy2 = min(by2, py2)
            if ix2 > ix1 and iy2 > iy1:
                score = (ix2 - ix1) * (iy2 - iy1)
                if score > best_score:
                    best_score = score
                    best_idx = i
        return best_idx, best_score

    @staticmethod
    def _ball_overlaps_any_player(ball_xyxy: np.ndarray,
                                  player_xyxy: np.ndarray) -> bool:
        """Cheap boolean check: does the ball bbox overlap ANY player bbox?

        Used as an extra gate by analytics that want to confirm the ball
        is "in play" near a player (e.g. possession / Voronoi). Mirrors
        the containment rule used by ``_find_ball_owner_by_bbox_overlap``.
        """
        if len(ball_xyxy) == 0 or len(player_xyxy) == 0:
            return False
        bx1 = float(ball_xyxy[0, 0]); by1 = float(ball_xyxy[0, 1])
        bx2 = float(ball_xyxy[0, 2]); by2 = float(ball_xyxy[0, 3])
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        for i in range(len(player_xyxy)):
            px1 = float(player_xyxy[i, 0]); py1 = float(player_xyxy[i, 1])
            px2 = float(player_xyxy[i, 2]); py2 = float(player_xyxy[i, 3])
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                return True
            if not (bx2 < px1 or px2 < bx1 or by2 < py1 or py2 < by1):
                return True
        return False

    @staticmethod
    def _build_pitch_mask(processed_segments, frame_h: int, frame_w: int
                          ) -> Optional[np.ndarray]:
        """Build a binary mask of the pitch from segmentation contours.

        All class contours (18Yard, 5Yard, Half Central Circle, 18Yard
        Circle, Half Field) are unioned and the result is dilated by a
        small kernel so a player standing right at the touchline isn't
        spuriously excluded. Returns ``None`` when no segmentation
        output is available — callers treat ``None`` as "no filter".
        """
        if not processed_segments:
            return None
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        any_drawn = False
        for seg in processed_segments:
            contour = seg.get("image_contour")
            if contour is None or len(contour) < 3:
                continue
            cv2.drawContours(mask, [contour], -1, 255, -1)
            any_drawn = True
        if not any_drawn:
            return None
        # Dilate so a player whose feet are right at the touchline (but
        # whose bbox slightly overhangs it) is not spuriously filtered.
        kernel = np.ones((15, 15), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    @staticmethod
    def _bbox_center_in_mask(xyxy: np.ndarray, mask: np.ndarray) -> bool:
        """Return True iff the bbox center lies inside ``mask``.

        Out-of-bounds centers are treated as off-pitch (return False).
        """
        if mask is None or len(xyxy) < 4:
            return True
        cx = (float(xyxy[0]) + float(xyxy[2])) / 2.0
        cy = (float(xyxy[1]) + float(xyxy[3])) / 2.0
        ix, iy = int(round(cx)), int(round(cy))
        if ix < 0 or iy < 0 or ix >= mask.shape[1] or iy >= mask.shape[0]:
            return False
        return bool(mask[iy, ix] > 0)

    @staticmethod
    def _filter_bboxes_by_mask(xyxy: np.ndarray, mask: np.ndarray,
                               anchor: str = "center") -> np.ndarray:
        """Return indices of bboxes whose anchor point is inside ``mask``.

        ``anchor="center"`` uses the bbox center (ball convention).
        ``anchor="lower_torso"`` uses a point 30% up from the bbox bottom
        (player convention — closer to the player's feet, which is where
        they actually touch the pitch and where the homography projects
        them).
        """
        if mask is None or len(xyxy) == 0:
            return np.arange(len(xyxy), dtype=np.int32)
        keep: list[int] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(xyxy[i, 0]), float(xyxy[i, 1]),
                               float(xyxy[i, 2]), float(xyxy[i, 3]))
            if anchor == "lower_torso":
                cx = (x1 + x2) / 2.0
                cy = y1 + 0.70 * (y2 - y1)
            else:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]:
                if mask[iy, ix] > 0:
                    keep.append(i)
        return np.array(keep, dtype=np.int32)

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

    _TEAM_ID_LABELS = {TEAM0: "T1", TEAM1: "T2", NO_TEAM: ""}

    def _draw_team_bboxes(self, frame, player_xyxy, team_colors,
                          player_conf=None, team_ids=None, role_ids=None,
                          track_ids=None):
        out = frame.copy()
        h, w = frame.shape[:2]
        for i in range(min(len(player_xyxy), len(team_colors))):
            x1, y1, x2, y2 = max(0, int(player_xyxy[i][0])), max(0, int(player_xyxy[i][1])), min(w - 1, int(player_xyxy[i][2])), min(h - 1, int(player_xyxy[i][3]))
            color = team_colors[i]
            ov = out.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, 0.25, out, 0.75, 0, out)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Build the label from membership + role: "T1", "T2",
            # "T1 GK", "T2 GK", "REF" (+ tid + conf).
            team_label = ""
            role = ROLE_UNKNOWN
            if role_ids is not None and i < len(role_ids):
                role = int(role_ids[i])
            if team_ids is not None and i < len(team_ids):
                team_id = int(team_ids[i])
                if role == ROLE_REF:
                    team_label = "REF"
                elif team_id == TEAM0:
                    team_label = "T1 GK" if role == ROLE_GK else "T1"
                elif team_id == TEAM1:
                    team_label = "T2 GK" if role == ROLE_GK else "T2"
                elif role == ROLE_GK:
                    team_label = "GK"
            tid_txt = ""
            if track_ids is not None and i < len(track_ids):
                tid_txt = f"#{int(track_ids[i])}"
            conf_txt = ""
            if player_conf is not None and i < len(player_conf):
                conf_txt = f"{float(player_conf[i]):.2f}"
            parts = [p for p in (team_label, tid_txt, conf_txt) if p]
            tag = " ".join(parts)
            if tag:
                # Filled black backing for legibility on any background.
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tx, ty = x1, max(y1 - 6, th + 4)
                cv2.rectangle(out, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2),
                              (0, 0, 0), -1)
                cv2.putText(out, tag, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            tuple(int(c) for c in color), 2, cv2.LINE_AA)
        return out

    def _draw_pass_flash(self, frame, pass_info: dict, remaining: int,
                         total: int):
        """Banner overlay shown on the Deep Analysis video for ~total
        frames after a pass is detected. Fades out as remaining → 0.
        """
        if not pass_info:
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
        # Fade in then out: alpha rises for the first 1/3 of the window,
        # then falls back to 0.
        progress = remaining / max(total, 1)
        if progress > 2.0 / 3.0:
            alpha = 1.0 - (progress - 2.0 / 3.0) / (1.0 / 3.0) * 0.3
        else:
            alpha = 0.3 + (progress / (2.0 / 3.0)) * 0.7
        alpha = float(max(0.0, min(1.0, alpha)))

        team_name = "T1" if int(pass_info.get("team", 0)) == 0 else "T2"
        text = (f"PASS DETECTED  ·  {team_name}  ·  "
                f"#{int(pass_info.get('from_tid', 0))} → "
                f"#{int(pass_info.get('to_tid', 0))}  ·  "
                f"{float(pass_info.get('distance_m', 0)):.1f} m")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        pad_x, pad_y = 16, 10
        box_w = tw + 2 * pad_x
        box_h = th + 2 * pad_y
        box_x = (w - box_w) // 2
        box_y = 18

        overlay = out.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55 * alpha, out, 1.0 - 0.55 * alpha, 0, out)
        # Bright yellow text + border so it reads at a glance.
        cv2.rectangle(out, (box_x, box_y), (box_x + box_w, box_y + box_h),
                      (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, text, (box_x + pad_x, box_y + pad_y + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return out

    def _draw_ball_bbox(self, frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR):
        if len(ball_xyxy) == 0:
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
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

    def _draw_defensive_lines(self, frame, H, tids_arr, team_ids_arr,
                              player_pitch_pts, team1_color, team2_color,
                              role_ids_arr=None):
        """Draw each team's defensive line as a perspective-correct line
        running ACROSS the pitch width at the depth of that team's deepest
        outfield defender.

        Args:
            frame: Deep-analysis frame to annotate.
            H: 3x3 homography mapping pitch (meters) → image (pixels).
            tids_arr: per-detection track_ids (parallel to player_pitch_pts).
            team_ids_arr: per-detection team membership labels (0, 1, NO_TEAM).
            role_ids_arr: per-detection role labels; GK/ref are excluded.
            player_pitch_pts: (N, 2) pitch coordinates (already flip-adjusted
                by ``process_frame`` to match the top-down canvas).
            team1_color, team2_color: BGR team colours.

        Method:
          * Goalkeepers and referees are excluded by ``role_ids_arr`` so the
            line reflects the back four / five, not the keeper.
          * Each team's defending goal is decided from a SMOOTHED running
            estimate of its mean pitch-X (``self._team_x_ema``) so the two
            lines stay on opposite halves instead of flipping when a player
            briefly crosses the halfway line.
          * The "deepest defender" is the outfield player nearest the goal
            the team defends: SMALLEST pitch-X when defending the left goal,
            LARGEST pitch-X when defending the right.
          * The line is drawn as a purely VERTICAL bar in image space at
            the depth of the deepest defender. The depth is measured in
            pitch metres, projected to a single pixel-x via the pitch
            midline (``line_x, PITCH_WIDTH/2``) → H⁻¹, then a vertical
            segment is drawn at that pixel-x spanning the frame height.
            This keeps the line upright instead of perspective-slanted.
            Because ``player_pitch_pts`` were flipped for display, the
            projection first UN-FLIPS so H⁻¹ receives the raw pitch
            coordinates it maps.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return out

        if role_ids_arr is None or len(role_ids_arr) == 0:
            role_ids_arr = np.full(len(team_ids_arr), ROLE_OUTFIELD, dtype=np.int32)
        role_ids_arr = np.asarray(role_ids_arr, dtype=np.int32)
        n = min(len(tids_arr), len(team_ids_arr), len(role_ids_arr), len(player_pitch_pts))
        if n == 0:
            return out

        def _pitch_to_image(px: float, py: float):
            """Map a (possibly flipped) pitch point to image pixels.

            ``player_pitch_pts`` are flipped to match the top-down canvas,
            but H⁻¹ expects the raw projection space, so un-flip first.
            Returns ``(x, y)`` floats or ``None`` if the point projects to
            (or behind) the camera horizon.
            """
            rx, ry = px, py
            if self.flip_projection_x:
                rx = PITCH_LENGTH - rx
            if self.flip_projection_y:
                ry = PITCH_WIDTH - ry
            homog = np.array([[rx, ry, 1.0]], dtype=np.float32)
            proj = homog @ H_inv.T
            wz = float(proj[0, 2])
            if abs(wz) < 1e-6:
                return None
            return (float(proj[0, 0] / wz), float(proj[0, 1] / wz))

        def _deepest_defender(team_id: int, defending_left: bool):
            """Return (pitch_x, track_id) for the team's deepest on-pitch
            outfield player, or None."""
            best_x = None
            best_track = None
            for i in range(n):
                if int(team_ids_arr[i]) != team_id:
                    continue
                if int(role_ids_arr[i]) in (ROLE_GK, ROLE_REF):
                    continue
                pt = player_pitch_pts[i]
                if not (-2 <= pt[0] <= PITCH_LENGTH + 2
                        and -2 <= pt[1] <= PITCH_WIDTH + 2):
                    continue
                px = float(pt[0])
                if best_x is None:
                    best_x, best_track = px, int(tids_arr[i])
                elif defending_left and px < best_x:
                    best_x, best_track = px, int(tids_arr[i])
                elif (not defending_left) and px > best_x:
                    best_x, best_track = px, int(tids_arr[i])
            if best_x is None:
                return None
            return (best_x, best_track)

        # Update the smoothed mean-X estimate for each team, then decide
        # which goal each team defends. The smaller smoothed mean-X defends
        # the left goal.
        for team_id in (0, 1):
            self._update_team_x_ema(team_id, team_ids_arr, player_pitch_pts, n,
                                    role_ids_arr=role_ids_arr)
        t1_defends_left = self._team_defends_left(0)

        for team_id, color, tag, defends_left in (
            (0, team1_color, "T1", t1_defends_left),
            (1, team2_color, "T2", not t1_defends_left),
        ):
            picked = _deepest_defender(team_id, defends_left)
            if picked is None:
                continue
            line_x, line_track = picked

            # Project the depth at the pitch midline and draw a purely
            # VERTICAL line in image space (constant pixel-x) at that depth.
            # Using a single projection point (mid-width) avoids the
            # perspective slant produced by connecting the two goal-line
            # endpoints, which is what the viewer asked to remove.
            p_mid = _pitch_to_image(line_x, PITCH_WIDTH * 0.5)
            if p_mid is None:
                continue
            vx = int(round(p_mid[0]))
            if not (0 <= vx < w):
                continue

            pt1 = (vx, 0)
            pt2 = (vx, h - 1)

            # Clip the (possibly off-screen) segment to the frame rectangle.
            inside, cp1, cp2 = cv2.clipLine((0, 0, w, h), pt1, pt2)
            if not inside:
                continue

            color_t = tuple(int(c) for c in color)
            cv2.line(out, cp1, cp2, color_t, 3, cv2.LINE_AA)

            # Label anchored at the top of the vertical line, nudged
            # inside the frame so it never clips off-screen.
            label = f"{tag} Def Line · {line_x:.0f}m · #{line_track}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            anchor = cp1 if cp1[1] <= cp2[1] else cp2
            tx = int(min(max(anchor[0] + 8, 4), max(4, w - tw - 4)))
            ty = int(min(max(anchor[1] + th + 4, th + 4), h - 4))
            cv2.rectangle(out, (tx - 4, ty - th - 4),
                          (tx + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(out, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_t, 2, cv2.LINE_AA)
        return out

    def _update_team_x_ema(self, team_id, team_ids_arr, player_pitch_pts, n,
                           role_ids_arr=None):
        """Fold this frame's mean pitch-X for ``team_id`` (outfield, on-pitch
        only) into the running EMA used for the defending-side decision."""
        xs = []
        if role_ids_arr is None or len(role_ids_arr) == 0:
            role_ids_arr = np.full(n, ROLE_OUTFIELD, dtype=np.int32)
        for i in range(n):
            if int(team_ids_arr[i]) != team_id:
                continue
            if int(role_ids_arr[i]) in (ROLE_GK, ROLE_REF):
                continue
            pt = player_pitch_pts[i]
            if -2 <= pt[0] <= PITCH_LENGTH + 2 and -2 <= pt[1] <= PITCH_WIDTH + 2:
                xs.append(float(pt[0]))
        if not xs:
            return
        mean_x = float(np.mean(xs))
        prev = self._team_x_ema.get(team_id)
        if prev is None:
            self._team_x_ema[team_id] = mean_x
        else:
            self._team_x_ema[team_id] = (
                (1.0 - TEAM_X_EMA_ALPHA) * prev + TEAM_X_EMA_ALPHA * mean_x
            )

    def _team_defends_left(self, team_id: int) -> bool:
        """True if ``team_id`` defends the LEFT goal, decided from the
        smoothed mean-X estimates. The team sitting further left (smaller
        mean-X) defends the left goal. Falls back to a stable default when
        one or both estimates are missing."""
        other = 1 - team_id
        a = self._team_x_ema.get(team_id)
        b = self._team_x_ema.get(other)
        if a is None and b is None:
            return team_id == 0  # deterministic default
        if a is None:
            return True
        if b is None:
            return False
        return a <= b


    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    def _reset_pass_state(self):
        """Reset pass detection without touching model, homography, or teams."""
        self._prev_ball_owner_tid: Optional[int] = None
        self._prev_ball_owner_display_tid: Optional[int] = None
        self._prev_ball_owner_team: Optional[int] = None
        self._prev_ball_owner_pitch: Optional[np.ndarray] = None
        self._prev_ball_owner_frames: int = 0

        self._owner_candidate_key: Optional[int] = None
        self._owner_candidate_display_tid: Optional[int] = None
        self._owner_candidate_team: Optional[int] = None
        self._owner_candidate_pitch: Optional[np.ndarray] = None
        self._owner_candidate_frames: int = 0

        self._pass_candidate: Optional[dict] = None
        self._pass_cooldown_frames: int = 0
        self._last_pass_edge: Optional[tuple[int, int]] = None
        self._missing_ball_frames: int = 0
        self._prev_ball_pitch: Optional[np.ndarray] = None
        self._pass_flash_counter: int = 0
        self._last_pass_info: dict = {}

    def _reset_video_state(self, reset_team_analyzer: bool = True,
                           reset_homography: bool = True):
        if reset_team_analyzer and self.team_analyzer is not None:
            self.team_analyzer.reset()
        if reset_homography:
            self.last_H = None
            if hasattr(self.keypoint_computer, "smoothed_H"):
                self.keypoint_computer.smoothed_H = None
        self.ball_trajectory = []
        self._reset_pass_state()
        self._team_x_ema = {TEAM0: None, TEAM1: None}

    def _analyze_frame_for_sequence(self, frame: np.ndarray, frame_idx: int):
        """First-pass inference only; stores compact metadata, no rendering."""
        frame_h, frame_w = frame.shape[:2]
        processed_segments = []

        seg_op = self.segmentor.model.predict(frame, conf=SEG_CONF, verbose=False)[0]
        if seg_op is not None and getattr(seg_op, 'masks', None) is not None:
            processed_segments = self.segmentor.extract(seg_op, frame_w, last_side=None)
        pitch_mask = self._build_pitch_mask(processed_segments, frame_h, frame_w)

        H, H_info = self.keypoint_computer.compute_homography(frame, last_H=self.last_H)
        if H is not None:
            self.last_H = H

        formatted = self.player_detector.track_players(frame, conf=PLAYER_CONF)
        if formatted is not None:
            player_xyxy, player_conf, _, track_ids = formatted
        else:
            player_xyxy = np.empty((0, 4), dtype=np.float32)
            player_conf = np.empty((0,), dtype=np.float32)
            track_ids = np.empty((0,), dtype=np.int32)

        if pitch_mask is not None and len(player_xyxy) > 0:
            keep = self._filter_bboxes_by_mask(player_xyxy, pitch_mask, anchor="lower_torso")
            if len(keep) < len(player_xyxy):
                orig_n = len(player_xyxy)
                player_xyxy = player_xyxy[keep]
                player_conf = player_conf[keep] if len(player_conf) == orig_n else player_conf[:0]
                track_ids = track_ids[keep] if len(track_ids) == orig_n else track_ids[:0]

        player_pitch_pts = np.empty((0, 2), dtype=np.float32)
        if H is not None and len(player_xyxy) > 0:
            player_pitch_pts = self.player_detector.project_points(player_xyxy, H)
            if self.flip_projection_x and len(player_pitch_pts) > 0:
                player_pitch_pts[:, 0] = PITCH_LENGTH - player_pitch_pts[:, 0]
            if self.flip_projection_y and len(player_pitch_pts) > 0:
                player_pitch_pts[:, 1] = PITCH_WIDTH - player_pitch_pts[:, 1]

        team_features = None
        if self.team_analyzer is not None and len(player_xyxy) > 0:
            team_features = self.team_analyzer.extract_detection_features(
                frame, player_xyxy, player_conf,
            )

        ball_xyxy = np.empty((0, 4), dtype=np.float32)
        ball_conf = np.empty((0,), dtype=np.float32)
        ball_pitch_pt = None
        if self.ball_detector is not None:
            ball_xyxy, ball_conf = self.ball_detector.detect_ball(frame)
            if pitch_mask is not None and len(ball_xyxy) > 0:
                if not self._bbox_center_in_mask(ball_xyxy[0], pitch_mask):
                    ball_xyxy = np.empty((0, 4), dtype=np.float32)
                    ball_conf = np.empty((0,), dtype=np.float32)
            if len(ball_xyxy) > 0 and H is not None:
                ball_pitch_pt = self.ball_detector.project_ball_to_pitch(
                    ball_xyxy, H,
                    flip_x=self.flip_projection_x,
                    flip_y=self.flip_projection_y,
                    pitch_length=PITCH_LENGTH,
                    pitch_width=PITCH_WIDTH,
                )

        return {
            'phase': 'analysis',
            'frame_idx': int(frame_idx),
            'H': H,
            'H_info': H_info,
            'player_xyxy': player_xyxy,
            'player_conf': player_conf,
            'track_ids': track_ids,
            'player_pitch_pts': player_pitch_pts,
            'keypoints_used': H_info.get('used_keypoints', []) if H_info else [],
            'processed_segments': processed_segments,
            'team_features': team_features,
            'ball_xyxy': ball_xyxy,
            'ball_conf': ball_conf,
            'ball_pitch_pt': ball_pitch_pt,
        }

    def _render_frame_from_record(self, frame: np.ndarray, record: dict,
                                  team_info: Optional[dict]):
        """Second-pass rendering from first-pass metadata and stable labels."""
        H = record.get('H')
        H_info = record.get('H_info') or {}
        player_xyxy = record.get('player_xyxy', np.empty((0, 4), dtype=np.float32))
        player_conf = record.get('player_conf', np.empty((0,), dtype=np.float32))
        track_ids = record.get('track_ids', np.empty((0,), dtype=np.int32))
        player_pitch_pts = record.get('player_pitch_pts', np.empty((0, 2), dtype=np.float32))
        processed_segments = record.get('processed_segments', [])
        ball_xyxy = record.get('ball_xyxy', np.empty((0, 4), dtype=np.float32))
        ball_conf = record.get('ball_conf', np.empty((0,), dtype=np.float32))
        ball_pitch_pt = record.get('ball_pitch_pt')
        used_kpts = record.get('keypoints_used', [])

        seg_overlay_frame = self._create_seg_overlay(frame, None, processed_segments) if processed_segments else frame.copy()

        if ball_pitch_pt is not None:
            self.ball_trajectory.append(np.asarray(ball_pitch_pt, dtype=np.float32).copy())
            if len(self.ball_trajectory) > BALL_TRAIL_LENGTH:
                self.ball_trajectory.pop(0)

        pitch_canvas = self.pitch_artist.draw_pitch_base()
        if processed_segments:
            pitch_canvas = self.pitch_artist.draw_seg_zones(pitch_canvas, processed_segments, alpha=0.25)
        if len(self.ball_trajectory) > 1:
            pitch_canvas = self.pitch_artist.draw_ball_trajectory(
                pitch_canvas, self.ball_trajectory, max_trail=BALL_TRAIL_LENGTH,
            )
        if len(player_pitch_pts) > 0:
            mask = ((player_pitch_pts[:, 0] >= -5) & (player_pitch_pts[:, 0] <= PITCH_LENGTH + 5)
                    & (player_pitch_pts[:, 1] >= -5) & (player_pitch_pts[:, 1] <= PITCH_WIDTH + 5))
            valid_pts = player_pitch_pts[mask]
            if len(valid_pts) > 0:
                colors = None
                if team_info is not None:
                    c = team_info.get('team_colors', [])
                    colors = [c[i] for i in range(len(c)) if mask[i]] if len(c) == len(mask) else None
                pitch_canvas = self.pitch_artist.draw_players_on_pitch(
                    pitch_canvas, valid_pts, colors=colors, default_color=(0, 0, 255),
                )
            if team_info is not None:
                pitch_canvas = self.pitch_artist.draw_team_legend(
                    pitch_canvas, team_info.get('team1_bgr', (255, 0, 0)),
                    team_info.get('team2_bgr', (0, 0, 255)),
                )
        if ball_pitch_pt is not None:
            pitch_canvas = self.pitch_artist.draw_ball_on_pitch(
                pitch_canvas, ball_pitch_pt, ball_color=BALL_DOT_COLOR,
            )

        tids_arr = np.asarray(track_ids) if len(track_ids) > 0 else np.empty((0,), dtype=np.int32)
        team_ids_arr = (np.asarray(team_info.get('team_ids', []), dtype=np.int32)
                        if team_info is not None else np.empty((0,), dtype=np.int32))
        role_ids_arr = (np.asarray(team_info.get('role_ids', []), dtype=np.int32)
                        if team_info is not None else np.empty((0,), dtype=np.int32))
        identity_ids_arr = (np.asarray(team_info.get('identity_ids', []), dtype=np.int32)
                            if team_info is not None else np.empty((0,), dtype=np.int32))

        pass_event = self._update_pass_state(
            ball_xyxy=ball_xyxy,
            player_xyxy=player_xyxy,
            tids_arr=tids_arr,
            owner_ids_arr=identity_ids_arr,
            team_ids_arr=team_ids_arr,
            player_pitch_pts=player_pitch_pts,
            ball_pitch_pt=ball_pitch_pt,
        )

        annotated_frame = self._draw_keypoints_on_frame(frame, used_kpts)
        if team_info is not None and len(player_xyxy) > 0:
            annotated_frame = self._draw_team_bboxes(
                annotated_frame, player_xyxy, team_info.get('team_colors', []),
                player_conf=player_conf,
                team_ids=team_info.get('team_ids'),
                role_ids=team_info.get('role_ids'),
            )
        if len(ball_xyxy) > 0:
            annotated_frame = self._draw_ball_bbox(annotated_frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR)

        deep_analysis_frame = self._draw_keypoints_on_frame(seg_overlay_frame, used_kpts) if used_kpts else seg_overlay_frame
        if team_info is not None and len(player_xyxy) > 0:
            deep_analysis_frame = self._draw_team_bboxes(
                deep_analysis_frame, player_xyxy, team_info.get('team_colors', []),
                player_conf=player_conf,
                team_ids=team_info.get('team_ids'),
                role_ids=team_info.get('role_ids'),
                track_ids=track_ids,
            )
        if len(ball_xyxy) > 0:
            deep_analysis_frame = self._draw_ball_bbox(deep_analysis_frame, ball_xyxy, ball_conf, color=BALL_BBOX_COLOR)
        if team_info is not None and H is not None and len(player_pitch_pts) > 0:
            deep_analysis_frame = self._draw_defensive_lines(
                deep_analysis_frame, H, tids_arr, team_ids_arr, player_pitch_pts,
                team_info.get("team1_bgr", (255, 0, 0)),
                team_info.get("team2_bgr", (0, 0, 255)),
                role_ids_arr=role_ids_arr,
            )
        if self._pass_flash_counter > 0:
            deep_analysis_frame = self._draw_pass_flash(
                deep_analysis_frame, self._last_pass_info, self._pass_flash_counter,
                PASS_FLASH_FRAMES,
            )

        return {
            'phase': 'render',
            'H': H, 'H_info': H_info, 'player_xyxy': player_xyxy, 'player_conf': player_conf,
            'track_ids': track_ids, 'player_pitch_pts': player_pitch_pts, 'keypoints_used': used_kpts,
            'seg_result': None, 'processed_segments': processed_segments, 'pitch_canvas': pitch_canvas,
            'annotated_frame': annotated_frame, 'deep_analysis_frame': deep_analysis_frame,
            'team_info': team_info,
            'ball_xyxy': ball_xyxy, 'ball_conf': ball_conf, 'ball_pitch_pt': ball_pitch_pt,
            'ball_trajectory': list(self.ball_trajectory), 'pass_event': pass_event,
            'pitch_mask': None,
        }

    def _clear_stable_ball_owner(self):
        self._prev_ball_owner_tid = None
        self._prev_ball_owner_display_tid = None
        self._prev_ball_owner_team = None
        self._prev_ball_owner_pitch = None
        self._prev_ball_owner_frames = 0

    def _clear_observed_owner_candidate(self):
        self._owner_candidate_key = None
        self._owner_candidate_display_tid = None
        self._owner_candidate_team = None
        self._owner_candidate_pitch = None
        self._owner_candidate_frames = 0

    def _clear_pass_candidate(self):
        self._pass_candidate = None

    def _tick_pass_flash(self):
        if self._pass_flash_counter > 0:
            self._pass_flash_counter = max(0, self._pass_flash_counter - 1)

    def _resolve_pass_owner(self, ball_xyxy, player_xyxy, tids_arr, owner_ids_arr,
                            team_ids_arr, player_pitch_pts) -> Optional[dict]:
        """Resolve current ball owner from bbox overlap, or None."""
        if ball_xyxy is None or player_xyxy is None:
            return None
        if len(ball_xyxy) == 0 or len(player_xyxy) == 0:
            return None

        ball_owner_idx, _owner_score = self._find_ball_owner_by_bbox_overlap(
            ball_xyxy, player_xyxy,
        )
        if ball_owner_idx is None:
            return None

        tids_arr = np.asarray(tids_arr, dtype=np.int32)
        team_ids_arr = np.asarray(team_ids_arr, dtype=np.int32)
        player_pitch_pts = np.asarray(player_pitch_pts, dtype=np.float32)
        if owner_ids_arr is None or len(owner_ids_arr) == 0:
            owner_ids_arr = tids_arr
        owner_ids_arr = np.asarray(owner_ids_arr, dtype=np.int32)

        n = min(len(tids_arr), len(owner_ids_arr), len(team_ids_arr),
                len(player_xyxy), len(player_pitch_pts))
        if ball_owner_idx >= n:
            return None

        owner_team = int(team_ids_arr[ball_owner_idx])
        if owner_team not in (TEAM0, TEAM1):
            return None

        proj_pt = player_pitch_pts[ball_owner_idx]
        if not (-2 <= proj_pt[0] <= PITCH_LENGTH + 2
                and -2 <= proj_pt[1] <= PITCH_WIDTH + 2):
            return None

        key = int(owner_ids_arr[ball_owner_idx])
        display_tid = int(tids_arr[ball_owner_idx])
        if key < 0:
            key = display_tid
        return {
            "key": key,
            "display_tid": display_tid,
            "team": owner_team,
            "pitch": np.asarray(proj_pt, dtype=np.float32).copy(),
        }

    def _observe_owner_candidate(self, owner: dict):
        key = int(owner["key"])
        if self._owner_candidate_key == key:
            self._owner_candidate_frames += 1
        else:
            self._owner_candidate_key = key
            self._owner_candidate_frames = 1
        self._owner_candidate_display_tid = int(owner["display_tid"])
        self._owner_candidate_team = int(owner["team"])
        self._owner_candidate_pitch = np.asarray(owner["pitch"], dtype=np.float32).copy()

    def _promote_observed_owner_to_stable(self):
        if self._owner_candidate_key is None:
            return
        self._prev_ball_owner_tid = int(self._owner_candidate_key)
        self._prev_ball_owner_display_tid = int(
            self._owner_candidate_display_tid
            if self._owner_candidate_display_tid is not None
            else self._owner_candidate_key
        )
        self._prev_ball_owner_team = int(self._owner_candidate_team)
        self._prev_ball_owner_pitch = np.asarray(
            self._owner_candidate_pitch, dtype=np.float32,
        ).copy()
        self._prev_ball_owner_frames = int(self._owner_candidate_frames)

    def _refresh_stable_owner(self, owner: dict):
        self._prev_ball_owner_tid = int(owner["key"])
        self._prev_ball_owner_display_tid = int(owner["display_tid"])
        self._prev_ball_owner_team = int(owner["team"])
        self._prev_ball_owner_pitch = np.asarray(owner["pitch"], dtype=np.float32).copy()
        self._prev_ball_owner_frames = max(
            int(self._prev_ball_owner_frames) + 1,
            int(self._owner_candidate_frames),
        )

    def _start_pass_candidate(self, ball_pitch_pt: Optional[np.ndarray],
                              initial_travel_m: float = 0.0):
        if self._prev_ball_owner_tid is None:
            return
        start_ball = None
        if ball_pitch_pt is not None:
            start_ball = np.asarray(ball_pitch_pt, dtype=np.float32).copy()
        elif self._prev_ball_pitch is not None:
            start_ball = np.asarray(self._prev_ball_pitch, dtype=np.float32).copy()
        self._pass_candidate = {
            "source_key": int(self._prev_ball_owner_tid),
            "source_display_tid": int(
                self._prev_ball_owner_display_tid
                if self._prev_ball_owner_display_tid is not None
                else self._prev_ball_owner_tid
            ),
            "source_team": int(self._prev_ball_owner_team),
            "source_pitch": np.asarray(
                self._prev_ball_owner_pitch, dtype=np.float32,
            ).copy() if self._prev_ball_owner_pitch is not None else None,
            "source_stable_frames": int(self._prev_ball_owner_frames),
            "start_ball_pitch": start_ball,
            "ball_travel_m": float(max(0.0, initial_travel_m)),
            "frames": 1,
            "missing_frames": 1 if ball_pitch_pt is None else 0,
        }

    def _age_pass_candidate(self, ball_step_m: float = 0.0, missing_ball: bool = False):
        if self._pass_candidate is None:
            return
        self._pass_candidate["frames"] = int(self._pass_candidate.get("frames", 0)) + 1
        self._pass_candidate["ball_travel_m"] = float(
            self._pass_candidate.get("ball_travel_m", 0.0)
            + max(0.0, float(ball_step_m))
        )
        if missing_ball:
            self._pass_candidate["missing_frames"] = int(
                self._pass_candidate.get("missing_frames", 0)
            ) + 1
        else:
            self._pass_candidate["missing_frames"] = 0

    def _pass_candidate_expired(self) -> bool:
        if self._pass_candidate is None:
            return False
        return (int(self._pass_candidate.get("frames", 0)) > MAX_PASS_TRANSIT_FRAMES
                or int(self._pass_candidate.get("missing_frames", 0)) > MAX_PASS_BALL_MISSING_FRAMES)

    def _finish_pass_frame(self, ball_pitch_pt: Optional[np.ndarray]):
        if ball_pitch_pt is not None:
            self._prev_ball_pitch = np.asarray(ball_pitch_pt, dtype=np.float32).copy()
        self._tick_pass_flash()

    def _update_pass_state(self, ball_xyxy, player_xyxy, tids_arr, owner_ids_arr,
                           team_ids_arr, player_pitch_pts, ball_pitch_pt) -> Optional[dict]:
        if self._pass_cooldown_frames > 0:
            self._pass_cooldown_frames = max(0, self._pass_cooldown_frames - 1)

        current_ball: Optional[np.ndarray] = None
        if ball_pitch_pt is not None:
            ball_arr = np.asarray(ball_pitch_pt, dtype=np.float32).reshape(-1)
            if ball_arr.size >= 2:
                current_ball = ball_arr[:2].copy()

        ball_step_m = 0.0
        if current_ball is not None and self._prev_ball_pitch is not None:
            ball_step_m = float(np.linalg.norm(
                np.asarray(self._prev_ball_pitch, dtype=np.float32) - current_ball,
            ))

        observed_owner = self._resolve_pass_owner(
            ball_xyxy, player_xyxy, tids_arr, owner_ids_arr, team_ids_arr,
            player_pitch_pts,
        ) if current_ball is not None else None

        if current_ball is None:
            self._missing_ball_frames += 1
            self._clear_observed_owner_candidate()
            if self._prev_ball_owner_tid is not None:
                if self._pass_candidate is None:
                    self._start_pass_candidate(None, initial_travel_m=0.0)
                else:
                    self._age_pass_candidate(missing_ball=True)
            if (self._missing_ball_frames > MAX_PASS_BALL_MISSING_FRAMES
                    or self._pass_candidate_expired()):
                self._clear_stable_ball_owner()
                self._clear_pass_candidate()
                self._prev_ball_pitch = None
            self._tick_pass_flash()
            return None

        self._missing_ball_frames = 0
        if self._pass_candidate is not None:
            self._age_pass_candidate(ball_step_m=ball_step_m, missing_ball=False)
            if self._pass_candidate_expired():
                self._clear_pass_candidate()
                self._clear_stable_ball_owner()

        if observed_owner is None:
            self._clear_observed_owner_candidate()
            if self._prev_ball_owner_tid is not None and self._pass_candidate is None:
                self._start_pass_candidate(current_ball, initial_travel_m=ball_step_m)
            if self._pass_candidate_expired():
                self._clear_pass_candidate()
                self._clear_stable_ball_owner()
            self._finish_pass_frame(current_ball)
            return None

        self._observe_owner_candidate(observed_owner)

        if self._prev_ball_owner_tid is None:
            if self._owner_candidate_frames >= MIN_PASSER_STABLE_FRAMES:
                self._promote_observed_owner_to_stable()
                self._clear_pass_candidate()
            self._finish_pass_frame(current_ball)
            return None

        if int(observed_owner["key"]) == int(self._prev_ball_owner_tid):
            self._refresh_stable_owner(observed_owner)
            self._clear_pass_candidate()
            self._finish_pass_frame(current_ball)
            return None

        if self._pass_candidate is None:
            self._start_pass_candidate(current_ball, initial_travel_m=ball_step_m)

        if self._owner_candidate_frames < MIN_RECEIVER_STABLE_FRAMES:
            self._finish_pass_frame(current_ball)
            return None

        pass_event: Optional[dict] = None
        source = self._pass_candidate or {}
        source_key = int(source.get("source_key", self._prev_ball_owner_tid))
        source_display_tid = int(source.get(
            "source_display_tid",
            self._prev_ball_owner_display_tid
            if self._prev_ball_owner_display_tid is not None
            else self._prev_ball_owner_tid,
        ))
        source_team = int(source.get("source_team", self._prev_ball_owner_team))
        source_pitch = source.get("source_pitch", self._prev_ball_owner_pitch)
        source_stable_frames = int(source.get(
            "source_stable_frames", self._prev_ball_owner_frames,
        ))

        pass_dist_m = 0.0
        if source_pitch is not None and observed_owner["pitch"] is not None:
            pass_dist_m = float(np.linalg.norm(
                np.asarray(source_pitch, dtype=np.float32)
                - np.asarray(observed_owner["pitch"], dtype=np.float32),
            ))
        ball_travel_m = float(source.get("ball_travel_m", 0.0))
        edge = (source_key, int(observed_owner["key"]))

        if (source_team == int(observed_owner["team"])
                and source_stable_frames >= MIN_PASSER_STABLE_FRAMES
                and pass_dist_m >= MIN_PASS_DISTANCE_M
                and ball_travel_m >= MIN_PASS_BALL_TRAVEL_M
                and self._pass_cooldown_frames == 0
                and source_key != int(observed_owner["key"])):
            def _pitch_point(prefix: str, point) -> dict:
                if point is None:
                    return {}
                arr = np.asarray(point, dtype=np.float32).reshape(-1)
                if arr.size < 2:
                    return {}
                x, y = float(arr[0]), float(arr[1])
                if not (np.isfinite(x) and np.isfinite(y)):
                    return {}
                return {f"{prefix}_x": round(x, 3), f"{prefix}_y": round(y, 3)}

            receiver_pitch = observed_owner.get("pitch")
            start_ball = source.get("start_ball_pitch")
            end_ball = current_ball
            if start_ball is not None and end_ball is not None:
                event_point = (
                    np.asarray(start_ball, dtype=np.float32).reshape(-1)[:2]
                    + np.asarray(end_ball, dtype=np.float32).reshape(-1)[:2]
                ) / 2.0
            elif end_ball is not None:
                event_point = end_ball
            elif source_pitch is not None and receiver_pitch is not None:
                event_point = (
                    np.asarray(source_pitch, dtype=np.float32).reshape(-1)[:2]
                    + np.asarray(receiver_pitch, dtype=np.float32).reshape(-1)[:2]
                ) / 2.0
            else:
                event_point = receiver_pitch

            pass_info = {
                "from_tid": int(source_display_tid),
                "to_tid": int(observed_owner["display_tid"]),
                "from_identity_id": int(source_key),
                "to_identity_id": int(observed_owner["key"]),
                "team": int(observed_owner["team"]),
                "distance_m": round(pass_dist_m, 1),
            }
            pass_info.update(_pitch_point("from", source_pitch))
            pass_info.update(_pitch_point("to", receiver_pitch))
            pass_info.update(_pitch_point("ball_start", start_ball))
            pass_info.update(_pitch_point("ball_end", end_ball))
            pass_info.update(_pitch_point("event", event_point))
            pass_event = dict(pass_info)
            self._pass_flash_counter = PASS_FLASH_FRAMES
            self._last_pass_info = pass_info
            self._pass_cooldown_frames = PASS_COOLDOWN_FRAMES
            self._last_pass_edge = edge

        # Once the receiver is stable, promote it even for non-pass events
        # (short touches or turnovers) so the same transition is not retried.
        self._promote_observed_owner_to_stable()
        self._clear_pass_candidate()
        self._finish_pass_frame(current_ball)
        return pass_event

    def _write_draft_frame(self, writer, frame, pitch_canvas, frame_w: int, frame_h: int):
        if pitch_canvas is None:
            return
        pip_w, pip_h = frame_w // 4, int((frame_w // 4) * CANVAS_H / CANVAS_W)
        pip = cv2.resize(pitch_canvas, (pip_w, pip_h))
        ox, oy = frame_w - pip_w - 15, frame_h - pip_h - 15
        draft = frame.copy()
        ov = draft.copy()
        cv2.rectangle(ov, (ox - 5, oy - 5), (ox + pip_w + 5, oy + pip_h + 5), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.4, draft, 0.6, 0, draft)
        draft[oy:oy + pip_h, ox:ox + pip_w] = pip
        cv2.rectangle(draft, (ox - 2, oy - 2), (ox + pip_w + 2, oy + pip_h + 2), (255, 255, 255), 2)
        writer.write(draft)

    def process_video(self, source_video_path: str, output_dir: str = "output", fps: float = 30.0,
                      start_frame: int = 0, max_frames: Optional[int] = None,
                      process_every_n: int = 1, team_calibration=None,
                      persist_data: bool = True):
        process_every_n = max(1, int(process_every_n))
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

        total_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        remaining_source = max(0, total_source - int(start_frame)) if total_source > 0 else 0
        expected_count = (remaining_source + max(process_every_n, 1) - 1) // max(process_every_n, 1) if remaining_source else 0
        if max_frames is not None:
            expected_count = min(expected_count if expected_count else int(max_frames), int(max_frames))
        expected_for_progress = max(expected_count, 1)

        self._reset_video_state(reset_team_analyzer=True, reset_homography=True)
        stabilizer = TeamSequenceStabilizer(
            calibration=TeamCalibration.from_any(team_calibration),
            feature_distance_fn=(self.team_analyzer._feature_distance if self.team_analyzer is not None else None),
        ) if self.team_analyzer is not None else None

        frame_records: list[dict] = []
        frame_idx = processed_count = 0
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
                record = self._analyze_frame_for_sequence(frame, frame_idx)
                frame_records.append(record)
                if stabilizer is not None:
                    features = record.get('team_features') or {}
                    stabilizer.add_frame_observations(
                        frame_idx=frame_idx,
                        track_ids=record.get('track_ids', np.empty((0,), dtype=np.int32)),
                        player_xyxy=record.get('player_xyxy', np.empty((0, 4), dtype=np.float32)),
                        player_conf=record.get('player_conf', np.empty((0,), dtype=np.float32)),
                        features=features.get('features'),
                        jersey_bgr=features.get('jersey_bgr'),
                        weights=features.get('weights'),
                        player_pitch_pts=record.get('player_pitch_pts'),
                    )
                processed_count += 1
                yield {
                    'phase': 'analysis',
                    'frame_idx': frame_idx,
                    'processed_count': processed_count,
                    'total_frames': expected_count or processed_count,
                    'progress_pct': min(50.0, 50.0 * processed_count / expected_for_progress),
                }
                frame_idx += 1
        finally:
            cap.release()

        if stabilizer is not None:
            assignments = stabilizer.fit()
            frame_team_infos = []
            for record in frame_records:
                team_info = assignments.team_info_for_frame(record.get('track_ids', np.empty((0,), dtype=np.int32)))
                frame_team_infos.append(team_info)
                record['team_info'] = team_info
            assignments.diagnostics['validation'] = validate_frame_team_infos(frame_team_infos)
            for team_info in frame_team_infos:
                team_info['stabilizer_diagnostics'] = assignments.diagnostics
            stabilizer_report = assignments.diagnostics
        else:
            for record in frame_records:
                record['team_info'] = None
            stabilizer_report = None

        yield {
            'phase': 'stabilizing',
            'processed_count': processed_count,
            'total_frames': expected_count or processed_count,
            'progress_pct': 50.0,
            'stabilizer_report': stabilizer_report,
        }

        cap_render = cv2.VideoCapture(source_video_path)
        if not cap_render.isOpened():
            raise RuntimeError(f"Could not reopen video for rendering: {source_video_path}")
        if start_frame > 0:
            cap_render.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pw = Director.make_video_writer(paths["pitch"], actual_fps, (CANVAS_W, CANVAS_H))
        aw = Director.make_video_writer(paths["annotated"], actual_fps, (frame_w, frame_h))
        dw = Director.make_video_writer(paths["deep"], actual_fps, (frame_w, frame_h))
        fw = Director.make_video_writer(paths["draft"], actual_fps, (frame_w, frame_h))
        kw = Director.make_video_writer(paths["keypoint"], actual_fps, (frame_w, frame_h))

        self._reset_video_state(reset_team_analyzer=False, reset_homography=False)
        frame_idx = rendered_count = record_idx = 0
        # Optional: accumulate per-frame records alongside the rendered
        # outputs so the dashboard can resume from disk after a restart and
        # so the demo cache can be populated without re-running inference.
        persisted_game_data: list[dict] = []
        persisted_analytics_data: list[dict] = []
        try:
            while record_idx < len(frame_records):
                ret, frame = cap_render.read()
                if not ret or frame is None:
                    break
                if frame_idx % process_every_n != 0:
                    frame_idx += 1
                    continue
                record = frame_records[record_idx]
                result = self._render_frame_from_record(frame, record, record.get('team_info'))
                result['stabilizer_report'] = stabilizer_report
                result['progress_pct'] = min(100.0, 50.0 + 50.0 * (rendered_count + 1) / max(len(frame_records), 1))
                if result['pitch_canvas'] is not None:
                    pw.write(result['pitch_canvas'])
                if result['annotated_frame'] is not None:
                    aw.write(result['annotated_frame'])
                if result['deep_analysis_frame'] is not None:
                    dw.write(result['deep_analysis_frame'])
                kpts = result.get('keypoints_used', [])
                kw.write(self._draw_keypoints_on_frame(frame.copy(), kpts))
                self._write_draft_frame(fw, frame, result.get('pitch_canvas'), frame_w, frame_h)
                if persist_data:
                    persisted_game_data.append(
                        self._build_game_data_entry(result, processed_count=rendered_count + 1)
                    )
                    segs = result.get('processed_segments', []) or []
                    if segs:
                        persisted_analytics_data.append({
                            "frame_idx": rendered_count + 1,
                            "segments": [
                                {"class_name": s.get("class_name"),
                                 "confidence": float(s.get("confidence", 0.0))}
                                for s in segs
                            ],
                        })
                rendered_count += 1
                record_idx += 1
                yield result
                frame_idx += 1
        finally:
            cap_render.release()
            for w in [pw, aw, dw, fw, kw]:
                w.release()
            if persist_data and persisted_game_data:
                self._persist_per_frame_data(
                    output_path, persisted_game_data, persisted_analytics_data,
                    fps=float(actual_fps), total_frames=int(rendered_count),
                )

    @staticmethod
    def _build_game_data_entry(result: dict, processed_count: int) -> dict:
        """Build the per-frame dict that the dashboard consumes.

        Same shape the Streamlit loop builds today, factored so the
        pipeline can also persist it without diverging from the live
        behaviour. All array fields default to empty arrays of the
        canonical dtype so missing detections don't break downstream
        analytics.
        """
        team_info = result.get("team_info") or {}
        return {
            "frame_idx": int(processed_count),
            "player_positions": result.get("player_pitch_pts",
                                           np.empty((0, 2), dtype=np.float32)),
            "player_xyxy": result.get("player_xyxy",
                                      np.empty((0, 4), dtype=np.float32)),
            "team_ids": team_info.get("team_ids") if team_info else None,
            "role_ids": team_info.get("role_ids") if team_info else None,
            "identity_ids": team_info.get("identity_ids") if team_info else None,
            "track_ids": result.get("track_ids",
                                    np.empty((0,), dtype=np.int32)),
            "track_quality": team_info.get("track_quality") if team_info else None,
            "team1_bgr": team_info.get("team1_bgr") if team_info else None,
            "team2_bgr": team_info.get("team2_bgr") if team_info else None,
            "ball_position": result.get("ball_pitch_pt"),
            "ball_xyxy": result.get("ball_xyxy",
                                    np.empty((0, 4), dtype=np.float32)),
            "ball_conf": result.get("ball_conf",
                                    np.empty((0,), dtype=np.float32)),
            "player_conf": result.get("player_conf",
                                      np.empty((0,), dtype=np.float32)),
            "pass_event": result.get("pass_event"),
        }

    @staticmethod
    def _persist_per_frame_data(output_path: Path,
                                game_data: list,
                                analytics_data: list,
                                fps: float,
                                total_frames: int) -> None:
        """Write ``game_data.npz`` + ``analytics_data.json`` + ``meta.json``
        under ``output_path/data/`` so the dashboard can reload without
        re-running inference. Object arrays with per-frame variable shape
        (e.g. ``player_positions``) are stored as ``dtype=object`` arrays
        so round-tripping preserves the ragged shape. ``meta.json`` carries
        the scalar fields needed by the demo picker.
        """
        data_dir = output_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # --- game_data.npz -------------------------------------------------
        # Per-frame arrays share a row count; columns with variable
        # per-row shape are stored as object arrays.
        array_keys_scalar = (
            "frame_idx", "team_ids", "role_ids", "identity_ids",
            "track_ids", "track_quality", "player_conf", "ball_conf",
        )
        array_keys_2d = (
            "player_positions", "player_xyxy", "ball_xyxy",
        )
        array_keys_misc = ("team1_bgr", "team2_bgr")

        save_kwargs: dict = {}
        n = len(game_data)
        # Build a per-key list, then convert to a single ndarray of the
        # right dtype. Empty frames still produce an empty row of the
        # correct shape so the archive round-trips.
        for key in array_keys_scalar:
            col = []
            for entry in game_data:
                v = entry.get(key)
                if v is None:
                    col.append(np.asarray([-1], dtype=np.int32))
                else:
                    arr = np.asarray(v)
                    if arr.ndim == 0:
                        arr = arr.reshape(1)
                    col.append(arr)
            try:
                save_kwargs[key] = np.asarray(col, dtype=np.int32)
            except (ValueError, TypeError):
                save_kwargs[key] = np.asarray(col, dtype=object)
        for key in array_keys_2d:
            col = [np.asarray(entry.get(key), dtype=np.float32)
                   if entry.get(key) is not None else np.empty((0, 2), dtype=np.float32)
                   for entry in game_data]
            try:
                save_kwargs[key] = np.asarray(col, dtype=np.float32)
            except (ValueError, TypeError):
                save_kwargs[key] = np.asarray(col, dtype=object)
        for key in array_keys_misc:
            col = []
            for entry in game_data:
                v = entry.get(key)
                if v is None:
                    col.append(np.asarray([-1, -1, -1], dtype=np.int32))
                else:
                    arr = np.asarray(v, dtype=np.int32).reshape(-1)
                    if arr.size < 3:
                        arr = np.concatenate(
                            [arr, np.full(3 - arr.size, -1, dtype=np.int32)]
                        )
                    col.append(arr[:3])
            save_kwargs[key] = np.asarray(col, dtype=np.int32)

        # Per-frame ball_position is a (2,) float or None. Store as object
        # array so the None entries survive.
        ball_pos_col = []
        for entry in game_data:
            v = entry.get("ball_position")
            if v is None:
                ball_pos_col.append(np.asarray([np.nan, np.nan], dtype=np.float32))
            else:
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                if arr.size < 2:
                    ball_pos_col.append(np.asarray([np.nan, np.nan], dtype=np.float32))
                else:
                    ball_pos_col.append(arr[:2])
        save_kwargs["ball_position"] = np.asarray(ball_pos_col, dtype=np.float32)

        # pass_event is a dict-or-None per frame; store as object array.
        save_kwargs["pass_event"] = np.asarray(
            [entry.get("pass_event") for entry in game_data], dtype=object,
        )

        try:
            np.savez_compressed(data_dir / "game_data.npz", **save_kwargs)
        except Exception as exc:
            print(f"[keypoint_pipeline] WARN: could not write game_data.npz: {exc}")

        # --- analytics_data.json -----------------------------------------
        try:
            with open(data_dir / "analytics_data.json", "w") as f:
                json.dump(analytics_data, f)
        except Exception as exc:
            print(f"[keypoint_pipeline] WARN: could not write analytics_data.json: {exc}")

        # --- meta.json ----------------------------------------------------
        try:
            with open(data_dir / "meta.json", "w") as f:
                json.dump({
                    "schema_version": 1,
                    "fps": float(fps),
                    "total_frames": int(total_frames),
                    "frame_count": int(n),
                }, f, indent=2)
        except Exception as exc:
            print(f"[keypoint_pipeline] WARN: could not write meta.json: {exc}")
