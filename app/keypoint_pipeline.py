"""
KeypointPipeline — End-to-end pipeline orchestrator.

Pipeline flow per frame:
    Keypoint detection → Homography matrix
    Player detection → Bottom-center projection via H
    Segmentation → Deep analysis overlay (on original frame + pitch canvas)

Outputs:
    - Annotated original frame (keypoints + seg masks)
    - Top-down pitch canvas (players projected + seg overlay)
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


class KeypointPipeline:
    """
    Orchestrates the full keypoint → homography → player projection →
    segmentation analysis pipeline for each video frame.
    """

    def __init__(
        self,
        keypoint_model_path: str,
        player_model_path: str,
        seg_model_path: str,
        flip_projection_x: bool = False,
    ):
        """
        Args:
            keypoint_model_path: Path to YOLO-Pose keypoint model.
            player_model_path: Path to YOLO player detection model.
            seg_model_path: Path to YOLO segmentation model.
            flip_projection_x: If True, mirror the x-axis of projected player
                               positions (PITCH_LENGTH - x). Fixes left-right
                               flip caused by camera being on opposite side.
        """
        # Models
        self.keypoint_computer = KeypointHomographyComputer(keypoint_model_path)
        self.player_detector = PlayerDetector(player_model_path)
        self.segmentor = Segmentor(seg_model_path)

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
        # 3. Player detection & projection
        # --------------------------------------------------------------
        player_results = self.player_detector.model.predict(
            frame, conf=PLAYER_CONF, verbose=False
        )
        formatted = self.player_detector.format_results(player_results[0])
        player_xyxy, player_conf, _ = formatted if formatted is not None else (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int))
        player_pitch_pts = np.empty((0, 2), dtype=np.float32)
        if H is not None and len(player_xyxy) > 0:
            player_pitch_pts = self.player_detector.project_points(player_xyxy, H)
            # Fix left-right flip if camera is on opposite side
            if self.flip_projection_x and len(player_pitch_pts) > 0:
                player_pitch_pts[:, 0] = PITCH_LENGTH - player_pitch_pts[:, 0]

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
                pitch_canvas = self.pitch_artist.draw_players_on_pitch(
                    pitch_canvas, valid_pts, color=(0, 0, 255)
                )

        # --------------------------------------------------------------
        # 5. Annotated frames with keypoints
        # --------------------------------------------------------------
        used_kpts = H_info.get('used_keypoints', [])
        annotated_frame = self._draw_keypoints_on_frame(frame, used_kpts)
        # Also add keypoints to the deep analysis overlay
        if used_kpts:
            deep_analysis_frame = self._draw_keypoints_on_frame(
                seg_overlay_frame, used_kpts
            )
        else:
            deep_analysis_frame = seg_overlay_frame

        return {
            'H': H,
            'H_info': H_info,
            'player_xyxy': player_xyxy,
            'player_conf': player_conf,
            'player_pitch_pts': player_pitch_pts,
            'keypoints_used': used_kpts,
            'seg_result': seg_op,
            'processed_segments': processed_segments,
            'pitch_canvas': pitch_canvas,
            'annotated_frame': annotated_frame,
            'deep_analysis_frame': deep_analysis_frame,
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
            print(f"\nDone. Processed {processed_count} frames.")
            print(f"  Pitch canvas:  {full_pitch_path}")
            print(f"  Annotated:     {annotated_path}")
            print(f"  Deep analysis: {deep_analysis_path}")