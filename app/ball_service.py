"""
BallDetector — Dedicated ball detection using a YOLO model trained on ball class only.

Pipeline per frame:
    YOLO inference → filter to class 0 (ball) → bottom-center projection via H

Outputs per frame:
    - ball_xyxy:    (1, 4) or empty — ball bbox in image coords
    - ball_conf:    (1,) or empty  — ball confidence
    - ball_pitch_pt: (2,) or None  — ball position on pitch in meters
"""

import numpy as np
import cv2
from ultralytics import YOLO
from constants import BALL_CONF


class BallDetector:
    """
    Wraps a YOLO model trained for ball-only detection (class_id == 0).
    Provides detection, center computation, and pitch projection methods.
    """

    def __init__(self, model_path: str, conf: float = BALL_CONF):
        """
        Args:
            model_path: Path to YOLO ball detection model (.pt file).
            conf: Detection confidence threshold.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        # Ball models are typically single-class (class 0 = ball)
        # but we check to be safe
        self.ball_class_id = self._find_ball_class()

    def _find_ball_class(self) -> int:
        """Identify the ball class ID from the model's class names."""
        names = self.model.names
        for cid, cname in names.items():
            if 'ball' in str(cname).lower():
                return int(cid)
        # Fallback: assume class 0
        return 0

    def detect_ball(self, frame: np.ndarray) -> tuple:
        """
        Run YOLO inference and return the highest-confidence ball detection.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            tuple: (xyxy, conf)
                xyxy: (1, 4) float32 array of [x1, y1, x2, y2] or
                      np.empty((0, 4), dtype=np.float32) if no ball.
                conf: (1,) float32 array or np.empty((0,), dtype=np.float32).
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            # Filter to ball class only
            ball_mask = classes == self.ball_class_id
            xyxy = xyxy[ball_mask]
            conf = conf[ball_mask]

            if len(xyxy) > 0:
                # Pick highest confidence detection
                best_idx = int(np.argmax(conf))
                return xyxy[best_idx:best_idx + 1], conf[best_idx:best_idx + 1]

        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    @staticmethod
    def get_ball_center(xyxy: np.ndarray) -> np.ndarray:
        """
        Get the (x, y) center point of the ball bbox.

        Args:
            xyxy: (1, 4) float32 array [x1, y1, x2, y2].

        Returns:
            (2,) float32 array [cx, cy].
        """
        cx = (xyxy[0, 0] + xyxy[0, 2]) / 2.0
        cy = (xyxy[0, 1] + xyxy[0, 3]) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    @staticmethod
    def project_ball_to_pitch(xyxy: np.ndarray, H: np.ndarray,
                              flip_x: bool = False, pitch_length: float = 105.0) -> np.ndarray:
        """
        Project ball position onto the pitch via homography H.

        Uses ball bottom-center (feet level) for projection, matching the
        player projection convention.

        Args:
            xyxy: (1, 4) float32 array [x1, y1, x2, y2].
            H: 3x3 homography matrix.
            flip_x: If True, mirror the x-axis (PITCH_LENGTH - x).
            pitch_length: Pitch length in meters (for flip_x correction).

        Returns:
            (2,) float32 array — (x, y) on pitch in meters, or
            (0,) empty array if H is None.
        """
        if H is None:
            return np.array([], dtype=np.float32)

        # Use bottom-center of bbox (feet position on ground)
        cx = (xyxy[0, 0] + xyxy[0, 2]) / 2.0
        cy = xyxy[0, 3]  # bottom edge

        pt = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pt, H)
        pitch_pt = warped.reshape(-1, 2).astype(np.float32)[0]

        if flip_x:
            pitch_pt[0] = pitch_length - pitch_pt[0]

        return pitch_pt

    def print_model_metadata(self) -> None:
        """Print ball model class names for debugging."""
        print("Ball model classes:")
        for cid, cname in self.model.names.items():
            print(f"  {cid}: {cname}")
        print(f"  Using ball_class_id={self.ball_class_id}")
