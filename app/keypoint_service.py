"""
Keypoint-based homography computation for football pitch registration.

Pipeline:
    YOLO-Pose inference → 29 keypoints
        ↓ filter by confidence threshold (KEYPOINT_MIN_CONF)
        ↓ map keypoint_id → real-world pitch coordinate
        ↓ cv2.findHomography(DLT) → H matrix
"""

from ultralytics import YOLO
import numpy as np
import cv2
from constants import (
    KEYPOINT_CONF,
    KEYPOINT_MIN_CONF,
    PITCH_LENGTH,
    PITCH_WIDTH,
    CENTER_X,
    CENTER_Y,
    CENTER_CIRCLE_RADIUS,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_WIDTH,
    PENALTY_Y_TOP,
    PENALTY_Y_BOTTOM,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    GOAL_AREA_Y_TOP,
    GOAL_AREA_Y_BOTTOM,
    PENALTY_SPOT_DISTANCE,
    PENALTY_ARC_RADIUS,
    LEFT_PENALTY_X,
    RIGHT_PENALTY_X,
    LEFT_GOAL_AREA_X,
    RIGHT_GOAL_AREA_X,
    LEFT_PENALTY_SPOT_X,
    RIGHT_PENALTY_SPOT_X,
    REUSE_LAST_HOMOGRAPHY,
    SMOOTHING_ALPHA,
    H_STABILITY_THRESHOLD,
)


class PitchKeypointMapper:
    """
    Maps YOLO-Pose keypoint IDs (0-28) to real-world pitch coordinates (meters).
    The pitch frame is 105m × 68m with origin (0,0) at top-left.
    """

    KEYPOINT_NAMES = {
        0: "sideline_top_left",
        1: "big_rect_left_top_pt1",
        2: "big_rect_left_top_pt2",
        3: "big_rect_left_bottom_pt1",
        4: "big_rect_left_bottom_pt2",
        5: "small_rect_left_top_pt1",
        6: "small_rect_left_top_pt2",
        7: "small_rect_left_bottom_pt1",
        8: "small_rect_left_bottom_pt2",
        9: "sideline_bottom_left",
        10: "left_semicircle_right",
        11: "center_line_top",
        12: "center_line_bottom",
        13: "center_circle_top",
        14: "center_circle_bottom",
        15: "field_center",
        16: "sideline_top_right",
        17: "big_rect_right_top_pt1",
        18: "big_rect_right_top_pt2",
        19: "big_rect_right_bottom_pt1",
        20: "big_rect_right_bottom_pt2",
        21: "small_rect_right_top_pt1",
        22: "small_rect_right_top_pt2",
        23: "small_rect_right_bottom_pt1",
        24: "small_rect_right_bottom_pt2",
        25: "sideline_bottom_right",
        26: "right_semicircle_left",
        27: "center_circle_left",
        28: "center_circle_right",
    }

    # Pre-computed pitch coordinates for each keypoint ID.
    # (x, y) in meters on the standardized 105m × 68m pitch.
    _PITCH_COORDS = {
        # Left sideline
        0: (0.0, 0.0),                       # sideline_top_left
        9: (0.0, PITCH_WIDTH),               # sideline_bottom_left
        # Left penalty area (big_rect)
        # pt1 = outer (sideline side, x=0), pt2 = inner (center side, x=LEFT_PENALTY_X)
        1: (0.0, PENALTY_Y_TOP),             # big_rect_left_top_pt1 (outer)
        2: (LEFT_PENALTY_X, PENALTY_Y_TOP),  # big_rect_left_top_pt2 (inner)
        3: (0.0, PENALTY_Y_BOTTOM),          # big_rect_left_bottom_pt1 (outer)
        4: (LEFT_PENALTY_X, PENALTY_Y_BOTTOM), # big_rect_left_bottom_pt2 (inner)
        # Left goal area (small_rect)
        # pt1 = outer (sideline side, x=0), pt2 = inner (center side, x=LEFT_GOAL_AREA_X)
        5: (0.0, GOAL_AREA_Y_TOP),           # small_rect_left_top_pt1 (outer)
        6: (LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP), # small_rect_left_top_pt2 (inner)
        7: (0.0, GOAL_AREA_Y_BOTTOM),        # small_rect_left_bottom_pt1 (outer)
        8: (LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM), # small_rect_left_bottom_pt2 (inner)
        # Left D-box arc
        10: (LEFT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS, CENTER_Y),  # left_semicircle_right
        # Center line & circle
        11: (CENTER_X, 0.0),                  # center_line_top
        12: (CENTER_X, PITCH_WIDTH),          # center_line_bottom
        13: (CENTER_X, CENTER_Y - CENTER_CIRCLE_RADIUS),  # center_circle_top
        14: (CENTER_X, CENTER_Y + CENTER_CIRCLE_RADIUS),  # center_circle_bottom
        15: (CENTER_X, CENTER_Y),             # field_center
        27: (CENTER_X - CENTER_CIRCLE_RADIUS, CENTER_Y),  # center_circle_left
        28: (CENTER_X + CENTER_CIRCLE_RADIUS, CENTER_Y),  # center_circle_right
        # Right sideline
        16: (PITCH_LENGTH, 0.0),              # sideline_top_right
        25: (PITCH_LENGTH, PITCH_WIDTH),      # sideline_bottom_right
        # Right penalty area (big_rect)
        # pt1 = outer (sideline side, x=PITCH_LENGTH), pt2 = inner (center side, x=RIGHT_PENALTY_X)
        17: (PITCH_LENGTH, PENALTY_Y_TOP),    # big_rect_right_top_pt1 (outer)
        18: (RIGHT_PENALTY_X, PENALTY_Y_TOP), # big_rect_right_top_pt2 (inner)
        19: (PITCH_LENGTH, PENALTY_Y_BOTTOM), # big_rect_right_bottom_pt1 (outer)
        20: (RIGHT_PENALTY_X, PENALTY_Y_BOTTOM), # big_rect_right_bottom_pt2 (inner)
        # Right goal area (small_rect)
        # pt1 = outer (sideline side, x=PITCH_LENGTH), pt2 = inner (center side, x=RIGHT_GOAL_AREA_X)
        21: (PITCH_LENGTH, GOAL_AREA_Y_TOP),        # small_rect_right_top_pt1 (outer)
        22: (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP),   # small_rect_right_top_pt2 (inner)
        23: (PITCH_LENGTH, GOAL_AREA_Y_BOTTOM),     # small_rect_right_bottom_pt1 (outer)
        24: (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM),# small_rect_right_bottom_pt2 (inner)
        # Right D-box arc
        26: (RIGHT_PENALTY_SPOT_X - PENALTY_ARC_RADIUS, CENTER_Y),  # right_semicircle_left
    }

    @staticmethod
    def get_pitch_coords(kpt_id: int):
        """
        Return the real-world pitch (x, y) in meters for a keypoint ID.

        Args:
            kpt_id: Keypoint index 0-28.

        Returns:
            (pitch_x, pitch_y) as floats, or None if the ID is out of range.
        """
        return PitchKeypointMapper._PITCH_COORDS.get(kpt_id, None)

    @staticmethod
    def get_keypoint_name(kpt_id: int) -> str:
        """Return the human-readable name for a keypoint ID."""
        return PitchKeypointMapper.KEYPOINT_NAMES.get(kpt_id, f"unknown_{kpt_id}")


class KeypointHomographyComputer:
    """
    Runs YOLO-Pose keypoint inference, filters keypoints by confidence,
    maps them to real-world pitch coordinates, and computes the
    homography matrix via DLT (Direct Linear Transform).

    Supports temporal smoothing of the homography matrix to reduce
    frame-to-frame jitter in the output video.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = KEYPOINT_CONF,
        min_conf_threshold: float = KEYPOINT_MIN_CONF,
        exclude_kpt_ids: set = None,
        smoothing_alpha: float = SMOOTHING_ALPHA,
        stability_threshold: float = H_STABILITY_THRESHOLD,
    ):
        """
        Args:
            model_path: Path to the YOLO-Pose model (.pt file).
            conf_threshold: Minimum overall detection confidence.
            min_conf_threshold: Minimum per-keypoint confidence (0-1).
            exclude_kpt_ids: Set of keypoint IDs to exclude (e.g. collinear
                   points that would make the homography degenerate).
                   Default excludes keypoint 15 (field_center).
            smoothing_alpha: EMA factor for temporal H smoothing
                   (0 = no update from new H, 1 = instant update, no smoothing).
            stability_threshold: Max relative Frobenius-norm change between
                   successive H matrices before rejecting a new H as unstable.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_conf_threshold = min_conf_threshold
        self.smoothing_alpha = smoothing_alpha
        self.stability_threshold = stability_threshold
        self.smoothed_H = None  # Accumulator for EMA smoothing
        # Exclude perfectly collinear center-line keypoints (all on same x=CENTER_X line),
        # which cause degenerate homography when over-represented.
        # Excluded: 11=center_line_top, 12=center_line_bottom, 15=field_center
        # Re-included: 13=center_circle_top, 14=center_circle_bottom (vertical constraint)
        # Keeping 27 (center_circle_left) and 28 (center_circle_right) since
        # they provide horizontal constraint on y=CENTER_Y.
        self.exclude_kpt_ids = exclude_kpt_ids if exclude_kpt_ids is not None else {11, 12, 15}

    def print_model_metadata(self):
        """Print keypoint model class names and configuration."""
        print("Keypoint classes:")
        for class_id, class_name in self.model.names.items():
            print(f"  {class_id}: {class_name}")
        # The pose model has keypoint shape info
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml'):
            yaml = self.model.model.yaml
            if yaml and 'kpt_shape' in yaml:
                print(f"  kpt_shape: {yaml['kpt_shape']}")

    def _extract_keypoints(self, result, debug=False):
        """
        Extract keypoints from a YOLO-Pose result.

        YOLO-Pose returns keypoints with shape (N, K, 3) where each keypoint
        has (x, y, confidence) with confidence in [0, 1].

        Args:
            result: A single YOLO result object (result.keypoints).
            debug: If True, print debug info.

        Returns:
            keypoints: np.ndarray of shape (K, 3) = (x, y, confidence)
                       or None if no keypoints detected.
        """
        if result is None:
            if debug: print("  [_extract] result is None")
            return None

        kps_obj = getattr(result, 'keypoints', None)
        if kps_obj is None:
            if debug: print("  [_extract] result.keypoints is None")
            return None

        # Check data
        kps_data = kps_obj.data
        if kps_data is None or len(kps_data) == 0:
            if debug: print("  [_extract] kps_obj.data is None/empty")
            return None

        if debug:
            print(f"  [_extract] kps_data shape: {kps_data.shape}")
            print(f"  [_extract] num detections (N): {kps_data.shape[0]}")
            print(f"  [_extract] keypoints per detection (K): {kps_data.shape[1]}")

        # Take the first (highest-confidence) detection's keypoints
        # kps_np shape: (K, 3)
        kps_np = kps_data[0].cpu().numpy().astype(np.float32)

        if debug:
            print(f"  [_extract] kps_np shape: {kps_np.shape}")
            # Show summary stats
            conf_values = kps_np[:, 2]
            print(f"  [_extract] confidence stats: min={conf_values.min():.3f}, max={conf_values.max():.3f}, mean={conf_values.mean():.3f}")
            print(f"  [_extract] num with conf>0: {(conf_values > 0).sum()}")
            print(f"  [_extract] num with conf>=0.5: {(conf_values >= 0.5).sum()}")
            # Show some sample keypoints
            valid_idx = np.where(conf_values > 0)[0]
            if len(valid_idx) > 0:
                print(f"  [_extract] Sample valid keypoints (first 5):")
                for idx in valid_idx[:5]:
                    name = PitchKeypointMapper.get_keypoint_name(int(idx))
                    print(f"    kpt {idx} ({name}): x={kps_np[idx,0]:.1f}, y={kps_np[idx,1]:.1f}, conf={kps_np[idx,2]:.3f}")

        return kps_np

    @staticmethod
    def _validate_keypoint_with_segmentation(
        kp_image_pt: tuple,
        processed_segments: list,
        kpt_id: int = -1,
        margin_px: float = 5.0,
    ) -> bool:
        """
        Validate a keypoint by checking if it falls inside any segmentation
        contour polygon. Keypoints outside all pitch region contours are
        likely incorrect model predictions.

        Uses cv2.pointPolygonTest which returns:
            +1 if point is inside
             0 if point is on the edge
            -1 if point is outside

        Args:
            kp_image_pt: (x, y) in image pixel coordinates.
            processed_segments: list from Segmentor.extract().
            kpt_id: Keypoint ID for debug logging.
            margin_px: Tolerance in pixels for pointPolygonTest.

        Returns:
            True if the keypoint is inside or on the edge of any segment.
        """
        if not processed_segments:
            return True  # no segmentation data → accept (fallback)

        point = (float(kp_image_pt[0]), float(kp_image_pt[1]))

        for seg in processed_segments:
            contour = seg.get('image_contour', None)
            if contour is None:
                continue
            # contour from YOLO is (N, 1, 2), pointPolygonTest expects that
            result = cv2.pointPolygonTest(contour, point, measureDist=False)
            # result >= 0 means inside or on edge
            if result >= 0:
                return True

        # Keypoint is outside all segments — log for debugging
        kpt_name = PitchKeypointMapper.get_keypoint_name(kpt_id)
        return False

    def compute_homography(
        self,
        frame: np.ndarray,
        last_H: np.ndarray = None,
        processed_segments: list = None,
    ):
        """
        Run keypoint inference and compute homography from valid keypoint
        correspondences.

        Pipeline:
            1. Run YOLO-Pose inference on frame.
            2. Extract all 29 keypoints with (x, y, confidence).
            3. Filter by confidence threshold (KEYPOINT_MIN_CONF).
            4. Validate keypoints against segmentation contours (if provided):
               only keep keypoints that fall inside a pitch segment polygon.
            5. Map validated keypoints to real-world pitch coordinates.
            6. Compute H via cv2.findHomography(DLT) using all valid keypoints.
            7. Fall back to last_H if computation fails.

        Args:
            frame: BGR image np.ndarray (H, W, 3).
            last_H: Previously computed homography matrix for fallback.
            processed_segments: list from Segmentor.extract() — used to
                                validate keypoints are on the pitch.

        Returns:
            (H, info_dict)
        """
        # Default info
        info = {
            'H': None,
            'mode': 'none',
            'used_keypoints': [],
            'inliers': 0,
            'total_points': 0,
            'confidences': [],
        }

        # ---- Step 1: Inference ----
        output = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        if not output:
            return self._fallback_or_none(last_H, info, 'no-detection')

        result = output[0]

        # ---- Step 2: Extract keypoints ----
        kps = self._extract_keypoints(result)
        if kps is None or len(kps) == 0:
            return self._fallback_or_none(last_H, info, 'no-keypoints')

        # ---- Step 3: Collect & validate keypoints ----
        candidates = []

        for kpt_id in range(len(kps)):
            # Skip excluded keypoints (e.g., field_center which is collinear)
            if kpt_id in self.exclude_kpt_ids:
                continue

            x_img = float(kps[kpt_id, 0])
            y_img = float(kps[kpt_id, 1])
            confidence = float(kps[kpt_id, 2])

            if confidence < self.min_conf_threshold:
                continue

            # Validate against segmentation contours
            if processed_segments is not None and len(processed_segments) > 0:
                if not self._validate_keypoint_with_segmentation(
                    (x_img, y_img), processed_segments, kpt_id=kpt_id
                ):
                    continue  # keypoint is outside all pitch segments → discard

            pitch_coords = PitchKeypointMapper.get_pitch_coords(kpt_id)
            if pitch_coords is None:
                continue

            kpt_name = PitchKeypointMapper.get_keypoint_name(kpt_id)

            candidates.append({
                'kpt_id': kpt_id,
                'name': kpt_name,
                'image_pt': (x_img, y_img),
                'pitch_pt': pitch_coords,
                'confidence': confidence,
            })

        info['total_points'] = len(candidates)
        info['used_keypoints'] = candidates
        info['confidences'] = [c['confidence'] for c in candidates]

        # ---- Step 4: Build correspondence arrays ----
        if len(candidates) < 4:
            return self._fallback_or_none(last_H, info,
                f'insufficient-keypoints ({len(candidates)} < 4)')

        src_pts = np.array(
            [[c['image_pt'][0], c['image_pt'][1]] for c in candidates],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        dst_pts = np.array(
            [[c['pitch_pt'][0], c['pitch_pt'][1]] for c in candidates],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        # ---- Step 5: Compute homography via RANSAC ----
        # RANSAC is robust to outliers: it randomly samples 4-point subsets,
        # finds the one maximizing inliers, and rejects outlier correspondences.
        # This handles low-confidence keypoints that passed the confidence filter
        # but are still geometrically inconsistent.
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )

        if H is not None:
            # Count inliers from the RANSAC mask
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                inliers = int(inlier_mask.sum())
                inlier_ratio = inliers / len(candidates) if len(candidates) > 0 else 0.0
            else:
                inliers = len(candidates)
                inlier_ratio = 1.0

            # Reject if insufficient inliers (degenerate geometry)
            if inliers < 4 or inlier_ratio < 0.3:
                return self._fallback_or_none(last_H, info,
                    f'ransac-insufficient-inliers ({inliers}/{len(candidates)}, ratio={inlier_ratio:.2f})')

            info['mode'] = 'keypoint-ransac'
            info['inliers'] = inliers
            info['H'] = H

            # ---- Step 6: Temporal smoothing (EMA + stability gate) ----
            H = self._apply_homography_smoothing(H, last_H)

            info['H'] = H
            return H, info
        else:
            return self._fallback_or_none(last_H, info, 'ransac-failed')

    def _apply_homography_smoothing(
        self, raw_H: np.ndarray, last_H: np.ndarray
    ) -> np.ndarray:
        """
        Apply temporal smoothing to the homography matrix using EMA
        (Exponential Moving Average), with a stability gate to reject
        large sudden changes.

        Strategy:
            1. If no previous smoothed_H exists, initialize it with raw_H.
            2. If a stability_threshold is set, compute the relative
               Frobenius-norm change between raw_H and smoothed_H.
               If the change exceeds the threshold, reject raw_H and
               keep the previous smoothed_H (or fall back to last_H).
            3. Apply EMA: smoothed_H = alpha * raw_H + (1 - alpha) * smoothed_H.
            4. Normalize H[2,2] to 1.0 to keep it a valid homography.

        Args:
            raw_H: Freshly computed homography from RANSAC.
            last_H: Previous frame's H for fallback (same as self.smoothed_H
                    in typical usage).

        Returns:
            Smoothed homography matrix (3x3).
        """
        # ---- Stability gate: reject H if it changes too drastically ----
        if self.smoothed_H is not None and self.stability_threshold > 0:
            change = np.linalg.norm(raw_H - self.smoothed_H) / (
                np.linalg.norm(self.smoothed_H) + 1e-10
            )
            if change > self.stability_threshold:
                # Reject — keep the previous smoothed matrix
                return self.smoothed_H

        # ---- EMA smoothing ----
        if self.smoothed_H is None:
            self.smoothed_H = raw_H.copy()
        else:
            alpha = self.smoothing_alpha
            self.smoothed_H = alpha * raw_H + (1.0 - alpha) * self.smoothed_H

        # Normalize H[2,2] to 1.0
        self.smoothed_H = self.smoothed_H / self.smoothed_H[2, 2]

        return self.smoothed_H

    def _fallback_or_none(self, last_H, info, reason: str):
        """Try fallback to last_H or return None."""
        info['mode'] = reason
        if REUSE_LAST_HOMOGRAPHY and last_H is not None:
            info['mode'] = 'fallback-last'
            info['H'] = last_H
            # When falling back, also sync the smoothed_H to prevent
            # a sudden jump when a new H is eventually accepted.
            if self.smoothed_H is not None:
                # Blend last_H into smoothed_H to keep it warm
                alpha = self.smoothing_alpha * 0.5  # gentle blend on fallback
                self.smoothed_H = (
                    alpha * last_H + (1.0 - alpha) * self.smoothed_H
                )
                self.smoothed_H = self.smoothed_H / self.smoothed_H[2, 2]
            else:
                self.smoothed_H = last_H.copy()
            return last_H, info
        return None, info
