"""
Keypoint-based homography computation for football pitch registration.

Pipeline:
    YOLO-Pose inference → 29 keypoints
        ↓ filter by confidence & visibility
        ↓ map keypoint_id → real-world pitch coordinate
        ↓ cv2.findHomography(RANSAC) → H matrix
"""

from ultralytics import YOLO
import numpy as np
import cv2
from constants import (
    KEYPOINT_CONF,
    KEYPOINT_VISIBILITY,
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
        # Left penalty area (big_rect) — clockwise from outer-top
        1: (0.0, PENALTY_Y_TOP),             # big_rect_left_top_pt1 (outer)
        2: (LEFT_PENALTY_X, PENALTY_Y_TOP),  # big_rect_left_top_pt2 (inner)
        3: (LEFT_PENALTY_X, PENALTY_Y_BOTTOM), # big_rect_left_bottom_pt1 (inner)
        4: (0.0, PENALTY_Y_BOTTOM),          # big_rect_left_bottom_pt2 (outer)
        # Left goal area (small_rect) — clockwise from outer-top
        5: (0.0, GOAL_AREA_Y_TOP),           # small_rect_left_top_pt1 (outer)
        6: (LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP), # small_rect_left_top_pt2 (inner)
        7: (LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM), # small_rect_left_bottom_pt1 (inner)
        8: (0.0, GOAL_AREA_Y_BOTTOM),        # small_rect_left_bottom_pt2 (outer)
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
        # Right penalty area (big_rect) — clockwise from inner-top
        17: (RIGHT_PENALTY_X, PENALTY_Y_TOP),  # big_rect_right_top_pt1 (inner)
        18: (PITCH_LENGTH, PENALTY_Y_TOP),    # big_rect_right_top_pt2 (outer)
        19: (PITCH_LENGTH, PENALTY_Y_BOTTOM), # big_rect_right_bottom_pt1 (outer)
        20: (RIGHT_PENALTY_X, PENALTY_Y_BOTTOM), # big_rect_right_bottom_pt2 (inner)
        # Right goal area (small_rect) — clockwise from inner-top
        21: (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP),   # small_rect_right_top_pt1 (inner)
        22: (PITCH_LENGTH, GOAL_AREA_Y_TOP),        # small_rect_right_top_pt2 (outer)
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
    Runs YOLO-Pose keypoint inference, filters keypoints by confidence and
    visibility, maps them to real-world pitch coordinates, and computes the
    homography matrix via RANSAC.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = KEYPOINT_CONF,
        visibility_threshold: float = KEYPOINT_VISIBILITY,
        top_k: int = 4,
        exclude_kpt_ids: set = None,
    ):
        """
        Args:
            model_path: Path to the YOLO-Pose model (.pt file).
            conf_threshold: Minimum overall detection confidence.
            visibility_threshold: Minimum per-keypoint visibility (0-1).
            top_k: Use only the top-k highest-confidence keypoints for
                   homography (after filtering by visibility_threshold).
            exclude_kpt_ids: Set of keypoint IDs to exclude (e.g. collinear
                   points that would make the homography degenerate).
                   Default excludes center-line points (11,12,13,14,15).
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.visibility_threshold = visibility_threshold
        self.top_k = top_k
        # Exclude keypoint 15 (field_center) which is collinear with 13 & 14
        # on the same vertical line (x=CENTER_X), causing degenerate homography
        self.exclude_kpt_ids = exclude_kpt_ids if exclude_kpt_ids is not None else {15}

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
        has (x, y, visibility). visibility=0 means not visible, 1 means
        occluded, 2 means fully visible. We treat > 0 as visible enough.

        Args:
            result: A single YOLO result object (result.keypoints).
            debug: If True, print debug info.

        Returns:
            keypoints: np.ndarray of shape (K, 3) = (x, y, visibility)
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
            vis_values = kps_np[:, 2]
            print(f"  [_extract] visibility stats: min={vis_values.min():.3f}, max={vis_values.max():.3f}, mean={vis_values.mean():.3f}")
            print(f"  [_extract] num with vis>0: {(vis_values > 0).sum()}")
            print(f"  [_extract] num with vis>=0.5: {(vis_values >= 0.5).sum()}")
            # Show some sample keypoints
            valid_idx = np.where(vis_values > 0)[0]
            if len(valid_idx) > 0:
                print(f"  [_extract] Sample valid keypoints (first 5):")
                for idx in valid_idx[:5]:
                    name = PitchKeypointMapper.get_keypoint_name(int(idx))
                    print(f"    kpt {idx} ({name}): x={kps_np[idx,0]:.1f}, y={kps_np[idx,1]:.1f}, vis={kps_np[idx,2]:.3f}")

        return kps_np

    def compute_homography(
        self,
        frame: np.ndarray,
        last_H: np.ndarray = None,
        ransac_reproj_threshold: float = 3.0,
    ):
        """
        Run keypoint inference and compute homography from valid keypoint
        correspondences.

        Pipeline:
            1. Run YOLO-Pose inference on frame.
            2. Extract all 29 keypoints with (x, y, visibility).
            3. Filter: keep keypoints with confidence >= conf_threshold AND
               visibility >= visibility_threshold.
            4. Map kept keypoints to real-world pitch coordinates.
            5. Compute H via cv2.findHomography(RANSAC) if >= 4 correspondences.
            6. Fall back to last_H if insufficient keypoints and
               REUSE_LAST_HOMOGRAPHY is set.

        Args:
            frame: BGR image np.ndarray (H, W, 3).
            last_H: Previously computed homography matrix for fallback.
            ransac_reproj_threshold: RANSAC reprojection error threshold.

        Returns:
            (H, info_dict)
                H: 3x3 homography matrix or None.
                info_dict: {
                    'H': H or None,
                    'mode': str - 'keypoint-ransac' | 'fallback-last' | 'insufficient' | 'none',
                    'used_keypoints': list of {kpt_id, name, confidence, visibility, image_pt, pitch_pt},
                    'inliers': int,
                    'total_points': int,
                    'confidences': list of floats,
                }
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

        # ---- Step 3: Filter & rank keypoints ----
        # Collect all keypoints that pass the low visibility threshold
        candidates = []

        for kpt_id in range(len(kps)):
            # Skip excluded collinear keypoints (e.g., center-line points)
            if kpt_id in self.exclude_kpt_ids:
                continue

            x_img = float(kps[kpt_id, 0])
            y_img = float(kps[kpt_id, 1])
            visibility = float(kps[kpt_id, 2])

            if visibility < self.visibility_threshold:
                continue

            pitch_coords = PitchKeypointMapper.get_pitch_coords(kpt_id)
            if pitch_coords is None:
                continue

            kpt_name = PitchKeypointMapper.get_keypoint_name(kpt_id)

            candidates.append({
                'kpt_id': kpt_id,
                'name': kpt_name,
                'image_pt': (x_img, y_img),
                'pitch_pt': pitch_coords,
                'visibility': visibility,
            })

        # Sort by visibility descending and take top_k
        candidates.sort(key=lambda c: c['visibility'], reverse=True)
        top_candidates = candidates[:self.top_k]

        # Build correspondence arrays from top-k candidates only
        image_pts = []
        pitch_pts = []
        used_kpts = []

        for c in top_candidates:
            image_pts.append([c['image_pt'][0], c['image_pt'][1]])
            pitch_pts.append([c['pitch_pt'][0], c['pitch_pt'][1]])
            used_kpts.append(c)

        info['total_points'] = len(candidates)  # total that passed threshold
        info['used_keypoints'] = used_kpts
        info['confidences'] = [c['visibility'] for c in top_candidates]

        # ---- Step 4 & 5: Compute homography ----
        if len(image_pts) < 4:
            return self._fallback_or_none(last_H, info, f'insufficient-keypoints ({len(image_pts)} < 4)')

        src_pts = np.array(image_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(pitch_pts, dtype=np.float32).reshape(-1, 1, 2)

        if len(src_pts) == 4:
            # Exact solve with 4 points
            H = cv2.getPerspectiveTransform(
                src_pts.reshape(-1, 2).astype(np.float32),
                dst_pts.reshape(-1, 2).astype(np.float32),
            )
            info['mode'] = 'keypoint-exact'
            info['inliers'] = 4
            info['H'] = H
            return H, info
        else:
            # RANSAC with > 4 points
            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_threshold,
            )
            if H is not None:
                info['mode'] = 'keypoint-ransac'
                info['inliers'] = int(mask.sum()) if mask is not None else 0
                info['H'] = H
                return H, info
            else:
                return self._fallback_or_none(last_H, info, 'ransac-failed')

    def _fallback_or_none(self, last_H, info, reason: str):
        """Try fallback to last_H or return None."""
        info['mode'] = reason
        if REUSE_LAST_HOMOGRAPHY and last_H is not None:
            info['mode'] = 'fallback-last'
            info['H'] = last_H
            return last_H, info
        return None, info
