"""Keypoint-based homography for football pitch registration. YOLO-Pose → 29 keypoints → filter → map to pitch coords → DLT homography."""
from ultralytics import YOLO
import numpy as np
import cv2
from constants import (
    KEYPOINT_CONF, KEYPOINT_MIN_CONF, PITCH_LENGTH, PITCH_WIDTH, CENTER_X, CENTER_Y,
    CENTER_CIRCLE_RADIUS, PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH, PENALTY_Y_TOP, PENALTY_Y_BOTTOM,
    GOAL_AREA_DEPTH, GOAL_AREA_WIDTH, GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM, PENALTY_SPOT_DISTANCE,
    PENALTY_ARC_RADIUS, LEFT_PENALTY_X, RIGHT_PENALTY_X, LEFT_GOAL_AREA_X, RIGHT_GOAL_AREA_X,
    LEFT_PENALTY_SPOT_X, RIGHT_PENALTY_SPOT_X, REUSE_LAST_HOMOGRAPHY, SMOOTHING_ALPHA, H_STABILITY_THRESHOLD,
)


class PitchKeypointMapper:
    """Maps YOLO-Pose keypoint IDs (0-28) to real-world pitch coordinates (meters) on a 105m×68m pitch."""

    KEYPOINT_NAMES = {
        0: "sideline_top_left", 1: "big_rect_left_top_pt1", 2: "big_rect_left_top_pt2",
        3: "big_rect_left_bottom_pt1", 4: "big_rect_left_bottom_pt2",
        5: "small_rect_left_top_pt1", 6: "small_rect_left_top_pt2",
        7: "small_rect_left_bottom_pt1", 8: "small_rect_left_bottom_pt2",
        9: "sideline_bottom_left", 10: "left_semicircle_right",
        11: "center_line_top", 12: "center_line_bottom", 13: "center_circle_top",
        14: "center_circle_bottom", 15: "field_center", 16: "sideline_top_right",
        17: "big_rect_right_top_pt1", 18: "big_rect_right_top_pt2",
        19: "big_rect_right_bottom_pt1", 20: "big_rect_right_bottom_pt2",
        21: "small_rect_right_top_pt1", 22: "small_rect_right_top_pt2",
        23: "small_rect_right_bottom_pt1", 24: "small_rect_right_bottom_pt2",
        25: "sideline_bottom_right", 26: "right_semicircle_left",
        27: "center_circle_left", 28: "center_circle_right",
    }

    _PITCH_COORDS = {
        0: (0.0, 0.0), 9: (0.0, PITCH_WIDTH),
        1: (0.0, PENALTY_Y_TOP), 2: (LEFT_PENALTY_X, PENALTY_Y_TOP),
        3: (0.0, PENALTY_Y_BOTTOM), 4: (LEFT_PENALTY_X, PENALTY_Y_BOTTOM),
        5: (0.0, GOAL_AREA_Y_TOP), 6: (LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP),
        7: (0.0, GOAL_AREA_Y_BOTTOM), 8: (LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM),
        10: (LEFT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS, CENTER_Y),
        11: (CENTER_X, 0.0), 12: (CENTER_X, PITCH_WIDTH),
        13: (CENTER_X, CENTER_Y - CENTER_CIRCLE_RADIUS),
        14: (CENTER_X, CENTER_Y + CENTER_CIRCLE_RADIUS),
        15: (CENTER_X, CENTER_Y),
        27: (CENTER_X - CENTER_CIRCLE_RADIUS, CENTER_Y),
        28: (CENTER_X + CENTER_CIRCLE_RADIUS, CENTER_Y),
        16: (PITCH_LENGTH, 0.0), 25: (PITCH_LENGTH, PITCH_WIDTH),
        17: (PITCH_LENGTH, PENALTY_Y_TOP), 18: (RIGHT_PENALTY_X, PENALTY_Y_TOP),
        19: (PITCH_LENGTH, PENALTY_Y_BOTTOM), 20: (RIGHT_PENALTY_X, PENALTY_Y_BOTTOM),
        21: (PITCH_LENGTH, GOAL_AREA_Y_TOP), 22: (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP),
        23: (PITCH_LENGTH, GOAL_AREA_Y_BOTTOM), 24: (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM),
        26: (RIGHT_PENALTY_SPOT_X - PENALTY_ARC_RADIUS, CENTER_Y),
    }

    @staticmethod
    def get_pitch_coords(kpt_id: int):
        return PitchKeypointMapper._PITCH_COORDS.get(kpt_id)

    @staticmethod
    def get_keypoint_name(kpt_id: int) -> str:
        return PitchKeypointMapper.KEYPOINT_NAMES.get(kpt_id, f"unknown_{kpt_id}")


class KeypointHomographyComputer:
    """Runs YOLO-Pose → filters keypoints → maps to pitch coords → DLT homography with temporal EMA smoothing."""

    RANSAC_REPROJ_THRESHOLD_M = 3.0
    SMOOTHING_MAX_EXTRA_ERROR_M = 1.0
    SMOOTHING_MAX_ERROR_RATIO = 1.35

    def __init__(self, model_path: str, conf_threshold: float = KEYPOINT_CONF,
                 min_conf_threshold: float = KEYPOINT_MIN_CONF, exclude_kpt_ids: set = None,
                 smoothing_alpha: float = SMOOTHING_ALPHA, stability_threshold: float = H_STABILITY_THRESHOLD):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_conf_threshold = min_conf_threshold
        self.smoothing_alpha = smoothing_alpha
        self.stability_threshold = stability_threshold
        self.smoothed_H = None
        self.exclude_kpt_ids = exclude_kpt_ids if exclude_kpt_ids is not None else {11, 12, 15}

    @staticmethod
    def _normalise_homography(H: np.ndarray):
        if H is None:
            return None
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return None
        if abs(H[2, 2]) > 1e-10:
            return H / H[2, 2]
        norm = np.linalg.norm(H)
        if norm <= 1e-10:
            return None
        return H / norm

    @staticmethod
    def _reprojection_rmse(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
        H = KeypointHomographyComputer._normalise_homography(H)
        if H is None or src is None or dst is None or len(src) == 0:
            return float("inf")
        try:
            projected = cv2.perspectiveTransform(
                np.asarray(src, dtype=np.float32).reshape(-1, 1, 2), H,
            ).reshape(-1, 2)
            target = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
        except cv2.error:
            return float("inf")
        if len(projected) != len(target) or len(projected) == 0:
            return float("inf")
        err = np.linalg.norm(projected - target, axis=1)
        return float(np.sqrt(np.mean(err ** 2)))

    def _extract_keypoints(self, result):
        kps_obj = getattr(result, 'keypoints', None)
        if kps_obj is None or kps_obj.data is None or len(kps_obj.data) == 0:
            return None
        return kps_obj.data[0].cpu().numpy().astype(np.float32)

    def compute_homography(self, frame: np.ndarray, last_H: np.ndarray = None):
        info = {'H': None, 'mode': 'none', 'used_keypoints': [], 'total_points': 0, 'confidences': []}
        output = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        if not output:
            return self._fallback_or_none(last_H, info, 'no-detection')
        kps = self._extract_keypoints(output[0])
        if kps is None or len(kps) == 0:
            return self._fallback_or_none(last_H, info, 'no-keypoints')

        candidates = []
        for kpt_id in range(len(kps)):
            if kpt_id in self.exclude_kpt_ids:
                continue
            conf = float(kps[kpt_id, 2])
            if conf < self.min_conf_threshold:
                continue
            pitch = PitchKeypointMapper.get_pitch_coords(kpt_id)
            if pitch is None:
                continue
            candidates.append({'kpt_id': kpt_id, 'name': PitchKeypointMapper.get_keypoint_name(kpt_id),
                               'image_pt': (float(kps[kpt_id, 0]), float(kps[kpt_id, 1])),
                               'pitch_pt': pitch, 'confidence': conf})

        candidates.sort(key=lambda c: c['confidence'], reverse=True)
        info['total_points'] = len(candidates)
        info['used_keypoints'] = candidates
        info['confidences'] = [c['confidence'] for c in candidates]
        if len(candidates) < 4:
            return self._fallback_or_none(last_H, info, f'insufficient-keypoints ({len(candidates)} < 4)')

        src = np.array([[c['image_pt'][0], c['image_pt'][1]] for c in candidates], dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([[c['pitch_pt'][0], c['pitch_pt'][1]] for c in candidates], dtype=np.float32).reshape(-1, 1, 2)
        H, inlier_mask = cv2.findHomography(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.RANSAC_REPROJ_THRESHOLD_M,
        )
        inlier_mask = None if inlier_mask is None else inlier_mask.reshape(-1).astype(bool)
        if H is None or inlier_mask is None or int(np.sum(inlier_mask)) < 4:
            H, _ = cv2.findHomography(src, dst, method=0)
            inlier_mask = np.ones(len(candidates), dtype=bool)
        if H is None:
            return self._fallback_or_none(last_H, info, 'dlt-failed')

        inliers = [c for c, keep in zip(candidates, inlier_mask) if keep]
        outliers = [c for c, keep in zip(candidates, inlier_mask) if not keep]
        fit_src = src[inlier_mask]
        fit_dst = dst[inlier_mask]

        info['mode'] = 'keypoint-ransac'
        info['used_keypoints'] = inliers
        info['outlier_keypoints'] = outliers
        info['inlier_count'] = len(inliers)
        info['outlier_count'] = len(outliers)
        info['H'] = H
        info['reprojection_rmse_raw'] = round(self._reprojection_rmse(H, fit_src, fit_dst), 3)
        H = self._apply_homography_smoothing(H, last_H, fit_src, fit_dst, info)
        info['H'] = H
        return H, info

    def _apply_homography_smoothing(self, raw_H: np.ndarray, last_H: np.ndarray,
                                    src: np.ndarray = None, dst: np.ndarray = None,
                                    info: dict = None) -> np.ndarray:
        raw_H = self._normalise_homography(raw_H)
        if raw_H is None:
            return last_H

        raw_error = self._reprojection_rmse(raw_H, src, dst)
        if info is not None:
            info['reprojection_rmse_raw'] = round(raw_error, 3) if np.isfinite(raw_error) else None

        if self.smoothed_H is None:
            self.smoothed_H = raw_H
            if info is not None:
                info['mode'] = f"{info.get('mode', 'keypoint')}-raw"
                info['reprojection_rmse'] = info.get('reprojection_rmse_raw')
                info['smoothing'] = 'initialised'
            return raw_H

        if self.smoothing_alpha >= 0.999:
            self.smoothed_H = raw_H
            if info is not None:
                info['mode'] = f"{info.get('mode', 'keypoint')}-raw"
                info['reprojection_rmse'] = info.get('reprojection_rmse_raw')
                info['smoothing'] = 'disabled'
            return raw_H

        prev_H = self._normalise_homography(self.smoothed_H)
        if prev_H is None:
            self.smoothed_H = raw_H
            return raw_H

        candidate_H = self.smoothing_alpha * raw_H + (1.0 - self.smoothing_alpha) * prev_H
        candidate_H = self._normalise_homography(candidate_H)
        smoothed_error = self._reprojection_rmse(candidate_H, src, dst)
        use_smoothed = True
        if np.isfinite(raw_error) and np.isfinite(smoothed_error):
            allowed = max(
                raw_error + self.SMOOTHING_MAX_EXTRA_ERROR_M,
                raw_error * self.SMOOTHING_MAX_ERROR_RATIO,
            )
            use_smoothed = smoothed_error <= allowed

        if use_smoothed and candidate_H is not None:
            self.smoothed_H = candidate_H
            if info is not None:
                info['mode'] = f"{info.get('mode', 'keypoint')}-smoothed"
                info['reprojection_rmse'] = round(smoothed_error, 3) if np.isfinite(smoothed_error) else None
                info['smoothing'] = 'accepted'
            return candidate_H

        # Current keypoints fit the raw H much better than the temporal H.
        # Reset instead of projecting the minimap with a stale transform.
        self.smoothed_H = raw_H
        if info is not None:
            info['mode'] = f"{info.get('mode', 'keypoint')}-raw"
            info['reprojection_rmse'] = info.get('reprojection_rmse_raw')
            info['reprojection_rmse_smoothed_rejected'] = (
                round(smoothed_error, 3) if np.isfinite(smoothed_error) else None
            )
            info['smoothing'] = 'rejected-current-frame-error'
        return raw_H

    def _fallback_or_none(self, last_H, info, reason: str):
        info['mode'] = reason
        if REUSE_LAST_HOMOGRAPHY and last_H is not None:
            info['mode'] = 'fallback-last'
            info['H'] = last_H
            if self.smoothed_H is not None:
                alpha = self.smoothing_alpha * 0.5
                self.smoothed_H = alpha * last_H + (1.0 - alpha) * self.smoothed_H
                self.smoothed_H /= self.smoothed_H[2, 2]
            else:
                self.smoothed_H = last_H.copy()
            return last_H, info
        return None, info
