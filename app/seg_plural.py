from constants import SEGMENTATION_PRIORITY, SEG_CONF
import cv2
import numpy as np


class BestSegmentPicker():
    # ------------------------------------------------------------------
    # Single-instance picker (legacy)
    # ------------------------------------------------------------------
    @staticmethod
    def choose_best_single_instance(used_instances):
        scored = []

        for inst in used_instances:
            if inst.get("confidence", 0.0) <= SEG_CONF:
                continue

            class_name = inst.get("class_name", "")
            score = SEGMENTATION_PRIORITY.get(class_name, 0)

            scored.append((score, inst))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    @staticmethod
    def compute_homography_from_single_instance(instance):
        if instance is None:
            return None

        image_quad = np.asarray(instance["image_bbox"], dtype=np.float32)
        pitch_quad = np.asarray(instance["canvas_bbox"], dtype=np.float32)

        if image_quad.shape != (4, 2) or pitch_quad.shape != (4, 2):
            return None

        return cv2.getPerspectiveTransform(image_quad, pitch_quad)

    # ------------------------------------------------------------------
    # Multi-instance homography (recommended)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_homography_from_all_instances(
        processed_results,
        min_conf=None,
        ransac_reproj_threshold=3.0,
    ):
        """
        Combine the 4-point correspondences from EVERY valid detected
        segment into a single homography using cv2.findHomography(RANSAC).
        """
        if min_conf is None:
            min_conf = SEG_CONF

        src_points = []
        dst_points = []
        used = []

        for inst in processed_results:
            if inst.get("confidence", 0.0) <= min_conf:
                continue

            image_quad = inst.get("image_bbox")
            canvas_quad = inst.get("canvas_bbox")

            if image_quad is None or canvas_quad is None:
                continue

            image_quad = np.asarray(image_quad, dtype=np.float32).reshape(-1, 2)
            canvas_quad = np.asarray(canvas_quad, dtype=np.float32).reshape(-1, 2)

            if image_quad.shape != (4, 2) or canvas_quad.shape != (4, 2):
                continue

            if not np.all(np.isfinite(image_quad)) or not np.all(np.isfinite(canvas_quad)):
                continue

            src_points.append(image_quad)
            dst_points.append(canvas_quad)

            used.append({
                "class_name": inst.get("class_name", "?"),
                "side_hint": inst.get("side_hint", "?"),
                "confidence": float(inst.get("confidence", 0.0)),
            })

        info = {
            "used": used,
            "mode": "none",
            "inliers": 0,
            "total_points": 0,
        }

        if len(src_points) == 0:
            return None, info

        src = np.vstack(src_points).astype(np.float32)
        dst = np.vstack(dst_points).astype(np.float32)

        info["total_points"] = int(len(src))

        if len(src) < 4:
            info["mode"] = "insufficient"
            return None, info

        if len(src) == 4:
            H = cv2.getPerspectiveTransform(src, dst)
            info["mode"] = "single-segment"
            info["inliers"] = 4
            return H, info

        H, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
        )

        info["mode"] = "multi-segment-ransac"
        info["inliers"] = int(mask.sum()) if mask is not None else 0

        return H, info

    # ------------------------------------------------------------------
    # Convenience: one call that picks the best strategy
    # ------------------------------------------------------------------
    @staticmethod
    def compute_homography(processed_results):
        """
        Try multi-segment RANSAC homography first; if that fails, fall
        back to the single best-scoring instance.

        Returns:
            (H, info)
                H    : 3x3 homography  or  None
                info : dict with debugging info
        """
        H, info = BestSegmentPicker.compute_homography_from_all_instances(
            processed_results
        )

        if H is not None:
            return H, info

        best = BestSegmentPicker.choose_best_single_instance(processed_results)

        if best is None:
            return None, info

        H = BestSegmentPicker.compute_homography_from_single_instance(best)

        info["mode"] = "fallback-single"
        info["used"] = [{
            "class_name": best.get("class_name", "?"),
            "side_hint": best.get("side_hint", "?"),
            "confidence": float(best.get("confidence", 0.0)),
        }]

        return H, info
