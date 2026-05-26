"""
segmentation mask
    ↓
YOLO mask polygon points
    ↓
largest contour
    ↓
max-area quadrilateral from actual mask/hull points
    ↓
matched with known pitch quadrilateral
    ↓
homography matrix H
"""

from ultralytics import YOLO
from seg_helpers import CanvasMapper
import numpy as np
import cv2


class Segmentor():
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.classes = self.model.names

    def print_model_metadata(self) -> None:
        print("Segmentation classes:")
        for class_id, class_name in self.model.names.items():
            print(f"  {class_id}: {class_name}")

    def process_class_name(self, class_name):
        class_name = class_name.strip()
        class_name = class_name.replace("First ", "").replace("Second ", "")
        return class_name.strip()

    def extract_arc_box(self, mask):
        contour = BboxManipulor.get_largest_contour_from_yolo_mask(mask)

        if contour is None:
            return None, None

        bbox = BboxManipulor.contour_to_axis_aligned_bbox_points(contour)

        return bbox, contour

    def extract_rectangle(self, mask):
        contour = BboxManipulor.get_largest_contour_from_yolo_mask(mask)

        if contour is None or len(contour) < 4:
            return None, contour

        bbox = BboxManipulor.contour_to_max_area_quad(
            contour,
            max_candidates=60,
            min_area_ratio=0.10,
        )

        if bbox is None:
            bbox = BboxManipulor.contour_to_extreme_points_quad(contour)

        return bbox, contour

    def extract(self, segmentation_result, frame_w, last_side=None):
        processed_segments = []

        for i, mask in enumerate(segmentation_result.masks):
            cls_id = int(segmentation_result.boxes.cls[i].item())
            conf = float(segmentation_result.boxes.conf[i].item())
            class_name = self.process_class_name(self.classes[cls_id])

            if "Circle" not in class_name.split():
                bbox, contour = self.extract_rectangle(mask)
            else:
                bbox, contour = self.extract_arc_box(mask)

            if bbox is None or contour is None:
                print(f"LOG THIS EDGE CASE: failed bbox extraction for {class_name}")
                continue

            area = cv2.contourArea(contour)

            processed_segments.append({
                "class_name": class_name,
                "confidence": conf,
                "area": area,
                "side_hint": CanvasMapper.suggest_side(bbox, frame_w),
                "image_contour": contour,
                "mask": mask,
                "image_bbox": bbox,
                "canvas_bbox": None,
            })

        for segment in processed_segments:
            segment["canvas_bbox"] = CanvasMapper.get_canvas_mapping(
                segment["class_name"],
                side_hint=segment["side_hint"],
            )

        processed_segments.sort(
            key=lambda s: {
                "18Yard Circle": 0,
                "Half Central Circle": 1,
                "18Yard": 2,
                "5Yard": 3,
                "Half Field": 4,
            }.get(s["class_name"], 999)
        )

        return processed_segments


class BboxManipulor():
    @staticmethod
    def get_largest_contour_from_yolo_mask(mask):
        """
        Important:
        Use YOLO polygon points first.

        Do NOT start with mask.data here. mask.data can be lower-res, jagged,
        or clipped strangely after resizing. mask.xy is already the polygon
        in image coordinates.
        """

        contour = BboxManipulor.get_largest_contour_from_yolo_polygon(mask)

        if contour is not None:
            return contour

        return BboxManipulor.get_largest_contour_from_dense_mask(mask)

    @staticmethod
    def get_largest_contour_from_yolo_polygon(mask):
        """
        Returns largest YOLO polygon as OpenCV contour:
            N x 1 x 2, float32
        """

        if not hasattr(mask, "xy") or mask.xy is None or len(mask.xy) == 0:
            return None

        contours = []

        for poly in mask.xy:
            pts = np.asarray(poly, dtype=np.float32)

            if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
                continue

            contour = pts.reshape(-1, 1, 2).astype(np.float32)

            if cv2.contourArea(contour) > 1.0:
                contours.append(contour)

        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)

        return contour

    @staticmethod
    def get_largest_contour_from_dense_mask(mask):
        """
        Fallback only. Uses mask.data if mask.xy is not available.
        """

        if not hasattr(mask, "data") or mask.data is None:
            return None

        data = mask.data

        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        else:
            data = np.asarray(data)

        while data.ndim > 2:
            data = data[0]

        if data.size == 0:
            return None

        binary = (data > 0.5).astype(np.uint8) * 255

        if hasattr(mask, "orig_shape") and mask.orig_shape is not None:
            h, w = mask.orig_shape[:2]

            if binary.shape[:2] != (h, w):
                binary = cv2.resize(
                    binary,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)

        if contour is None or len(contour) < 4:
            return None

        return contour.astype(np.float32)

    @staticmethod
    def contour_to_max_area_quad(
        contour,
        max_candidates=60,
        min_area_ratio=0.10,
    ):
        """
        Extracts a quadrilateral from actual mask points.

        No minAreaRect.
        No approxPolyDP.
        No line intersections.

        It finds the convex hull, then searches for the 4 hull points that
        produce the largest quadrilateral area.

        This works better than TL/TR/BR/BL extremes when the mask is slanted,
        clipped, or has noisy boundary points.
        """

        if contour is None or len(contour) < 4:
            return None

        pts = contour.reshape(-1, 2).astype(np.float32)
        pts = BboxManipulor.remove_duplicate_points(pts)

        if len(pts) < 4:
            return None

        hull = cv2.convexHull(
            pts.reshape(-1, 1, 2).astype(np.float32),
            returnPoints=True,
        ).reshape(-1, 2).astype(np.float32)

        hull = BboxManipulor.remove_duplicate_points(hull)

        if len(hull) < 4:
            return None

        if len(hull) == 4:
            return GeometryManipulor.order_points_tl_tr_br_bl(hull)

        candidates = BboxManipulor.build_hull_candidates(
            hull,
            max_candidates=max_candidates,
        )

        if candidates is None or len(candidates) < 4:
            return BboxManipulor.contour_to_extreme_points_quad(contour)

        quad = BboxManipulor.find_max_area_quad_from_ordered_points(
            candidates,
            min_area_ratio=min_area_ratio,
        )

        if quad is None:
            return BboxManipulor.contour_to_extreme_points_quad(contour)

        return GeometryManipulor.order_points_tl_tr_br_bl(quad)

    @staticmethod
    def build_hull_candidates(hull, max_candidates=60):
        """
        Reduces hull points while keeping important extreme points.

        The selected points still come from the hull.
        """

        hull = np.asarray(hull, dtype=np.float32).reshape(-1, 2)
        n = len(hull)

        if n <= max_candidates:
            return hull

        idx_set = set()

        # Uniform samples around the hull.
        uniform_count = max(8, max_candidates // 2)
        uniform_idx = np.linspace(0, n - 1, uniform_count, dtype=int)

        for idx in uniform_idx:
            idx_set.add(int(idx))

        x = hull[:, 0]
        y = hull[:, 1]

        scores = [
            x,
            y,
            x + y,
            x - y,
            -x + y,
            x + 2.0 * y,
            x - 2.0 * y,
            2.0 * x + y,
            2.0 * x - y,
        ]

        # Keep top/bottom score candidates.
        k = 3

        for score in scores:
            order = np.argsort(score)

            for idx in order[:k]:
                idx_set.add(int(idx))

            for idx in order[-k:]:
                idx_set.add(int(idx))

        idx = np.array(sorted(idx_set), dtype=int)

        if len(idx) > max_candidates:
            keep = np.linspace(0, len(idx) - 1, max_candidates, dtype=int)
            idx = idx[keep]

        candidates = hull[idx]

        return candidates.astype(np.float32)

    @staticmethod
    def find_max_area_quad_from_ordered_points(points, min_area_ratio=0.10):
        """
        Brute-force search over ordered hull candidates.

        Since points are ordered around the convex hull, any i < j < k < l
        forms a valid cyclic quadrilateral.
        """

        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        m = len(points)

        if m < 4:
            return None

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        bbox_area = max(float((x_max - x_min) * (y_max - y_min)), 1.0)
        min_area = min_area_ratio * bbox_area

        best_quad = None
        best_score = -1.0

        for i in range(0, m - 3):
            for j in range(i + 1, m - 2):
                for k in range(j + 1, m - 1):
                    for l in range(k + 1, m):
                        quad = np.array(
                            [points[i], points[j], points[k], points[l]],
                            dtype=np.float32,
                        )

                        area = abs(GeometryManipulor.polygon_area(quad))

                        if area < min_area:
                            continue

                        edge_lengths = GeometryManipulor.edge_lengths(quad)

                        if np.min(edge_lengths) < 2.0:
                            continue

                        # Prefer large area, but penalize nearly collapsed edges.
                        score = area * (np.min(edge_lengths) / (np.max(edge_lengths) + 1e-6))

                        if score > best_score:
                            best_score = score
                            best_quad = quad

        return best_quad

    @staticmethod
    def contour_to_extreme_points_quad(contour):
        """
        Fallback using actual contour/hull points.

        Corners:
            TL = min x+y
            TR = max x-y
            BR = max x+y
            BL = min x-y
        """

        if contour is None or len(contour) < 4:
            return None

        pts = contour.reshape(-1, 2).astype(np.float32)
        pts = BboxManipulor.remove_duplicate_points(pts)

        if len(pts) < 4:
            return None

        hull = cv2.convexHull(
            pts.reshape(-1, 1, 2).astype(np.float32),
            returnPoints=True,
        ).reshape(-1, 2).astype(np.float32)

        if len(hull) >= 4:
            pts = hull

        x = pts[:, 0]
        y = pts[:, 1]

        tl = pts[np.argmin(x + y)]
        tr = pts[np.argmax(x - y)]
        br = pts[np.argmax(x + y)]
        bl = pts[np.argmin(x - y)]

        quad = np.array([tl, tr, br, bl], dtype=np.float32)

        return GeometryManipulor.order_points_tl_tr_br_bl(quad)

    @staticmethod
    def contour_to_axis_aligned_bbox_points(contour):
        """
        For circle/arc classes only.
        """

        if contour is None or len(contour) < 4:
            return None

        pts = contour.reshape(-1, 2).astype(np.float32)

        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))

        bbox = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ], dtype=np.float32)

        return bbox

    @staticmethod
    def remove_duplicate_points(pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

        if len(pts) <= 1:
            return pts

        rounded = np.round(pts, decimals=2)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)

        return pts[unique_idx].astype(np.float32)


class GeometryManipulor():
    @staticmethod
    def order_points_tl_tr_br_bl(pts):
        """
        Returns:
            [top_left, top_right, bottom_right, bottom_left]

        This version is better for broadcast-camera football views than pure
        sum/diff ordering.

        It first separates top two points and bottom two points by y-coordinate,
        then orders each pair by x-coordinate.
        """

        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

        y_order = np.argsort(pts[:, 1])

        top = pts[y_order[:2]]
        bottom = pts[y_order[2:]]

        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        tl = top[0]
        tr = top[1]
        bl = bottom[0]
        br = bottom[1]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def order_points(pts):
        return GeometryManipulor.order_points_tl_tr_br_bl(pts)

    @staticmethod
    def polygon_area(pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

        x = pts[:, 0]
        y = pts[:, 1]

        return 0.5 * float(
            np.dot(x, np.roll(y, -1)) -
            np.dot(y, np.roll(x, -1))
        )

    @staticmethod
    def edge_lengths(pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

        shifted = np.roll(pts, -1, axis=0)

        return np.linalg.norm(shifted - pts, axis=1)