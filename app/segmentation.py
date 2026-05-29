
"""
segmentation mask
    ↓
binary mask # Skipped 
    ↓
largest contour # From the polygons in the masks 
    ↓
quadrilateral in the image
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
    def __init__(self,model_path) -> None:
        self.model=YOLO(model_path)
        self.classes=self.model.names

    def process_class_name(self, class_name):
        class_name = class_name.strip()
        class_name = class_name.replace("First ", "").replace("Second ", "")
        return class_name.strip()

    def extract_arc_box(self,mask):
        contour = BboxManipulor.get_largest_contour_from_yolo_mask(mask)
        bbox=BboxManipulor.contour_to_bbox(contour)
        return bbox,contour

    
    def extract(self,segmentation_result,frame_w,last_side=None):
        processed_segments=[]
        for i,mask in enumerate(segmentation_result.masks):
            cls_id = int(segmentation_result.boxes.cls[i].item())
            conf = float(segmentation_result.boxes.conf[i].item())
            class_name = self.process_class_name(self.classes[cls_id])

            bbox,contour=self.extract_arc_box(mask)
            if contour is None:
                continue
            area = cv2.contourArea(contour)
            processed_segments.append({
                "class_name": class_name,
                "confidence": conf,
                "area": area,
                "side_hint": CanvasMapper.suggest_side(bbox,frame_w),
                "image_contour":contour,
                "mask":mask,
                "image_bbox": bbox,
                "canvas_bbox":None
            })

        for segment in processed_segments:
            segment["canvas_bbox"] = CanvasMapper.get_canvas_mapping(
                segment["class_name"],
                side_hint=segment["side_hint"],
            )
        processed_segments.sort(key=lambda s: {"18Yard Circle": 0, "Half Central Circle": 1, "18Yard": 2, "5Yard": 3, "Half Field": 4}.get(s["class_name"], 999))
        return processed_segments


class BboxManipulor():
    @staticmethod
    def get_largest_contour_from_yolo_mask(mask):
        if mask.xy is None or len(mask.xy) == 0:
            return None
        contour=max(mask.xy, key=cv2.contourArea)
        return contour.astype(np.int32).reshape(-1, 1, 2) # Opencv is (N,1,2) Yolo is (N,2) yoloy to opencv format

    @staticmethod
    def contour_to_bbox(contour):
        hull = cv2.convexHull(contour) # Removed noise to get a more clearn bounday 
        rect = cv2.minAreaRect(hull) # Get the recatangle that can 
        box = cv2.boxPoints(rect).astype(np.float32)
        return GeometryManipulor.order_points(box)

class GeometryManipulor():
    @staticmethod
    def order_points(pts):
        """Order quad as top-left, top-right, bottom-right, bottom-left."""
        pts = np.asarray(pts, dtype=np.float32)

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

