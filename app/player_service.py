from ultralytics import YOLO
import numpy as np 
import cv2

class PlayerDetector():
    def __init__(self,model_path) -> None:
        self.model=YOLO(model_path)
        self.remove_names = {"ball", "other"}

    
    def print_model_metadate(self) -> None:
        print("Player classes:")
        for class_id, class_name in self.model.names.items():
            print(f"  {class_id}: {class_name}")

    def format_results(self,result):
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = result.boxes.conf.cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            return self.remove_classes(xyxy, conf, classes,self.model.names)
        
    def remove_classes(self, xyxy, conf, classes, class_names):
        if xyxy is None or len(xyxy) == 0:
            return xyxy, conf, classes
        keep = []
        for cls_id in classes:
            cls_id = int(cls_id)
            class_name = str(class_names[cls_id]).lower()
            keep.append(class_name not in self.remove_names)
        keep = np.array(keep, dtype=bool)
        if len(conf[keep])==len(conf):
            print('NOTHING REMOVED')
        return xyxy[keep], conf[keep], classes[keep]
    
    def get_player_bottom_center_points(self,xyxy):
        if len(xyxy) == 0:
            return np.empty((0, 2), dtype=np.float32)

        x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        y = xyxy[:, 3]
        return np.stack([x, y], axis=1).astype(np.float32)

    def project_points(self,points,H):
        points=self.get_player_bottom_center_points(points)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2).astype(np.float32)

