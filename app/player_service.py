from ultralytics import YOLO
import numpy as np
import cv2

class PlayerDetector():
    def __init__(self,model_path) -> None:
        self.model=YOLO(model_path)
        self.remove_names = {"ball", "other"}
        self._tracking_active = False

    
    def format_results(self, result, has_track_ids: bool = False):
        """
        Format YOLO detection results.
        
        Args:
            result: Ultralytics Results object.
            has_track_ids: If True, extracts track_ids from result.boxes.id.
        
        Returns:
            tuple: (xyxy, conf, classes, track_ids) or None if no detections.
            track_ids is an int32 array of track IDs; if has_track_ids is False,
            fallback sequential IDs are assigned.
        """
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = result.boxes.conf.cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            # Extract track IDs if available
            if has_track_ids and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(np.int32)
            else:
                track_ids = np.arange(len(xyxy), dtype=np.int32)

            # Remove unwanted classes (ball, other) and filter track_ids accordingly
            xyxy, conf, classes, track_ids = self.remove_classes(
                xyxy, conf, classes, track_ids, self.model.names
            )

            return xyxy, conf, classes, track_ids
        
        return None

    def track_players(self, frame, conf=0.25):
        """
        Run YOLO detection + ByteTrack tracking on a single frame.
        Uses ultralytics' built-in model.track() with persist=True for
        cross-frame track ID persistence.
        
        Args:
            frame: BGR image (H, W, 3).
            conf: Detection confidence threshold.
        
        Returns:
            tuple: (xyxy, conf, classes, track_ids) or None if no detections.
        """
        results = self.model.track(frame, persist=True, conf=conf, verbose=False)
        return self.format_results(results[0], has_track_ids=True)
        
    def remove_classes(self, xyxy, conf, classes, track_ids, class_names):
        """
        Remove detections for unwanted classes (ball, other).
        Returns filtered arrays with track_ids matched to kept detections.
        """
        if xyxy is None or len(xyxy) == 0:
            return xyxy, conf, classes, track_ids
        keep = []
        for cls_id in classes:
            cls_id = int(cls_id)
            class_name = str(class_names[cls_id]).lower()
            keep.append(class_name not in self.remove_names)
        keep = np.array(keep, dtype=bool)
        return xyxy[keep], conf[keep], classes[keep], track_ids[keep]
    
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

