from ultralytics import YOLO 

class Segmentor():
    def __init__(self,model_path) -> None:
        self.model=YOLO(model_path)
        self.classes=self.model.names

    def print_model_metadata(self) -> None:
        print("Keypoint classes:")
        for class_id, class_name in self.model.names.items():
            print(f"  {class_id}: {class_name}")
