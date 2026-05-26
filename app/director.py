import cv2 
class Director:
    @staticmethod
    def make_video_writer(path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")

        return writer