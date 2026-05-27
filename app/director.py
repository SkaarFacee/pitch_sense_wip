import cv2 
class Director:
    @staticmethod
    def make_video_writer(path, fps, size):
        # Use avc1 (H.264) for browser-compatible MP4 playback
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")

        return writer