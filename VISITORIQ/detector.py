from ultralytics import YOLO
import cv2


class PersonDetector:
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls[0])

                # COCO class 0 = person
                if cls == 0:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    confidence = float(box.conf[0])

                    detections.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "confidence": confidence,
                            "class_id": cls
                        }
                    )
                    

        return detections