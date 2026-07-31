import cv2

from modules.detector import PersonDetector
from modules.tracker import PersonTracker
from modules.color_detector import ColorDetector
from modules.build_classifier import BuildClassifier
from modules.height_estimator import HeightEstimator
from modules.reidentifier import ReIdentifier
from modules.database import VisitorDatabase



detector = PersonDetector()
tracker = PersonTracker()
color_detector = ColorDetector()
build_classifier = BuildClassifier()
height_estimator = HeightEstimator()
reidentifier = ReIdentifier()
database = VisitorDatabase()
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)
    #print("Tracks found:", len(tracks))

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        estimated_height = height_estimator.estimate(height)

        build = build_classifier.classify(width, height)
        person_crop = frame[y1:y2, x1:x2]
        shirt_crop = person_crop[0:int(person_crop.shape[0] * 0.4), :]
        color = color_detector.detect_color(shirt_crop)

        visitor_id, is_new = reidentifier.identify(color,build,estimated_height)

        if is_new:
            database.save_visitor(visitor_id, color, build, estimated_height)
        
        cv2.imwrite(
            f"data/snapshots/visitor_{visitor_id}.jpg",
            person_crop
        )
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Visitor:{visitor_id} | {color} | {build} | ~{estimated_height}cm",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # for det in detections:

    #     x1, y1, x2, y2 = det["bbox"]
    #     conf = det["confidence"]

    #     cv2.rectangle(
    #         frame,
    #         (x1, y1),
    #         (x2, y2),
    #         (0, 255, 0),
    #         2
    #     )

    #     cv2.putText(
    #         frame,
    #         f"Person {conf:.2f}",
    #         (x1, y1 - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.6,
    #         (0, 255, 0),
    #         2
    #     )

    cv2.imshow("VisitorIQ - Person Detection", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()