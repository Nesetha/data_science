from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            max_cosine_distance=0.4
        )

    def update(self, detections, frame):

        tracker_input = []

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            width = x2 - x1
            height = y2 - y1

            tracker_input.append(
                (
                    [x1, y1, width, height],
                    det["confidence"],
                    "person"
                )
            )

        tracks = self.tracker.update_tracks(
            tracker_input,
            frame=frame
        )

        return tracks