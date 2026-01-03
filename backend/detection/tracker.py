from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            embedder="mobilenet",
            half=True,
            bgr=True
        )

    def update(self, detections, frame):
        """
        detections: list of [x1, y1, x2, y2, confidence, class_name]
        returns: list of tracks
        """
        formatted = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            formatted.append(([x1, y1, x2-x1, y2-y1], conf, cls))

        tracks = self.tracker.update_tracks(formatted, frame=frame)
        return tracks