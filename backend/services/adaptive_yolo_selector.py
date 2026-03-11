import os
from ultralytics import YOLO
import cv2
import numpy as np
import torch


class AdaptiveYOLOSelector:
    """Helper class that loads multiple YOLO models and picks the best one
    for a particular video.

    Usage:
        selector = AdaptiveYOLOSelector()
        results = selector.run_best_model("/path/to/video.mp4")

    The returned dictionary contains:
        * selected_model: the key/name of the chosen model
        * model_scores: summary metrics used during evaluation
        * detected_entities: list of unique object names seen in the video
        * frame_detections: list of label lists for each frame (full video)
    """

    DEFAULT_MODELS = {
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8m-pose": "yolov8m-pose.pt",
        "yolov8n": "yolov8n.pt",
    }

    def __init__(self, model_paths: dict = None, device: str = None):
        # map human name->path
        self.model_paths = model_paths or AdaptiveYOLOSelector.DEFAULT_MODELS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.initialize_models()

    def initialize_models(self):
        """Load each YOLO model once and hold on to it.
        """
        for name, path in self.model_paths.items():
            # allow absolute or relative path; models should exist in working dir
            model = YOLO(path)
            # try to move to desired device if API supports it
            try:
                model.to(self.device)
            except Exception:
                pass
            self.models[name] = model

    def extract_frames(self, video_path: str):
        """Read all frames from a video file and return as a list.

        The caller may sample the frames later.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames: list, step: int = 5):
        """Return a subsampled list of frames to speed up evaluation."""
        return frames[::step]

    def evaluate_models(self, frames: list):
        """Run each model on a subset of frames and compute metrics.

        Returns a tuple (scores, details) where:
            * scores: name -> combined score used for ranking
            * details: name -> dict of avg_confidence, avg_count, consistency
        """
        sampled = self._sample_frames(frames)
        scores = {}
        details = {}

        for name, model in self.models.items():
            confidences = []
            counts = []
            detections_per_frame = []

            for frame in sampled:
                results = model(frame)
                if not results:
                    confidences.append(0.0)
                    counts.append(0)
                    detections_per_frame.append(set())
                    continue

                r = results[0]
                frame_confs = []
                frame_classes = []
                if hasattr(r, "boxes"):
                    for box in r.boxes:
                        frame_confs.append(float(box.conf))
                        frame_classes.append(int(box.cls))
                else:
                    # fallback for older result object
                    frame_confs = [float(c) for c in getattr(r, "conf", [])]
                    frame_classes = [int(c) for c in getattr(r, "cls", [])]

                confidences.append(np.mean(frame_confs) if frame_confs else 0.0)
                counts.append(len(frame_confs))
                detections_per_frame.append(set(frame_classes))

            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            avg_count = float(np.mean(counts)) if counts else 0.0

            # consistency as mean IoU of class sets between consecutive sampled frames
            consistency_scores = []
            for i in range(len(detections_per_frame) - 1):
                a = detections_per_frame[i]
                b = detections_per_frame[i + 1]
                if a or b:
                    consistency_scores.append(len(a & b) / len(a | b))
                else:
                    consistency_scores.append(1.0)
            consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0

            # simple weighted aggregation: confidence (0.4) + count*0.1 + consistency*0.5
            score = avg_conf * 0.4 + avg_count * 0.1 + consistency * 0.5

            scores[name] = score
            details[name] = {
                "avg_confidence": avg_conf,
                "avg_count": avg_count,
                "consistency": consistency,
            }

        return scores, details

    def select_best_model(self, model_scores: dict):
        """Return the key of the model with the highest score."""
        if not model_scores:
            return None
        return max(model_scores, key=model_scores.get)

    def run_best_model(self, video_path: str):
        """Process the full video with the selected model.

        This method extracts frames, evaluates models, picks the best one,
        and then runs the selected model on every frame to produce detection
        data that can be consumed by the rest of the backend.
        """
        frames = self.extract_frames(video_path)
        model_scores, details = self.evaluate_models(frames)
        best = self.select_best_model(model_scores)
        if best is None:
            return {
                "selected_model": None,
                "model_scores": details,
                "detected_entities": [],
                "frame_detections": [],
            }

        selected_model = self.models[best]

        frame_detections = []
        detected_entities = set()

        for frame in frames:
            results = selected_model(frame)
            labels = []
            if results:
                r = results[0]
                if hasattr(r, "boxes"):
                    for box in r.boxes:
                        cls = int(box.cls)
                        label = selected_model.names.get(cls, str(cls))
                        labels.append(label)
                        detected_entities.add(label)
                else:
                    for cls in getattr(r, "cls", []):
                        label = selected_model.names.get(int(cls), str(cls))
                        labels.append(label)
                        detected_entities.add(label)
            frame_detections.append(labels)

        return {
            "selected_model": best,
            "model_scores": details,
            "detected_entities": list(detected_entities),
            "frame_detections": frame_detections,
        }
