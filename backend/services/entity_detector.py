from ultralytics import YOLO
import cv2
# import selector using relative package import; backend is on path when Uvicorn
from .adaptive_yolo_selector import AdaptiveYOLOSelector

# default lightweight model for simple use-cases
_basic_model = YOLO("yolov8n.pt")


def detect_entities(frames, selector: AdaptiveYOLOSelector = None):
    """Detect entities in a sequence of frames.

    Args:
        frames: list of BGR images (as numpy arrays)
        selector: optional AdaptiveYOLOSelector instance. If provided,
            it will be used to pick the most appropriate YOLO model based on
            a quick evaluation of the frames.

    Returns:
        list of unique label strings detected in the sampled frames.
    """
    model = _basic_model
    if selector is not None:
        # run a quick evaluation to choose the best model
        model_scores, _ = selector.evaluate_models(frames)
        best_name = selector.select_best_model(model_scores)
        model = selector.models.get(best_name, _basic_model)

    detected_entities = set()
    for frame in frames[::10]:  # analyze every 10th frame for speed
        results = model(frame)
        for r in results:
            # newer results have r.boxes with cls attribute
            if hasattr(r, "boxes"):
                for box in r.boxes:
                    class_id = int(box.cls)
                    label = model.names[class_id]
                    detected_entities.add(label)
    return list(detected_entities)
