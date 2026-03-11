from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_entities(frames):

    entities = set()

    for frame in frames[::10]:

        results = model(frame)

        for r in results:
            for box in r.boxes.cls:
                cls = int(box)
                label = model.names[cls]
                entities.add(label)

    return list(entities)