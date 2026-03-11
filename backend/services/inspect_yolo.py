from ultralytics import YOLO
import inspect

print('YOLO signature:', inspect.signature(YOLO))
print('YOLO doc:', YOLO.__doc__)
