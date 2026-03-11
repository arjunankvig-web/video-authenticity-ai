import sys
import os
# make sure backend/services is on sys.path
sys.path.append(os.path.dirname(__file__))
from adaptive_yolo_selector import AdaptiveYOLOSelector

selector = AdaptiveYOLOSelector()
print('models loaded:', list(selector.models.keys()))
