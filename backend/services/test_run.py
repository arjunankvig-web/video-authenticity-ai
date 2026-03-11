import cv2
import numpy as np
from adaptive_yolo_selector import AdaptiveYOLOSelector

# create a dummy video with random frames
video_path = 'dummy_video.mp4'
frame_width, frame_height = 320, 240
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
for _ in range(30):  # 3 seconds at 10 fps
    frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
    out.write(frame)
out.release()

# run selector
selector = AdaptiveYOLOSelector()
print('Loaded models:', list(selector.models.keys()))
report = selector.run_best_model(video_path)
print('Report:')
print(report)
