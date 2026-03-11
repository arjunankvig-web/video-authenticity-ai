import cv2
import numpy as np


def validate_motion(frames):

    motion_values = []

    for i in range(1, len(frames)):

        prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        magnitude = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))

        motion_values.append(magnitude)

    if len(motion_values) == 0:
        return 0

    variance = np.var(motion_values)

    score = 1 / (1 + variance)

    return float(score)