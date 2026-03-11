import cv2
import numpy as np


def validate_motion(frames):
    """Detect unnatural motion patterns typical of AI-generated videos.
    Returns 0-1 where lower scores indicate more likely AI generation."""
    if len(frames) < 2:
        return 0.5
    
    motion_magnitudes = []
    
    # Sample frames for efficiency
    for i in range(1, min(len(frames), 50)):
        prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        motion_magnitudes.append(np.mean(magnitude))
    
    if len(motion_magnitudes) < 2:
        return 0.5
    
    # AI videos are often TOO smooth or have sudden jerks
    motion_variance = np.var(motion_magnitudes)
    
    # Real motion has natural variance. Too smooth indicates AI
    if motion_variance < 0.1:
        return 0.2  # AI artifact: too smooth
    
    if motion_variance > 50:
        return 0.3  # AI artifact: too jerky
    
    # Score based on natural motion characteristics
    score = 1 - (abs(motion_variance - 5) / 50)
    return max(0.0, min(1.0, float(score)))