import cv2
import numpy as np


def validate_physics(frames):
    """Analyze frame consistency and temporal artifacts.
    AI videos often show jitter, flicker, or compression inconsistencies."""
    if len(frames) < 3:
        return 0.5
    
    flicker_scores = []
    
    # Sample frames for analysis
    for i in range(1, min(len(frames), 30)):
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Analyze intensity stability (flicker detection)
        intensity_diff = np.mean(np.abs(prev_gray.astype(float) - curr_gray.astype(float)))
        flicker_scores.append(intensity_diff)
    
    if not flicker_scores:
        return 0.5
    
    avg_flicker = np.mean(flicker_scores)
    
    # AI videos show excessive flicker/jitter or unnatural smoothness
    if avg_flicker > 30:  # Too much frame difference
        return 0.2
    if avg_flicker < 2:   # Too little change (suspicious)
        return 0.3
    
    # Natural range is ~5-15
    physics_score = 1 - (abs(avg_flicker - 10) / 25)
    return max(0.0, min(1.0, float(physics_score)))