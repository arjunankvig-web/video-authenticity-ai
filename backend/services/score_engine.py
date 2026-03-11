def compute_score(motion, pose, physics):
    """Compute overall authenticity score.
    Lower scores indicate AI-generated content."""
    # More weight on motion since AI videos have characteristic motion artifacts
    weights = {
        "motion": 0.5,
        "pose": 0.25,
        "physics": 0.25
    }
    
    # Clamp scores to valid range
    motion = max(0, min(1, motion))
    pose = max(0, min(1, pose))
    physics = max(0, min(1, physics))
    
    score = (
        motion * weights["motion"] +
        pose * weights["pose"] +
        physics * weights["physics"]
    )
    
    return round(score, 3)