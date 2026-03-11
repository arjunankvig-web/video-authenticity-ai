def compute_score(motion, pose, physics):

    weights = {
        "motion": 0.4,
        "pose": 0.3,
        "physics": 0.3
    }

    score = (
        motion * weights["motion"] +
        pose * weights["pose"] +
        physics * weights["physics"]
    )

    return round(score, 3)