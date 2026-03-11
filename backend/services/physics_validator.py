import numpy as np


def validate_physics(frames):

    # placeholder physics check
    # future: track objects and evaluate acceleration

    values = np.random.normal(0.8, 0.05, 10)

    score = np.mean(values)

    return float(score)