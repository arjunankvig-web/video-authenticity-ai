import cv2

from services.entity_detector import detect_entities
from services.motion_validator import validate_motion
from services.pose_validator import validate_pose
from services.physics_validator import validate_physics
from services.score_engine import compute_score


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()

    motion_score = validate_motion(frames)

    entities = detect_entities(frames)

    pose_score = validate_pose(frames)

    physics_score = validate_physics(frames)

    final_score = compute_score(
        motion_score,
        pose_score,
        physics_score
    )

    return {
        "motion_score": motion_score,
        "pose_score": pose_score,
        "physics_score": physics_score,
        "entities": entities,
        "authenticity_score": final_score
    }