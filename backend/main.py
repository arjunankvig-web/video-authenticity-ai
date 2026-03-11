from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os
import cv2
import json
from datetime import datetime
from services.motion_validator import validate_motion
from services.pose_validator import validate_pose
from services.physics_validator import validate_physics
from services.score_engine import compute_score
from database import VideoRecord, get_db

app = FastAPI(title="Hybrid Explainable Video Authenticity AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load YOLO model lazily on first use to avoid startup errors
model = None

def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        try:
            model = YOLO("yolov8m.pt")
        except Exception as e:
            print(f"Warning: Failed to load yolov8m.pt: {e}")
            print("Attempting to download fresh copy...")
            import os
            if os.path.exists("yolov8m.pt"):
                os.remove("yolov8m.pt")
            model = YOLO("yolov8m.pt")
    return model

CONF_THRESHOLD = 0.4


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames


def detect_entities(frames):

    detected_entities = set()
    detections = []
    
    model = get_model()

    # Sample frames every 5 frames
    sampled_frames = frames[::5]

    for frame in sampled_frames:

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb)

        for r in results:

            boxes = r.boxes.xyxy
            scores = r.boxes.conf
            classes = r.boxes.cls

            for box, score, cls in zip(boxes, scores, classes):

                if score < CONF_THRESHOLD:
                    continue

                label = model.names[int(cls)]

                detected_entities.add(label)

                detections.append({
                    "label": label,
                    "confidence": float(score),
                    "bbox": list(map(int, box))
                })

    return list(detected_entities), detections


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static/index.html"))


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), db: Session = Depends(get_db)):

    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get file size
    file_size = os.path.getsize(filepath)

    frames = extract_frames(filepath)
    
    # Calculate video duration (assuming 30 fps by default from cv2)
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = len(frames) / fps if fps > 0 else 0
    cap.release()

    entities, detections = detect_entities(frames)
    
    # Validate authenticity
    try:
        motion_score = validate_motion(frames)
        pose_score = validate_pose(frames)
        physics_score = validate_physics(frames)
        authenticity_score = compute_score(motion_score, pose_score, physics_score)
        
        verdict = "Real" if authenticity_score > 0.4 else "AI-Generated"
        
        print(f"Authenticity Analysis:")
        print(f"  Motion: {motion_score:.3f}")
        print(f"  Pose: {pose_score:.3f}")
        print(f"  Physics: {physics_score:.3f}")
        print(f"  Overall Score: {authenticity_score:.3f}")
        print(f"  Verdict: {verdict}")
        
        analysis_status = "completed"
    except Exception as e:
        motion_score = 0
        pose_score = 0
        physics_score = 0
        authenticity_score = 0
        verdict = f"Error: {str(e)}"
        analysis_status = "error"
        print(f"Error during analysis: {e}")

    # Save to database
    try:
        db_record = VideoRecord(
            filename=file.filename,
            filepath=filepath,
            file_size=file_size,
            duration=duration,
            num_frames=len(frames),
            motion_score=motion_score,
            pose_score=pose_score,
            physics_score=physics_score,
            authenticity_score=authenticity_score,
            verdict=verdict,
            detected_entities=json.dumps(list(entities)),
            analysis_status=analysis_status
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        video_id = db_record.id
    except Exception as e:
        print(f"Database error: {e}")
        video_id = None

    result = {
        "video_id": video_id,
        "filename": file.filename,
        "detected_entities": list(entities),
        "num_frames": len(frames),
        "num_detections": len(detections),
        "authenticity_analysis": {
            "motion_score": motion_score,
            "pose_score": pose_score,
            "physics_score": physics_score,
            "overall_score": authenticity_score,
            "verdict": verdict
        },
        "status": "analysis_complete"
    }

    return result


@app.get("/videos")
def get_all_videos(db: Session = Depends(get_db)):
    """Get all uploaded videos from database"""
    videos = db.query(VideoRecord).all()
    return {
        "total": len(videos),
        "videos": [
            {
                "id": v.id,
                "filename": v.filename,
                "upload_date": v.upload_date,
                "verdict": v.verdict,
                "authenticity_score": v.authenticity_score,
                "num_frames": v.num_frames,
                "file_size": v.file_size
            }
            for v in videos
        ]
    }


@app.get("/videos/{video_id}")
def get_video_details(video_id: int, db: Session = Depends(get_db)):
    """Get detailed analysis of a specific video"""
    video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    
    if not video:
        return {"error": "Video not found"}
    
    return {
        "id": video.id,
        "filename": video.filename,
        "upload_date": video.upload_date,
        "file_size": video.file_size,
        "duration": video.duration,
        "num_frames": video.num_frames,
        "detected_entities": json.loads(video.detected_entities) if video.detected_entities else [],
        "authenticity_analysis": {
            "motion_score": video.motion_score,
            "pose_score": video.pose_score,
            "physics_score": video.physics_score,
            "overall_score": video.authenticity_score,
            "verdict": video.verdict
        },
        "analysis_status": video.analysis_status
    }
