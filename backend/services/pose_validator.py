import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os

pose_landmarker = None

def _initialize_pose_landmarker():
    """Lazily initialize the pose landmarker when first needed."""
    global pose_landmarker
    
    if pose_landmarker is not None:
        return
    
    try:
        # Try to get model from models directory
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/pose_landmarker_lite.tflite")
        
        # If model doesn't exist, download it
        if not os.path.exists(MODEL_PATH):
            print("Downloading MediaPipe Pose Landmarker model...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Try multiple model URLs
            model_urls = [
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float32/pose_landmarker_lite.tflite",
                "https://storage.googleapis.com/download.tensorflow.org/models/tflite/pose_estimation/full/pose_landmark_lite.tflite"
            ]
            
            import urllib.request
            for url in model_urls:
                try:
                    print(f"Trying: {url}")
                    urllib.request.urlretrieve(url, MODEL_PATH, timeout=30)
                    print(f"Model downloaded to {MODEL_PATH}")
                    break
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
                    continue
            
            if not os.path.exists(MODEL_PATH):
                print(f"Warning: Could not download model. Ensure {MODEL_PATH} exists.")
                raise FileNotFoundError(f"Pose landmarker model not found at {MODEL_PATH}")
        
        # Initialize pose landmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        print("Pose landmarker initialized successfully")
        
    except Exception as e:
        print(f"Error initializing pose landmarker: {e}")
        raise


def validate_pose(frames):
    """
    Validates pose smoothness across frames.
    Returns a score between 0 and 1 where 1 is perfect pose smoothness.
    """
    try:
        _initialize_pose_landmarker()
    except FileNotFoundError:
        # model failed to download; skip pose validation rather than crash
        print("Warning: pose landmarker model missing, skipping pose validation")
        return 0.0
    
    if not frames or len(frames) == 0:
        return 0.0
    
    movement = []
    
    # Process every 5th frame for efficiency
    for frame in frames[::5]:
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect pose
            result = pose_landmarker.detect(mp_image)
            
            if result.landmarks and len(result.landmarks) > 0:
                # Extract x, y coordinates from all landmarks
                coords = []
                for lm in result.landmarks[0]:
                    coords.append([lm.x, lm.y])
                
                coords = np.array(coords)
                # Calculate mean position of all landmarks
                movement.append(np.mean(coords))
        except Exception as e:
            print(f"Error detecting pose in frame: {e}")
            continue
    
    # If not enough movement data, return partial score
    if len(movement) < 2:
        return 0.5 if len(movement) > 0 else 0.0
    
    # Calculate smoothness using variance of movement differences
    smoothness = np.var(np.diff(movement))
    
    # Convert to score (lower smoothness variance = higher score)
    score = 1 / (1 + smoothness)
    
    return float(score)