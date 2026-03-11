from fastapi import FastAPI, UploadFile, File
import shutil
import os

from services.video_processor import process_video

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_video(filepath)

    return result