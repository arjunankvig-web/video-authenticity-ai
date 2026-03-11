import requests
import sys
import os

"""Utility script to upload a video to the local FastAPI server.
Usage:
    python upload_video.py path/to/video.mp4
"""

# if no argument provided, fall back to dummy_video.mp4 created earlier
if len(sys.argv) == 1:
    video_path = "dummy_video.mp4"
elif len(sys.argv) == 2:
    video_path = sys.argv[1]
else:
    print("Usage: python upload_video.py [path/to/video]")
    sys.exit(1)

url = "http://127.0.0.1:8000/analyze-video"

# ensure video_path exists; search in several likely locations
if not os.path.exists(video_path):
    base = os.path.dirname(__file__)
    candidates = [
        video_path,
        os.path.join(base, video_path),
        os.path.join(base, "..", video_path),
        os.path.join(base, "uploads", video_path),
    ]
    found = False
    for cand in candidates:
        cand = os.path.abspath(cand)
        if os.path.exists(cand):
            video_path = cand
            found = True
            break
    if not found:
        print(f"Video file not found in any of {candidates}")
        sys.exit(1)


with open(video_path, "rb") as f:
    files = {"file": (video_path, f, "video/mp4")}
    resp = requests.post(url, files=files)

print("Status code:", resp.status_code)

try:
    data = resp.json()
    print(data)
    # interpret authenticity
    score = data.get("authenticity_score")
    if score is not None:
        if score < 0.5:
            verdict = "likely AI-generated or tampered"
        else:
            verdict = "likely generic/real"
        print(f"Verdict based on authenticity_score: {verdict} (score={score})")
except Exception:
    print(resp.text)
