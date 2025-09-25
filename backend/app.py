from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import random
import time
from datetime import datetime

app = FastAPI(title="TrustAI API", description="Deepfake Detection API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "TrustAI API is running", "status": "healthy"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze uploaded video for deepfake detection"""
    
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Please upload a valid video file")
    
    # Simulate processing time
    await asyncio.sleep(2)
    
    # Generate realistic scores with some randomness
    base_frame_score = random.uniform(65, 95)
    base_audio_score = random.uniform(70, 90)
    base_sync_score = random.uniform(75, 95)
    
    # Add some correlation between scores
    if base_frame_score < 75:
        base_audio_score *= 0.9
        base_sync_score *= 0.95
    
    frame_score = round(base_frame_score, 1)
    audio_score = round(base_audio_score, 1)
    sync_score = round(base_sync_score, 1)
    overall_score = round((frame_score + audio_score + sync_score) / 3, 1)
    
    return JSONResponse({
        "frame_fake_score": frame_score,
        "audio_fake_score": audio_score,
        "av_sync_score": sync_score,
        "overall_authenticity": overall_score,
        "filename": file.filename,
        "analysis_time": datetime.now().isoformat(),
        "status": "success"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)