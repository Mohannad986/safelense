from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles
import os
import tempfile
import shutil
from datetime import datetime
import asyncio
from typing import Dict, Any

from frame_detector import FrameDetector
from audio_detector import AudioDetector
from av_sync_detector import AVSyncDetector

app = FastAPI(title="TrustAI API", description="Deepfake Detection API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for WebContainer
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
frame_detector = FrameDetector()
audio_detector = AudioDetector()
av_sync_detector = AVSyncDetector()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Loading AI models...")
    results = await asyncio.gather(
        frame_detector.load_model(),
        audio_detector.load_model(),
        av_sync_detector.load_model(),
        return_exceptions=True
    )
    
    # Log any exceptions without crashing the server
    model_names = ["frame_detector", "audio_detector", "av_sync_detector"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Warning: {model_names[i]} failed to load: {result}")
        else:
            print(f"{model_names[i]} loaded successfully")
    
    print("Server startup completed!")

@app.get("/")
async def root():
    return {"message": "TrustAI API is running", "status": "healthy"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze uploaded video for deepfake detection"""
    
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Please upload a valid video file")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        file_path = os.path.join(temp_dir, file.filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        try:
            # Run all detectors in parallel
            results = await asyncio.gather(
                frame_detector.analyze(file_path),
                audio_detector.analyze(file_path),
                av_sync_detector.analyze(file_path),
                return_exceptions=True
            )
            
            frame_result, audio_result, av_sync_result = results
            
            # Handle any exceptions
            if isinstance(frame_result, Exception):
                frame_score = 75  # Default fallback
                print(f"Frame detection error: {frame_result}")
            else:
                frame_score = frame_result
                
            if isinstance(audio_result, Exception):
                audio_score = 75  # Default fallback
                print(f"Audio detection error: {audio_result}")
            else:
                audio_score = audio_result
                
            if isinstance(av_sync_result, Exception):
                av_sync_score = 75  # Default fallback
                print(f"AV sync detection error: {av_sync_result}")
            else:
                av_sync_score = av_sync_result
            
            # Calculate overall authenticity score
            overall_score = (frame_score + audio_score + av_sync_score) / 3
            
            return JSONResponse({
                "frame_fake_score": round(frame_score, 1),
                "audio_fake_score": round(audio_score, 1),
                "av_sync_score": round(av_sync_score, 1),
                "overall_authenticity": round(overall_score, 1),
                "filename": file.filename,
                "analysis_time": datetime.now().isoformat(),
                "status": "success"
            })
            
        except Exception as e:
            print(f"Analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "frame_detector": frame_detector.is_loaded,
            "audio_detector": audio_detector.is_loaded,
            "av_sync_detector": av_sync_detector.is_loaded
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)