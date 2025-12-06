"""
Minimal FastAPI adapter for Music AI Detector (test-only).
"""

import os
import asyncio
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import soundfile as sf

try:
    # Preferred when installed as a package
    from .predictor import MusicAIPredictor
except Exception:  # pragma: no cover - fallback for direct script usage
    from predictor import MusicAIPredictor
import uvicorn

app = FastAPI(title="Music AI Detector API", version="1.0.0")

# Predictor instance
try:
    predictor = MusicAIPredictor()
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model_loaded = False
    predictor = None

# Upload directory and limits (env configurable)
UPLOAD_DIR = Path(os.getenv("MUSIC_API_UPLOAD_DIR", "backend/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MUSIC_API_MAX_UPLOAD_MB", "25"))
MAX_DURATION_SEC = int(os.getenv("MUSIC_API_MAX_DURATION_SEC", "600"))
MAX_CHANNELS = int(os.getenv("MUSIC_API_MAX_CHANNELS", "2"))

# Simple in-memory rate limit (per-process, env configurable)
RATE_LIMIT_WINDOW_SEC = int(os.getenv("MUSIC_API_RATE_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("MUSIC_API_RATE_MAX", "30"))
_rate_limit_hits = defaultdict(list)
_rate_limit_lock = asyncio.Lock()

# Concurrency guard for heavy vocal separation
VOCAL_SEP_CONCURRENCY = int(os.getenv("MUSIC_API_VOCAL_SEP_CONCURRENCY", "2"))
VOCAL_SEP_SEMAPHORE = asyncio.Semaphore(VOCAL_SEP_CONCURRENCY)


@app.get("/")
def root():
    """Basic API info."""
    return {
        "name": "Music AI Detector API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
        },
    }


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": model_loaded,
    }


@app.post("/predict")
async def predict_music(
    request: Request,
    file: UploadFile = File(...),
    separate_vocals: bool = False,
):
    """
    Analyze a music file and predict AI vs Human.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    allowed_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}",
        )

    # Rate limit check
    client_id = request.client.host if request.client else "unknown"
    now = time.time()
    async with _rate_limit_lock:
        hits = _rate_limit_hits[client_id]
        # prune old hits
        _rate_limit_hits[client_id] = [t for t in hits if now - t < RATE_LIMIT_WINDOW_SEC]
        if len(_rate_limit_hits[client_id]) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down.",
            )
        _rate_limit_hits[client_id].append(now)

    # Save uploaded file with a safe random name
    target_name = f"{uuid4().hex}{file_ext or '.bin'}"
    file_path = UPLOAD_DIR / target_name
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    bytes_written = 0

    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max {MAX_UPLOAD_MB} MB allowed.",
                    )
                buffer.write(chunk)

        # Basic audio sanity: duration / channels
        try:
            with sf.SoundFile(str(file_path)) as snd:
                duration_sec = len(snd) / float(snd.samplerate)
                if duration_sec > MAX_DURATION_SEC:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too long ({duration_sec:.0f}s). Max {MAX_DURATION_SEC}s allowed.",
                    )
                if snd.channels > MAX_CHANNELS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Too many channels ({snd.channels}). Max {MAX_CHANNELS} allowed.",
                    )
        except HTTPException:
            raise
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"Unreadable audio file: {err}")

        # Heavy sync work -> run in threadpool (with optional vocal-separation guard)
        if separate_vocals:
            async with VOCAL_SEP_SEMAPHORE:
                result = await run_in_threadpool(
                    predictor.predict,
                    str(file_path),
                    separate_vocals=separate_vocals,
                )
        else:
            result = await run_in_threadpool(
                predictor.predict,
                str(file_path),
                separate_vocals=separate_vocals,
            )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}",
        )
    finally:
        file_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Music AI Detector API")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("\nEndpoints:")
    print("  GET  /          - API info")
    print("  GET  /health    - Health check")
    print("  POST /predict   - Predict if music is AI or Human")
    print("\nExample usage:")
    print('  curl -X POST "http://localhost:8000/predict" \\')
    print('       -F "file=@your_music.mp3"')
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
