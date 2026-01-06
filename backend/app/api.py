"""
Minimal FastAPI adapter for Music AI Detector (test-only).
"""

import asyncio
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

try:
    from .config import get_config
    from .feature_extractor import FEATURE_EXTRACTOR_VERSION
    from .logging_config import get_logger
    from .predictor import MusicAIPredictor, PredictionError
    from .utils import ResourceManager
except Exception:
    from config import get_config
    from feature_extractor import FEATURE_EXTRACTOR_VERSION
    from logging_config import get_logger
    from predictor import MusicAIPredictor, PredictionError
    from utils import ResourceManager

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title="Music AI Detector API", version="1.0.0")

# Predictor instance
try:
    predictor = MusicAIPredictor()
    model_loaded = True
    logger.info("Model loaded successfully for API")
except Exception as e:
    logger.warning(f"Could not load model: {e}")
    model_loaded = False
    predictor = None

# Upload directory and limits (env configurable)
UPLOAD_DIR = cfg.uploads_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MUSIC_API_MAX_UPLOAD_MB", "25"))
MAX_DURATION_SEC = int(os.getenv("MUSIC_API_MAX_DURATION_SEC", "600"))
MAX_CHANNELS = int(os.getenv("MUSIC_API_MAX_CHANNELS", "2"))

# Simple in-memory rate limit (per-process, env configurable)
RATE_LIMIT_WINDOW_SEC = int(os.getenv("MUSIC_API_RATE_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("MUSIC_API_RATE_MAX", "30"))
_rate_limit_hits: Dict[str, List[float]] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()

# Concurrency guard for heavy vocal separation
VOCAL_SEP_CONCURRENCY = int(os.getenv("MUSIC_API_VOCAL_SEP_CONCURRENCY", "2"))
VOCAL_SEP_SEMAPHORE = asyncio.Semaphore(VOCAL_SEP_CONCURRENCY)


@app.get("/")
def root() -> Dict:
    """Basic API info."""
    return {
        "name": "Music AI Detector API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "feature_extractor_version": getattr(
            predictor, "metadata_version", FEATURE_EXTRACTOR_VERSION
        ),
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_duration_sec": MAX_DURATION_SEC,
            "max_channels": MAX_CHANNELS,
            "rate_limit_window_sec": RATE_LIMIT_WINDOW_SEC,
            "rate_limit_max": RATE_LIMIT_MAX_REQUESTS,
            "vocal_sep_concurrency": VOCAL_SEP_CONCURRENCY,
        },
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
        },
    }


@app.get("/health")
def health() -> Dict:
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

    # Save uploaded file with a safe random name using context manager
    target_name = f"{uuid4().hex}{file_ext or '.bin'}"
    file_path = UPLOAD_DIR / target_name
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    bytes_written = 0

    try:
        # Write file
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

        logger.info(
            f"Prediction complete: {result.get('prediction')} "
            f"(confidence: {result.get('confidence', 0):.2%})"
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}",
        )
    finally:
        # Clean up uploaded file
        ResourceManager.safe_remove_file(file_path)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Music AI Detector API")
    logger.info("=" * 60)
    logger.info("Starting server on http://localhost:8000")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /          - API info")
    logger.info("  GET  /health    - Health check")
    logger.info("  POST /predict   - Predict if music is AI or Human")
    logger.info("")
    logger.info("Example usage:")
    logger.info('  curl -X POST "http://localhost:8000/predict" \\')
    logger.info('       -F "file=@your_music.mp3"')
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
