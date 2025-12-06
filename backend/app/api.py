"""
Minimal FastAPI adapter for Music AI Detector (test-only).
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from uuid import uuid4

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

# Upload directory
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = 25


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

        # Heavy sync work -> run in threadpool to avoid blocking the event loop
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
