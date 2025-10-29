"""
Minimal API - Sadece test için
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
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


@app.get("/")
def root():
    """API bilgisi"""
    return {
        "name": "Music AI Detector API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


@app.post("/predict")
async def predict_music(
    file: UploadFile = File(...),
    separate_vocals: bool = True
):
    """
    Müzik dosyasını analiz et

    Args:
        file: Müzik dosyası (mp3, wav, etc.)
        separate_vocals: Vocal'leri ayır mı?

    Returns:
        {
            "prediction": "AI" veya "Human",
            "confidence": güven skoru,
            "ai_probability": AI olasılığı,
            "human_probability": Human olasılığı
        }
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    # Dosya kontrolü
    allowed_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )

    # Dosyayı kaydet
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict
        result = predictor.predict(str(file_path), separate_vocals=separate_vocals)

        # Cleanup (opsiyonel)
        # file_path.unlink()

        return JSONResponse(content=result)

    except Exception as e:
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


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
