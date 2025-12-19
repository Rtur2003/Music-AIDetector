# Configuration Guide

## Overview

The Music AI Detector uses a centralized configuration system that allows customization through environment variables. This provides flexibility for different deployment scenarios without modifying code.

## Configuration Module

Location: `backend/app/config.py`

The `Config` class provides:
- Centralized path management
- Environment variable support
- Lazy property evaluation
- Directory creation utilities

## Environment Variables

### Core Directories

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSIC_DETECTOR_DATA_DIR` | `backend/data` | Main data directory |
| `MUSIC_DETECTOR_MODELS_DIR` | `backend/data/models` | Trained models location |
| `MUSIC_DETECTOR_TEMP_DIR` | `backend/temp` | Temporary files |
| `MUSIC_DETECTOR_UPLOADS_DIR` | `backend/uploads` | API uploads |

### Audio Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSIC_DETECTOR_SAMPLE_RATE` | `22050` | Audio sample rate (Hz) |
| `MUSIC_DETECTOR_DEMUCS_MODEL` | `htdemucs` | Vocal separation model |

### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSIC_API_MAX_UPLOAD_MB` | `25` | Maximum upload size (MB) |
| `MUSIC_API_MAX_DURATION_SEC` | `600` | Maximum audio duration (seconds) |
| `MUSIC_API_MAX_CHANNELS` | `2` | Maximum audio channels |
| `MUSIC_API_RATE_WINDOW_SEC` | `60` | Rate limit window (seconds) |
| `MUSIC_API_RATE_MAX` | `30` | Max requests per window |
| `MUSIC_API_VOCAL_SEP_CONCURRENCY` | `2` | Concurrent vocal separations |

## Usage

### Using Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your custom values:

```bash
MUSIC_DETECTOR_DATA_DIR=/path/to/data
MUSIC_DETECTOR_SAMPLE_RATE=44100
```

### In Python Code

```python
from config import get_config

cfg = get_config()

# Access configured paths
models_dir = cfg.models_dir
sample_rate = cfg.sample_rate

# Ensure directories exist
cfg.ensure_directories()
```

### Module Integration

All modules have been updated to use the configuration:

- `feature_extractor.py` - Uses configured sample rate
- `vocal_separator.py` - Uses configured Demucs model
- `dataset_processor.py` - Uses configured data directories
- `model_trainer.py` - Uses configured models directory
- `predictor.py` - Uses configured paths for model loading
- `detailed_analyzer.py` - Uses configured analysis directory
- `api.py` - Uses configured upload limits and paths

## Directory Structure

The configuration maintains this structure:

```
{data_dir}/
├── raw/
│   ├── ai_generated/
│   └── human_made/
├── processed/
│   ├── features.csv
│   └── metadata.json
├── models/
│   ├── latest_model.pkl
│   ├── latest_scaler.pkl
│   └── latest_metadata.json
└── analysis/
    ├── detailed_analysis_*.json
    ├── comparison_report_*.json
    └── analysis_report_*.txt
```

## Benefits

1. **Flexibility** - Easy to customize without code changes
2. **Portability** - Works across different environments
3. **Testing** - Easy to configure test vs production paths
4. **Deployment** - Environment-specific configuration
5. **Maintainability** - Single source of truth for paths

## Migration

All hardcoded paths have been replaced with configuration references. The configuration is backward-compatible with default values matching the original hardcoded paths.
