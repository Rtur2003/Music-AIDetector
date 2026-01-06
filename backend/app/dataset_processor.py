"""
Dataset Processor - prepares the dataset for training.

Usage:
1. Put AI music into backend/data/raw/ai_generated/
2. Put Human music into backend/data/raw/human_made/
3. Run this script.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from .config import get_config
    from .feature_extractor import MusicFeatureExtractor, FeatureExtractionError
    from .logging_config import get_logger
    from .utils import ResourceManager
    from .validators import AudioValidator, ValidationError
    from .vocal_separator import VocalSeparator, VocalSeparatorError
except Exception:
    from config import get_config
    from feature_extractor import MusicFeatureExtractor, FeatureExtractionError
    from logging_config import get_logger
    from utils import ResourceManager
    from validators import AudioValidator, ValidationError
    from vocal_separator import VocalSeparator, VocalSeparatorError

logger = get_logger(__name__)


class DatasetProcessingError(Exception):
    """Exception raised for dataset processing errors."""

    pass


class DatasetProcessor:
    """Processes raw audio files and extracts features for training."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize dataset processor.

        Args:
            data_dir: Data directory path (None = use config default)
        """
        cfg = get_config()
        self.data_dir = Path(data_dir) if data_dir is not None else cfg.data_dir
        self.raw_dir = cfg.raw_dir
        self.processed_dir = cfg.processed_dir
        self.temp_root = cfg.temp_dir

        # Create directories
        cfg.ensure_directories()

        self.separator = VocalSeparator()
        self.extractor = MusicFeatureExtractor()
        logger.info("DatasetProcessor initialized")

    def process_dataset(
        self, separate_vocals: bool = True, skip_errors: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Process the complete dataset and write features.csv + metadata.json.

        Args:
            separate_vocals: Whether to perform vocal separation
            skip_errors: Whether to skip files with errors or fail immediately

        Returns:
            DataFrame with features, or None if no files processed

        Raises:
            DatasetProcessingError: If processing fails (when skip_errors=False)
        """
        logger.info("=" * 60)
        logger.info("DATASET PROCESSING STARTED")
        logger.info("=" * 60)

        all_features = []
        all_labels = []
        metadata = []

        # Process AI Generated music
        ai_dir = self.raw_dir / "ai_generated"
        if ai_dir.exists():
            logger.info(f"[1/2] Processing AI-generated music from: {ai_dir}")
            try:
                ai_files = AudioValidator.validate_audio_files_in_directory(ai_dir)
                ai_features, ai_metadata = self._process_category(
                    ai_files,
                    label=1,
                    category="AI",
                    separate_vocals=separate_vocals,
                    skip_errors=skip_errors,
                )
                all_features.extend(ai_features)
                all_labels.extend([1] * len(ai_features))
                metadata.extend(ai_metadata)
            except ValidationError as e:
                logger.warning(f"AI directory validation: {e}")
        else:
            logger.warning(f"AI directory not found: {ai_dir}")

        # Process Human Made music
        human_dir = self.raw_dir / "human_made"
        if human_dir.exists():
            logger.info(f"[2/2] Processing Human-made music from: {human_dir}")
            try:
                human_files = AudioValidator.validate_audio_files_in_directory(human_dir)
                human_features, human_metadata = self._process_category(
                    human_files,
                    label=0,
                    category="Human",
                    separate_vocals=separate_vocals,
                    skip_errors=skip_errors,
                )
                all_features.extend(human_features)
                all_labels.extend([0] * len(human_features))
                metadata.extend(human_metadata)
            except ValidationError as e:
                logger.warning(f"Human directory validation: {e}")
        else:
            logger.warning(f"Human directory not found: {human_dir}")

        if len(all_features) == 0:
            logger.error("No audio files processed successfully!")
            logger.error(f"Please add music files to:")
            logger.error(f"  - {ai_dir}")
            logger.error(f"  - {human_dir}")
            return None

        # Create DataFrame
        df = pd.DataFrame(all_features)
        df["label"] = all_labels

        # Save features
        output_file = self.processed_dir / "features.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Features saved to: {output_file}")

        # Save metadata
        metadata_file = self.processed_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to: {metadata_file}")

        # Summary
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"  - AI generated: {sum(all_labels)}")
        logger.info(f"  - Human made: {len(all_labels) - sum(all_labels)}")
        logger.info(f"Total features: {len(df.columns) - 1}")

        return df

    def _process_category(
        self,
        files: List[Path],
        label: int,
        category: str,
        separate_vocals: bool = True,
        skip_errors: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process one category (AI or Human).

        Args:
            files: List of audio file paths
            label: Label value (0=Human, 1=AI)
            category: Category name for logging
            separate_vocals: Whether to perform vocal separation
            skip_errors: Whether to skip files with errors

        Returns:
            Tuple of (features_list, metadata_list)

        Raises:
            DatasetProcessingError: If processing fails (when skip_errors=False)
        """
        features_list = []
        metadata_list = []
        error_count = 0

        for audio_file in tqdm(files, desc=f"Processing {category}"):
            try:
                # Use context manager for temp directory
                with ResourceManager.temporary_directory(
                    base_dir=self.temp_root, prefix=f"{audio_file.stem}_"
                ) as session_dir:
                    # Vocal separation (optional)
                    if separate_vocals:
                        logger.debug(f"Separating vocals: {audio_file.name}")
                        result = self.separator.separate(str(audio_file), str(session_dir))
                        audio_to_analyze = result["instrumental"]
                    else:
                        audio_to_analyze = audio_file

                    # Feature extraction
                    logger.debug(f"Extracting features: {audio_file.name}")
                    features = self.extractor.extract_all_features(str(audio_to_analyze))

                    features_list.append(features)

                    # Metadata (temp files auto-cleaned by context manager)
                    metadata_list.append(
                        {
                            "filename": audio_file.name,
                            "category": category,
                            "label": label,
                            "original_path": str(audio_file),
                            "vocal_separated": separate_vocals,
                        }
                    )

                    logger.debug(f"Successfully processed: {audio_file.name}")

            except (ValidationError, VocalSeparatorError, FeatureExtractionError) as e:
                error_count += 1
                logger.error(f"Error processing {audio_file.name}: {e}")
                if not skip_errors:
                    raise DatasetProcessingError(f"Failed to process {audio_file.name}") from e

            except Exception as e:
                error_count += 1
                logger.error(f"Unexpected error processing {audio_file.name}: {e}")
                if not skip_errors:
                    raise DatasetProcessingError(f"Failed to process {audio_file.name}") from e

        if error_count > 0:
            logger.warning(
                f"Completed {category} processing with {error_count} error(s). "
                f"Successfully processed: {len(features_list)}/{len(files)}"
            )

        return features_list, metadata_list

    def get_dataset_info(self) -> Optional[pd.DataFrame]:
        """
        Display information about the processed dataset.

        Returns:
            DataFrame with dataset info, or None if not found
        """
        features_file = self.processed_dir / "features.csv"

        if not features_file.exists():
            logger.warning("No processed dataset found. Run process_dataset() first.")
            return None

        df = pd.read_csv(features_file)

        logger.info("=" * 60)
        logger.info("DATASET INFO")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Features: {len(df.columns) - 1}")
        logger.info("")
        logger.info("Class distribution:")
        ai_count = sum(df["label"])
        human_count = len(df) - ai_count
        logger.info(f"  - AI (label=1): {ai_count} ({ai_count/len(df)*100:.1f}%)")
        logger.info(f"  - Human (label=0): {human_count} ({human_count/len(df)*100:.1f}%)")

        logger.info("")
        logger.info("Feature summary (first 5 features):")
        logger.info(str(df.describe().iloc[:, :5]))

        return df


def create_sample_readme() -> None:
    """Create README for dataset structure."""
    readme = """# Music AI Detector - Dataset

## Directory Structure

```
backend/data/
├── raw/
│   ├── ai_generated/     # AI-generated music
│   │   ├── song1.mp3
│   │   ├── song2.wav
│   │   └── ...
│   └── human_made/       # Human-made music
│       ├── song1.mp3
│       ├── song2.wav
│       └── ...
├── processed/
│   ├── features.csv      # Extracted features
│   └── metadata.json     # File metadata
└── models/
    └── trained_model.pkl # Trained model
```

## Data Collection Suggestions

### AI-Generated Music:
- Suno AI
- Udio
- MusicGen (Meta)
- AIVA
- Soundraw
- Boomy

### Human-Made Music:
- Spotify playlists (indie/underground artists)
- SoundCloud
- Bandcamp
- YouTube (original compositions)

## Processing

1. Put music files into the relevant folders.
2. Run:
```bash
python backend/app/dataset_processor.py
```

3. Features include:
   - Tempo stability and variance
   - Pitch metrics
   - Spectral features (MFCC, spectral centroid, etc.)
   - Micro-timing variations
   - Dynamic range
   - Harmonic-percussive ratio

## Note

At least 50-100 AI and 50-100 Human tracks are recommended.
More data = better model.
"""

    readme_path = Path("backend/data/README.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    logger.info(f"README created: {readme_path}")


if __name__ == "__main__":
    # Create README
    create_sample_readme()

    # Dataset processor
    processor = DatasetProcessor()

    # Info message
    logger.info("\nIMPORTANT: Make sure you have audio files in:")
    logger.info("  - backend/data/raw/ai_generated/")
    logger.info("  - backend/data/raw/human_made/")
    logger.info("\nPress Enter to start processing (or Ctrl+C to cancel)...")
    input()

    try:
        df = processor.process_dataset(separate_vocals=True)

        if df is not None:
            # Show info
            processor.get_dataset_info()
    except DatasetProcessingError as e:
        logger.error(f"Dataset processing failed: {e}")
    except KeyboardInterrupt:
        logger.info("\nProcessing cancelled by user")
