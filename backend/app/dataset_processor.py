"""
Dataset Processor - Veri setini işler ve eğitim için hazırlar

Kullanım:
1. backend/data/raw/ai_generated/ klasörüne AI müziklerini koy
2. backend/data/raw/human_made/ klasörüne insan müziklerini koy
3. Bu script'i çalıştır
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from vocal_separator import VocalSeparator
from feature_extractor import MusicFeatureExtractor


class DatasetProcessor:
    def __init__(self, data_dir="backend/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.temp_dir = Path("backend/temp")

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.separator = VocalSeparator()
        self.extractor = MusicFeatureExtractor()

    def process_dataset(self, separate_vocals=True):
        """
        Tüm veri setini işler

        Args:
            separate_vocals: Vocal'leri ayır mı? (False ise direkt kullan)
        """
        print("=" * 60)
        print("DATASET PROCESSING STARTED")
        print("=" * 60)

        all_features = []
        all_labels = []
        metadata = []

        # AI Generated müzikler
        ai_dir = self.raw_dir / "ai_generated"
        if ai_dir.exists():
            print(f"\n[1/2] Processing AI-generated music from: {ai_dir}")
            ai_files = list(ai_dir.glob("*.mp3")) + list(ai_dir.glob("*.wav"))
            ai_features, ai_metadata = self._process_category(
                ai_files, label=1, category="AI", separate_vocals=separate_vocals
            )
            all_features.extend(ai_features)
            all_labels.extend([1] * len(ai_features))
            metadata.extend(ai_metadata)
        else:
            print(f"Warning: {ai_dir} not found!")

        # Human Made müzikler
        human_dir = self.raw_dir / "human_made"
        if human_dir.exists():
            print(f"\n[2/2] Processing Human-made music from: {human_dir}")
            human_files = list(human_dir.glob("*.mp3")) + list(human_dir.glob("*.wav"))
            human_features, human_metadata = self._process_category(
                human_files, label=0, category="Human", separate_vocals=separate_vocals
            )
            all_features.extend(human_features)
            all_labels.extend([0] * len(human_features))
            metadata.extend(human_metadata)
        else:
            print(f"Warning: {human_dir} not found!")

        if len(all_features) == 0:
            print("\nERROR: No audio files found!")
            print(f"Please add music files to:")
            print(f"  - {ai_dir}")
            print(f"  - {human_dir}")
            return None

        # DataFrame oluştur
        df = pd.DataFrame(all_features)
        df['label'] = all_labels

        # Save
        output_file = self.processed_dir / "features.csv"
        df.to_csv(output_file, index=False)

        # Metadata kaydet
        metadata_file = self.processed_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETED")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"  - AI generated: {sum(all_labels)}")
        print(f"  - Human made: {len(all_labels) - sum(all_labels)}")
        print(f"Total features: {len(df.columns) - 1}")
        print(f"\nSaved to: {output_file}")
        print(f"Metadata: {metadata_file}")

        return df

    def _process_category(self, files, label, category, separate_vocals=True):
        """
        Bir kategoriyi işler (AI veya Human)
        """
        features_list = []
        metadata_list = []

        for audio_file in tqdm(files, desc=f"Processing {category}"):
            try:
                # Vocal separation (opsiyonel)
                if separate_vocals:
                    print(f"  Separating vocals: {audio_file.name}")
                    result = self.separator.separate(audio_file, self.temp_dir)
                    # Sadece instrumental kullan
                    audio_to_analyze = result['instrumental']
                else:
                    audio_to_analyze = audio_file

                # Feature extraction
                print(f"  Extracting features: {audio_file.name}")
                features = self.extractor.extract_all_features(str(audio_to_analyze))

                features_list.append(features)

                # Metadata
                metadata_list.append({
                    'filename': audio_file.name,
                    'category': category,
                    'label': label,
                    'original_path': str(audio_file),
                    'instrumental_path': str(audio_to_analyze) if separate_vocals else None
                })

            except Exception as e:
                print(f"  Error processing {audio_file.name}: {e}")
                continue

        return features_list, metadata_list

    def get_dataset_info(self):
        """
        İşlenmiş veri seti hakkında bilgi
        """
        features_file = self.processed_dir / "features.csv"

        if not features_file.exists():
            print("No processed dataset found. Run process_dataset() first.")
            return None

        df = pd.read_csv(features_file)

        print("\n" + "=" * 60)
        print("DATASET INFO")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(df.columns) - 1}")
        print(f"\nClass distribution:")
        print(f"  - AI (label=1): {sum(df['label'])} ({sum(df['label'])/len(df)*100:.1f}%)")
        print(f"  - Human (label=0): {len(df) - sum(df['label'])} ({(len(df) - sum(df['label']))/len(df)*100:.1f}%)")

        print(f"\nFeature summary:")
        print(df.describe())

        return df


def create_sample_readme():
    """
    Veri seti için README oluştur
    """
    readme = """# Music AI Detector - Dataset

## Veri Seti Yapısı

```
backend/data/
├── raw/
│   ├── ai_generated/     # AI tarafından üretilen müzikler
│   │   ├── song1.mp3
│   │   ├── song2.wav
│   │   └── ...
│   └── human_made/       # İnsan tarafından yapılan müzikler
│       ├── song1.mp3
│       ├── song2.wav
│       └── ...
├── processed/
│   ├── features.csv      # İşlenmiş özellikler
│   └── metadata.json     # Dosya metadata'sı
└── models/
    └── trained_model.pkl # Eğitilmiş model
```

## Veri Toplama Önerileri

### AI-Generated Müzikler:
- Suno AI
- Udio
- MusicGen (Meta)
- AIVA
- Soundraw
- Boomy

### Human-Made Müzikler:
- Spotify playlists (indie/underground artists)
- SoundCloud
- Bandcamp
- YouTube (original compositions)

## Veri Seti İşleme

1. Müzik dosyalarını ilgili klasörlere koy
2. Script'i çalıştır:
```bash
python backend/app/dataset_processor.py
```

3. Features extracted:
   - Tempo stability ve variance
   - Pitch perfection metrics
   - Spectral features (MFCC, spectral centroid, etc.)
   - Micro-timing variations
   - Dynamic range
   - Harmonic-percussive ratio

## Not

En az 50-100 AI ve 50-100 Human müzik önerilir.
Daha fazla veri = daha iyi model.
"""

    with open("backend/data/README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print("README created: backend/data/README.md")


if __name__ == "__main__":
    # README oluştur
    create_sample_readme()

    # Dataset processor
    processor = DatasetProcessor()

    # Veri setini işle
    print("\nIMPORTANT: Make sure you have audio files in:")
    print("  - backend/data/raw/ai_generated/")
    print("  - backend/data/raw/human_made/")
    print("\nPress Enter to start processing (or Ctrl+C to cancel)...")
    input()

    df = processor.process_dataset(separate_vocals=True)

    if df is not None:
        # Info göster
        processor.get_dataset_info()
