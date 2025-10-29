# Music AI Detector

AI tarafından üretilen müziklerle insan yapımı müzikleri ayırt eden makine öğrenmesi sistemi.

## Özellikler

- **Vocal Separation**: Sözlü müziklerden vocal'i ayırıp sadece melodi analizi
- **Deep Feature Extraction**: 60+ özellik çıkarma (tempo, pitch, harmony, spectral, timing vb.)
- **Multiple ML Models**: Random Forest, XGBoost, SVM, Neural Network ve Ensemble
- **Detailed Analysis**: Her müziği detaylı analiz edip AI vs Human farkları gösterir
- **Minimal API**: Test için basit FastAPI endpoint

## Proje Yapısı

```
Music-AIDetector/
├── backend/
│   ├── app/
│   │   ├── vocal_separator.py      # Demucs ile vocal ayırma
│   │   ├── feature_extractor.py    # Feature extraction (60+ özellik)
│   │   ├── dataset_processor.py    # Veri seti işleme pipeline
│   │   ├── detailed_analyzer.py    # Detaylı analiz ve rapor
│   │   ├── model_trainer.py        # Model eğitimi (5 farklı algoritma)
│   │   ├── predictor.py            # Tahmin yapma
│   │   └── api.py                  # Minimal FastAPI
│   ├── data/
│   │   ├── raw/
│   │   │   ├── ai_generated/       # AI müzikler buraya
│   │   │   └── human_made/         # İnsan müzikleri buraya
│   │   ├── processed/              # İşlenmiş features
│   │   ├── models/                 # Eğitilmiş modeller
│   │   └── analysis/               # Detaylı analiz raporları
│   ├── temp/                       # Geçici dosyalar
│   └── uploads/                    # API upload'ları
└── requirements.txt
```

## Kurulum

### 1. Python ve Dependencies

```bash
# Python 3.8+ gerekli
python -m pip install -r requirements.txt
```

### 2. Demucs Kurulumu (Vocal Separation)

```bash
pip install demucs
```

## Kullanım - Adım Adım

### Adım 1: Veri Seti Hazırlama

100 AI + 100 Human müzik toplayın:

**AI Müzik Kaynakları:**
- Suno AI (suno.ai)
- Udio (udio.com)
- Meta MusicGen
- AIVA (aiva.ai)
- Soundraw (soundraw.io)

**Human Müzik Kaynakları:**
- Spotify playlists (indie/underground)
- SoundCloud
- Bandcamp
- YouTube (orijinal kompozisyonlar)

Dosyaları yerleştirin:
```
backend/data/raw/ai_generated/     ← AI müzikleri buraya (.mp3, .wav)
backend/data/raw/human_made/       ← İnsan müzikleri buraya
```

### Adım 2: Detaylı Analiz (İnceleme için)

Önce her müziği detaylı analiz edin, AI vs Human farkları görmek için:

```bash
cd backend/app
python detailed_analyzer.py
```

Bu:
- Her parçayı vocal'lerden ayırır
- 60+ özellik çıkarır
- AI vs Human karşılaştırmalı rapor oluşturur
- Görselleştirmeler yapar
- Text rapor oluşturur

**Çıktılar:**
- `backend/data/analysis/detailed_analysis_*.json` - Her track'in detaylı analizi
- `backend/data/analysis/comparison_report_*.json` - Karşılaştırma istatistikleri
- `backend/data/analysis/analysis_report_*.txt` - İnsan okunabilir rapor
- `backend/data/analysis/feature_comparison_*.png` - Box plots
- `backend/data/analysis/correlation_heatmap_*.png` - Correlation heatmaps

### Adım 3: Veri Seti İşleme

Analizden sonra, modeli eğitmek için veri setini işleyin:

```bash
cd backend/app
python dataset_processor.py
```

Bu:
- Tüm müziklerden vocal'leri ayırır
- Feature extraction yapar
- CSV formatında kaydeder

**Çıktı:**
- `backend/data/processed/features.csv` - Tüm features
- `backend/data/processed/metadata.json` - Dosya bilgileri

### Adım 4: Model Eğitimi

5 farklı algoritma dener ve en iyisini seçer:

```bash
cd backend/app
python model_trainer.py
```

**Eğitilen Modeller:**
1. Random Forest
2. XGBoost
3. SVM
4. Neural Network
5. Ensemble (Voting)

**Çıktı:**
- `backend/data/models/latest_model.pkl` - En iyi model
- `backend/data/models/latest_scaler.pkl` - Feature scaler
- `backend/data/models/metadata_*.json` - Model bilgileri
- `backend/data/models/feature_importance.png` - Önemli özellikler

### Adım 5: Tahmin Yapma

Yeni bir müzik dosyasını test edin:

```bash
cd backend/app
python predictor.py path/to/music.mp3
```

**Örnek Çıktı:**
```
============================================================
PREDICTION EXPLANATION
============================================================
File: unknown_song.mp3
Prediction: AI
Confidence: 87.34%
  - AI probability: 87.34%
  - Human probability: 12.66%

Top 10 Contributing Features:
  tempo_stability              : 0.0023 (importance: 0.0845)
  pitch_std                    : 12.3456 (importance: 0.0782)
  semitone_deviation           : 3.4567 (importance: 0.0721)
  ...
============================================================
```

### Adım 6: API Kullanımı (Opsiyonel)

Minimal API başlatın:

```bash
cd backend/app
python api.py
```

Test:
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@music.mp3"
```

## Çıkarılan Özellikler (60+)

### 1. Tempo & Rhythm (6 özellik)
- `tempo`: BPM
- `tempo_stability`: Tempo ne kadar stabil (AI'da çok düşük)
- `tempo_variance`: Tempo varyasyonu
- `tempo_cv`: Coefficient of variation
- `onset_variation`: Onset strength variation
- `num_beats`: Beat sayısı

### 2. Pitch & Harmony (7 özellik)
- `pitch_std`: Pitch standart sapması (AI çok düşük)
- `pitch_variance`: Pitch varyansı
- `semitone_deviation`: Semitone'dan sapma (AI çok düşük)
- `vibrato_strength`: Vibrato gücü (Human'da yüksek)
- `chroma_std/mean`: Harmony özellikleri
- `tonnetz_variance`: Tonal space variance

### 3. Spectral Features (26 özellik)
- `spectral_centroid_mean/std`: Brightness
- `spectral_rolloff_mean/std`: Frequency rolloff
- `spectral_contrast_mean/std`: Frequency band contrast
- `spectral_flatness_mean/std`: Noise vs tonal
- `zcr_mean/std`: Zero crossing rate
- `mfcc_0..12_mean/std`: Timbre (26 features)

### 4. Timing (4 özellik)
- `ioi_variance`: Inter-onset interval variance (AI'da çok düşük)
- `ioi_cv`: IOI coefficient of variation
- `ioi_entropy`: Timing randomness (Human'da yüksek)
- `num_onsets`: Onset sayısı

### 5. Dynamics (5 özellik)
- `rms_mean/std/range`: RMS energy
- `dynamic_range`: dB range
- `peak_to_avg_ratio`: Peak-to-average

### 6. Harmonic-Percussive (4 özellik)
- `hp_ratio`: Harmonic/Percussive ratio
- `harmonic_energy`: Harmonic component energy
- `percussive_energy`: Percussive component energy
- `harmonic_std`: Harmonic consistency

## AI vs Human - Temel Farklar

### AI Müziğin Karakteristikleri:
1. **Tempo**: Çok stabil, neredeyse hiç varyasyon yok
2. **Pitch**: Mükemmel pitch alignment, vibrato yok
3. **Timing**: Mikro-timing variations yok, çok mekanik
4. **Spectral**: Pattern tekrarları, uniform timbre
5. **Dynamics**: Aşırı compressed veya aşırı wide

### İnsan Müziğinin Karakteristikleri:
1. **Tempo**: Natural variations, "groove" var
2. **Pitch**: İnsan hataları, vibrato var
3. **Timing**: Mikro-timing variations (groove)
4. **Spectral**: Organic, varied timbre
5. **Dynamics**: Natural dynamic range

## Model Performansı (Örnek)

Tipik performans (100 AI + 100 Human ile):
```
Random Forest:    Accuracy: 0.92, F1: 0.91
XGBoost:         Accuracy: 0.94, F1: 0.93
SVM:             Accuracy: 0.88, F1: 0.87
Neural Network:  Accuracy: 0.91, F1: 0.90
Ensemble:        Accuracy: 0.95, F1: 0.94
```

## Gelişmiş Kullanım

### Sadece Belirli Feature'ları Kullan

```python
from feature_extractor import MusicFeatureExtractor

extractor = MusicFeatureExtractor()
features = extractor.extract_all_features("song.mp3")

# Sadece tempo features
tempo_features = extractor._extract_tempo_features(y, sr)
```

### Custom Model Eğit

```python
from model_trainer import MusicAIDetectorTrainer

trainer = MusicAIDetectorTrainer()
X, y = trainer.load_data()

# Kendi modelinizi ekleyin
from sklearn.ensemble import GradientBoostingClassifier
custom_model = GradientBoostingClassifier()
# ...
```

## Troubleshooting

### Demucs çalışmıyor
```bash
# Manuel test
python -m demucs --two-stems=vocals test_song.mp3
```

### Memory hatası
```python
# feature_extractor.py'de sample rate'i düşür
sr=16000  # default 22050 yerine
```

### Model bulunamadı
```bash
# Önce model eğitin
python model_trainer.py
```

## Notlar

- **Minimum veri**: 50 AI + 50 Human önerilir, 100+100 ideal
- **Vocal separation**: İlk çalıştırmada Demucs modelini indirir (~300MB)
- **İşlem süresi**: Her parça ~30-60 saniye (vocal separation dahil)
- **Disk alanı**: 100+100 parça için ~10GB (separated stems dahil)

## Lisans

MIT License

## Katkıda Bulunma

Pull request'ler kabul edilir.

## İletişim

Sorular için issue açın.
