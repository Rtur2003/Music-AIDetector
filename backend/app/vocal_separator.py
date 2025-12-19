"""
Vocal Separator - Müzik dosyalarından vocal'i ayırıp melodiyi çıkarır
Demucs kullanarak 4 stem'e ayırır: vocals, drums, bass, other
"""

import os
import torch
import torchaudio
from pathlib import Path
import subprocess
import shutil

try:
    from .config import get_config
except ImportError:
    from config import get_config


class VocalSeparator:
    def __init__(self, model_name=None):
        """
        Args:
            model_name: Demucs model ismi (htdemucs, htdemucs_ft, mdx_extra vb.)
        """
        cfg = get_config()
        self.model_name = model_name if model_name is not None else cfg.demucs_model

    def separate(self, audio_path, output_dir):
        """
        Ses dosyasını stem'lere ayırır

        Args:
            audio_path: Giriş ses dosyası yolu
            output_dir: Çıkış klasörü

        Returns:
            dict: Her stem için dosya yolları
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Demucs komutu
        cmd = [
            "python", "-m", "demucs",
            "--two-stems=vocals",  # Sadece vocal ve instrumental ayır (daha hızlı)
            "-n", self.model_name,
            "-o", str(output_dir),
            str(audio_path)
        ]

        try:
            print(f"Separating vocals from: {audio_path.name}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Demucs çıktı yapısı: output_dir/model_name/song_name/vocals.wav
            song_name = audio_path.stem
            separated_dir = output_dir / self.model_name / song_name

            return {
                'vocals': separated_dir / 'vocals.wav',
                'instrumental': separated_dir / 'no_vocals.wav',
                'original': audio_path
            }

        except subprocess.CalledProcessError as e:
            print(f"Error separating audio: {e.stderr}")
            raise

    def get_instrumental_only(self, audio_path, output_dir):
        """
        Sadece instrumental (melodi) kısmını döndürür

        Args:
            audio_path: Giriş ses dosyası
            output_dir: Çıkış klasörü

        Returns:
            str: Instrumental dosya yolu
        """
        result = self.separate(audio_path, output_dir)
        return result['instrumental']


class SimpleStemExtractor:
    """
    Alternatif: Daha basit bir yaklaşım - spleeter kullanmadan
    Eğer demucs çalışmazsa bu kullanılabilir
    """

    @staticmethod
    def extract_instrumental_basic(audio_path, output_path):
        """
        Basit spectral subtraction ile vocal azaltma
        Not: Bu method demucs kadar iyi değil, sadece fallback
        """
        import librosa
        import soundfile as sf
        import numpy as np

        # Ses dosyasını yükle
        y, sr = librosa.load(audio_path, sr=22050, mono=False)

        if len(y.shape) == 1:
            # Mono ise stereo yap
            y = np.stack([y, y])

        # Basit vocal reduction: center channel subtraction
        # Stereo farkı al (vocals genelde center'da)
        if y.shape[0] == 2:
            # Side signal (instrumental çoğunlukla)
            instrumental = y[0] - y[1]
        else:
            instrumental = y[0]

        # Normalize
        instrumental = librosa.util.normalize(instrumental)

        # Kaydet
        sf.write(output_path, instrumental, sr)
        return output_path


if __name__ == "__main__":
    # Test
    separator = VocalSeparator()

    # Örnek kullanım
    test_file = "backend/data/raw/test_song.mp3"
    if os.path.exists(test_file):
        result = separator.separate(test_file, "backend/temp/separated")
        print(f"Instrumental saved to: {result['instrumental']}")
