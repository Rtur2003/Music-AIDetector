"""
Detailed Analyzer - Her müziği detaylı analiz edip rapor oluşturur
AI vs Human tartışması için tüm feature'ları gösterir
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from vocal_separator import VocalSeparator
from feature_extractor import MusicFeatureExtractor
from datetime import datetime


class DetailedMusicAnalyzer:
    def __init__(self, output_dir="backend/data/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.separator = VocalSeparator()
        self.extractor = MusicFeatureExtractor()

    def analyze_single_track(self, audio_path, category="Unknown", separate_vocals=True):
        """
        Tek bir parçayı derinlemesine analiz et

        Returns:
            dict: Tüm analizler
        """
        audio_path = Path(audio_path)
        print(f"\n{'='*60}")
        print(f"Analyzing: {audio_path.name}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        analysis = {
            'filename': audio_path.name,
            'category': category,
            'path': str(audio_path),
            'timestamp': datetime.now().isoformat()
        }

        # 1. Vocal Separation
        if separate_vocals:
            print("\n[1/2] Separating audio components...")
            try:
                sep_result = self.separator.separate(audio_path, self.output_dir / "separated")
                analysis['separated_files'] = {
                    'vocals': str(sep_result['vocals']),
                    'instrumental': str(sep_result['instrumental'])
                }
                # Instrumental'ı analiz et
                audio_to_analyze = sep_result['instrumental']
            except Exception as e:
                print(f"Warning: Vocal separation failed: {e}")
                audio_to_analyze = audio_path
                analysis['separated_files'] = None
        else:
            audio_to_analyze = audio_path

        # 2. Feature Extraction
        print("\n[2/2] Extracting features...")
        features = self.extractor.extract_all_features(str(audio_to_analyze))
        analysis['features'] = features

        # 3. Feature Categorization
        analysis['feature_categories'] = self._categorize_features(features)

        # 4. Anomaly Detection (her feature için z-score)
        # Bu, diğer parçalarla karşılaştırıldığında anormal olan feature'ları gösterir

        return analysis

    def _categorize_features(self, features):
        """
        Feature'ları kategorilere ayır - analiz kolaylığı için
        """
        categories = {
            'tempo_rhythm': {},
            'pitch_harmony': {},
            'spectral': {},
            'timing': {},
            'dynamics': {},
            'harmonic_percussive': {}
        }

        for key, value in features.items():
            if any(x in key for x in ['tempo', 'beat', 'onset']):
                categories['tempo_rhythm'][key] = value
            elif any(x in key for x in ['pitch', 'chroma', 'tonnetz', 'vibrato']):
                categories['pitch_harmony'][key] = value
            elif any(x in key for x in ['spectral', 'mfcc', 'zcr', 'flatness', 'contrast', 'rolloff', 'centroid']):
                categories['spectral'][key] = value
            elif any(x in key for x in ['ioi', 'timing']):
                categories['timing'][key] = value
            elif any(x in key for x in ['rms', 'dynamic', 'peak']):
                categories['dynamics'][key] = value
            elif any(x in key for x in ['harmonic', 'percussive', 'hp_']):
                categories['harmonic_percussive'][key] = value

        return categories

    def analyze_dataset(self, ai_dir, human_dir, separate_vocals=True):
        """
        Tüm veri setini analiz et ve karşılaştırmalı rapor oluştur

        Args:
            ai_dir: AI müzikler klasörü
            human_dir: Human müzikler klasörü
        """
        ai_dir = Path(ai_dir)
        human_dir = Path(human_dir)

        print("\n" + "="*60)
        print("DETAILED DATASET ANALYSIS")
        print("="*60)

        all_analyses = []

        # AI müzikleri
        if ai_dir.exists():
            ai_files = list(ai_dir.glob("*.mp3")) + list(ai_dir.glob("*.wav"))
            print(f"\nFound {len(ai_files)} AI tracks")

            for audio_file in tqdm(ai_files, desc="Analyzing AI tracks"):
                try:
                    analysis = self.analyze_single_track(audio_file, "AI", separate_vocals)
                    all_analyses.append(analysis)
                except Exception as e:
                    print(f"Error analyzing {audio_file.name}: {e}")

        # Human müzikler
        if human_dir.exists():
            human_files = list(human_dir.glob("*.mp3")) + list(human_dir.glob("*.wav"))
            print(f"\nFound {len(human_files)} Human tracks")

            for audio_file in tqdm(human_files, desc="Analyzing Human tracks"):
                try:
                    analysis = self.analyze_single_track(audio_file, "Human", separate_vocals)
                    all_analyses.append(analysis)
                except Exception as e:
                    print(f"Error analyzing {audio_file.name}: {e}")

        # Sonuçları kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"detailed_analysis_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_analyses, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed analysis saved to: {output_file}")

        # Comparative report oluştur
        self._create_comparative_report(all_analyses, timestamp)

        return all_analyses

    def _create_comparative_report(self, analyses, timestamp):
        """
        AI vs Human karşılaştırmalı rapor
        """
        # DataFrame'e çevir
        ai_data = []
        human_data = []

        for analysis in analyses:
            features = analysis['features']
            if analysis['category'] == 'AI':
                ai_data.append(features)
            else:
                human_data.append(features)

        if not ai_data or not human_data:
            print("Not enough data for comparison")
            return

        ai_df = pd.DataFrame(ai_data)
        human_df = pd.DataFrame(human_data)

        # İstatistiksel karşılaştırma
        report = {
            'timestamp': timestamp,
            'ai_count': len(ai_df),
            'human_count': len(human_df),
            'feature_comparison': {}
        }

        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS: AI vs HUMAN")
        print("="*60)

        # Her feature için karşılaştırma
        print("\n{:<35} {:>12} {:>12} {:>12}".format(
            "Feature", "AI Mean", "Human Mean", "Difference"
        ))
        print("-"*75)

        for column in ai_df.columns:
            ai_mean = ai_df[column].mean()
            human_mean = human_df[column].mean()
            diff_pct = ((ai_mean - human_mean) / (human_mean + 1e-10)) * 100

            report['feature_comparison'][column] = {
                'ai_mean': float(ai_mean),
                'ai_std': float(ai_df[column].std()),
                'human_mean': float(human_mean),
                'human_std': float(human_df[column].std()),
                'difference_percent': float(diff_pct)
            }

            # En büyük farkları göster
            if abs(diff_pct) > 20:  # %20'den fazla fark varsa
                print("{:<35} {:>12.4f} {:>12.4f} {:>11.1f}%".format(
                    column[:34], ai_mean, human_mean, diff_pct
                ))

        # Raporu kaydet
        report_file = self.output_dir / f"comparison_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"\nComparison report saved to: {report_file}")

        # Görselleştirme
        self._create_visualizations(ai_df, human_df, timestamp)

        # Text rapor
        self._create_text_report(report, timestamp)

    def _create_visualizations(self, ai_df, human_df, timestamp):
        """
        AI vs Human görselleştirmeleri
        """
        print("\nCreating visualizations...")

        # En önemli farkları bul
        differences = {}
        for col in ai_df.columns:
            ai_mean = ai_df[col].mean()
            human_mean = human_df[col].mean()
            diff = abs(ai_mean - human_mean) / (abs(human_mean) + 1e-10)
            differences[col] = diff

        # En büyük 15 farkı göster
        top_features = sorted(differences.items(), key=lambda x: x[1], reverse=True)[:15]
        top_feature_names = [f[0] for f in top_features]

        # Plot 1: Box plots for top features
        fig, axes = plt.subplots(5, 3, figsize=(20, 25))
        axes = axes.flatten()

        for idx, feature in enumerate(top_feature_names):
            ax = axes[idx]

            data_to_plot = [
                ai_df[feature].dropna(),
                human_df[feature].dropna()
            ]

            ax.boxplot(data_to_plot, labels=['AI', 'Human'])
            ax.set_title(feature, fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / f"feature_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {plot_file}")
        plt.close()

        # Plot 2: Correlation heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # AI correlation
        ai_corr = ai_df.corr()
        sns.heatmap(ai_corr, cmap='coolwarm', center=0, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('AI Music - Feature Correlations', fontsize=14)

        # Human correlation
        human_corr = human_df.corr()
        sns.heatmap(human_corr, cmap='coolwarm', center=0, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Human Music - Feature Correlations', fontsize=14)

        plt.tight_layout()
        corr_file = self.output_dir / f"correlation_heatmap_{timestamp}.png"
        plt.savefig(corr_file, dpi=150, bbox_inches='tight')
        print(f"Correlation heatmap saved: {corr_file}")
        plt.close()

    def _create_text_report(self, report, timestamp):
        """
        İnsan okunabilir text rapor
        """
        report_text = f"""
{'='*80}
MUSIC AI DETECTOR - COMPARATIVE ANALYSIS REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
{'='*80}
AI-Generated Tracks:  {report['ai_count']}
Human-Made Tracks:    {report['human_count']}
Total Tracks:         {report['ai_count'] + report['human_count']}

KEY FINDINGS
{'='*80}

Below are the features with the most significant differences between AI and Human music:

"""

        # En büyük farkları bul
        features = report['feature_comparison']
        sorted_features = sorted(
            features.items(),
            key=lambda x: abs(x[1]['difference_percent']),
            reverse=True
        )

        report_text += "\n{:<40} {:>12} {:>12} {:>12}\n".format(
            "Feature", "AI Mean", "Human Mean", "Diff %"
        )
        report_text += "-"*80 + "\n"

        for feature_name, stats in sorted_features[:30]:
            report_text += "{:<40} {:>12.4f} {:>12.4f} {:>11.1f}%\n".format(
                feature_name[:39],
                stats['ai_mean'],
                stats['human_mean'],
                stats['difference_percent']
            )

        report_text += "\n" + "="*80 + "\n"
        report_text += "INTERPRETATION GUIDE\n"
        report_text += "="*80 + "\n"
        report_text += """
TEMPO & RHYTHM FEATURES:
- tempo_stability: AI genelde daha düşük (çok stabil tempo)
- tempo_variance: AI'da minimal, Human'da daha yüksek
- onset_variation: AI çok düzgün, Human daha varied

PITCH & HARMONY FEATURES:
- pitch_std/variance: AI çok düşük (perfect pitch), Human daha yüksek
- semitone_deviation: AI çok düşük (quantized), Human daha natural
- vibrato_strength: Human'da daha belirgin

SPECTRAL FEATURES:
- mfcc_*: Timbre özellikleri, AI genelde daha uniform
- spectral_contrast: AI'da pattern'ler tekrarlayıcı

TIMING FEATURES:
- ioi_variance: Inter-onset intervals, AI'da çok consistent
- ioi_entropy: Randomness, Human'da daha yüksek

DYNAMICS:
- dynamic_range: AI çok geniş veya çok dar olabilir
- rms_std: AI genelde daha düşük (over-compressed)

"""

        report_text += "\nCONCLUSION\n"
        report_text += "="*80 + "\n"
        report_text += """
Bu analize dayanarak, AI ve Human müziği ayırt etmek için en önemli özellikler:
1. Tempo stability ve variance (AI çok stabil)
2. Pitch perfection metrics (AI çok perfect)
3. Micro-timing variations (Human'da groove var, AI'da yok)
4. Spectral pattern consistency (AI tekrarlayıcı)
5. Dynamic range handling (AI aşırı compressed veya aşırı wide)

Makine öğrenmesi modeli bu özellikleri kombine ederek yüksek doğrulukla
AI vs Human ayırımı yapabilir.
"""

        # Text dosyasını kaydet
        text_file = self.output_dir / f"analysis_report_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"Text report saved: {text_file}")

        # Ekrana da yazdır
        print(report_text)


def main():
    """
    Ana analiz fonksiyonu
    """
    analyzer = DetailedMusicAnalyzer()

    ai_dir = "backend/data/raw/ai_generated"
    human_dir = "backend/data/raw/human_made"

    print("\n" + "="*60)
    print("DETAILED MUSIC ANALYZER")
    print("="*60)
    print("\nThis tool will:")
    print("1. Analyze each track individually")
    print("2. Extract ALL features")
    print("3. Create detailed comparison reports")
    print("4. Generate visualizations")
    print("5. Provide insights for AI vs Human differences")
    print("\nMake sure you have music files in:")
    print(f"  - {ai_dir}")
    print(f"  - {human_dir}")
    print("\nPress Enter to start (or Ctrl+C to cancel)...")
    input()

    # Analiz yap
    analyses = analyzer.analyze_dataset(
        ai_dir=ai_dir,
        human_dir=human_dir,
        separate_vocals=True
    )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Total tracks analyzed: {len(analyses)}")
    print(f"Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
