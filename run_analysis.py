"""
Quick Start Script - Tüm analiz pipeline'ını tek komutla çalıştır
"""

import sys
from pathlib import Path

# Add backend/app to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "app"))

from detailed_analyzer import DetailedMusicAnalyzer
from config import get_config


def main():
    """
    Detaylı analiz pipeline
    """
    cfg = get_config()
    
    print("\n" + "="*80)
    print(" "*20 + "MUSIC AI DETECTOR - DETAILED ANALYSIS")
    print("="*80)

    # Veri setini kontrol et
    ai_dir = cfg.ai_generated_dir
    human_dir = cfg.human_made_dir

    if not ai_dir.exists():
        print(f"\nERROR: {ai_dir} klasörü bulunamadı!")
        print("Lütfen AI müziklerini bu klasöre koyun.")
        sys.exit(1)

    if not human_dir.exists():
        print(f"\nERROR: {human_dir} klasörü bulunamadı!")
        print("Lütfen insan müziklerini bu klasöre koyun.")
        sys.exit(1)

    ai_files = list(ai_dir.glob("*.mp3")) + list(ai_dir.glob("*.wav"))
    human_files = list(human_dir.glob("*.mp3")) + list(human_dir.glob("*.wav"))

    if len(ai_files) == 0:
        print(f"\nWARNING: {ai_dir} klasöründe müzik bulunamadı!")

    if len(human_files) == 0:
        print(f"\nWARNING: {human_dir} klasöründe müzik bulunamadı!")

    print(f"\nBulunan dosyalar:")
    print(f"  - AI müzikler: {len(ai_files)}")
    print(f"  - Human müzikler: {len(human_files)}")
    print(f"  - Toplam: {len(ai_files) + len(human_files)}")

    if len(ai_files) + len(human_files) == 0:
        print("\nHiç müzik dosyası bulunamadı. Çıkılıyor...")
        sys.exit(1)

    print("\n" + "-"*80)
    print("Bu analiz şunları yapacak:")
    print("  1. Her müziği vocal'lerden ayıracak (Demucs)")
    print("  2. 60+ özellik çıkaracak")
    print("  3. AI vs Human karşılaştırması yapacak")
    print("  4. Detaylı raporlar oluşturacak")
    print("  5. Görselleştirmeler yapacak")
    print("\nTahmini süre: ~1 dakika per track")
    print(f"Toplam süre: ~{len(ai_files) + len(human_files)} dakika")
    print("-"*80)

    response = input("\nDevam etmek istiyor musunuz? (y/n): ")
    if response.lower() != 'y':
        print("İptal edildi.")
        sys.exit(0)

    # Analizi başlat
    analyzer = DetailedMusicAnalyzer()

    try:
        analyses = analyzer.analyze_dataset(
            ai_dir=ai_dir,
            human_dir=human_dir,
            separate_vocals=True
        )

        print("\n" + "="*80)
        print(" "*30 + "ANALİZ TAMAMLANDI!")
        print("="*80)
        print(f"\nToplam {len(analyses)} parça analiz edildi.")
        print(f"\nRaporlar: {cfg.analysis_dir}")
        print("  - detailed_analysis_*.json     (Her track'in detayları)")
        print("  - comparison_report_*.json     (AI vs Human istatistikleri)")
        print("  - analysis_report_*.txt        (İnsan okunabilir rapor)")
        print("  - feature_comparison_*.png     (Box plot görselleştirmeleri)")
        print("  - correlation_heatmap_*.png    (Correlation heatmaps)")

        print("\n" + "="*80)
        print("SONRAKİ ADIMLAR:")
        print("="*80)
        print(f"1. Raporları inceleyin ({cfg.analysis_dir}/analysis_report_*.txt)")
        print("2. Görselleri kontrol edin (*.png dosyaları)")
        print("3. AI vs Human farkları üzerine tartışın")
        print("4. Veri setini işleyin: python backend/app/dataset_processor.py")
        print("5. Modeli eğitin: python backend/app/model_trainer.py")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\nAnaliz kullanıcı tarafından durduruldu.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nHATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
