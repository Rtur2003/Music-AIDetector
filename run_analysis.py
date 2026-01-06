"""
Quick Start Script - Tüm analiz pipeline'ını tek komutla çalıştır
"""

import sys
from pathlib import Path

# Add backend/app to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "backend" / "app"))

# Local application imports (must be after path modification)
from config import get_config
from detailed_analyzer import DetailedMusicAnalyzer
from logging_config import get_logger

logger = get_logger(__name__)


def main():
    """
    Detaylı analiz pipeline
    """
    cfg = get_config()

    logger.info("=" * 80)
    logger.info(" " * 20 + "MUSIC AI DETECTOR - DETAILED ANALYSIS")
    logger.info("=" * 80)

    # Veri setini kontrol et
    ai_dir = cfg.ai_generated_dir
    human_dir = cfg.human_made_dir

    if not ai_dir.exists():
        logger.error(f"{ai_dir} klasörü bulunamadı!")
        logger.error("Lütfen AI müziklerini bu klasöre koyun.")
        sys.exit(1)

    if not human_dir.exists():
        logger.error(f"{human_dir} klasörü bulunamadı!")
        logger.error("Lütfen insan müziklerini bu klasöre koyun.")
        sys.exit(1)

    ai_files = list(ai_dir.glob("*.mp3")) + list(ai_dir.glob("*.wav"))
    human_files = list(human_dir.glob("*.mp3")) + list(human_dir.glob("*.wav"))

    if len(ai_files) == 0:
        logger.warning(f"{ai_dir} klasöründe müzik bulunamadı!")

    if len(human_files) == 0:
        logger.warning(f"{human_dir} klasöründe müzik bulunamadı!")

    logger.info("Bulunan dosyalar:")
    logger.info(f"  - AI müzikler: {len(ai_files)}")
    logger.info(f"  - Human müzikler: {len(human_files)}")
    logger.info(f"  - Toplam: {len(ai_files) + len(human_files)}")

    if len(ai_files) + len(human_files) == 0:
        logger.error("Hiç müzik dosyası bulunamadı. Çıkılıyor...")
        sys.exit(1)

    logger.info("-" * 80)
    logger.info("Bu analiz şunları yapacak:")
    logger.info("  1. Her müziği vocal'lerden ayıracak (Demucs)")
    logger.info("  2. 60+ özellik çıkaracak")
    logger.info("  3. AI vs Human karşılaştırması yapacak")
    logger.info("  4. Detaylı raporlar oluşturacak")
    logger.info("  5. Görselleştirmeler yapacak")
    logger.info("Tahmini süre: ~1 dakika per track")
    logger.info(f"Toplam süre: ~{len(ai_files) + len(human_files)} dakika")
    logger.info("-" * 80)

    response = input("\nDevam etmek istiyor musunuz? (y/n): ")
    if response.lower() != 'y':
        logger.info("İptal edildi.")
        sys.exit(0)

    # Analizi başlat
    analyzer = DetailedMusicAnalyzer()

    try:
        analyses = analyzer.analyze_dataset(
            ai_dir=ai_dir,
            human_dir=human_dir,
            separate_vocals=True
        )

        logger.info("=" * 80)
        logger.info(" " * 30 + "ANALİZ TAMAMLANDI!")
        logger.info("=" * 80)
        logger.info(f"Toplam {len(analyses)} parça analiz edildi.")
        logger.info(f"Raporlar: {cfg.analysis_dir}")
        logger.info("  - detailed_analysis_*.json     (Her track'in detayları)")
        logger.info("  - comparison_report_*.json     (AI vs Human istatistikleri)")
        logger.info("  - analysis_report_*.txt        (İnsan okunabilir rapor)")
        logger.info("  - feature_comparison_*.png     (Box plot görselleştirmeleri)")
        logger.info("  - correlation_heatmap_*.png    (Correlation heatmaps)")

        logger.info("=" * 80)
        logger.info("SONRAKİ ADIMLAR:")
        logger.info("=" * 80)
        logger.info(f"1. Raporları inceleyin ({cfg.analysis_dir}/analysis_report_*.txt)")
        logger.info("2. Görselleri kontrol edin (*.png dosyaları)")
        logger.info("3. AI vs Human farkları üzerine tartışın")
        logger.info("4. Veri setini işleyin: python backend/app/dataset_processor.py")
        logger.info("5. Modeli eğitin: python backend/app/model_trainer.py")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("Analiz kullanıcı tarafından durduruldu.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"HATA: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
