"""
run_pipeline.py — Master Orchestrator
Chạy toàn bộ pipeline theo thứ tự:
  Step 1: Crawl VnExpress news
  Step 2: NLP (sentiment + LDA topic modeling)
  Step 3: Causal discovery + forecasting (main.py)

Cách chạy:
  python run_pipeline.py                   # full pipeline
  python run_pipeline.py --demo            # test nhanh (~10 phút, ít data)
  python run_pipeline.py --freq daily      # dùng dữ liệu daily
  python run_pipeline.py --skip-crawl      # bỏ qua step 1 (đã có raw news)
  python run_pipeline.py --skip-nlp        # bỏ qua step 2 (đã có news features)
  python run_pipeline.py --gpu             # dùng GPU cho sentiment
  python run_pipeline.py --topics 15       # tuỳ chỉnh số LDA topics
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "Data"
PYTHON     = sys.executable


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def check_done(label: str, output: Path, min_kb: float = 1.0) -> bool:
    """True nếu output file tồn tại và đủ lớn (không phải file rỗng)."""
    if output.exists() and output.stat().st_size / 1024 >= min_kb:
        kb = output.stat().st_size / 1024
        print(f"  [OK] {label} — {output.name} ({kb:.0f} KB)")
        return True
    return False


def run(label: str, args: list[str], allow_fail: bool = False) -> bool:
    """Chạy subprocess, in output trực tiếp ra terminal."""
    print(f"\n  $ {' '.join(args)}\n")
    t0 = time.time()
    result = subprocess.run(args, cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "DONE" if ok else "FAILED"
    print(f"\n  [{status}] {label}  ({elapsed:.0f}s)")
    if not ok and not allow_fail:
        print(f"\n❌ Pipeline dừng tại: {label}")
        sys.exit(1)
    return ok


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Causal News→Financial Pipeline")
    parser.add_argument("--demo",       action="store_true",
                        help="Test nhanh: crawl 5 trang/category, train LDA nhỏ")
    parser.add_argument("--freq",       default="monthly", choices=["monthly", "daily"],
                        help="Tần suất dữ liệu tài chính (default: monthly)")
    parser.add_argument("--skip-crawl", action="store_true",
                        help="Bỏ qua Step 1 (đã có vnexpress_raw.csv)")
    parser.add_argument("--skip-nlp",   action="store_true",
                        help="Bỏ qua Step 2 (đã có news_features_daily.csv)")
    parser.add_argument("--gpu",        action="store_true",
                        help="Dùng GPU cho PhoBERT sentiment")
    parser.add_argument("--topics",     type=int, default=10,
                        help="Số LDA topics (default: 10)")
    parser.add_argument("--fresh",      action="store_true",
                        help="Crawl lại từ đầu (xóa raw news cũ)")
    args = parser.parse_args()

    RAW_NEWS    = DATA_DIR / "vnexpress_raw.csv"
    NEWS_FEATS  = DATA_DIR / "news_features_daily.csv"
    DATASET     = DATA_DIR / f"dataset{'_daily' if args.freq == 'daily' else ''}.csv"

    header("CAUSAL NEWS → FINANCIAL PIPELINE")
    print(f"  Mode      : {'DEMO (test nhanh)' if args.demo else 'FULL'}")
    print(f"  Frequency : {args.freq}")
    print(f"  Topics    : {args.topics}")
    print(f"  GPU       : {'yes' if args.gpu else 'no (CPU)'}")
    print(f"  Python    : {PYTHON}")

    # ----------------------------------------------------------------
    # Step 1 — Crawl VnExpress
    # ----------------------------------------------------------------
    header("Step 1 / 3 — Crawl VnExpress News")

    if args.skip_crawl and check_done("Crawler", RAW_NEWS, min_kb=5):
        print("  → Bỏ qua (--skip-crawl)")
    elif check_done("Raw news cache", RAW_NEWS, min_kb=5) and not args.fresh and not args.demo:
        print("  → Đã có file, tiếp tục (resume mode).")
        print("     Dùng --fresh để crawl lại từ đầu.")
    else:
        crawl_args = [PYTHON, "news_crawler.py"]
        if args.demo:
            crawl_args.append("--demo")
        if args.fresh:
            crawl_args.append("--fresh")
        run("Crawl VnExpress", crawl_args)

    if not RAW_NEWS.exists():
        print("❌ Không có raw news data. Pipeline dừng.")
        sys.exit(1)

    rows = sum(1 for _ in open(RAW_NEWS)) - 1
    print(f"\n  Tổng bài báo đã crawl: {rows:,}")

    # ----------------------------------------------------------------
    # Step 2 — NLP: Sentiment + Topic Modeling
    # ----------------------------------------------------------------
    header("Step 2 / 3 — NLP Pipeline (Sentiment + LDA)")

    if args.skip_nlp and check_done("News features", NEWS_FEATS, min_kb=1):
        print("  → Bỏ qua (--skip-nlp)")
    elif check_done("News features cache", NEWS_FEATS, min_kb=1) and not args.demo:
        print("  → Đã có file. Dùng --skip-nlp để giữ nguyên hoặc xóa file để train lại.")
    else:
        nlp_args = [PYTHON, "news_processor.py", f"--topics={args.topics}"]
        if args.gpu:
            nlp_args.append("--gpu")
        if args.demo:
            # demo: LDA nhỏ hơn, không retrain nếu đã có
            nlp_args.append("--retrain-lda")
        run("NLP Pipeline", nlp_args)

    if not NEWS_FEATS.exists():
        print("⚠️  Không có news_features_daily.csv — tiếp tục với chỉ dữ liệu tài chính.")

    # ----------------------------------------------------------------
    # Step 3 — Causal Discovery + Forecasting
    # ----------------------------------------------------------------
    header("Step 3 / 3 — Causal Discovery + Forecasting (PCMCI+)")

    # Xoá dataset cache để rebuild với news features mới nhất
    if DATASET.exists():
        DATASET.unlink()
        print(f"  Đã xóa cache dataset cũ: {DATASET.name}")

    main_args = [PYTHON, "main.py", args.freq, "--fresh"]
    run("Causal Discovery + Forecast", main_args)

    # ----------------------------------------------------------------
    # Tổng kết
    # ----------------------------------------------------------------
    header("PIPELINE HOÀN THÀNH!")
    for label, path in [
        ("Raw news",      RAW_NEWS),
        ("News features", NEWS_FEATS),
        ("Dataset",       DATASET),
    ]:
        if path.exists():
            kb = path.stat().st_size / 1024
            print(f"  ✓ {label:<16} {path.name}  ({kb:.0f} KB)")
        else:
            print(f"  ✗ {label:<16} (không tồn tại)")

    print("\n  Causal graph  → causal_graph.png")
    print("  Model compare → model_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
