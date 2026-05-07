"""
backfill_content.py — Lấy content cho các bài chưa có trong vneconomy_raw.csv

Cách chạy:
  python backfill_content.py          # back-fill tất cả bài thiếu content
  python backfill_content.py --demo   # chỉ 20 bài để test
"""

import sys
import time
import random
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent / "Data"
OUTPUT_FILE = DATA_DIR / "vneconomy_raw.csv"
CKPT_FILE   = DATA_DIR / ".backfill_checkpoint.txt"  # lưu index đã xong

DELAY    = (0.5, 1.0)
MAX_RETRY = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": "https://vneconomy.vn/",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _fetch_content(session: requests.Session, url: str) -> str:
    """Fetch bài và trả về full body text."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 404:
                return ""
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")

            # Thử selector cụ thể trước
            body = None
            for sel in [".article__body", ".detail__body", ".cms-body"]:
                el = soup.select_one(sel)
                if el:
                    body = el
                    break

            if body:
                paras = [p.get_text(strip=True)
                         for p in body.find_all("p")
                         if len(p.get_text(strip=True)) > 20]
            else:
                container = soup.select_one("article, main, .article, .detail")
                if container:
                    paras = [p.get_text(strip=True)
                             for p in container.find_all("p")
                             if len(p.get_text(strip=True)) > 20]
                else:
                    paras = []

            return " ".join(paras)

        except Exception as e:
            log.debug(f"  Lỗi (lần {attempt}): {e}")
            time.sleep(2 ** attempt)
    return ""


def backfill(demo: bool = False) -> None:
    # Load CSV
    log.info(f"Đọc {OUTPUT_FILE}...")
    df = pd.read_csv(OUTPUT_FILE, dtype=str)
    df["content"] = df["content"].fillna("")

    # Tìm các hàng thiếu content
    missing_idx = df.index[df["content"] == ""].tolist()
    log.info(f"Tổng bài thiếu content: {len(missing_idx):,}")

    if demo:
        missing_idx = missing_idx[:20]
        log.info("DEMO MODE: chỉ back-fill 20 bài.")

    # Resume từ checkpoint
    start_pos = 0
    if CKPT_FILE.exists():
        try:
            start_pos = int(CKPT_FILE.read_text().strip())
            log.info(f"Resume từ vị trí {start_pos}/{len(missing_idx)}")
        except Exception:
            pass

    missing_idx = missing_idx[start_pos:]
    if not missing_idx:
        log.info("Không còn bài nào cần back-fill.")
        return

    session = requests.Session()
    session.headers.update(HEADERS)

    filled = 0
    SAVE_EVERY = 100

    for pos, idx in enumerate(missing_idx, start=start_pos):
        url = df.at[idx, "url"]
        time.sleep(random.uniform(*DELAY))

        content = _fetch_content(session, url)
        df.at[idx, "content"] = content

        if content:
            filled += 1

        # Lưu định kỳ
        if (pos + 1) % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_FILE, index=False)
            CKPT_FILE.write_text(str(pos + 1))
            log.info(
                f"  [{pos+1}/{start_pos + len(missing_idx)}] "
                f"Đã fill: {filled} | Lưu checkpoint."
            )

    # Lưu lần cuối
    df.to_csv(OUTPUT_FILE, index=False)
    if CKPT_FILE.exists():
        CKPT_FILE.unlink()

    log.info(f"\n{'='*50}")
    log.info(f"HOÀN THÀNH!")
    log.info(f"  Đã back-fill: {filled:,}/{len(missing_idx):,} bài")
    log.info(f"  File: {OUTPUT_FILE}")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    backfill(demo=demo)
