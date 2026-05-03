"""
cafef_crawler.py — CafeF Financial News Crawler (2015-2026)

Chiến lược:
  1. Query Wayback Machine CDX API theo category + từng năm → lấy list URL
  2. Fetch bài từ live CafeF (nhanh)
  3. Nếu live fail → fallback Wayback Machine snapshot
  4. Extract title, description, date từ meta tags
  5. Auto-resume checkpoint, lưu mỗi SAVE_EVERY bài

Cách chạy:
  python cafef_crawler.py              # full crawl (tiếp tục nếu đã có)
  python cafef_crawler.py --fresh      # crawl lại từ đầu
  python cafef_crawler.py --demo       # chỉ 50 URL/category để test
"""

import re
import sys
import json
import time
import random
import logging
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent / "Data"
OUTPUT_FILE = DATA_DIR / "cafef_raw.csv"
CKPT_FILE   = DATA_DIR / ".cafef_checkpoint.json"

CATEGORIES = {
    "thi-truong-chung-khoan": "Chứng khoán",
    "tai-chinh-ngan-hang":    "Tài chính - Ngân hàng",
    "vi-mo-dau-tu":           "Vĩ mô - Đầu tư",
    "doanh-nghiep":           "Doanh nghiệp",
}

START_YEAR  = 2015
END_YEAR    = 2026
SAVE_EVERY  = 100
MAX_RETRIES = 3

DELAY_CDX   = (1.0, 2.0)   # giữa các CDX API call
DELAY_FETCH = (0.5, 1.2)   # giữa các fetch bài

CDX_API = "http://web.archive.org/cdx/search/cdx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://cafef.vn/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CDX API — lấy danh sách URL từ Wayback Machine
# ------------------------------------------------------------------

def _cdx_get_urls(session: requests.Session, slug: str,
                  year: int, demo: bool = False) -> list[str]:
    """
    Query CDX API để lấy tất cả URL bài báo của 1 category trong 1 năm.
    Trả về list URL (live CafeF, không phải Wayback URL).
    """
    params = {
        "url":      f"cafef.vn/{slug}/*.chn",
        "output":   "json",
        "fl":       "original",
        "collapse": "urlkey",
        "filter":   ["statuscode:200", "mimetype:text/html"],
        "from":     f"{year}0101",
        "to":       f"{year}1231",
        "limit":    50 if demo else 50000,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(CDX_API, params=params, timeout=30)
            if r.status_code != 200:
                log.warning(f"  CDX HTTP {r.status_code}, thử lại...")
                time.sleep(2 ** attempt)
                continue

            data = r.json()
            if not data or len(data) <= 1:
                return []

            # Bỏ header row đầu tiên ["original"]
            urls = [row[0] for row in data[1:]]

            # Lọc chỉ lấy bài báo (có ID cuối URL), bỏ trang category
            urls = [u for u in urls if re.search(r"-\d{10,}\.chn$", u)]

            # Normalize về https://cafef.vn/...
            normalized = []
            for u in urls:
                if not u.startswith("http"):
                    u = "https://" + u
                u = re.sub(r"^http://", "https://", u)
                normalized.append(u)

            return normalized

        except Exception as e:
            log.warning(f"  CDX lỗi (lần {attempt}): {e}")
            time.sleep(2 ** attempt)

    return []


# ------------------------------------------------------------------
# Fetch bài báo
# ------------------------------------------------------------------

def _get(session: requests.Session, url: str,
         timeout: int = 15) -> BeautifulSoup | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            log.debug(f"    Fetch lỗi (lần {attempt}): {e}")
            time.sleep(2 ** attempt)
    return None


def _wayback_url(original_url: str) -> str:
    """Chuyển live URL sang Wayback Machine URL (snapshot gần nhất)."""
    return f"https://web.archive.org/web/{original_url}"


def _parse_article(soup: BeautifulSoup, url: str, label: str) -> dict | None:
    """Trích xuất title, description, date từ HTML bài báo CafeF."""
    # --- Title ---
    title = None
    for sel in ["h1.title", "h1.detail-title", "h1", 'meta[property="og:title"]']:
        el = soup.select_one(sel)
        if el:
            title = el.get("content") or el.get_text(strip=True)
            break
    if not title:
        return None

    # --- Date ---
    art_date = None
    for sel in [
        'meta[property="article:published_time"]',
        'meta[name="pubdate"]',
        'meta[itemprop="datePublished"]',
    ]:
        el = soup.select_one(sel)
        if el and el.get("content"):
            try:
                art_date = date.fromisoformat(el["content"][:10])
                break
            except ValueError:
                pass

    # Fallback: lấy ngày từ URL (cafef URL có timestamp 10 chữ số cuối)
    if art_date is None:
        m = re.search(r"-(\d{10})\.chn", url)
        if m:
            try:
                import datetime
                art_date = datetime.datetime.fromtimestamp(
                    int(m.group(1))
                ).date()
            except Exception:
                pass

    if art_date is None:
        return None

    # --- Description ---
    desc = ""
    for sel in [
        'meta[property="og:description"]',
        'meta[name="description"]',
        ".sapo", ".FlyTitle", "h2.sapo",
    ]:
        el = soup.select_one(sel)
        if el:
            desc = el.get("content") or el.get_text(strip=True)
            break

    return {
        "date":        art_date.isoformat(),
        "title":       title,
        "description": desc,
        "url":         url,
        "category":    label,
    }


def _fetch_article(session: requests.Session, url: str,
                   label: str) -> dict | None:
    """Fetch từ live CafeF, fallback Wayback Machine nếu fail."""
    time.sleep(random.uniform(*DELAY_FETCH))

    soup = _get(session, url)
    if soup is None:
        # Fallback Wayback
        wb_url = _wayback_url(url)
        soup = _get(session, wb_url)
        if soup is None:
            return None

    return _parse_article(soup, url, label)


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if CKPT_FILE.exists():
        try:
            return json.loads(CKPT_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(ckpt: dict) -> None:
    CKPT_FILE.write_text(json.dumps(ckpt, indent=2))


def _append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not OUTPUT_FILE.exists()
    df.to_csv(OUTPUT_FILE, mode="a", index=False, header=write_header)


# ------------------------------------------------------------------
# Crawl 1 category
# ------------------------------------------------------------------

def _crawl_category(slug: str, label: str,
                    session: requests.Session,
                    seen_urls: set,
                    ckpt: dict,
                    demo: bool = False) -> None:

    years = range(START_YEAR, END_YEAR + 1)
    buffer: list[dict] = []

    for year in years:
        ckpt_key = f"{slug}:{year}"

        if ckpt.get(ckpt_key) == "done":
            log.info(f"  [SKIP] {label} năm {year} — đã hoàn thành.")
            continue

        log.info(f"\n  [{label}] Năm {year} — query CDX API...")
        time.sleep(random.uniform(*DELAY_CDX))

        urls = _cdx_get_urls(session, slug, year, demo=demo)
        new_urls = [u for u in urls if u not in seen_urls]

        log.info(f"    CDX: {len(urls):,} URL tổng | {len(new_urls):,} URL mới")

        if not new_urls:
            ckpt[ckpt_key] = "done"
            _save_checkpoint(ckpt)
            continue

        # Resume từ offset nếu có
        start_idx = ckpt.get(f"{ckpt_key}:idx", 0)
        if start_idx > 0:
            log.info(f"    Resume từ bài {start_idx}/{len(new_urls)}")
        new_urls = new_urls[start_idx:]

        for i, url in enumerate(new_urls, start=start_idx):
            row = _fetch_article(session, url, label)
            if row:
                buffer.append(row)
                seen_urls.add(url)

            # Lưu định kỳ
            if (i + 1) % SAVE_EVERY == 0 and buffer:
                _append_rows(buffer)
                buffer = []
                ckpt[f"{ckpt_key}:idx"] = i + 1
                _save_checkpoint(ckpt)
                log.info(f"    [checkpoint] {i+1}/{len(urls)} bài — đã lưu")

        if buffer:
            _append_rows(buffer)
            buffer = []

        ckpt[ckpt_key] = "done"
        if f"{ckpt_key}:idx" in ckpt:
            del ckpt[f"{ckpt_key}:idx"]
        _save_checkpoint(ckpt)
        log.info(f"    Hoàn thành năm {year}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def crawl_cafef(resume: bool = True, demo: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)

    if demo:
        log.info("DEMO MODE: 50 URL/category/năm.")

    # Load seen URLs
    seen_urls: set = set()
    if resume and OUTPUT_FILE.exists():
        df_old = pd.read_csv(OUTPUT_FILE, usecols=["url"])
        seen_urls = set(df_old["url"].dropna())
        log.info(f"Resume: đã có {len(seen_urls):,} URLs.")
    elif not resume:
        for f in [OUTPUT_FILE, CKPT_FILE]:
            if f.exists():
                f.unlink()
        log.info("Fresh start: xóa data cũ.")

    ckpt = _load_checkpoint()
    session = requests.Session()
    session.headers.update(HEADERS)

    for slug, label in CATEGORIES.items():
        log.info(f"\n{'='*55}")
        log.info(f"  Category: {label}  ({slug})")
        log.info(f"{'='*55}")
        _crawl_category(slug, label, session, seen_urls, ckpt, demo=demo)
        time.sleep(random.uniform(2.0, 4.0))

    if not OUTPUT_FILE.exists():
        log.warning("Không crawl được bài nào.")
        return pd.DataFrame()

    # Dedup + sort
    df = pd.read_csv(OUTPUT_FILE)
    df = df.drop_duplicates("url")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    log.info(f"\n{'='*55}")
    log.info(f"HOÀN THÀNH!")
    log.info(f"  Tổng bài  : {len(df):,}")
    if not df.empty:
        log.info(f"  Từ        : {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"  Lưu tại   : {OUTPUT_FILE}")
    log.info(f"{'='*55}")
    return df


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    resume = "--fresh" not in sys.argv
    demo   = "--demo"  in sys.argv

    df = crawl_cafef(resume=resume, demo=demo)

    if not df.empty:
        print(f"\n--- Mẫu 5 bài ---")
        print(df[["date", "category", "title"]].head().to_string(index=False))

        print(f"\n--- Số bài theo category ---")
        print(df["category"].value_counts().to_string())

        print(f"\n--- Số bài theo năm ---")
        print(df["date"].dt.year.value_counts().sort_index().to_string())
