"""
vneconomy_crawler.py — VnEconomy News Crawler (2015-2026)

Chiến lược:
  1. Đọc sitemap theo tháng: vneconomy.vn/sitemap/news-YYYY-M.xml
  2. Lấy URL + date trực tiếp từ sitemap (không cần đoán)
  3. Fetch từng bài để lấy title, description, category
  4. Checkpoint theo tháng, auto-resume

Cách chạy:
  python vneconomy_crawler.py              # full crawl (tiếp tục nếu đã có)
  python vneconomy_crawler.py --fresh      # crawl lại từ đầu
  python vneconomy_crawler.py --demo       # chỉ 3 tháng đầu để test
"""

import re
import sys
import json
import time
import random
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent / "Data"
OUTPUT_FILE = DATA_DIR / "vneconomy_raw.csv"
CKPT_FILE   = DATA_DIR / ".vneconomy_checkpoint.json"

START_YEAR, START_MONTH = 2015, 1
END_YEAR,   END_MONTH   = 2026, 4

SAVE_EVERY  = 200     # lưu mỗi N bài
MAX_RETRIES = 3
DELAY_ART   = (0.6, 1.2)
DELAY_SITEMAP = (1.0, 2.0)

SITEMAP_BASE = "https://vneconomy.vn/sitemap/news-{year}-{month}.xml"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://vneconomy.vn/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# HTTP helper
# ------------------------------------------------------------------

def _get(session: requests.Session, url: str,
         timeout: int = 15) -> BeautifulSoup | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                log.debug(f"  HTTP {r.status_code}: {url}")
                time.sleep(2 ** attempt)
                continue
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            log.debug(f"  Lỗi (lần {attempt}): {e}")
            time.sleep(2 ** attempt)
    return None


# ------------------------------------------------------------------
# Sitemap reader
# ------------------------------------------------------------------

def _read_sitemap(session: requests.Session,
                  year: int, month: int) -> list[dict]:
    """
    Đọc sitemap tháng, trả về list {url, date}.
    Date lấy từ <lastmod> — chính xác, không cần fetch thêm.
    """
    url = SITEMAP_BASE.format(year=year, month=month)
    soup = _get(session, url, timeout=20)
    if soup is None:
        return []

    entries = []
    for tag in soup.find_all("url"):
        loc = tag.find("loc")
        mod = tag.find("lastmod")
        if loc and mod:
            u = loc.get_text(strip=True)
            d = mod.get_text(strip=True)[:10]   # YYYY-MM-DD
            if u.endswith(".htm"):
                entries.append({"url": u, "date": d})
    return entries


# ------------------------------------------------------------------
# Article parser
# ------------------------------------------------------------------

def _parse_article(soup: BeautifulSoup, url: str, date: str) -> dict | None:
    # Title
    title = None
    for sel in ["h1.article__title", "h1.detail__title", "h1"]:
        el = soup.select_one(sel)
        if el:
            title = el.get_text(strip=True)
            break
    if not title:
        el = soup.select_one('meta[property="og:title"]')
        title = el.get("content", "").strip() if el else None
    if not title:
        return None

    # Description / sapo
    desc = ""
    for sel in [
        "h2.article__sapo", ".article__sapo",
        ".detail__summary", "h2.detail__sapo",
        'meta[name="description"]', 'meta[property="og:description"]',
    ]:
        el = soup.select_one(sel)
        if el:
            desc = el.get("content") or el.get_text(strip=True)
            desc = desc.strip()
            break

    # Category
    category = ""
    for sel in [
        'meta[property="article:section"]',
        ".breadcrumb a:last-child",
        ".article__category a",
        ".detail__category",
    ]:
        el = soup.select_one(sel)
        if el:
            category = el.get("content") or el.get_text(strip=True)
            category = category.strip()
            break

    # Content — thử selector cụ thể, fallback lấy <p> từ article/main
    content = ""
    body = None
    for sel in [".article__body", ".detail__body", ".cms-body"]:
        el = soup.select_one(sel)
        if el:
            body = el
            break
    if body:
        content = " ".join(
            p.get_text(strip=True)
            for p in body.find_all("p")
            if len(p.get_text(strip=True)) > 20
        )
    else:
        container = soup.select_one("article, main, .article, .detail")
        if container:
            content = " ".join(
                p.get_text(strip=True)
                for p in container.find_all("p")
                if len(p.get_text(strip=True)) > 20
            )

    return {
        "date":        date,
        "title":       title,
        "description": desc,
        "category":    category,
        "content":     content,
        "url":         url,
    }


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
# Month iterator
# ------------------------------------------------------------------

def _months(start_y, start_m, end_y, end_m):
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ------------------------------------------------------------------
# Main crawl
# ------------------------------------------------------------------

def crawl_vneconomy(resume: bool = True, demo: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)

    if demo:
        log.info("DEMO MODE: chỉ crawl 3 tháng (2015-01 → 2015-03).")

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

    ckpt    = _load_checkpoint()
    session = requests.Session()
    session.headers.update(HEADERS)

    buffer: list[dict] = []
    total_saved = 0

    end_y = 2015 if demo else END_YEAR
    end_m = 3    if demo else END_MONTH

    for year, month in _months(START_YEAR, START_MONTH, end_y, end_m):
        ckpt_key = f"{year}-{month:02d}"

        if ckpt.get(ckpt_key) == "done" and resume:
            log.info(f"  [SKIP] {ckpt_key} — đã hoàn thành.")
            continue

        log.info(f"\n  [{ckpt_key}] Đọc sitemap...")
        time.sleep(random.uniform(*DELAY_SITEMAP))

        entries = _read_sitemap(session, year, month)
        new_entries = [e for e in entries if e["url"] not in seen_urls]
        log.info(f"    Sitemap: {len(entries):,} URL | {len(new_entries):,} mới")

        if not new_entries:
            ckpt[ckpt_key] = "done"
            _save_checkpoint(ckpt)
            continue

        # Resume từ idx nếu bị crash giữa chừng
        start_idx = ckpt.get(f"{ckpt_key}:idx", 0)
        if start_idx:
            log.info(f"    Resume từ bài {start_idx}")
        new_entries = new_entries[start_idx:]

        for i, entry in enumerate(new_entries, start=start_idx):
            time.sleep(random.uniform(*DELAY_ART))
            soup = _get(session, entry["url"])
            if soup is None:
                continue

            row = _parse_article(soup, entry["url"], entry["date"])
            if row:
                buffer.append(row)
                seen_urls.add(entry["url"])

            if (i + 1) % SAVE_EVERY == 0 and buffer:
                _append_rows(buffer)
                total_saved += len(buffer)
                buffer = []
                ckpt[f"{ckpt_key}:idx"] = i + 1
                _save_checkpoint(ckpt)
                log.info(f"    [checkpoint] {i+1}/{len(entries)} bài — tổng đã lưu: {total_saved:,}")

        if buffer:
            _append_rows(buffer)
            total_saved += len(buffer)
            buffer = []

        ckpt[ckpt_key] = "done"
        if f"{ckpt_key}:idx" in ckpt:
            del ckpt[f"{ckpt_key}:idx"]
        _save_checkpoint(ckpt)
        log.info(f"    Xong {ckpt_key} — tổng đã lưu: {total_saved:,}")

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

    df = crawl_vneconomy(resume=resume, demo=demo)

    if not df.empty:
        print(f"\n--- Mẫu 5 bài ---")
        print(df[["date", "category", "title"]].head().to_string(index=False))

        print(f"\n--- Số bài theo category ---")
        print(df["category"].value_counts().head(10).to_string())

        print(f"\n--- Số bài theo năm ---")
        print(df["date"].dt.year.value_counts().sort_index().to_string())
