"""
news_crawler.py — VnExpress News Crawler
Thu thập tin tức tài chính/kinh tế từ VnExpress (2015 → nay)

Chiến lược:
  - Phân trang category dùng URL /p{N} (đúng format)
  - Lấy ngày chính xác từ meta tag article:published_time
  - Delay 2-3s giữa request để tránh throttle
  - Auto-resume: skip URL đã crawl, tiếp tục từ page chưa xong
  - Lưu checkpoint mỗi 10 trang để không mất data khi crash

Cách chạy:
  python news_crawler.py            # full crawl (tiếp tục nếu đã có data)
  python news_crawler.py --fresh    # crawl lại từ đầu
  python news_crawler.py --demo     # chỉ 5 trang/category để test
"""

import re
import sys
import json
import time
import random
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent / "Data"
OUTPUT_FILE = DATA_DIR / "vnexpress_raw.csv"
CKPT_FILE   = DATA_DIR / ".crawler_checkpoint.json"  # {slug: last_page_done}

# URL format đúng: /p{N} cho trang N>1
#
# Lưu ý category overlap:
#   "kinh-doanh" (parent) chứa ALL bài của "kinh-doanh/tai-chinh"
#   → KHÔNG dùng tai-chinh để tránh 0 bài mới.
#   "chung-khoan", "vi-mo", "quoc-te" có bài RIÊNG không nằm trên parent.
CATEGORIES = {
    "kinh-doanh":              "Kinh doanh",        # parent — bao gồm tài chính
    "kinh-doanh/chung-khoan": "Chứng khoán",        # có bài riêng
    "kinh-doanh/vi-mo":       "Vĩ mô",              # có bài riêng
    "kinh-doanh/quoc-te":     "Kinh tế quốc tế",   # có bài riêng
}

START_DATE    = date(2015, 1, 1)
MAX_PAGES     = 500      # giới hạn an toàn (VnExpress sub-cat thường ~20-30 trang)
SAVE_EVERY    = 10       # lưu mỗi N trang
EMPTY_STOP    = 5        # dừng nếu N trang liên tiếp đều 0 bài mới

DELAY_PAGE  = (2.0, 3.5)
DELAY_ART   = (0.8, 1.5)
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://vnexpress.net/",
}

_DATE_IMG = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def _get(session: requests.Session, url: str,
         timeout: int = 15) -> BeautifulSoup | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                log.warning(f"    HTTP {r.status_code}: {url}")
                time.sleep(2 ** attempt)
                continue
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"    Lỗi (lần {attempt}): {e} — thử lại sau {wait}s")
            time.sleep(wait)
    return None


def _category_url(slug: str, page: int) -> str:
    """URL đúng format của VnExpress cho category pagination."""
    if page == 1:
        return f"https://vnexpress.net/{slug}"
    return f"https://vnexpress.net/{slug}/p{page}"     # /p{N} không phải -p{N}.html


# ------------------------------------------------------------------
# Date extraction
# ------------------------------------------------------------------

def _date_from_img(tag) -> date | None:
    """Lấy ngày từ URL ảnh thumbnail CDN."""
    if tag is None:
        return None
    for attr in ("src", "srcset", "data-src"):
        val = tag.get(attr, "") or ""
        if isinstance(val, list):
            val = " ".join(val)
        m = _DATE_IMG.search(val)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
    return None


def _date_from_article_page(url: str,
                             session: requests.Session) -> date | None:
    """Fetch bài báo để lấy ngày chính xác từ meta tag."""
    time.sleep(random.uniform(*DELAY_ART))
    soup = _get(session, url)
    if soup is None:
        return None
    for sel in [
        'meta[name="pubdate"]',
        'meta[itemprop="datePublished"]',
        'meta[property="article:published_time"]',
    ]:
        el = soup.select_one(sel)
        if el and el.get("content"):
            try:
                return date.fromisoformat(el["content"][:10])
            except ValueError:
                pass
    return None


# ------------------------------------------------------------------
# Parse category listing page
# ------------------------------------------------------------------

def _parse_listing(soup: BeautifulSoup, label: str) -> list[dict]:
    """Trích xuất articles từ một trang category listing."""
    # Bỏ qua placeholder (lazy-load skeleton)
    items = [
        a for a in soup.select("article.item-news")
        if "box-placeholder" not in (a.get("class") or [])
    ]

    rows = []
    for item in items:
        # Title + URL
        title_el = item.select_one(
            "h1.title-news a, h2.title-news a, h3.title-news a, h4.title-news a"
        )
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        href  = title_el.get("href", "")
        if not href:
            continue
        url = href if href.startswith("http") else f"https://vnexpress.net{href}"

        # Bỏ link không phải bài báo
        if any(x in url for x in ["/video/", "/anh/", "/tag/", "/chu-de/"]):
            continue

        # Description
        desc_el = item.select_one("p.description, .description")
        desc = desc_el.get_text(strip=True) if desc_el else ""

        # Date từ ảnh thumbnail (nhanh, không cần fetch thêm)
        img = item.find("img") or item.find("source")
        art_date = _date_from_img(img)

        rows.append({
            "date":        art_date.isoformat() if art_date else None,
            "title":       title,
            "description": desc,
            "url":         url,
            "category":    label,
        })
    return rows


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
# Crawl một category
# ------------------------------------------------------------------

def _crawl_category(slug: str, label: str,
                    session: requests.Session,
                    seen_urls: set,
                    start_page: int = 1,
                    max_pages: int = MAX_PAGES) -> None:
    """
    Crawl từ start_page đến hết (hoặc đến MAX_PAGES / gặp bài quá cũ).
    Lưu checkpoint + CSV theo từng SAVE_EVERY trang.
    """
    ckpt          = _load_checkpoint()
    buffer: list[dict] = []
    consecutive_empty = 0   # số trang liên tiếp không có bài mới

    for page in range(start_page, max_pages + 1):
        url = _category_url(slug, page)
        log.info(f"  [{label}] trang {page:4d}  {url}")

        soup = _get(session, url)
        if soup is None:
            log.info(f"  → 404 / lỗi tại trang {page}, dừng category.")
            break

        rows      = _parse_listing(soup, label)
        real_rows = [r for r in rows if r["url"] not in seen_urls]

        # Trang không có article nào (placeholder / hết data)
        if not rows:
            log.info(f"  → Không có bài nào (placeholder hoặc hết trang), dừng.")
            break

        stop     = False
        new_rows = []
        for r in real_rows:
            # Chưa có ngày → fetch article page lấy meta tag (chậm hơn)
            if r["date"] is None:
                art_date = _date_from_article_page(r["url"], session)
                if art_date:
                    r["date"] = art_date.isoformat()

            if r["date"]:
                try:
                    art_date = date.fromisoformat(r["date"])
                    if art_date < START_DATE:
                        stop = True
                        continue
                except ValueError:
                    pass

            new_rows.append(r)
            seen_urls.add(r["url"])

        buffer.extend(new_rows)
        log.info(f"    +{len(new_rows):3d} bài mới  (buffer: {len(buffer)})")

        # Early-stop: tất cả bài trên trang đều đã seen (overlap / hết lịch sử)
        if len(new_rows) == 0:
            consecutive_empty += 1
            if consecutive_empty >= EMPTY_STOP:
                log.info(
                    f"  → {EMPTY_STOP} trang liên tiếp không có bài mới "
                    f"(category đã crawl hết hoặc overlap với category khác). Dừng."
                )
                break
        else:
            consecutive_empty = 0

        # Lưu định kỳ
        if page % SAVE_EVERY == 0 and buffer:
            _append_rows(buffer)
            buffer = []
            ckpt[slug] = page
            _save_checkpoint(ckpt)
            log.info(f"    [checkpoint] lưu tại trang {page}")

        if stop:
            log.info(f"  → Gặp bài trước {START_DATE}, dừng.")
            break

        time.sleep(random.uniform(*DELAY_PAGE))

    # Lưu phần còn lại trong buffer
    if buffer:
        _append_rows(buffer)
    ckpt[slug] = "done"
    _save_checkpoint(ckpt)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def crawl_vnexpress(resume: bool = True, demo: bool = False) -> pd.DataFrame:
    """
    Crawl VnExpress tài chính 2015 → nay.

    resume=True : tiếp tục từ checkpoint (bỏ qua URL đã có)
    resume=False: xóa output + checkpoint, crawl lại
    demo=True   : chỉ 5 trang/category
    """
    DATA_DIR.mkdir(exist_ok=True)
    max_pages = 5 if demo else MAX_PAGES

    if demo:
        log.info("DEMO MODE: 5 trang/category, chỉ lấy 3 tháng gần nhất.")
        global START_DATE
        START_DATE = date.today() - timedelta(days=90)

    # Load seen URLs từ file cũ
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
        if ckpt.get(slug) == "done" and resume:
            log.info(f"\n[SKIP] {label} — đã hoàn thành trước đó.")
            continue

        start_page = 1
        if resume and isinstance(ckpt.get(slug), int):
            start_page = ckpt[slug] + 1
            log.info(f"\n[RESUME] {label} từ trang {start_page}")
        else:
            log.info(f"\n{'='*55}")
            log.info(f"  Category: {label}  ({slug})")
            log.info(f"{'='*55}")

        _crawl_category(slug, label, session, seen_urls,
                        start_page=start_page, max_pages=max_pages)

        time.sleep(random.uniform(3.0, 5.0))

    # Đọc lại, dedup, sort, lưu cuối
    if not OUTPUT_FILE.exists():
        log.warning("Không crawl được bài nào.")
        return pd.DataFrame()

    df = pd.read_csv(OUTPUT_FILE)
    df = df.drop_duplicates("url")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    log.info(f"\n{'='*55}")
    log.info(f"HOÀN THÀNH!")
    log.info(f"  Tổng bài  : {len(df):,}")
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

    df = crawl_vnexpress(resume=resume, demo=demo)

    if not df.empty:
        print(f"\n--- Mẫu 5 bài ---")
        print(df[["date", "category", "title"]].head().to_string(index=False))

        print(f"\n--- Số bài theo category ---")
        print(df["category"].value_counts().to_string())

        print(f"\n--- Số bài theo năm ---")
        print(df["date"].dt.year.value_counts().sort_index().to_string())
