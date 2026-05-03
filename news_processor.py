"""
news_processor.py — Phase 2: NLP Pipeline
Sentiment Analysis + LDA Topic Modeling trên VnExpress articles

Input : Data/vnexpress_raw.csv      (từ news_crawler.py)
Output: Data/news_features_daily.csv

Cách chạy:
  python news_processor.py                    # full pipeline (CPU)
  python news_processor.py --gpu              # dùng GPU nếu có
  python news_processor.py --topics-only      # chỉ LDA (bỏ qua sentiment)
  python news_processor.py --topics=15        # tuỳ chỉnh số topics (mặc định 10)
  python news_processor.py --retrain-lda      # train LDA lại từ đầu
"""

import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR        = Path(__file__).parent / "Data"
RAW_FILE        = DATA_DIR / "vnexpress_raw.csv"
SENT_CACHE      = DATA_DIR / ".sentiment_cache.csv"
LDA_MODEL_DIR   = DATA_DIR / "lda_model"
OUTPUT_FILE     = DATA_DIR / "news_features_daily.csv"

SENTIMENT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"
N_TOPICS        = 10
BATCH_SIZE      = 32
MAX_SEQ_LEN     = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Vietnamese stopwords (inline, không cần file ngoài)
# ------------------------------------------------------------------
VIET_STOPWORDS = set("""
và của để trong từ với tại về cho đến khi được là có không phải
các một những này đây đó bị làm theo như thì đã sẽ còn rồi
nhưng nên vì mà hay hoặc nếu cũng vẫn chỉ rất quá nhiều ít
tôi bạn anh chị ông bà họ chúng ta mình ai đâu gì sao thế nào
năm tháng ngày giờ tuần quý kỳ hôm qua nay mai đây kia ở trên dưới
được đã sẽ bằng qua lại thêm cần phải đang còn mới cũng vẫn
""".split())

# ------------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------------
_HTML_RE  = re.compile(r"<[^>]+>")
_URL_RE   = re.compile(r"https?://\S+")
_NUM_RE   = re.compile(r"\b\d+([.,]\d+)*\b")
_PUNC_RE  = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNC_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def tokenize_vi(text: str) -> list[str]:
    """
    Tokenize tiếng Việt.
    Ưu tiên underthesea nếu cài sẵn, fallback sang split đơn giản.
    """
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text, format="text").split()
    except Exception:
        tokens = text.split()
    return [t for t in tokens if t not in VIET_STOPWORDS and len(t) > 1]


def build_text(row: pd.Series) -> str:
    """Ghép title + description làm input cho NLP."""
    title = str(row.get("title", "") or "")
    desc  = str(row.get("description", "") or "")
    return (title + ". " + desc).strip()


# ------------------------------------------------------------------
# Sentiment Analysis
# ------------------------------------------------------------------

# Chuẩn hoá label từ các model khác nhau → NEG / NEU / POS
_LABEL_MAP = {
    "negative": "NEG", "neg": "NEG", "label_0": "NEG", "0": "NEG",
    "neutral":  "NEU", "neu": "NEU", "label_1": "NEU", "1": "NEU",
    "positive": "POS", "pos": "POS", "label_2": "POS", "2": "POS",
}
_SCORE_MAP = {"NEG": -1.0, "NEU": 0.0, "POS": 1.0}


def _normalize_label(label: str) -> str:
    return _LABEL_MAP.get(label.lower(), "NEU")


def _batch_sentiment(texts: list[str], clf) -> tuple[list[float], list[str]]:
    """Chạy sentiment pipeline trên một batch, trả (scores, labels)."""
    scores, labels = [], []
    try:
        results = clf(texts)
        for res in results:
            # top_k=None → list of dicts; top_k=1 → list of dict
            if isinstance(res, list):
                score_map = {_normalize_label(r["label"]): r["score"] for r in res}
            else:
                # top_k=1 mode
                score_map = {_normalize_label(res["label"]): res["score"]}
                for lbl in ["NEG", "NEU", "POS"]:
                    score_map.setdefault(lbl, 0.0)

            weighted = sum(score_map.get(lbl, 0) * v for lbl, v in _SCORE_MAP.items())
            best = max(score_map, key=score_map.get)
            scores.append(round(weighted, 4))
            labels.append(best)
    except Exception as e:
        log.warning(f"  Batch lỗi: {e}")
        scores.extend([0.0] * len(texts))
        labels.extend(["NEU"] * len(texts))
    return scores, labels


def run_sentiment(df: pd.DataFrame, device: int = -1) -> pd.DataFrame:
    """
    Thêm cột sentiment_score ∈ [-1, 1] và sentiment_label (NEG/NEU/POS).

    Kết quả được cache vào SENT_CACHE để không chạy lại khi bị ngắt.
    """
    from transformers import pipeline as hf_pipeline

    # ---- Load cache ----
    cached_urls: set = set()
    cache_rows: list[dict] = []

    if SENT_CACHE.exists():
        df_cache = pd.read_csv(SENT_CACHE)
        cached_urls = set(df_cache["url"].dropna())
        cache_rows = df_cache.to_dict("records")
        log.info(f"  Cache: {len(cached_urls):,} bài đã xử lý")

    todo = df[~df["url"].isin(cached_urls)].copy()
    log.info(f"  Cần xử lý: {len(todo):,} bài")

    if len(todo) == 0:
        df_cache = pd.read_csv(SENT_CACHE)
        return df.merge(df_cache[["url", "sentiment_score", "sentiment_label"]],
                        on="url", how="left")

    # ---- Load model ----
    log.info(f"Loading model: {SENTIMENT_MODEL}  (device={'GPU' if device >= 0 else 'CPU'})")
    clf = hf_pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        device=device,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        top_k=None,
    )

    # ---- Inference ----
    texts = [build_text(row) for _, row in todo.iterrows()]
    urls  = todo["url"].tolist()

    all_scores, all_labels = [], []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Sentiment"):
        batch_texts = texts[i: i + BATCH_SIZE]
        s, l = _batch_sentiment(batch_texts, clf)
        all_scores.extend(s)
        all_labels.extend(l)

        # Lưu checkpoint mỗi 100 batch
        if (i // BATCH_SIZE) % 100 == 0 and i > 0:
            new_rows = [{"url": u, "sentiment_score": sc, "sentiment_label": lb}
                        for u, sc, lb in zip(urls[:len(all_scores)], all_scores, all_labels)]
            _save_cache(cache_rows + new_rows)

    # ---- Merge + save cache ----
    new_rows = [{"url": u, "sentiment_score": sc, "sentiment_label": lb}
                for u, sc, lb in zip(urls, all_scores, all_labels)]
    all_cache = cache_rows + new_rows
    _save_cache(all_cache)

    df_sent = pd.DataFrame(all_cache).drop_duplicates("url")
    return df.merge(df_sent[["url", "sentiment_score", "sentiment_label"]],
                    on="url", how="left")


def _save_cache(rows: list[dict]) -> None:
    pd.DataFrame(rows).drop_duplicates("url").to_csv(SENT_CACHE, index=False)


# ------------------------------------------------------------------
# LDA Topic Modeling
# ------------------------------------------------------------------

def run_lda(df: pd.DataFrame, n_topics: int = N_TOPICS,
            retrain: bool = False) -> pd.DataFrame:
    """
    Thêm các cột topic_0 … topic_{n-1} (trọng số mỗi topic).

    Nếu LDA model đã tồn tại và retrain=False thì load lại thay vì train.
    """
    from gensim import corpora
    from gensim.models import LdaMulticore

    model_path = LDA_MODEL_DIR / "lda.model"
    dict_path  = LDA_MODEL_DIR / "dictionary.dict"
    LDA_MODEL_DIR.mkdir(exist_ok=True)

    # ---- Tokenize ----
    log.info("Tokenizing for LDA...")
    raw_texts  = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    cleaned    = [clean_text(t) for t in raw_texts]
    tokenized  = [tokenize_vi(t) for t in tqdm(cleaned, desc="Tokenize")]

    # ---- Dictionary ----
    if model_path.exists() and not retrain:
        log.info(f"Loading existing LDA model: {model_path}")
        lda  = LdaMulticore.load(str(model_path))
        dct  = corpora.Dictionary.load(str(dict_path))
        n_topics = lda.num_topics
    else:
        log.info("Building dictionary...")
        dct = corpora.Dictionary(tokenized)
        dct.filter_extremes(no_below=5, no_above=0.8)
        dct.save(str(dict_path))
        log.info(f"  Vocab size: {len(dct):,} tokens")

        corpus = [dct.doc2bow(t) for t in tokenized]
        log.info(f"Training LDA: {n_topics} topics, {len(corpus):,} docs...")
        lda = LdaMulticore(
            corpus,
            num_topics=n_topics,
            id2word=dct,
            passes=15,
            workers=2,
            random_state=42,
            alpha="asymmetric",
        )
        lda.save(str(model_path))
        log.info(f"LDA model saved: {model_path}")

    # ---- Print topics ----
    log.info("Top words per topic:")
    for idx, topic in lda.print_topics(num_words=8):
        log.info(f"  Topic {idx:2d}: {topic}")

    # ---- Get distributions ----
    corpus = [dct.doc2bow(t) for t in tokenized]
    topic_data = {f"topic_{k}": [] for k in range(n_topics)}

    for bow in tqdm(corpus, desc="Topic dist"):
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        for k in range(n_topics):
            topic_data[f"topic_{k}"].append(round(dist.get(k, 0.0), 4))

    df = df.copy()
    for col, vals in topic_data.items():
        df[col] = vals

    return df, n_topics


# ------------------------------------------------------------------
# Daily Aggregation
# ------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame, n_topics: int) -> pd.DataFrame:
    """
    Gộp article-level features → daily time series.

    Output columns:
      sentiment_mean   : trung bình điểm sentiment [-1, 1]
      sentiment_std    : độ lệch chuẩn (cao = ngày có nhiều ý kiến trái chiều)
      sentiment_pos_pct: % bài báo tích cực
      sentiment_neg_pct: % bài báo tiêu cực
      article_count    : số bài trong ngày (chỉ số khối lượng tin)
      topic_0..N       : trung bình trọng số mỗi topic
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["date", "sentiment_score"])

    topic_cols = [f"topic_{k}" for k in range(n_topics) if f"topic_{k}" in df.columns]

    # Sentiment aggregation
    sent_agg = df.groupby("date").agg(
        sentiment_mean    = ("sentiment_score", "mean"),
        sentiment_std     = ("sentiment_score", "std"),
        article_count     = ("sentiment_score", "count"),
    )
    sent_agg["sentiment_pos_pct"] = (
        df.groupby("date")["sentiment_label"].apply(lambda x: (x == "POS").mean())
    )
    sent_agg["sentiment_neg_pct"] = (
        df.groupby("date")["sentiment_label"].apply(lambda x: (x == "NEG").mean())
    )

    # Topic aggregation
    topic_agg = df.groupby("date")[topic_cols].mean() if topic_cols else pd.DataFrame()

    daily = pd.concat([sent_agg, topic_agg], axis=1)

    # Forward-fill ngày nghỉ (max 3 ngày)
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily    = daily.reindex(full_idx).ffill(limit=3)
    daily.index.name = "date"

    return daily.round(4)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def process(use_gpu: bool = False,
            n_topics: int = N_TOPICS,
            skip_sentiment: bool = False,
            retrain_lda: bool = False) -> pd.DataFrame:
    """
    Chạy toàn bộ NLP pipeline.

    Parameters
    ----------
    use_gpu       : dùng GPU (device=0) nếu True
    n_topics      : số LDA topics
    skip_sentiment: bỏ qua sentiment, chỉ chạy LDA
    retrain_lda   : train LDA lại từ đầu dù model đã tồn tại
    """
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {RAW_FILE}.\n"
            "Hãy chạy `python news_crawler.py` trước."
        )

    # ---- Load raw articles ----
    log.info(f"Loading: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    log.info(f"  {len(df):,} articles  "
             f"({df['date'].min().date()} → {df['date'].max().date()})")

    # ---- Step 1: Sentiment ----
    if not skip_sentiment:
        log.info("\n--- Step 1: Sentiment Analysis ---")
        device = 0 if use_gpu else -1
        df = run_sentiment(df, device=device)
        log.info(f"  sentiment_score: mean={df['sentiment_score'].mean():.3f}  "
                 f"std={df['sentiment_score'].std():.3f}")
        dist = df["sentiment_label"].value_counts(normalize=True) * 100
        log.info(f"  Label distribution: {dist.to_dict()}")
    else:
        log.info("\n--- Step 1: Sentiment — BỎ QUA (--topics-only) ---")
        df["sentiment_score"] = np.nan
        df["sentiment_label"] = "NEU"

    # ---- Step 2: LDA ----
    log.info(f"\n--- Step 2: LDA Topic Modeling ({n_topics} topics) ---")
    df, n_topics = run_lda(df, n_topics=n_topics, retrain=retrain_lda)

    # ---- Step 3: Daily aggregation ----
    log.info("\n--- Step 3: Daily Aggregation ---")
    daily = aggregate_daily(df, n_topics=n_topics)

    daily.to_csv(OUTPUT_FILE)
    log.info(f"\n{'='*50}")
    log.info(f"HOÀN THÀNH!")
    log.info(f"  Output : {OUTPUT_FILE}")
    log.info(f"  Shape  : {daily.shape}")
    log.info(f"  Columns: {list(daily.columns)}")
    log.info(f"  Range  : {daily.index.min().date()} → {daily.index.max().date()}")
    log.info(f"{'='*50}")

    return daily


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    use_gpu      = "--gpu"          in args
    skip_sent    = "--topics-only"  in args
    retrain_lda  = "--retrain-lda"  in args
    n_topics     = int(next(
        (a.split("=")[1] for a in args if a.startswith("--topics=")),
        N_TOPICS
    ))

    daily = process(
        use_gpu=use_gpu,
        n_topics=n_topics,
        skip_sentiment=skip_sent,
        retrain_lda=retrain_lda,
    )

    print(f"\n--- Mẫu 5 ngày đầu ---")
    print(daily.head().to_string())

    print(f"\n--- Thống kê ---")
    print(daily.describe().round(3).to_string())
