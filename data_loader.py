"""
Data Loader - Thị trường chứng khoán Việt Nam + Macro toàn cầu
Nguồn 1: vnstock (KBS)  — VN-Index + blue-chip VN
Nguồn 2: yfinance       — S&P500, VIX, Giá dầu Brent, Gold, DXY

Hỗ trợ 2 tần suất:
  freq="daily"   → dữ liệu ngày giao dịch (lưu Data/dataset_daily.csv)
  freq="monthly" → dữ liệu tháng (lưu Data/dataset.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from vnstock import Vnstock
import yfinance as yf

# ------------------------------------------------------------------
# Cấu hình
# ------------------------------------------------------------------
START_DATE = "2015-01-01"   # KBS chỉ có từ ~2015 cho hầu hết mã
END_DATE   = "2026-04-01"
DATA_DIR   = Path(__file__).parent / "Data"

TARGET_SYMBOL = "VNINDEX"

VN_SYMBOLS = {
    "VCB": "Vietcombank",       # IPO 2009
    "BID": "BIDV",              # IPO 2013
    "VIC": "Vingroup",          # IPO 2007
    "HPG": "Hoa Phat Group",    # IPO 2007
    "MSN": "Masan Group",       # IPO 2010
    "MWG": "Mobile World",      # IPO 2014
    "GAS": "PetroVietnam Gas",  # IPO 2012
    # TCB (Techcombank) IPO 2018 — bỏ để không làm cắt ngắn dữ liệu
    # VHM (Vinhomes)    IPO 2018 — bỏ cùng lý do
}

MACRO_SYMBOLS = {
    "SP500": "^GSPC",    # S&P 500 — thị trường Mỹ dẫn dắt VN
    "VIX":   "^VIX",     # CBOE VIX — chỉ số sợ hãi toàn cầu
    "OIL":   "BZ=F",     # Dầu Brent — ảnh hưởng GAS, HPG
    "GOLD":  "GC=F",     # Giá vàng — tài sản trú ẩn an toàn
    "DXY":   "DX-Y.NYB", # US Dollar Index — áp lực tỷ giá toàn cầu
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_log_return(close: pd.Series, name: str) -> pd.Series:
    """Tính log return hàng ngày, chuẩn hoá index về date."""
    close.index = pd.to_datetime(close.index).normalize()
    ret = np.log(close / close.shift(1)).dropna()
    ret.name = name
    return ret


def _to_monthly_return(close: pd.Series, name: str) -> pd.Series:
    """Chuẩn hoá index về month-end và tính log return."""
    close.index = pd.to_datetime(close.index).to_period("M").to_timestamp("M")
    ret = np.log(close / close.shift(1)).dropna()
    ret.name = name
    return ret


# ------------------------------------------------------------------
# Tải dữ liệu VN (vnstock)
# ------------------------------------------------------------------

def _download_vn_symbol(symbol: str, freq: str = "monthly") -> tuple[pd.Series | None, pd.Series | None]:
    """
    Trả về (log_return_series, close_price_series).
    freq="daily"   → interval 1D
    freq="monthly" → interval 1M
    """
    interval = "1D" if freq == "daily" else "1M"
    try:
        stock = Vnstock().stock(symbol=symbol, source="KBS")
        df = stock.quote.history(start=START_DATE, end=END_DATE, interval=interval)
        if df is None or df.empty:
            print(f"  [WARNING] Không có dữ liệu: {symbol}")
            return None, None
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df.set_index("time").sort_index()
        close = df["close"].astype(float)
        close.name = symbol
        if freq == "daily":
            return _to_log_return(close.copy(), symbol), close
        else:
            close_monthly = close.copy()
            close_monthly.index = pd.to_datetime(close_monthly.index).to_period("M").to_timestamp("M")
            return _to_monthly_return(close.copy(), symbol), close_monthly
    except Exception as e:
        print(f"  [WARNING] Lỗi khi tải {symbol}: {e}")
        return None, None


def get_latest_prices(freq: str = "daily") -> pd.Series:
    """
    Trả về giá đóng cửa mới nhất của VNINDEX + các blue-chip.
    Dùng để tính giá dự báo từ predicted return.
    """
    suffix = "_daily" if freq == "daily" else ""
    price_path = DATA_DIR / f"prices{suffix}.csv"
    if price_path.exists():
        df = pd.read_csv(price_path, index_col=0, parse_dates=True)
        return df.iloc[-1]

    # Nếu chưa có file, tải lại
    load_vn_data(freq)
    if price_path.exists():
        df = pd.read_csv(price_path, index_col=0, parse_dates=True)
        return df.iloc[-1]
    return pd.Series(dtype=float)


def load_vn_data(freq: str = "monthly") -> pd.DataFrame:
    """Tải VNINDEX + blue-chip VN. Lưu thêm prices[_daily].csv (giá đóng cửa thực)."""
    label = "daily" if freq == "daily" else "monthly"
    print(f"\n[1/2] Tải dữ liệu VN ({label}, vnstock)...")

    series_list = []
    price_list  = []

    print(f"  → VNINDEX (target)")
    s, p = _download_vn_symbol(TARGET_SYMBOL, freq)
    if s is None:
        raise RuntimeError("Không tải được VN-Index.")
    s.name = "VNINDEX_Return"
    series_list.append(s)
    if p is not None:
        p.name = "VNINDEX"
        price_list.append(p)

    for symbol, name in VN_SYMBOLS.items():
        print(f"  → {symbol} ({name})")
        s, p = _download_vn_symbol(symbol, freq)
        if s is not None:
            series_list.append(s)
        if p is not None:
            price_list.append(p)

    df = pd.concat(series_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.loc[START_DATE:END_DATE]

    # Bỏ cột thiếu > 30% dữ liệu
    thresh = int(len(df) * 0.7)
    df = df.dropna(axis=1, thresh=thresh)
    # Bỏ hàng chưa đủ 80% cột
    df = df.dropna(thresh=int(len(df.columns) * 0.8))

    suffix = "_daily" if freq == "daily" else ""
    path = DATA_DIR / f"vn_stocks{suffix}.csv"
    df.to_csv(path)
    print(f"  Đã lưu: {path}  {df.shape}")

    # Lưu giá đóng cửa thực (để tính giá dự báo từ predicted return)
    if price_list:
        df_prices = pd.concat(price_list, axis=1)
        df_prices.index = pd.to_datetime(df_prices.index)
        df_prices = df_prices.loc[START_DATE:END_DATE].ffill(limit=3)
        price_path = DATA_DIR / f"prices{suffix}.csv"
        df_prices.to_csv(price_path)
        print(f"  Đã lưu giá: {price_path}  {df_prices.shape}")

    return df


# ------------------------------------------------------------------
# Tải dữ liệu Macro (yfinance)
# ------------------------------------------------------------------

def load_macro_data(freq: str = "monthly") -> pd.DataFrame:
    """Tải S&P500, VIX, Giá dầu, Gold, DXY."""
    label = "daily" if freq == "daily" else "monthly"
    print(f"\n[2/2] Tải dữ liệu Macro ({label}, yfinance)...")

    yf_interval = "1d" if freq == "daily" else "1mo"

    series_list = []
    for col_name, ticker in MACRO_SYMBOLS.items():
        print(f"  → {col_name} ({ticker})")
        try:
            raw = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval=yf_interval,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print(f"    [WARNING] Không có dữ liệu: {ticker}")
                continue

            close = raw["Close"].squeeze().dropna()
            if freq == "daily":
                s = _to_log_return(close, col_name)
            else:
                s = _to_monthly_return(close, col_name)
            series_list.append(s)
        except Exception as e:
            print(f"    [WARNING] Lỗi khi tải {ticker}: {e}")

    if not series_list:
        print("  [WARNING] Không tải được macro data nào.")
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.loc[START_DATE:END_DATE]

    if freq == "daily":
        # Forward-fill tối đa 3 ngày (ngày nghỉ lễ VN/US không trùng nhau)
        df = df.ffill(limit=3).dropna()
    else:
        df = df.ffill(limit=1).dropna()

    suffix = "_daily" if freq == "daily" else ""
    path = DATA_DIR / f"macro{suffix}.csv"
    df.to_csv(path)
    print(f"  Đã lưu: {path}  {df.shape}")
    return df


# ------------------------------------------------------------------
# Ghép dataset tổng hợp
# ------------------------------------------------------------------

def load_news_features(freq: str = "daily") -> pd.DataFrame:
    """
    Đọc Data/news_features_daily.csv (output của news_processor.py).
    Nếu freq="monthly" thì resample về tháng.
    Trả về DataFrame rỗng nếu file chưa tồn tại (news optional).
    """
    news_path = DATA_DIR / "news_features_daily.csv"
    if not news_path.exists():
        print("  [INFO] Chưa có news_features_daily.csv — bỏ qua news features.")
        return pd.DataFrame()

    df = pd.read_csv(news_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()

    if freq == "monthly":
        # Resample về tháng (mean), align index về month-end
        df = df.resample("ME").mean()

    print(f"  [News] {df.shape}  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_dataset(use_cache: bool = True, freq: str = "monthly",
                 with_news: bool = True) -> pd.DataFrame:
    """
    Ghép VN stocks + macro (+ news features tuỳ chọn) thành dataset cuối cùng.

    freq="monthly" → Data/dataset.csv
    freq="daily"   → Data/dataset_daily.csv
    with_news      → tích hợp news_features nếu file tồn tại
    """
    suffix = "_daily" if freq == "daily" else ""
    cache_path = DATA_DIR / f"dataset{suffix}.csv"

    if use_cache and cache_path.exists():
        print(f"Đọc từ cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"  Shape: {df.shape}")
        print(f"  Thời gian: {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Columns: {list(df.columns)}")
        return df

    # Tải mới
    df_vn    = load_vn_data(freq)
    df_macro = load_macro_data(freq)

    if df_macro.empty:
        df = df_vn.copy()
    else:
        df = df_vn.join(df_macro, how="inner")
        macro_cols = df_macro.columns.tolist()
        df[macro_cols] = df[macro_cols].ffill(limit=2)

    # Tích hợp news features (optional)
    if with_news:
        df_news = load_news_features(freq)
        if not df_news.empty:
            # Left join: giữ tất cả ngày giao dịch, news forward-fill tối đa 5 ngày
            df = df.join(df_news, how="left")
            news_cols = df_news.columns.tolist()
            df[news_cols] = df[news_cols].ffill(limit=5)
            n_filled = df[news_cols[0]].notna().sum()
            print(f"  [News] Tích hợp {len(news_cols)} news features "
                  f"({n_filled}/{len(df)} hàng có dữ liệu)")

    # Loại cột thiếu > 30% dữ liệu
    thresh = int(len(df) * 0.7)
    df = df.dropna(axis=1, thresh=thresh)

    # Loại hàng còn NaN
    df = df.dropna()

    # Volatility features
    if "VNINDEX_Return" in df.columns:
        df["VNINDEX_Vol5"]  = df["VNINDEX_Return"].rolling(5).std()
        df["VNINDEX_Vol20"] = df["VNINDEX_Return"].rolling(20).std()
        df = df.dropna()

    # VNINDEX_Return luôn là cột cuối (target)
    cols = [c for c in df.columns if c != "VNINDEX_Return"] + ["VNINDEX_Return"]
    df = df[cols]

    df.to_csv(cache_path)

    print(f"\nDataset tổng hợp ({freq}):")
    print(f"  Shape   : {df.shape}")
    print(f"  Thời gian: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Đã lưu  : {cache_path}")
    return df


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    freq = sys.argv[1] if len(sys.argv) > 1 else "monthly"
    print(f"=== Test Data Loader ({freq}) ===\n")
    df = load_dataset(use_cache=False, freq=freq)
    print("\nMẫu 5 dòng đầu:")
    print(df.head())
    print("\nThống kê mô tả:")
    print(df.describe().round(4))
