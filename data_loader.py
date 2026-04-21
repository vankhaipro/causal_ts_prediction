"""
Data Loader - Thị trường chứng khoán Việt Nam + Macro toàn cầu
Nguồn 1: vnstock (KBS)  — VN-Index + 9 blue-chip VN
Nguồn 2: yfinance       — S&P500, VIX, Giá dầu Brent, Tỷ giá USD/VND

Dữ liệu được lưu vào thư mục Data/ dưới dạng CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from vnstock import Vnstock
import yfinance as yf

# ------------------------------------------------------------------
# Cấu hình
# ------------------------------------------------------------------
START_DATE = "2012-01-01"
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

# Yahoo Finance tickers cho macro features
# USDVND bỏ ra vì ticker VNDUSD=X trên Yahoo không đáng tin (dữ liệu nhiễu)
# Thay bằng USDJPY và Gold làm proxy rủi ro toàn cầu
MACRO_SYMBOLS = {
    "SP500": "^GSPC",   # S&P 500 — thị trường Mỹ dẫn dắt VN
    "VIX":   "^VIX",    # CBOE VIX — chỉ số sợ hãi toàn cầu
    "OIL":   "BZ=F",    # Dầu Brent — ảnh hưởng GAS, HPG
    "GOLD":  "GC=F",    # Giá vàng — tài sản trú ẩn an toàn
    "DXY":   "DX-Y.NYB",# US Dollar Index — áp lực tỷ giá toàn cầu
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_monthly_return(close: pd.Series, name: str) -> pd.Series:
    """Chuẩn hoá index về month-end và tính log return."""
    close.index = pd.to_datetime(close.index).to_period("M").to_timestamp("M")
    ret = np.log(close / close.shift(1)).dropna()
    ret.name = name
    return ret


# ------------------------------------------------------------------
# Tải dữ liệu VN (vnstock)
# ------------------------------------------------------------------

def _download_vn_symbol(symbol: str) -> pd.Series | None:
    try:
        stock = Vnstock().stock(symbol=symbol, source="KBS")
        df = stock.quote.history(start=START_DATE, end=END_DATE, interval="1M")
        if df is None or df.empty:
            print(f"  [WARNING] Không có dữ liệu: {symbol}")
            return None
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df.set_index("time").sort_index()
        close = df["close"].astype(float)
        return _to_monthly_return(close, symbol)
    except Exception as e:
        print(f"  [WARNING] Lỗi khi tải {symbol}: {e}")
        return None


def load_vn_data() -> pd.DataFrame:
    """Tải VNINDEX + 9 blue-chip, lưu Data/vn_stocks.csv."""
    print("\n[1/2] Tải dữ liệu VN (vnstock)...")

    series_list = []

    # Target: VNINDEX
    print(f"  → VNINDEX (target)")
    s = _download_vn_symbol(TARGET_SYMBOL)
    if s is None:
        raise RuntimeError("Không tải được VN-Index.")
    s.name = "VNINDEX_Return"
    series_list.append(s)

    # Features: blue-chip
    for symbol, name in VN_SYMBOLS.items():
        print(f"  → {symbol} ({name})")
        s = _download_vn_symbol(symbol)
        if s is not None:
            series_list.append(s)

    df = pd.concat(series_list, axis=1)
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.loc[START_DATE:END_DATE]

    # Bỏ cột thiếu > 30% dữ liệu (xử lý cổ phiếu IPO muộn còn sót)
    thresh = int(len(df) * 0.7)
    df = df.dropna(axis=1, thresh=thresh)
    # Bỏ hàng đầu chuỗi chưa có đủ dữ liệu (yêu cầu >= 80% cột)
    df = df.dropna(thresh=int(len(df.columns) * 0.8))

    path = DATA_DIR / "vn_stocks.csv"
    df.to_csv(path)
    print(f"  Đã lưu: {path}  {df.shape}")
    return df


# ------------------------------------------------------------------
# Tải dữ liệu Macro (yfinance)
# ------------------------------------------------------------------

def load_macro_data() -> pd.DataFrame:
    """Tải S&P500, VIX, Giá dầu, Tỷ giá USD/VND, DXY, lưu Data/macro.csv."""
    print("\n[2/2] Tải dữ liệu Macro (yfinance)...")

    series_list = []
    for col_name, ticker in MACRO_SYMBOLS.items():
        print(f"  → {col_name} ({ticker})")
        try:
            raw = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1mo",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print(f"    [WARNING] Không có dữ liệu: {ticker}")
                continue

            close = raw["Close"].squeeze().dropna()
            s = _to_monthly_return(close, col_name)
            series_list.append(s)
        except Exception as e:
            print(f"    [WARNING] Lỗi khi tải {ticker}: {e}")

    if not series_list:
        print("  [WARNING] Không tải được macro data nào.")
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.loc[START_DATE:END_DATE]

    # Forward-fill tối đa 1 tháng để lấp gaps nhỏ (macro thay đổi chậm)
    # Sau đó bỏ hàng vẫn còn NaN (đầu chuỗi chưa có data)
    df = df.ffill(limit=1).dropna()

    path = DATA_DIR / "macro.csv"
    df.to_csv(path)
    print(f"  Đã lưu: {path}  {df.shape}")
    return df


# ------------------------------------------------------------------
# Ghép dataset tổng hợp
# ------------------------------------------------------------------

def load_dataset(use_cache: bool = True) -> pd.DataFrame:
    """
    Ghép VN stocks + macro thành dataset cuối cùng.

    use_cache=True : đọc từ Data/dataset.csv nếu đã tồn tại (nhanh hơn).
    use_cache=False: tải lại từ đầu và ghi đè file cache.
    """
    cache_path = DATA_DIR / "dataset.csv"

    if use_cache and cache_path.exists():
        print(f"Đọc từ cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"  Shape: {df.shape}")
        print(f"  Thời gian: {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Columns: {list(df.columns)}")
        return df

    # Tải mới
    df_vn    = load_vn_data()
    df_macro = load_macro_data()

    # Ghép: giữ tất cả tháng VN làm gốc, macro ffill lấp chỗ trống
    if df_macro.empty:
        df = df_vn.copy()
    else:
        df = df_vn.join(df_macro, how="left")
        # Macro thay đổi chậm — ffill tối đa 2 tháng cho các gaps nhỏ
        macro_cols = df_macro.columns.tolist()
        df[macro_cols] = df[macro_cols].ffill(limit=2)

    # Loại cột thiếu > 30% dữ liệu
    thresh = int(len(df) * 0.7)
    df = df.dropna(axis=1, thresh=thresh)

    # Loại hàng còn NaN
    df = df.dropna()

    # Đảm bảo VNINDEX_Return là cột cuối (target)
    cols = [c for c in df.columns if c != "VNINDEX_Return"] + ["VNINDEX_Return"]
    df = df[cols]

    # Lưu cache
    df.to_csv(cache_path)

    print(f"\nDataset tổng hợp:")
    print(f"  Shape   : {df.shape}")
    print(f"  Thời gian: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Đã lưu  : {cache_path}")
    return df


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test Data Loader ===\n")
    df = load_dataset(use_cache=False)
    print("\nMẫu 5 dòng đầu:")
    print(df.head())
    print("\nThống kê mô tả:")
    print(df.describe().round(4))
