"""
Data Loader - Thị trường chứng khoán Việt Nam
Nguồn: vnstock (HOSE/HNX)

Target  : VN-Index monthly log return
Features: 10 cổ phiếu blue-chip đại diện các ngành + HNX-Index
"""

import pandas as pd
import numpy as np
from vnstock import Vnstock

START_DATE = "2012-01-01"   # vnstock thường có data từ 2012
END_DATE   = "2026-04-01"

# -------------------------------------------------------------------
# Danh sách symbols: VN-Index + blue-chip đại diện từng ngành
# -------------------------------------------------------------------
TARGET_SYMBOL = "VNINDEX"

FEATURE_SYMBOLS = {
    # Ngân hàng
    "VCB":  "Vietcombank",
    "BID":  "BIDV",
    "TCB":  "Techcombank",
    # Bất động sản
    "VIC":  "Vingroup",
    "VHM":  "Vinhomes",
    # Thép / Công nghiệp
    "HPG":  "Hoa Phat Group",
    # Hàng tiêu dùng
    "MSN":  "Masan Group",
    "MWG":  "Mobile World",
    # Dầu khí
    "GAS":  "PetroVietnam Gas",
    # Index phụ
    "HNX":  "HNX-Index",
}


def _download_symbol(symbol: str, is_index: bool = False) -> pd.Series:
    """
    Tải lịch sử giá monthly của một symbol, trả về log return.
    """
    try:
        stock = Vnstock().stock(symbol=symbol, source="KBS")
        df = stock.quote.history(
            start=START_DATE,
            end=END_DATE,
            interval="1M",
        )

        if df is None or df.empty:
            print(f"  [WARNING] Không có dữ liệu: {symbol}")
            return None

        # KBS trả về cột 'time' thay vì dùng index
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df.set_index("time").sort_index()

        close = df["close"].astype(float)
        close.index = close.index.to_period("M").to_timestamp("M")

        # Log return
        ret = np.log(close / close.shift(1)).dropna()
        ret.name = symbol
        return ret

    except Exception as e:
        print(f"  [WARNING] Lỗi khi tải {symbol}: {e}")
        return None


def load_vnindex_return() -> pd.Series:
    """Tải VN-Index monthly log return (target)."""
    print(f"  Tải {TARGET_SYMBOL} (target)...")
    s = _download_symbol(TARGET_SYMBOL, is_index=True)
    if s is not None:
        s.name = "VNINDEX_Return"
    return s


def load_feature_data() -> pd.DataFrame:
    """Tải log return của các cổ phiếu/index feature."""
    print("  Tải feature symbols...")
    series_list = []

    for symbol, name in FEATURE_SYMBOLS.items():
        print(f"    → {symbol} ({name})")
        s = _download_symbol(symbol, is_index=(symbol == "HNX"))
        if s is not None:
            series_list.append(s)

    if not series_list:
        raise RuntimeError("Không tải được dữ liệu nào từ vnstock.")

    return pd.concat(series_list, axis=1)


def load_dataset() -> pd.DataFrame:
    """
    Ghép VN-Index return + feature returns thành DataFrame.
    Index : monthly end-of-month (normalize về ngày cuối tháng)
    Target: 'VNINDEX_Return'
    """
    target = load_vnindex_return()
    if target is None:
        raise RuntimeError("Không tải được VN-Index.")

    features = load_feature_data()

    # Chuẩn hoá index về Month-End để đảm bảo align đúng
    target.index  = target.index.to_period("M").to_timestamp("M")
    features.index = features.index.to_period("M").to_timestamp("M")

    # Dùng reindex để align features theo index của target
    features = features.reindex(target.index)

    df = pd.concat([features, target], axis=1)
    df = df.loc[START_DATE:END_DATE]

    # Loại cột thiếu quá 30% dữ liệu
    thresh = int(len(df) * 0.7)
    df = df.dropna(axis=1, thresh=thresh)

    # Loại hàng còn NaN
    df = df.dropna()

    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Thời gian: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    print("=== Test Data Loader (Vietnam Stock Market) ===\n")
    df = load_dataset()
    print("\nMẫu 5 dòng đầu:")
    print(df.head())
    print("\nThống kê mô tả:")
    print(df.describe().round(4))
