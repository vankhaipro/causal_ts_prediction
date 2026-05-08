"""
Pipeline chính — Causal TS Prediction VN-Index

Cách chạy:
  python main.py              # dùng dữ liệu monthly (mặc định)
  python main.py daily        # dùng dữ liệu daily
  python main.py monthly      # dùng dữ liệu monthly (tường minh)
"""

import sys
import numpy as np
import pandas as pd
from causal_forecaster import CausalForecaster
from data_loader import load_dataset


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nhóm 1 — thêm autoregression + technical indicators từ data giá có sẵn.
    Tất cả tính với lag để tránh look-ahead bias.
    """
    df = df.copy()
    ret = df["VNINDEX_Return"]

    # 1. Autoregression lag 1-3 (momentum ngắn hạn)
    df["VNINDEX_lag1"] = ret.shift(1)
    df["VNINDEX_lag2"] = ret.shift(2)
    df["VNINDEX_lag3"] = ret.shift(3)

    # 2. Momentum 3 tháng (tổng return 3 tháng trước)
    df["VNINDEX_mom3"] = ret.shift(1).rolling(3).sum()

    # 3. Volatility 3 tháng (độ biến động gần đây)
    df["VNINDEX_vol3"] = ret.shift(1).rolling(3).std()

    # 4. RSI-style momentum: tỷ lệ tháng tăng / tổng tháng (6 tháng)
    up   = ret.shift(1).rolling(6).apply(lambda x: (x > 0).sum())
    df["VNINDEX_rsi6"] = up / 6.0

    # 5. Trend signal: MA3 so với MA6 (cross signal)
    ma3 = ret.shift(1).rolling(3).mean()
    ma6 = ret.shift(1).rolling(6).mean()
    df["VNINDEX_trend"] = (ma3 - ma6)

    return df


# ------------------------------------------------------------------
# Cấu hình theo tần suất
# ------------------------------------------------------------------
FREQ_CONFIG = {
    "monthly": {
        "tau_max":     3,    # lag tối đa: 3 tháng
        "pc_alpha":    0.1,  # nới lỏng hơn vì monthly chỉ ~132 điểm
        "window_size": 60,   # rolling window: 60 tháng (~5 năm)
        "unit":        "tháng",
    },
    "daily": {
        "tau_max":     5,    # lag tối đa: 5 ngày giao dịch (1 tuần)
        "pc_alpha":    0.01, # alpha chặt hơn vì nhiều quan sát hơn
        "window_size": 252,  # rolling window: 252 ngày (~1 năm giao dịch)
        "unit":        "ngày",
    },
}


def main(freq: str = "monthly", use_cache: bool = True):
    cfg = FREQ_CONFIG[freq]
    print(f"=== Causal TS Prediction — VN-Index ({freq}) ===\n")

    # 1. Load data
    print("--- 1. Load Data ---")
    df = load_dataset(freq=freq, use_cache=use_cache)
    print(f"  Shape: {df.shape}")

    forecaster = CausalForecaster(target_col="VNINDEX_Return")

    # 2. Stationarity check
    print("\n--- 2. Kiểm tra Stationarity (ADF) ---")
    non_stat = forecaster.check_stationarity(df)
    if non_stat:
        print(f"  Non-stationary columns: {non_stat}")
        print("  → Log returns thường đã stationary; nếu không thì cần diff thêm.")
    else:
        print("  Tất cả columns đã stationary. ✓")

    # 2b. Fix non-stationary + NaN trước khi đưa vào PCMCI+
    df_clean = df.copy()
    if non_stat:
        print(f"  → Diff các cột non-stationary: {non_stat}")
        df_clean[non_stat] = df_clean[non_stat].diff()

    # 2b+. Thêm autoregression + technical indicators (Nhóm 1)
    df_clean = add_technical_features(df_clean)
    new_cols = ["VNINDEX_lag1","VNINDEX_lag2","VNINDEX_lag3",
                "VNINDEX_mom3","VNINDEX_vol3","VNINDEX_rsi6","VNINDEX_trend"]
    print(f"  → Thêm {len(new_cols)} technical features: {new_cols}")

    before = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"  → Sau khi dropna: {before} → {len(df_clean)} hàng")

    # 2c. LASSO pre-selection — giảm features trước khi PCMCI+
    # Lý do: 31 features × 98 điểm monthly → statistical power thấp
    # LASSO chọn ~10 features → PCMCI+ có power tìm causal links tốt hơn
    print(f"\n--- 2c. LASSO Pre-selection (31 → ~10 features) ---")
    df_pre = forecaster.reduce_dimensions(df_clean, method='lasso')
    pre_cols = [c for c in df_pre.columns if c != forecaster.target_col]
    print(f"  → Còn lại: {len(pre_cols)} features: {pre_cols}")

    # 3. Causal discovery — PCMCI+ trên tập features đã lọc
    #    tau_min=1 đảm bảo chỉ tìm X(t-τ) → Y(t), không look-ahead bias.
    print(f"\n--- 3. Causal Discovery (PCMCI+, tau_max={cfg['tau_max']}) ---")
    forecaster.perform_causal_discovery(
        df_pre,
        tau_max=cfg["tau_max"],
        pc_alpha=cfg["pc_alpha"],
    )

    # 4. Trích xuất causal features + visualize
    print("\n--- 4. Causal Features & Graph (Phase 5) ---")
    causal_pairs = forecaster.extract_features()
    print(f"  PCMCI+ features: {forecaster.selected_features}")
    forecaster.visualize_graph(freq=freq)
    forecaster.plot_effect_heatmap(freq=freq)
    forecaster.plot_lag_effects(freq=freq)
    impact_df = forecaster.news_impact_report()

    # 4b. VAR-LiNGAM — Paper 2: union với PCMCI+ để feature set ổn định hơn
    print("\n--- 4b. VAR-LiNGAM Causal Discovery ---")
    lingam_features = forecaster.perform_causal_discovery_lingam(
        df_pre, lags=cfg["tau_max"]
    )
    print(f"  LiNGAM features : {lingam_features}")

    union_features = list(set(forecaster.selected_features) | set(lingam_features))
    print(f"  Union features  : {union_features}")
    forecaster.selected_features = union_features  # ghi đè để compare_models dùng

    # 5. So sánh các model với rolling window
    window = min(cfg["window_size"], len(df_pre) // 3)
    print(f"\n--- 5. Model Comparison (Rolling Window = {window} {cfg['unit']}) ---")
    results = forecaster.compare_models(df_pre, window_size=window, freq=freq)

    # 6. Tổng kết
    print("\n=== KẾT QUẢ TỔNG KẾT ===")
    print(f"{'Model':<30} {'MSE':>10} {'MAE':>10} {'Dir.Acc':>10}")
    print("-" * 62)
    for res in results.values():
        if res:
            print(
                f"{res['model']:<30} "
                f"{res['mse']:>10.6f} "
                f"{res['mae']:>10.6f} "
                f"{res['da']:>10.2%}"
            )

    # 7. Backtest chiến lược — model vs Buy & Hold
    print("\n--- 7. Backtest Strategy ---")
    best = results.get('all') or results.get('lasso')
    if best:
        forecaster.backtest_strategy(
            best['actuals'], best['predictions'], freq=freq
        )

    # 8. Dự báo tháng tiếp theo
    print("\n--- 7. Dự báo tháng tiếp theo ---")
    # Nhập giá VN-Index hiện tại (hoặc None nếu không biết)
    try:
        import vnstock as vns
        vnindex = vns.Vnstock().stock(symbol='VNINDEX', source='VCI')
        price_df = vnindex.quote.history(start='2026-01-01', end='2026-04-30', interval='1M')
        current_price = float(price_df['close'].iloc[-1]) if not price_df.empty else None
    except Exception:
        current_price = None

    forecaster.predict_next_month(
        df_pre,
        window_size=window,
        current_price=current_price,
    )


if __name__ == "__main__":
    freq_arg  = next((a for a in sys.argv[1:] if a in FREQ_CONFIG), "monthly")
    use_cache = "--fresh" not in sys.argv
    main(freq_arg, use_cache=use_cache)
