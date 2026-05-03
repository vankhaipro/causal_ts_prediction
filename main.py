"""
Pipeline chính — Causal TS Prediction VN-Index

Cách chạy:
  python main.py              # dùng dữ liệu monthly (mặc định)
  python main.py daily        # dùng dữ liệu daily
  python main.py monthly      # dùng dữ liệu monthly (tường minh)
"""

import sys
from causal_forecaster import CausalForecaster
from data_loader import load_dataset


# ------------------------------------------------------------------
# Cấu hình theo tần suất
# ------------------------------------------------------------------
FREQ_CONFIG = {
    "monthly": {
        "tau_max":     3,    # lag tối đa: 3 tháng
        "pc_alpha":    0.05,
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

    # 3. Causal discovery — PCMCI+
    #    tau_min=1 đảm bảo chỉ tìm X(t-τ) → Y(t), không look-ahead bias.
    print(f"\n--- 3. Causal Discovery (PCMCI+, tau_max={cfg['tau_max']}) ---")
    forecaster.perform_causal_discovery(
        df,
        tau_max=cfg["tau_max"],
        pc_alpha=cfg["pc_alpha"],
    )

    # 4. Trích xuất causal features + visualize
    print("\n--- 4. Causal Features & Graph (Phase 5) ---")
    causal_pairs = forecaster.extract_features()
    print(f"  Số features nhân quả: {len(forecaster.selected_features)}")
    forecaster.visualize_graph(freq=freq)
    forecaster.plot_effect_heatmap(freq=freq)
    forecaster.plot_lag_effects(freq=freq)
    impact_df = forecaster.news_impact_report()

    # 5. So sánh 3 baselines với rolling window
    window = min(cfg["window_size"], len(df) // 3)
    print(f"\n--- 5. Model Comparison (Rolling Window = {window} {cfg['unit']}) ---")
    results = forecaster.compare_models(df, window_size=window, freq=freq)

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


if __name__ == "__main__":
    freq_arg  = next((a for a in sys.argv[1:] if a in FREQ_CONFIG), "monthly")
    use_cache = "--fresh" not in sys.argv
    main(freq_arg, use_cache=use_cache)
