from causal_forecaster import CausalForecaster
from data_loader import load_dataset


def main():
    print("=== Causal TS Prediction — VN-Index ===\n")

    # 1. Load data
    print("--- 1. Load Data ---")
    df = load_dataset()
    print(f"Shape: {df.shape}")

    forecaster = CausalForecaster(target_col='VNINDEX_Return')

    # 2. Stationarity check
    print("\n--- 2. Kiểm tra Stationarity (ADF) ---")
    non_stat = forecaster.check_stationarity(df)
    if non_stat:
        print(f"  Non-stationary columns: {non_stat}")
        print("  → Log returns thường đã stationary; nếu không thì cần diff thêm.")
    else:
        print("  Tất cả columns đã stationary. ✓")

    # 3. Causal discovery — PCMCI+ thay PC algorithm
    #    tau_min=1 đảm bảo chỉ tìm lagged relationships X(t-τ) → Y(t),
    #    không cần áp dụng publication lag thủ công.
    print("\n--- 3. Causal Discovery (PCMCI+) ---")
    forecaster.perform_causal_discovery(df, tau_max=3, pc_alpha=0.05)

    # 4. Trích xuất causal features + visualize graph
    print("\n--- 4. Causal Features & Graph ---")
    causal_pairs = forecaster.extract_features()
    print(f"  Số features nhân quả: {len(forecaster.selected_features)}")
    forecaster.visualize_graph()

    # 5. So sánh 3 baselines với rolling window
    print("\n--- 5. Model Comparison (Rolling Window) ---")
    window = min(60, len(df) // 3)
    print(f"  Window size: {window} tháng")
    results = forecaster.compare_models(df, window_size=window)

    # 6. Tổng kết
    print("\n=== KẾT QUẢ TỔNG KẾT ===")
    print(f"{'Model':<30} {'MSE':>10} {'MAE':>10} {'Dir.Acc':>10}")
    print("-" * 62)
    for res in results.values():
        if res:
            print(f"{res['model']:<30} {res['mse']:>10.6f} {res['mae']:>10.6f} {res['da']:>10.2%}")


if __name__ == "__main__":
    main()
