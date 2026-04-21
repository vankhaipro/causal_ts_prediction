from causal_forecaster import CausalForecaster
from data_loader import load_dataset
import pandas as pd
import numpy as np


def main():
    print("--- 1. Initializing Refined Causal Framework ---")
    forecaster = CausalForecaster(target_col='VNINDEX_Return')

    print("\n--- 2. Loading Real Data (Vietnam Stock Market, 2012-2026) ---")
    df = load_dataset()
    print(f"Initial dataset shape: {df.shape}")

    print("\n--- 3. Preprocessing: Stationarity & Alignment ---")
    non_stationary = forecaster.check_stationarity(df)
    print(f"Non-stationary columns found: {len(non_stationary)}")
    if non_stationary:
        print(f"  → {non_stationary}")

    # Publication lag: macro data thường công bố trễ 1 tháng
    macro_cols = [c for c in df.columns if c != 'SPY_Return']
    df = forecaster.apply_publication_lag(df, macro_cols, lag=1)
    print(f"Data after publication lag alignment. New shape: {df.shape}")

    print("\n--- 4. Dimensionality Reduction ---")
    n_features = df.shape[1] - 1  # trừ cột target
    if n_features >= 10:
        df_reduced = forecaster.reduce_dimensions(df, method='lasso')
    else:
        # Ít features rồi, không cần reduce
        print(f"  Chỉ có {n_features} features, bỏ qua LASSO reduction.")
        df_reduced = df.copy()
    print(f"Reduced dataset shape: {df_reduced.shape}")

    print("\n--- 5. Causal Discovery (PC Algorithm) ---")
    forecaster.perform_causal_discovery(df_reduced)
    print("Causal structure learned.")

    print("\n--- 6. Feature Selection & Visualization ---")
    selected = forecaster.extract_features()
    print(f"Causal Features selected: {selected}")
    forecaster.visualize_graph()

    print("\n--- 7. Rolling Window Forecasting ---")
    # Window 60 tháng (~5 năm) phù hợp với 190 tháng dữ liệu thực
    window = min(60, len(df_reduced) // 3)
    actuals, predictions = forecaster.forecast_rolling_window(df_reduced, window_size=window)

    print("\n--- 8. Evaluation ---")
    forecaster.evaluate(actuals, predictions)


if __name__ == "__main__":
    main()
