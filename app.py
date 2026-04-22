"""
Streamlit Web App — Causal TS Prediction VN-Index

Chạy:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from data_loader import load_dataset, DATA_DIR, get_latest_prices
from causal_forecaster import CausalForecaster, TORCH_AVAILABLE

# ------------------------------------------------------------------
# Cấu hình trang
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Causal TS Prediction — VN-Index",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar — Settings
# ------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Cài đặt")

    freq = st.selectbox(
        "Tần suất dữ liệu",
        ["monthly", "daily"],
        format_func=lambda x: "📅 Monthly (tháng)" if x == "monthly" else "📆 Daily (ngày)",
    )

    use_cache = st.checkbox("Dùng cache (không tải lại)", value=True)

    st.divider()
    st.subheader("PCMCI+ Settings")
    tau_max = st.slider(
        "tau_max (lag tối đa)",
        min_value=1,
        max_value=10 if freq == "daily" else 6,
        value=5 if freq == "daily" else 3,
    )
    pc_alpha = st.select_slider(
        "pc_alpha (ngưỡng ý nghĩa)",
        options=[0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01 if freq == "daily" else 0.05,
    )

    st.divider()
    st.subheader("Forecasting Settings")
    default_window = 252 if freq == "daily" else 60
    window_size = st.number_input(
        "Rolling window size",
        min_value=30,
        max_value=500,
        value=default_window,
        step=10,
    )

    use_lstm = st.checkbox(
        "Thêm Causal LSTM",
        value=False,
        disabled=not TORCH_AVAILABLE,
        help="Yêu cầu PyTorch. Chạy chậm hơn đáng kể." if TORCH_AVAILABLE
             else "PyTorch chưa được cài đặt.",
    )
    if use_lstm:
        lstm_epochs = st.slider("LSTM epochs (mỗi window)", 5, 50, 15)
    else:
        lstm_epochs = 15

    st.divider()
    st.caption(f"PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    st.caption("Data: vnstock (KBS) + yfinance")

# ------------------------------------------------------------------
# Session state — lưu kết quả đắt tiền giữa các lần tương tác
# ------------------------------------------------------------------
for key in ["df", "forecaster", "causal_pairs", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📈 Causal TS Prediction — VN-Index")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dữ liệu",
    "🔗 Causal Discovery",
    "📈 So sánh mô hình",
    "🎯 Dự báo kỳ tiếp theo",
    "💹 Khuyến nghị cổ phiếu",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Dữ liệu
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("1. Tải & Khám phá dữ liệu")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        load_btn = st.button("🔄 Tải dữ liệu", type="primary", use_container_width=True)

    if load_btn:
        with st.spinner("Đang tải dữ liệu..."):
            try:
                df = load_dataset(use_cache=use_cache, freq=freq)
                st.session_state.df = df
                # Reset downstream khi đổi data
                st.session_state.forecaster = None
                st.session_state.causal_pairs = None
                st.session_state.results = None
                st.success(f"Tải thành công! {df.shape[0]} quan sát × {df.shape[1]} cột")
            except Exception as e:
                st.error(f"Lỗi: {e}")

    # Hiển thị data nếu đã tải
    if st.session_state.df is not None:
        df = st.session_state.df

        # Metrics tổng quan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số quan sát", f"{len(df):,}")
        c2.metric("Số features", df.shape[1] - 1)
        c3.metric("Từ ngày", str(df.index[0].date()))
        c4.metric("Đến ngày", str(df.index[-1].date()))

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Mẫu dữ liệu (10 hàng cuối)**")
            st.dataframe(df.tail(10).round(4), use_container_width=True)

        with col_right:
            st.markdown("**Thống kê mô tả**")
            st.dataframe(df.describe().round(4), use_container_width=True)

        # ADF Stationarity
        st.divider()
        st.markdown("**Kiểm tra Stationarity (ADF Test)**")
        forecaster_tmp = CausalForecaster(target_col="VNINDEX_Return")
        non_stat = forecaster_tmp.check_stationarity(df)
        if non_stat:
            st.warning(f"Non-stationary: {non_stat}")
        else:
            st.success("Tất cả cột đã stationary (p < 0.05) ✓")

        # Correlation heatmap
        st.divider()
        st.markdown("**Ma trận tương quan**")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        corr = df.corr()
        im = ax_corr.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        ax_corr.set_xticks(range(len(corr.columns)))
        ax_corr.set_yticks(range(len(corr.columns)))
        ax_corr.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax_corr.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax_corr.text(j, i, f"{corr.values[i, j]:.2f}",
                             ha='center', va='center', fontsize=6,
                             color='black' if abs(corr.values[i, j]) < 0.7 else 'white')
        plt.colorbar(im, ax=ax_corr)
        ax_corr.set_title("Correlation Matrix", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_corr, use_container_width=True)
        plt.close(fig_corr)

        # VNINDEX return chart
        st.divider()
        st.markdown("**VN-Index Log Return theo thời gian**")
        fig_ret, ax_ret = plt.subplots(figsize=(12, 3))
        ax_ret.plot(df.index, df["VNINDEX_Return"], color='#C44E52', linewidth=0.8)
        ax_ret.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_ret.fill_between(df.index, df["VNINDEX_Return"], 0,
                            where=df["VNINDEX_Return"] >= 0,
                            alpha=0.3, color='#2ca02c', label='Tăng')
        ax_ret.fill_between(df.index, df["VNINDEX_Return"], 0,
                            where=df["VNINDEX_Return"] < 0,
                            alpha=0.3, color='#d62728', label='Giảm')
        ax_ret.set_ylabel("Log Return")
        ax_ret.legend(loc='upper right', fontsize=8)
        ax_ret.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_ret, use_container_width=True)
        plt.close(fig_ret)

    else:
        st.info("Nhấn **Tải dữ liệu** để bắt đầu.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Causal Discovery
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("2. Causal Discovery — PCMCI+")

    if st.session_state.df is None:
        st.warning("Vui lòng tải dữ liệu ở tab **Dữ liệu** trước.")
    else:
        df = st.session_state.df

        st.info(
            f"PCMCI+ sẽ chạy với tau_max={tau_max}, pc_alpha={pc_alpha}. "
            f"{'Daily với 2,500+ quan sát có thể mất 10–30 phút.' if freq == 'daily' else 'Monthly thường mất 1–3 phút.'}"
        )

        run_causal = st.button("🔍 Chạy PCMCI+", type="primary")

        if run_causal:
            forecaster = CausalForecaster(target_col="VNINDEX_Return")
            with st.spinner("Đang chạy PCMCI+... (có thể mất vài phút)"):
                forecaster.perform_causal_discovery(df, tau_max=tau_max, pc_alpha=pc_alpha)
                causal_pairs = forecaster.extract_features()
                st.session_state.forecaster = forecaster
                st.session_state.causal_pairs = causal_pairs
                st.session_state.results = None  # reset forecast
                st.success("PCMCI+ hoàn thành!")

        if st.session_state.forecaster is not None:
            forecaster = st.session_state.forecaster
            causal_pairs = st.session_state.causal_pairs

            col_l, col_r = st.columns([1, 1])

            with col_l:
                st.markdown("**Causal Features → VNINDEX_Return**")
                if causal_pairs:
                    df_pairs = pd.DataFrame(causal_pairs, columns=["Feature", "Lag (τ)"])
                    df_pairs["Ý nghĩa"] = df_pairs["Lag (τ)"].map(
                        lambda t: f"X(t-{t}) → Y(t)"
                    )
                    st.dataframe(df_pairs, use_container_width=True, hide_index=True)
                    st.metric("Số causal features", len(forecaster.selected_features))
                    st.markdown(f"**Features:** {', '.join(forecaster.selected_features)}")
                else:
                    st.warning("Không tìm được causal features. Thử tăng pc_alpha.")

            with col_r:
                st.markdown("**Causal Graph**")
                G = forecaster.build_causal_graph()
                if G:
                    fig_g, ax_g = plt.subplots(figsize=(8, 6))
                    pos = nx.spring_layout(G, k=0.8, seed=42)
                    colors = [
                        '#FF4B4B' if n == "VNINDEX_Return" else '#1E90FF'
                        for n in G.nodes()
                    ]
                    nx.draw(G, pos, with_labels=True, node_color=colors,
                            node_size=2000, font_size=7, font_weight='bold',
                            arrows=True, arrowsize=15, edge_color='gray',
                            alpha=0.85, ax=ax_g)
                    nx.draw_networkx_edge_labels(
                        G, pos, nx.get_edge_attributes(G, 'label'),
                        font_size=6, ax=ax_g
                    )
                    ax_g.set_title(
                        f"PCMCI+ Causal Graph\n(tau_max={tau_max}, α={pc_alpha})",
                        fontsize=10
                    )
                    plt.tight_layout()
                    st.pyplot(fig_g, use_container_width=True)
                    plt.close(fig_g)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — So sánh mô hình
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("3. So sánh mô hình (Rolling Window)")

    if st.session_state.forecaster is None:
        st.warning("Vui lòng chạy **PCMCI+** ở tab Causal Discovery trước.")
    else:
        forecaster = st.session_state.forecaster
        df = st.session_state.df

        model_labels = ["All-features Ridge", "LASSO Ridge", "Causal Ridge (PCMCI+)"]
        if use_lstm and TORCH_AVAILABLE:
            model_labels.append("Causal LSTM (PCMCI+)")

        st.info(
            f"Sẽ chạy {len(model_labels)} mô hình với rolling window = {window_size}. "
            f"{'LSTM sẽ mất rất lâu với daily data.' if use_lstm and freq == 'daily' else ''}"
        )

        run_compare = st.button("🚀 Chạy so sánh mô hình", type="primary")

        if run_compare:
            with st.spinner("Đang chạy rolling window forecast... (có thể mất vài phút)"):
                results = forecaster.compare_models(
                    df,
                    window_size=window_size,
                    freq=freq,
                    use_lstm=use_lstm,
                    lstm_epochs=lstm_epochs,
                )
                st.session_state.results = results
                st.success("Hoàn thành!")

        if st.session_state.results is not None:
            results = st.session_state.results

            # Metrics table
            st.markdown("**Bảng kết quả**")
            rows = []
            for key, r in results.items():
                if r is None:
                    continue
                rows.append({
                    "Mô hình": r['model'],
                    "MSE ↓": round(r['mse'], 6),
                    "MAE ↓": round(r['mae'], 6),
                    "Directional Accuracy ↑": f"{r['da']:.2%}",
                })
            df_results = pd.DataFrame(rows)

            best_da_idx = max(
                range(len(rows)),
                key=lambda i: float(rows[i]["Directional Accuracy ↑"].strip('%')) / 100
            )

            # Hiển thị từng hàng, hàng tốt nhất dùng st.success
            for idx, row in df_results.iterrows():
                cols_row = st.columns([3, 2, 2, 2])
                is_best = (idx == best_da_idx)
                label = f"{'🏆 ' if is_best else ''}{row['Mô hình']}"
                cols_row[0].markdown(f"**{label}**")
                cols_row[1].metric("MSE ↓", row["MSE ↓"])
                cols_row[2].metric("MAE ↓", row["MAE ↓"])
                cols_row[3].metric("DA ↑", row["Directional Accuracy ↑"])
            st.caption("🏆 = Directional Accuracy cao nhất")

            st.divider()

            # Time series plot
            st.markdown("**Dự báo vs Thực tế**")
            fig_ts, ax_ts = plt.subplots(figsize=(14, 5))
            actuals = results['all']['actuals']
            ax_ts.plot(actuals, label='Actual', color='black', linewidth=1.5)

            colors_map = {
                'all':    ('#4C72B0', '--'),
                'lasso':  ('#55A868', '--'),
                'causal': ('#C44E52', '-'),
                'lstm':   ('#FF7F0E', '-.'),
            }
            for key, (color, ls) in colors_map.items():
                if results.get(key):
                    ax_ts.plot(
                        results[key]['predictions'],
                        label=results[key]['model'],
                        color=color, linestyle=ls, alpha=0.85,
                        linewidth=1.5 if key in ('causal', 'lstm') else 1.0,
                    )
            ax_ts.set_ylabel("Log Return")
            ax_ts.legend(fontsize=8)
            ax_ts.grid(True, linestyle=':', alpha=0.4)
            ax_ts.set_title("VN-Index Return: Dự báo vs Thực tế", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_ts, use_container_width=True)
            plt.close(fig_ts)

            # Bar chart DA
            st.divider()
            st.markdown("**Directional Accuracy theo mô hình**")
            fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
            bar_rows = [r for r in rows]
            names_bar = [r['Mô hình'] for r in bar_rows]
            das_bar   = [float(r['Directional Accuracy ↑'].strip('%')) / 100
                         for r in bar_rows]
            bar_colors = ['#C44E52' if i == best_da_idx else '#4C72B0'
                          for i in range(len(bar_rows))]
            bars = ax_bar.bar(names_bar, das_bar, color=bar_colors, alpha=0.8)
            ax_bar.axhline(0.5, color='gray', linestyle='--', linewidth=1,
                           label='Baseline ngẫu nhiên (50%)')
            for bar, val in zip(bars, das_bar):
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.1%}", ha='center', va='bottom', fontsize=9)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel("Directional Accuracy")
            ax_bar.set_title("So sánh Directional Accuracy", fontsize=11)
            ax_bar.legend()
            ax_bar.tick_params(axis='x', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — Dự báo kỳ tiếp theo
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("4. Dự báo kỳ tiếp theo")

    if st.session_state.forecaster is None or st.session_state.df is None:
        st.warning("Vui lòng hoàn thành tab **Causal Discovery** trước.")
    else:
        df = st.session_state.df
        forecaster = st.session_state.forecaster

        last_date = df.index[-1].date()
        unit_label = "ngày giao dịch" if freq == "daily" else "tháng"
        st.info(
            f"Dự báo VNINDEX_Return cho kỳ tiếp theo sau **{last_date}** "
            f"({unit_label} kế tiếp)."
        )

        predict_btn = st.button("🎯 Dự báo ngay", type="primary")

        if predict_btn:
            all_cols = [c for c in df.columns if c != "VNINDEX_Return"]

            if forecaster.selected_features:
                pred_causal = forecaster.forecast_next(
                    df, forecaster.selected_features, window_size=window_size
                )
            else:
                pred_causal = None

            pred_all = forecaster.forecast_next(df, all_cols, window_size=window_size)

            c1, c2 = st.columns(2)
            with c1:
                direction = "📈 TĂNG" if pred_all >= 0 else "📉 GIẢM"
                st.metric(
                    "All-features Ridge",
                    f"{pred_all:.4f}",
                    delta=direction,
                    delta_color="normal" if pred_all >= 0 else "inverse",
                )

            with c2:
                if pred_causal is not None:
                    direction_c = "📈 TĂNG" if pred_causal >= 0 else "📉 GIẢM"
                    st.metric(
                        "Causal Ridge (PCMCI+)",
                        f"{pred_causal:.4f}",
                        delta=direction_c,
                        delta_color="normal" if pred_causal >= 0 else "inverse",
                    )
                else:
                    st.info("Không có causal features để dự báo.")

            st.divider()
            st.caption(
                "**Lưu ý:** Đây là dự báo log return. "
                "Giá trị dương → kỳ vọng VN-Index tăng; âm → giảm. "
                "Đây là mô hình học thuật, không phải tư vấn đầu tư."
            )

            # Lịch sử gần đây
            st.markdown(f"**VNINDEX_Return 20 kỳ gần nhất**")
            fig_hist, ax_hist = plt.subplots(figsize=(12, 3))
            recent = df["VNINDEX_Return"].tail(20)
            bar_c = ['#2ca02c' if v >= 0 else '#d62728' for v in recent.values]
            ax_hist.bar(range(len(recent)), recent.values, color=bar_c, alpha=0.8)
            ax_hist.axhline(0, color='black', linewidth=0.5)

            if pred_causal is not None:
                ax_hist.bar([len(recent)], [pred_causal], color='#FF7F0E',
                            alpha=0.9, label='Dự báo (Causal)', width=0.8)
            ax_hist.bar([len(recent) + (1 if pred_causal is not None else 0)],
                        [pred_all], color='#9467bd',
                        alpha=0.9, label='Dự báo (All-feat)', width=0.8)

            ax_hist.set_xticks(range(len(recent)))
            ax_hist.set_xticklabels(
                [str(d.date()) for d in recent.index], rotation=45, ha='right', fontsize=6
            )
            ax_hist.set_ylabel("Log Return")
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, linestyle=':', alpha=0.4, axis='y')
            plt.tight_layout()
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — Khuyến nghị cổ phiếu
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("5. Khuyến nghị cổ phiếu — Dự báo giá kỳ tiếp theo")

    if st.session_state.df is None:
        st.warning("Vui lòng tải dữ liệu ở tab **Dữ liệu** trước.")
    else:
        df = st.session_state.df
        forecaster_tmp = st.session_state.forecaster or CausalForecaster(target_col="VNINDEX_Return")

        st.info(
            f"Model dự báo **log return** kỳ tiếp theo cho từng mã, "
            f"sau đó tính **giá dự báo = giá hiện tại × exp(predicted return)**. "
            f"Causal parents từ PCMCI+ được dùng làm features (nếu đã chạy)."
        )

        # Ngưỡng MUA/BÁN
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            buy_threshold = st.number_input(
                "Ngưỡng MUA (return tối thiểu)",
                min_value=0.001, max_value=0.05,
                value=0.005 if freq == "daily" else 0.01,
                step=0.001, format="%.3f",
                help="Return dự báo ≥ ngưỡng này → khuyến nghị MUA"
            )
        with col_th2:
            sell_threshold = st.number_input(
                "Ngưỡng BÁN (return tối đa)",
                min_value=-0.05, max_value=-0.001,
                value=-0.005 if freq == "daily" else -0.01,
                step=0.001, format="%.3f",
                help="Return dự báo ≤ ngưỡng này → khuyến nghị BÁN"
            )

        predict_stocks_btn = st.button("💹 Phân tích & Dự báo tất cả cổ phiếu", type="primary")

        if predict_stocks_btn:
            with st.spinner("Đang tải giá hiện tại và tính dự báo..."):
                latest_prices = get_latest_prices(freq=freq)

                if latest_prices.empty:
                    st.error("Không lấy được giá hiện tại. Thử tải lại dữ liệu với use_cache=False.")
                else:
                    df_rec = forecaster_tmp.predict_all_stocks(
                        df,
                        latest_prices,
                        window_size=window_size,
                        buy_threshold=buy_threshold,
                        sell_threshold=sell_threshold,
                    )
                    st.session_state["stock_recommendations"] = df_rec

        if st.session_state.get("stock_recommendations") is not None:
            df_rec = st.session_state["stock_recommendations"]

            # Tổng quan khuyến nghị
            n_buy  = (df_rec["Khuyến nghị"].str.contains("MUA")).sum()
            n_sell = (df_rec["Khuyến nghị"].str.contains("BÁN")).sum()
            n_hold = (df_rec["Khuyến nghị"].str.contains("GIỮ")).sum()
            last_date = df.index[-1].date()

            st.markdown(f"### Tổng quan — Kỳ tiếp theo sau {last_date}")
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Khuyến nghị MUA", n_buy)
            c2.metric("🟡 Khuyến nghị GIỮ", n_hold)
            c3.metric("🔴 Khuyến nghị BÁN", n_sell)

            st.divider()

            # Bảng khuyến nghị chi tiết
            st.markdown("**Chi tiết từng mã**")
            for _, row in df_rec.iterrows():
                with st.container():
                    cols = st.columns([1, 1.5, 1.5, 1.5, 1.5, 2, 1.5])
                    cols[0].markdown(f"**{row['Mã']}**")
                    cols[1].metric("Giá hiện tại", f"{row['Giá hiện tại']:,.0f}")
                    cols[2].metric(
                        "Giá dự báo",
                        f"{row['Giá dự báo']:,.0f}",
                        delta=f"{row['Thay đổi (%)']:+.2f}%",
                        delta_color="normal" if row["Thay đổi (%)"] >= 0 else "inverse",
                    )
                    cols[3].metric("Return dự báo", f"{row['Predicted Return']:+.4f}")
                    cols[4].markdown(f"**{row['Khuyến nghị']}**")
                    cols[5].progress(
                        int(row["Độ tin cậy"]),
                        text=f"Tin cậy: {row['Độ tin cậy']:.0f}%"
                    )
                st.divider()

            # Biểu đồ giá hiện tại vs dự báo
            st.markdown("**Biểu đồ so sánh Giá hiện tại vs Giá dự báo**")
            fig_price, ax_price = plt.subplots(figsize=(12, 5))
            x = np.arange(len(df_rec))
            w = 0.35
            bar1 = ax_price.bar(x - w/2, df_rec["Giá hiện tại"], w,
                                 label="Giá hiện tại", color="#4C72B0", alpha=0.8)
            bar2 = ax_price.bar(x + w/2, df_rec["Giá dự báo"], w,
                                 label="Giá dự báo", color="#C44E52", alpha=0.8)
            ax_price.set_xticks(x)
            ax_price.set_xticklabels(df_rec["Mã"], fontsize=11)
            ax_price.set_ylabel("Giá (VND)")
            ax_price.set_title(f"Giá hiện tại vs Dự báo kỳ tiếp theo", fontsize=12)
            ax_price.legend()
            ax_price.grid(True, linestyle=':', alpha=0.4, axis='y')
            for bar in bar2:
                h = bar.get_height()
                ax_price.text(bar.get_x() + bar.get_width()/2, h * 1.005,
                              f"{h:,.0f}", ha='center', va='bottom', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_price, use_container_width=True)
            plt.close(fig_price)

            # Lịch sử 30 kỳ gần nhất cho mã được chọn
            st.divider()
            st.markdown("**Lịch sử giá mã cụ thể**")
            symbols_available = df_rec["Mã"].tolist()
            selected_sym = st.selectbox("Chọn mã cổ phiếu", symbols_available)

            if selected_sym and selected_sym in df.columns:
                price_file = DATA_DIR / f"prices{'_daily' if freq == 'daily' else ''}.csv"
                if price_file.exists():
                    df_prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
                    if selected_sym in df_prices.columns:
                        recent_prices = df_prices[selected_sym].dropna().tail(60)
                        pred_row = df_rec[df_rec["Mã"] == selected_sym].iloc[0]

                        fig_sym, ax_sym = plt.subplots(figsize=(12, 4))
                        ax_sym.plot(recent_prices.index, recent_prices.values,
                                    color="#4C72B0", linewidth=1.5, label=f"{selected_sym} giá thực")
                        # Thêm điểm dự báo
                        last_real_date = recent_prices.index[-1]
                        ax_sym.scatter([last_real_date], [pred_row["Giá hiện tại"]],
                                       color="#4C72B0", s=60, zorder=5)
                        ax_sym.scatter(
                            [last_real_date], [pred_row["Giá dự báo"]],
                            color="#C44E52", s=120, zorder=6,
                            marker="*", label=f"Dự báo: {pred_row['Giá dự báo']:,.0f}"
                        )
                        ax_sym.set_ylabel("Giá đóng cửa (VND)")
                        ax_sym.set_title(f"{selected_sym} — Lịch sử & Dự báo kỳ tiếp theo", fontsize=12)
                        ax_sym.legend()
                        ax_sym.grid(True, linestyle=':', alpha=0.4)
                        plt.xticks(rotation=30, ha='right', fontsize=7)
                        plt.tight_layout()
                        st.pyplot(fig_sym, use_container_width=True)
                        plt.close(fig_sym)

            st.caption(
                "⚠️ **Tuyên bố miễn trách:** Đây là dự báo từ mô hình học máy phục vụ mục đích "
                "nghiên cứu học thuật, KHÔNG phải tư vấn đầu tư. "
                "Thị trường chứng khoán có rủi ro cao — luôn tự nghiên cứu trước khi đầu tư."
            )
