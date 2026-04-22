"""
Streamlit App — Stock Causal Analysis

Chạy:
  streamlit run stock_causal_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_dataset
from stock_causal_engine import (
    VNIndexForecaster, CounterfactualStock, DoubleMLinear,
    ScenarioAnalyzer, FEATURE_LABELS, MACRO_COLS, STOCK_COLS,
)

# ------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Causal Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Cài đặt")
    freq = st.selectbox(
        "Tần suất dữ liệu",
        ["daily", "monthly"],
        format_func=lambda x: "📆 Daily" if x == "daily" else "📅 Monthly",
    )
    use_cache = st.checkbox("Dùng cache", value=True)
    st.divider()
    st.caption("Dữ liệu: vnstock (KBS) + yfinance")
    st.caption("Model: GradientBoosting + Double ML")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
for k in ["df", "forecaster", "cf_engine", "scenario_engine",
          "latest_features", "dml_results"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📊 Stock Causal Analysis — VN-Index")
st.caption(
    "'Nếu S&P500 hôm qua tăng 3%, VN-Index hôm nay thay đổi bao nhiêu?' "
    "| Counterfactual Analysis + Double ML + Scenario Simulation"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Huấn luyện",
    "🔁 Counterfactual",
    "⚖️ Double ML",
    "🌐 Kịch bản thị trường",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Huấn luyện
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("1. Tải dữ liệu & Huấn luyện mô hình dự báo")

    if st.button("🚀 Bắt đầu", type="primary"):
        with st.spinner("Đang tải dữ liệu..."):
            df = load_dataset(use_cache=use_cache, freq=freq)
            st.session_state.df = df

        with st.spinner("Đang huấn luyện GradientBoosting..."):
            forecaster = VNIndexForecaster()
            forecaster.fit(df)

            cf_engine       = CounterfactualStock(forecaster)
            scenario_engine = ScenarioAnalyzer(forecaster, cf_engine)

            # Features mới nhất (ngày/tháng cuối) làm baseline
            feat_cols = [c for c in df.columns if c != "VNINDEX_Return"]
            latest = df[feat_cols].iloc[-1].to_dict()

            st.session_state.forecaster      = forecaster
            st.session_state.cf_engine       = cf_engine
            st.session_state.scenario_engine = scenario_engine
            st.session_state.latest_features = latest
            st.success(f"Xong! Dataset: {df.shape[0]} quan sát × {df.shape[1]-1} features")

    if st.session_state.forecaster is not None:
        df         = st.session_state.df
        forecaster = st.session_state.forecaster
        eval_res   = forecaster.evaluate(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE",  f"{eval_res['mse']:.6f}")
        c2.metric("MAE",  f"{eval_res['mae']:.6f}")
        c3.metric("RMSE", f"{eval_res['rmse']:.6f}")
        c4.metric("Directional Accuracy", f"{eval_res['da']:.1%}")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Feature Importance (Permutation)**")
            fi = forecaster.feature_importance_df.head(10)
            fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
            colors = ['#C44E52' if i < 3 else '#4C72B0' for i in range(len(fi))]
            ax_fi.barh(fi["Label"][::-1], fi["Importance"][::-1],
                       xerr=fi["Std"][::-1], color=colors[::-1],
                       alpha=0.85, capsize=3)
            ax_fi.set_xlabel("Permutation Importance")
            ax_fi.set_title("Top 10 Features quan trọng nhất", fontsize=11)
            ax_fi.grid(True, linestyle=':', alpha=0.4, axis='x')
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

        with col_r:
            st.markdown("**VN-Index Return: Thực tế vs Dự báo**")
            feat_cols = [c for c in df.columns if c != "VNINDEX_Return"]
            df_lag = df[feat_cols].shift(1).copy()
            df_lag["VNINDEX_Return"] = df["VNINDEX_Return"]
            df_lag = df_lag.dropna()
            X_all = df_lag[feat_cols].values
            X_s   = forecaster.scaler.transform(X_all)
            y_pred = forecaster.model.predict(X_s)
            y_true = df_lag["VNINDEX_Return"].values

            fig_pred, ax_pred = plt.subplots(figsize=(7, 4))
            ax_pred.plot(y_true[-100:], color='black',
                         linewidth=1.2, label='Thực tế', alpha=0.8)
            ax_pred.plot(y_pred[-100:], color='#C44E52',
                         linewidth=1.2, linestyle='--', label='Dự báo', alpha=0.8)
            ax_pred.axhline(0, color='gray', linewidth=0.5)
            ax_pred.set_title("100 kỳ gần nhất", fontsize=11)
            ax_pred.set_ylabel("Log Return")
            ax_pred.legend(fontsize=8)
            ax_pred.grid(True, linestyle=':', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_pred, use_container_width=True)
            plt.close(fig_pred)

        # Dữ liệu mới nhất
        st.divider()
        st.markdown(f"**Dữ liệu mới nhất ({df.index[-1].date()}) — Dùng làm baseline**")
        latest = st.session_state.latest_features
        df_latest = pd.DataFrame([{
            "Feature": k,
            "Label": FEATURE_LABELS.get(k, k),
            "Giá trị (log return)": round(v, 5),
            "Đổi (%)": f"{v*100:+.3f}%",
        } for k, v in latest.items()])
        st.dataframe(df_latest, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Counterfactual
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("2. Counterfactual Analysis — 'What If?'")

    if st.session_state.cf_engine is None:
        st.warning("Vui lòng huấn luyện mô hình ở tab **Huấn luyện** trước.")
    else:
        cf_engine  = st.session_state.cf_engine
        forecaster = st.session_state.forecaster
        latest     = st.session_state.latest_features.copy()
        df         = st.session_state.df

        st.markdown(
            "Thay đổi **một biến** hôm qua và xem VN-Index hôm nay thay đổi thế nào. "
            "Causal propagation tự động điều chỉnh các biến liên quan "
            "(vd: SP500 tăng → VIX giảm)."
        )

        # Chỉnh baseline nếu muốn
        with st.expander("🔧 Tùy chỉnh giá trị baseline (mặc định = ngày cuối dataset)"):
            custom = {}
            c1, c2, c3 = st.columns(3)
            for i, feat in enumerate([f for f in latest if f in MACRO_COLS]):
                col = [c1, c2, c3][i % 3]
                custom[feat] = col.number_input(
                    FEATURE_LABELS.get(feat, feat),
                    value=float(latest[feat]),
                    format="%.5f",
                    step=0.001,
                    key=f"cf_base_{feat}"
                )
            baseline = {**latest, **custom}
        baseline = latest  # dùng mặc định

        st.divider()
        col_feat, col_delta, col_causal = st.columns([2, 2, 1])
        cf_feature = col_feat.selectbox(
            "Biến cần can thiệp",
            MACRO_COLS + STOCK_COLS,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
        )
        cf_delta = col_delta.slider(
            "Thay đổi (điểm phần trăm, vd: +3 = tăng thêm 3%)",
            min_value=-10.0, max_value=10.0,
            value=3.0, step=0.5,
        )
        cf_causal = col_causal.checkbox("Causal\nPropagation", value=True)

        if st.button("🔁 Tính Counterfactual", type="primary"):
            result = cf_engine.what_if(baseline, cf_feature, cf_delta, causal=cf_causal)

            pred_orig = result["pred_original"]
            pred_new  = result["pred_modified"]

            st.divider()
            col_a, col_mid, col_b = st.columns([2, 1, 2])

            with col_a:
                st.markdown("#### Baseline (thực tế)")
                st.metric(
                    "VN-Index Return dự báo",
                    f"{pred_orig*100:+.3f}%",
                    delta="📈 TĂNG" if pred_orig >= 0 else "📉 GIẢM",
                    delta_color="normal" if pred_orig >= 0 else "inverse",
                )
                st.caption(f"Log return: {pred_orig:+.5f}")

            with col_mid:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"**{FEATURE_LABELS.get(cf_feature, cf_feature)}**")
                st.markdown(f"**{cf_delta:+.1f}%** →")
                if result["causal_changes"] and cf_causal:
                    st.caption("Lan truyền:")
                    for f, chg in result["causal_changes"].items():
                        arrow = "↑" if chg["delta"] > 0 else "↓"
                        st.caption(f"{FEATURE_LABELS.get(f, f)[:10]}: {arrow}{abs(chg['delta']*100):.2f}%")

            with col_b:
                st.markdown("#### Counterfactual")
                delta_label = f"{result['pred_delta']*100:+.3f}%"
                st.metric(
                    "VN-Index Return dự báo",
                    f"{pred_new*100:+.3f}%",
                    delta=delta_label,
                    delta_color="normal" if result["pred_delta"] >= 0 else "inverse",
                )
                st.caption(f"Log return: {pred_new:+.5f}")

            if result["direction_changed"]:
                st.warning(
                    f"⚠️ **Đảo chiều!** Thay đổi {FEATURE_LABELS.get(cf_feature, cf_feature)} "
                    f"{cf_delta:+.1f}% làm VN-Index đổi từ "
                    f"{'tăng' if pred_orig >= 0 else 'giảm'} sang "
                    f"{'tăng' if pred_new >= 0 else 'giảm'}."
                )

            # Sensitivity sweep
            st.divider()
            st.markdown(f"**Sensitivity: VN-Index phản ứng với {FEATURE_LABELS.get(cf_feature, cf_feature)} ở các mức độ khác nhau**")
            sweep = cf_engine.sensitivity_sweep(
                baseline, cf_feature,
                deltas=np.arange(-8, 8.5, 0.5).tolist()
            )
            fig_sw, ax_sw = plt.subplots(figsize=(10, 4))
            pos = sweep["pred_vnindex"] >= 0
            ax_sw.fill_between(sweep["delta_pct"], sweep["pred_vnindex"],
                                where=pos, alpha=0.3, color='#2ca02c', label='Dự báo tăng')
            ax_sw.fill_between(sweep["delta_pct"], sweep["pred_vnindex"],
                                where=~pos, alpha=0.3, color='#d62728', label='Dự báo giảm')
            ax_sw.plot(sweep["delta_pct"], sweep["pred_vnindex"],
                       color='#1f77b4', linewidth=2)
            ax_sw.axvline(cf_delta, color='orange', linestyle='--',
                          linewidth=1.5, label=f'Điểm can thiệp ({cf_delta:+.1f}%)')
            ax_sw.axhline(0, color='black', linewidth=0.8)
            ax_sw.axvline(0, color='black', linewidth=0.5, linestyle=':')
            ax_sw.scatter([cf_delta], [pred_new], color='orange', s=100, zorder=5)
            ax_sw.set_xlabel(f"Thay đổi {FEATURE_LABELS.get(cf_feature, cf_feature)} (điểm %)")
            ax_sw.set_ylabel("VN-Index Return dự báo")
            ax_sw.set_title(f"Sensitivity Curve — {FEATURE_LABELS.get(cf_feature, cf_feature)} → VN-Index", fontsize=11)
            ax_sw.legend(fontsize=8)
            ax_sw.grid(True, linestyle=':', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_sw, use_container_width=True)
            plt.close(fig_sw)

        # Lịch sử phản ứng thực tế
        st.divider()
        st.markdown("**Phân tích lịch sử: quan hệ thực tế giữa feature và VN-Index**")
        hist_feat = st.selectbox(
            "Chọn feature", MACRO_COLS,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            key="hist_feat"
        )
        if hist_feat in df.columns:
            fig_sc, ax_sc = plt.subplots(figsize=(8, 4))
            ax_sc.scatter(
                df[hist_feat].shift(1).dropna() * 100,
                df["VNINDEX_Return"][1:] * 100,
                alpha=0.3, s=15, color='#4C72B0'
            )
            # Trend line
            x_clean = df[hist_feat].shift(1).dropna().values * 100
            y_clean = df["VNINDEX_Return"].iloc[1:].values * 100
            if len(x_clean) == len(y_clean):
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                ax_sc.plot(x_line, p(x_line), color='#C44E52', linewidth=2,
                           label=f'Trend (slope={z[0]:.3f})')
            ax_sc.axhline(0, color='black', linewidth=0.5)
            ax_sc.axvline(0, color='black', linewidth=0.5)
            ax_sc.set_xlabel(f"{FEATURE_LABELS.get(hist_feat, hist_feat)} t-1 (%)")
            ax_sc.set_ylabel("VN-Index Return t (%)")
            ax_sc.set_title(f"Quan hệ lịch sử: {FEATURE_LABELS.get(hist_feat, hist_feat)}(t-1) → VN-Index(t)", fontsize=10)
            ax_sc.legend(fontsize=8)
            ax_sc.grid(True, linestyle=':', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_sc, use_container_width=True)
            plt.close(fig_sc)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Double ML
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("3. Double ML — Hiệu ứng nhân quả thuần")

    if st.session_state.df is None:
        st.warning("Vui lòng huấn luyện mô hình trước.")
    else:
        df = st.session_state.df

        st.markdown("""
        **Double ML** loại bỏ confounders để đo hiệu ứng nhân quả *thực sự*.

        > Ví dụ: *SP500 tăng 1% → VN-Index thay đổi bao nhiêu %,*
        > *sau khi đã kiểm soát VIX, OIL, DXY, Gold?*

        Khác với hồi quy thường: hệ số OLS bị nhiễu confounders.
        Double ML dùng **2 bước residual** để tách hiệu ứng thuần.
        """)

        col_t, col_c = st.columns(2)
        all_features = MACRO_COLS + STOCK_COLS
        available    = [f for f in all_features if f in df.columns]

        treatment = col_t.selectbox(
            "Treatment (T) — Biến muốn đo tác động nhân quả",
            available,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            index=available.index("SP500") if "SP500" in available else 0,
        )
        confounders = col_c.multiselect(
            "Confounders (X) — Biến kiểm soát",
            [f for f in available if f != treatment],
            default=[f for f in available if f != treatment and f in MACRO_COLS][:4],
            format_func=lambda x: FEATURE_LABELS.get(x, x),
        )

        if st.button("⚙️ Chạy Double ML", type="primary"):
            if len(confounders) < 2:
                st.error("Cần ít nhất 2 confounders.")
            else:
                with st.spinner("Đang chạy Double ML..."):
                    dml = DoubleMLinear()
                    dml.fit(df, treatment=treatment,
                            outcome="VNINDEX_Return",
                            confounders=confounders)
                    st.session_state.dml_results = dml.summary()

        if st.session_state.dml_results:
            r = st.session_state.dml_results
            st.divider()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Causal Effect (θ)", f"{r['theta']*100:+.4f}%")
            col2.metric("Std Error",          f"{r['theta_se']*100:.4f}%")
            col3.metric("t-stat",             f"{r['t_stat']:.3f}")
            col4.metric("p-value",            f"{r['p_value']:.5f}",
                        delta="✅ Có ý nghĩa" if r["significant"] else "❌ Không ý nghĩa",
                        delta_color="normal" if r["significant"] else "off")

            st.info(r["interpretation"])
            st.caption(
                f"95% CI: [{r['ci_lo']*100:+.4f}%, {r['ci_hi']*100:+.4f}%] | "
                f"R² E[T|X]={r['r2_T']:.3f} | R² E[Y|X]={r['r2_Y']:.3f}"
            )

            # Visualize CI
            fig_dml, ax_dml = plt.subplots(figsize=(8, 2.5))
            ax_dml.barh(
                [FEATURE_LABELS.get(r["treatment"], r["treatment"])],
                [r["theta"] * 100],
                xerr=[1.96 * r["theta_se"] * 100],
                color='#C44E52' if r["theta"] > 0 else '#2ca02c',
                alpha=0.8, capsize=6, height=0.4,
            )
            ax_dml.axvline(0, color='black', linewidth=1, linestyle='--')
            ax_dml.set_xlabel("Causal Effect trên VNINDEX_Return (%)")
            ax_dml.set_title(
                f"Double ML: Tác động nhân quả của {FEATURE_LABELS.get(r['treatment'], r['treatment'])}",
                fontsize=10
            )
            ax_dml.grid(True, linestyle=':', alpha=0.4, axis='x')
            plt.tight_layout()
            st.pyplot(fig_dml, use_container_width=True)
            plt.close(fig_dml)

            # So sánh OLS vs DML
            st.divider()
            st.markdown("**So sánh: OLS thường vs Double ML**")
            from sklearn.linear_model import LinearRegression
            feat_cols = [treatment] + confounders
            sub = df[feat_cols + ["VNINDEX_Return"]].dropna()
            X_ols = sub[feat_cols].values
            y_ols = sub["VNINDEX_Return"].values
            s = StandardScaler()
            coef_ols = LinearRegression().fit(s.fit_transform(X_ols), y_ols).coef_[0]

            col_ols, col_dml = st.columns(2)
            col_ols.metric(
                "OLS coefficient (biased)",
                f"{coef_ols*100:+.4f}%",
                help="Bị nhiễu bởi confounders"
            )
            col_dml.metric(
                "Double ML (causal, unbiased)",
                f"{r['theta']*100:+.4f}%",
                delta=f"Chênh lệch: {(r['theta'] - coef_ols)*100:+.4f}%",
                help="Đã loại bỏ confounders"
            )


# ══════════════════════════════════════════════════════════════════
# TAB 4 — Kịch bản thị trường
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("4. Mô phỏng kịch bản thị trường")

    if st.session_state.scenario_engine is None:
        st.warning("Vui lòng huấn luyện mô hình trước.")
    else:
        scenario_engine = st.session_state.scenario_engine
        latest          = st.session_state.latest_features
        df              = st.session_state.df

        st.markdown(
            "Thay đổi nhiều biến cùng lúc để mô phỏng các kịch bản thị trường. "
            "Có thể dùng kịch bản có sẵn hoặc tự tạo."
        )

        # Kịch bản tất cả presets
        if st.button("📊 So sánh tất cả kịch bản có sẵn", type="primary"):
            df_presets = scenario_engine.compare_all_presets(latest)
            st.dataframe(df_presets, use_container_width=True, hide_index=True)

            # Bar chart
            fig_sc2, ax_sc2 = plt.subplots(figsize=(10, 4))
            preds  = [float(r.split('%')[0]) for r in df_presets["VN-Index dự báo"]]
            names  = df_presets["Kịch bản"].tolist()
            colors = ['#2ca02c' if p >= 0 else '#d62728' for p in preds]
            bars   = ax_sc2.barh(names, preds, color=colors, alpha=0.8)
            ax_sc2.axvline(0, color='black', linewidth=1)
            for bar, val in zip(bars, preds):
                ax_sc2.text(
                    val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}%",
                    va='center', ha='left' if val >= 0 else 'right', fontsize=9
                )
            ax_sc2.set_xlabel("VN-Index Return dự báo (%)")
            ax_sc2.set_title("So sánh tác động các kịch bản lên VN-Index", fontsize=11)
            ax_sc2.grid(True, linestyle=':', alpha=0.4, axis='x')
            plt.tight_layout()
            st.pyplot(fig_sc2, use_container_width=True)
            plt.close(fig_sc2)

        st.divider()

        # Kịch bản tùy chỉnh
        st.markdown("**🎛️ Tự tạo kịch bản**")
        col_a, col_b = st.columns(2)
        custom_deltas = {}
        macro_avail = [f for f in MACRO_COLS if f in latest]
        half = len(macro_avail) // 2
        for i, feat in enumerate(macro_avail):
            col = col_a if i < half + 1 else col_b
            custom_deltas[feat] = col.slider(
                FEATURE_LABELS.get(feat, feat),
                min_value=-10.0, max_value=10.0,
                value=0.0, step=0.5,
                key=f"custom_{feat}",
                help="Điểm phần trăm thêm vào (vd: +3 = tăng thêm 3%)"
            )

        if st.button("🎛️ Chạy kịch bản tùy chỉnh"):
            result = scenario_engine.run_scenario(latest, custom_deltas)
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(
                "VN-Index Return dự báo",
                f"{result['pred_modified']*100:+.3f}%",
                delta=f"Thay đổi: {result['pred_delta']*100:+.3f}%",
                delta_color="normal" if result["pred_delta"] >= 0 else "inverse",
            )
            col_res2.markdown(f"## {result['direction']}")

            st.markdown("**Chi tiết thay đổi áp dụng:**")
            for feat, chg in result["changes"].items():
                if chg["delta_pct"] != 0:
                    st.write(f"- **{chg['label']}**: {chg['delta_pct']:+.1f}% "
                             f"({chg['old']:+.5f} → {chg['new']:+.5f})")

        # Heatmap nhạy cảm 2 biến
        st.divider()
        st.markdown("**🌡️ Heatmap: VN-Index phản ứng khi 2 biến cùng thay đổi**")
        col_x, col_y = st.columns(2)
        feat_x = col_x.selectbox(
            "Trục X", macro_avail,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            index=0, key="hm_x"
        )
        feat_y = col_y.selectbox(
            "Trục Y", macro_avail,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            index=1, key="hm_y"
        )

        if st.button("🌡️ Vẽ Heatmap"):
            deltas_range = np.arange(-5, 5.5, 1.0)
            matrix = np.zeros((len(deltas_range), len(deltas_range)))
            for i, dx in enumerate(deltas_range):
                for j, dy in enumerate(deltas_range):
                    r = scenario_engine.run_scenario(
                        latest, {feat_x: dx, feat_y: dy}
                    )
                    matrix[i, j] = r["pred_modified"] * 100

            fig_hm, ax_hm = plt.subplots(figsize=(9, 7))
            im = ax_hm.imshow(
                matrix, cmap='RdYlGn', aspect='auto',
                vmin=-matrix.__abs__().max(),
                vmax=matrix.__abs__().max()
            )
            ax_hm.set_xticks(range(len(deltas_range)))
            ax_hm.set_yticks(range(len(deltas_range)))
            ax_hm.set_xticklabels([f"{d:+.0f}%" for d in deltas_range], fontsize=8)
            ax_hm.set_yticklabels([f"{d:+.0f}%" for d in deltas_range], fontsize=8)
            ax_hm.set_xlabel(FEATURE_LABELS.get(feat_x, feat_x))
            ax_hm.set_ylabel(FEATURE_LABELS.get(feat_y, feat_y))
            ax_hm.set_title(
                f"VN-Index Return (%) khi {FEATURE_LABELS.get(feat_x, feat_x)} "
                f"và {FEATURE_LABELS.get(feat_y, feat_y)} thay đổi",
                fontsize=10
            )
            plt.colorbar(im, ax=ax_hm, label="VN-Index Return (%)")
            plt.tight_layout()
            st.pyplot(fig_hm, use_container_width=True)
            plt.close(fig_hm)
