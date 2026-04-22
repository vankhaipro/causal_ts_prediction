"""
Streamlit App — Causal Credit Scoring

Chạy:
  streamlit run credit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from credit_scorer import (
    CreditScorer, CounterfactualEngine, DoubleMLinear, FairnessAuditor,
    load_german_credit, FEATURE_LABELS, NUMERIC_COLS,
)

# ------------------------------------------------------------------
st.set_page_config(
    page_title="Causal Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
for k in ["scorer", "X", "y", "X_test", "y_test", "cf_engine",
          "dml_results", "fairness_results"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🏦 Causal Credit Scoring")
st.caption(
    "Counterfactual Credit Models — 'Nếu thu nhập tăng 10%, khách hàng có trả được nợ không?' "
    "| German Credit Dataset (1000 mẫu)"
)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dữ liệu & Huấn luyện",
    "🔮 Chấm điểm khách hàng",
    "🔁 Counterfactual Analysis",
    "⚖️ Double ML (Causal Effect)",
    "🛡️ Fairness Audit",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Dữ liệu & Huấn luyện
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("1. Tải dữ liệu & Huấn luyện mô hình")

    st.info(
        "**German Credit Dataset** (UCI): 1000 khách hàng vay ngân hàng Đức. "
        "Features: tuổi, thu nhập, lịch sử tín dụng, tài sản,... "
        "Target: good credit (1) / bad credit (0)."
    )

    if st.button("🚀 Tải dữ liệu & Huấn luyện", type="primary"):
        with st.spinner("Đang tải German Credit dataset..."):
            X, y = load_german_credit()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            scorer = CreditScorer()
            scorer.fit(X_train, y_train)
            cf_engine = CounterfactualEngine(scorer)

            st.session_state.update({
                "scorer": scorer, "X": X, "y": y,
                "X_test": X_test, "y_test": y_test,
                "cf_engine": cf_engine,
            })
            st.success(f"Huấn luyện xong! Dataset: {X.shape[0]} mẫu × {X.shape[1]} features")

    if st.session_state.scorer is not None:
        scorer = st.session_state.scorer
        X, y   = st.session_state.X, st.session_state.y
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Metrics
        eval_res = scorer.evaluate(X_test, y_test)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC (test)", f"{eval_res['auc']:.4f}")
        c2.metric("Precision (good)", f"{eval_res['report']['1']['precision']:.3f}")
        c3.metric("Recall (good)",    f"{eval_res['report']['1']['recall']:.3f}")
        c4.metric("F1 (good)",        f"{eval_res['report']['1']['f1-score']:.3f}")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Top 10 Feature Importance (Permutation)**")
            fi = scorer.feature_importance_df.head(10)
            fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
            colors = ['#C44E52' if i < 3 else '#4C72B0' for i in range(len(fi))]
            ax_fi.barh(fi["Label"][::-1], fi["Importance"][::-1],
                       xerr=fi["Std"][::-1], color=colors[::-1],
                       alpha=0.85, capsize=3)
            ax_fi.set_xlabel("Permutation Importance")
            ax_fi.set_title("Feature Importance", fontsize=11)
            ax_fi.grid(True, linestyle=':', alpha=0.4, axis='x')
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

        with col_r:
            st.markdown("**Confusion Matrix (Test set)**")
            cm = eval_res["confusion_matrix"]
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            im = ax_cm.imshow(cm, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                               fontsize=16, fontweight='bold',
                               color='white' if cm[i, j] > cm.max() / 2 else 'black')
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Dự báo BAD", "Dự báo GOOD"])
            ax_cm.set_yticklabels(["Thực tế BAD", "Thực tế GOOD"])
            ax_cm.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

        # Phân phối điểm tín dụng
        st.divider()
        st.markdown("**Phân phối điểm tín dụng (Credit Score 300–850)**")
        probs = scorer.predict_proba(X)[:, 1]
        scores_all = (300 + probs * 550).astype(int)
        fig_dist, ax_dist = plt.subplots(figsize=(10, 3))
        ax_dist.hist(scores_all[y == 1], bins=40, alpha=0.7,
                     color='#2ca02c', label='Good credit', density=True)
        ax_dist.hist(scores_all[y == 0], bins=40, alpha=0.7,
                     color='#d62728', label='Bad credit', density=True)
        for thresh, label, color in [(500, 'Kém', '#d62728'), (580, 'TB', '#ff7f0e'),
                                      (670, 'Tốt', '#2ca02c'), (740, 'Xuất sắc', '#1f77b4')]:
            ax_dist.axvline(thresh, color=color, linestyle='--', linewidth=1, alpha=0.7)
            ax_dist.text(thresh + 2, ax_dist.get_ylim()[1] * 0.9,
                         label, fontsize=7, color=color)
        ax_dist.set_xlabel("Credit Score")
        ax_dist.set_ylabel("Mật độ")
        ax_dist.legend()
        ax_dist.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Chấm điểm khách hàng
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("2. Nhập thông tin khách hàng — Chấm điểm tín dụng")

    if st.session_state.scorer is None:
        st.warning("Vui lòng huấn luyện mô hình ở tab **Dữ liệu & Huấn luyện** trước.")
    else:
        scorer = st.session_state.scorer
        X = st.session_state.X

        st.markdown("**Thông tin cơ bản**")
        c1, c2, c3 = st.columns(3)
        age             = c1.slider("Tuổi", 18, 75, 35)
        duration        = c2.slider("Thời hạn vay (tháng)", 6, 72, 24)
        credit_amount   = c3.number_input("Số tiền vay (DM)", 500, 20000, 3000, step=500)

        c4, c5, c6 = st.columns(3)
        installment     = c4.slider("Tỷ lệ trả góp / thu nhập (%)", 1, 4, 2)
        existing_credits= c5.slider("Số khoản vay hiện có", 1, 4, 1)
        num_dependents  = c6.slider("Số người phụ thuộc", 1, 2, 1)

        st.markdown("**Thông tin định tính**")
        c7, c8 = st.columns(2)
        # Dùng giá trị trung bình của dataset cho các categorical features
        median_vals = X.median().to_dict()
        customer = {col: median_vals.get(col, 0) for col in scorer.feature_names}
        customer.update({
            "age": age,
            "duration": duration,
            "credit_amount": credit_amount,
            "installment_commitment": installment,
            "existing_credits": existing_credits,
            "num_dependents": num_dependents,
        })

        if st.button("🔍 Chấm điểm ngay", type="primary"):
            result = scorer.score_customer(customer)

            st.divider()
            st.markdown("### Kết quả chấm điểm")

            col_score, col_detail = st.columns([1, 2])
            with col_score:
                st.metric(
                    f"{result['risk_color']} Credit Score",
                    result["credit_score"],
                    delta=result["risk_tier"],
                )
                st.metric("Xác suất trả được nợ", f"{result['prob_good']:.1%}")
                st.metric("Xác suất vỡ nợ",       f"{result['prob_default']:.1%}")
                if result["approved"]:
                    st.success("✅ CHẤP THUẬN vay")
                else:
                    st.error("❌ TỪ CHỐI vay")

            with col_detail:
                # Gauge chart
                score = result["credit_score"]
                fig_g, ax_g = plt.subplots(figsize=(6, 3),
                                            subplot_kw={"projection": "polar"})
                ax_g.set_theta_offset(np.pi)
                ax_g.set_theta_direction(-1)
                theta = np.linspace(0, np.pi, 300)
                ranges = [(300, 500, '#d62728'), (500, 580, '#ff7f0e'),
                           (580, 670, '#ffdd57'), (670, 740, '#2ca02c'),
                           (740, 850, '#1f77b4')]
                for lo, hi, color in ranges:
                    t_lo = (lo - 300) / 550 * np.pi
                    t_hi = (hi - 300) / 550 * np.pi
                    ax_g.fill_between(np.linspace(t_lo, t_hi, 50),
                                       0.6, 1.0, color=color, alpha=0.7)
                needle = (score - 300) / 550 * np.pi
                ax_g.annotate("", xy=(needle, 0.9), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="-|>", color='black', lw=2))
                ax_g.set_ylim(0, 1.1)
                ax_g.set_yticklabels([])
                ax_g.set_xticklabels([])
                ax_g.set_title(f"Score: {score}  |  {result['risk_tier']}", fontsize=12, pad=15)
                plt.tight_layout()
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)

            # Top factors
            st.divider()
            st.markdown("**Top 5 yếu tố ảnh hưởng nhất đến điểm của khách hàng này**")
            fi_top = scorer.feature_importance_df.head(5)
            for _, row in fi_top.iterrows():
                val = customer.get(row["Feature"], "N/A")
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(f"**{row['Label']}**: {val:.2f}" if isinstance(val, float)
                               else f"**{row['Label']}**: {val}")
                col_b.progress(max(0, min(1, row["Importance"] * 20)),
                               text=f"Imp: {row['Importance']:.3f}")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Counterfactual Analysis
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("3. Counterfactual Analysis — 'What If?'")

    if st.session_state.cf_engine is None:
        st.warning("Vui lòng huấn luyện mô hình trước.")
    else:
        cf_engine = st.session_state.cf_engine
        scorer    = st.session_state.scorer
        X         = st.session_state.X
        median_vals = X.median().to_dict()

        st.markdown(
            "Thay đổi **một feature** và xem điểm tín dụng thay đổi như thế nào. "
            "Câu hỏi dạng: *'Nếu giảm số tiền vay 20%, điểm có đủ để được chấp thuận không?'*"
        )

        col_f1, col_f2, col_f3 = st.columns(3)
        cf_age    = col_f1.slider("Tuổi khách hàng", 18, 75, 35)
        cf_dur    = col_f2.slider("Thời hạn vay", 6, 72, 36)
        cf_amount = col_f3.number_input("Số tiền vay", 500, 20000, 5000, step=500)

        customer_cf = {col: median_vals.get(col, 0) for col in scorer.feature_names}
        customer_cf.update({
            "age": cf_age, "duration": cf_dur, "credit_amount": cf_amount,
            "installment_commitment": 2, "existing_credits": 1,
        })

        st.divider()
        st.markdown("**Chọn can thiệp counterfactual**")
        col_i1, col_i2, col_i3 = st.columns(3)

        cf_feature = col_i1.selectbox(
            "Feature cần thay đổi",
            ["duration", "credit_amount", "installment_commitment",
             "existing_credits", "savings_status"],
            format_func=lambda x: FEATURE_LABELS.get(x, x)
        )
        cf_delta = col_i2.slider(
            "Thay đổi (%)", -50, 100, 10, step=5,
            help="Giá trị âm = giảm, dương = tăng"
        )
        cf_causal = col_i3.checkbox(
            "Áp dụng Causal Propagation",
            value=True,
            help="Tự động điều chỉnh các features liên quan (vd: vay nhiều → tỷ lệ trả góp tăng)"
        )

        if st.button("🔁 Tính Counterfactual", type="primary"):
            orig_score = scorer.score_customer(customer_cf)
            cf_result  = cf_engine.what_if(
                customer_cf, cf_feature, delta_pct=cf_delta, causal=cf_causal
            )

            col_before, col_arrow, col_after = st.columns([2, 1, 2])
            with col_before:
                st.markdown("#### Trước can thiệp")
                st.metric("Credit Score", orig_score["credit_score"])
                st.metric("Xác suất trả nợ", f"{orig_score['prob_good']:.1%}")
                st.markdown(f"**{orig_score['risk_color']} {orig_score['risk_tier']}**")
                if orig_score["approved"]:
                    st.success("✅ Được chấp thuận")
                else:
                    st.error("❌ Bị từ chối")

            with col_arrow:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                direction = "→ Tăng" if cf_delta > 0 else "→ Giảm"
                st.markdown(f"**{FEATURE_LABELS.get(cf_feature, cf_feature)}**")
                st.markdown(f"**{cf_delta:+d}%**")
                if cf_result["causal_changes"]:
                    st.caption("Causal effects:")
                    for feat, chg in cf_result["causal_changes"].items():
                        st.caption(f"  {FEATURE_LABELS.get(feat, feat)}: {chg['old']} → {chg['new']}")

            with col_after:
                st.markdown("#### Sau can thiệp")
                new_score = cf_result["score_new"]
                delta_label = f"{cf_result['score_delta']:+d}"
                st.metric("Credit Score", new_score["credit_score"], delta=delta_label)
                st.metric("Xác suất trả nợ", f"{new_score['prob_good']:.1%}",
                          delta=f"{cf_result['prob_delta']:+.1%}")
                st.markdown(f"**{new_score['risk_color']} {new_score['risk_tier']}**")
                if new_score["approved"]:
                    st.success("✅ Được chấp thuận")
                else:
                    st.error("❌ Bị từ chối")

            if cf_result["decision_flipped"]:
                st.balloons()
                st.success(
                    "🎉 **Quyết định đảo chiều!** "
                    f"Thay đổi {FEATURE_LABELS.get(cf_feature, cf_feature)} "
                    f"{cf_delta:+d}% đã giúp khách hàng "
                    f"{'được chấp thuận' if new_score['approved'] else 'bị từ chối'}."
                )

        # Roadmap cải thiện
        st.divider()
        st.markdown("**🗺️ Lộ trình cải thiện điểm tín dụng**")
        st.caption("Tìm tập thay đổi tối thiểu để đạt điểm 670 (ngưỡng 'Tốt').")

        if st.button("🗺️ Tìm lộ trình cải thiện"):
            steps = cf_engine.find_minimal_changes(customer_cf, target_score=670)
            if steps:
                for step in steps:
                    icon = "✅" if step["score_delta"] > 10 else "➡️"
                    st.markdown(
                        f"{icon} **Bước {step['step']}:** "
                        f"Điều chỉnh **{step['label']}** "
                        f"từ {step['old_value']:.1f} → {step['new_value']:.1f} "
                        f"(điểm tăng **+{step['score_delta']}**, "
                        f"đạt {step['new_score']})"
                    )
            else:
                st.info("Điểm hiện tại đã đạt ngưỡng 670.")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — Double ML
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("4. Double ML — Hiệu ứng nhân quả thuần")

    if st.session_state.scorer is None:
        st.warning("Vui lòng huấn luyện mô hình trước.")
    else:
        X, y = st.session_state.X, st.session_state.y
        df_full = X.copy()
        df_full["default"] = (1 - y).values   # 1 = bad (default), 0 = good

        st.markdown("""
        **Double ML** tách hiệu ứng nhân quả *thực sự* của một biến ra khỏi confounders.

        Ví dụ: Tác động của **thời hạn vay** lên rủi ro vỡ nợ sau khi đã kiểm soát
        tuổi, thu nhập, lịch sử tín dụng,...

        > *Khác với hồi quy thường: hệ số OLS bị nhiễu bởi confounders.*
        """)

        col_t, col_c = st.columns(2)
        all_numeric = [c for c in X.columns if c in NUMERIC_COLS]
        treatment = col_t.selectbox(
            "Treatment (T) — biến cần đo tác động nhân quả",
            all_numeric,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            index=all_numeric.index("duration") if "duration" in all_numeric else 0,
        )
        confounders = col_c.multiselect(
            "Confounders (X) — biến kiểm soát",
            [c for c in all_numeric if c != treatment],
            default=[c for c in all_numeric if c != treatment][:4],
            format_func=lambda x: FEATURE_LABELS.get(x, x),
        )

        if st.button("⚙️ Chạy Double ML", type="primary"):
            if not confounders:
                st.error("Chọn ít nhất 1 confounder.")
            else:
                with st.spinner("Đang chạy Double ML..."):
                    dml = DoubleMLinear()
                    dml.fit(df_full, treatment=treatment,
                            outcome="default", confounders=confounders)
                    summary = dml.summary()
                    st.session_state.dml_results = summary

        if st.session_state.dml_results:
            r = st.session_state.dml_results
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Causal Effect (θ)", f"{r['theta']:+.5f}")
            c2.metric("Standard Error",    f"{r['theta_se']:.5f}")
            c3.metric("p-value",           f"{r['p_value']:.4f}",
                      delta="Có ý nghĩa" if r["significant"] else "Không ý nghĩa",
                      delta_color="normal" if r["significant"] else "off")

            st.info(r["interpretation"])

            # Confidence interval
            lo = r["theta"] - 1.96 * r["theta_se"]
            hi = r["theta"] + 1.96 * r["theta_se"]
            st.markdown(f"**95% CI:** [{lo:.5f}, {hi:.5f}]")

            # Visualize
            fig_dml, ax_dml = plt.subplots(figsize=(7, 2))
            ax_dml.barh([r["treatment"]], [r["theta"]],
                        xerr=[1.96 * r["theta_se"]],
                        color='#C44E52' if r["theta"] > 0 else '#2ca02c',
                        alpha=0.8, capsize=5)
            ax_dml.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax_dml.set_xlabel("Causal Effect on P(Default)")
            ax_dml.set_title(f"Double ML: Tác động nhân quả của "
                             f"{FEATURE_LABELS.get(r['treatment'], r['treatment'])}", fontsize=10)
            ax_dml.set_yticklabels(
                [FEATURE_LABELS.get(r["treatment"], r["treatment"])], fontsize=9
            )
            plt.tight_layout()
            st.pyplot(fig_dml, use_container_width=True)
            plt.close(fig_dml)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — Fairness Audit
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("5. Fairness Audit — Kiểm tra phân biệt đối xử")

    if st.session_state.scorer is None:
        st.warning("Vui lòng huấn luyện mô hình trước.")
    else:
        scorer = st.session_state.scorer
        X, y   = st.session_state.X, st.session_state.y
        auditor = FairnessAuditor(scorer)

        st.markdown("""
        Kiểm tra mô hình có phân biệt đối xử theo **tuổi** hoặc **tình trạng cá nhân** không.

        | Metric | Ý nghĩa | Ngưỡng công bằng |
        |--------|---------|-----------------|
        | **Demographic Parity** | Tỷ lệ chấp thuận bằng nhau giữa các nhóm | Chênh lệch < 5% |
        | **Equal Opportunity** | TPR (True Positive Rate) bằng nhau | Chênh lệch < 5% |
        | **Average Odds** | Cả TPR và FPR bằng nhau | Chênh lệch < 5% |
        """)

        protected_col = st.selectbox(
            "Thuộc tính bảo vệ",
            ["age", "personal_status"],
            format_func=lambda x: FEATURE_LABELS.get(x, x),
        )

        if st.button("🛡️ Chạy Fairness Audit", type="primary"):
            with st.spinner("Đang kiểm tra fairness..."):
                result = auditor.audit(X, y, protected_col=protected_col)
                st.session_state.fairness_results = result

        if st.session_state.fairness_results:
            r = st.session_state.fairness_results

            st.divider()
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"**Nhóm A: {r['group_a_label']}** (n={r['group_a']['n']})")
                st.metric("Tỷ lệ chấp thuận", f"{r['group_a']['approval_rate']:.1%}")
                st.metric("TPR (Recall)", f"{r['group_a']['tpr']:.1%}")
                st.metric("AUC",         str(r['group_a']['auc']))

            with col_b:
                st.markdown(f"**Nhóm B: {r['group_b_label']}** (n={r['group_b']['n']})")
                st.metric("Tỷ lệ chấp thuận", f"{r['group_b']['approval_rate']:.1%}")
                st.metric("TPR (Recall)", f"{r['group_b']['tpr']:.1%}")
                st.metric("AUC",         str(r['group_b']['auc']))

            st.divider()
            st.markdown("**Kết quả kiểm tra**")

            metrics_fairness = [
                ("Demographic Parity",  r["dp_diff"], r["dp_verdict"]),
                ("Equal Opportunity",   r["eo_diff"], r["eo_verdict"]),
                ("Average Odds",        r["ao_diff"], r["ao_verdict"]),
            ]
            for name, diff, verdict in metrics_fairness:
                col_n, col_d, col_v = st.columns([2, 1, 2])
                col_n.markdown(f"**{name}**")
                col_d.metric("Chênh lệch", f"{diff:.1%}")
                col_v.markdown(f"**{verdict}**")

            # Bar chart so sánh approval rate
            fig_fair, ax_fair = plt.subplots(figsize=(8, 4))
            groups  = [r["group_a_label"], r["group_b_label"]]
            metrics = ["approval_rate", "tpr", "fpr"]
            labels  = ["Approval Rate", "TPR", "FPR"]
            x = np.arange(len(groups))
            w = 0.25
            for k, (m, label) in enumerate(zip(metrics, labels)):
                vals = [r["group_a"][m], r["group_b"][m]]
                ax_fair.bar(x + k * w, vals, w, label=label, alpha=0.8)
            ax_fair.set_xticks(x + w)
            ax_fair.set_xticklabels(groups, fontsize=9)
            ax_fair.set_ylim(0, 1.1)
            ax_fair.set_ylabel("Tỷ lệ")
            ax_fair.set_title(f"Fairness Metrics theo {FEATURE_LABELS.get(r['protected_col'], r['protected_col'])}")
            ax_fair.legend()
            ax_fair.axhline(0.8, color='gray', linestyle=':', linewidth=1,
                            label='80% rule threshold')
            ax_fair.grid(True, linestyle=':', alpha=0.4, axis='y')
            plt.tight_layout()
            st.pyplot(fig_fair, use_container_width=True)
            plt.close(fig_fair)

            st.caption(
                "Quy tắc 80%: nếu tỷ lệ chấp thuận nhóm thiểu số < 80% tỷ lệ nhóm đa số "
                "→ có dấu hiệu phân biệt đối xử theo EEOC (US Equal Employment Opportunity Commission)."
            )
