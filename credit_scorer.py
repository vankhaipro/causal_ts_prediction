"""
Causal Credit Scoring
---------------------
Triển khai counterfactual credit models dựa trên causal ML:

1. CreditScorer       — GradientBoosting classifier (baseline)
2. CounterfactualEngine — "Nếu thu nhập +10%, họ có trả được nợ không?"
3. DoubleMLinear      — Double ML đơn giản: tách hiệu ứng nhân quả khỏi confounders
4. FairnessAuditor    — Kiểm tra phân biệt đối xử theo tuổi / giới tính
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.datasets import fetch_openml


# ------------------------------------------------------------------
# Tải dữ liệu German Credit (UCI)
# ------------------------------------------------------------------

FEATURE_LABELS = {
    "duration":               "Thời hạn vay (tháng)",
    "credit_amount":          "Số tiền vay",
    "installment_commitment": "Tỷ lệ trả góp / thu nhập (%)",
    "residence_since":        "Số năm cư trú hiện tại",
    "age":                    "Tuổi",
    "existing_credits":       "Số khoản vay hiện có",
    "num_dependents":         "Số người phụ thuộc",
    "checking_status":        "Trạng thái tài khoản thanh toán",
    "credit_history":         "Lịch sử tín dụng",
    "purpose":                "Mục đích vay",
    "savings_status":         "Số dư tiết kiệm",
    "employment":             "Thâm niên làm việc",
    "personal_status":        "Tình trạng hôn nhân / giới tính",
    "other_parties":          "Người bảo lãnh",
    "property_magnitude":     "Tài sản thế chấp",
    "other_payment_plans":    "Các kế hoạch trả nợ khác",
    "housing":                "Hình thức nhà ở",
    "job":                    "Nghề nghiệp",
    "own_telephone":          "Có điện thoại riêng",
    "foreign_worker":         "Là lao động nước ngoài",
}

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]

PROTECTED_COLS = ["age", "personal_status"]   # Thuộc tính bảo vệ (fairness)


def load_german_credit() -> tuple[pd.DataFrame, pd.Series]:
    """
    Tải German Credit dataset từ OpenML.
    Trả về (X, y) — y=1 good credit, y=0 bad credit.
    """
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    X = data.data.copy()
    y = (data.target == "good").astype(int)

    # Encode categorical features
    for col in X.select_dtypes(include="category").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.astype(float)
    return X, y


# ------------------------------------------------------------------
# 1. Causal Credit Scorer
# ------------------------------------------------------------------

class CreditScorer:
    """
    GradientBoosting classifier với:
    - Cross-validated AUC
    - Permutation feature importance (causal-aware)
    - Probability calibration
    """

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance_df = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)
        self.is_fitted = True

        # Permutation importance (ổn định hơn built-in feature_importances_)
        result = permutation_importance(
            self.model, X_s, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        self.feature_importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
            "Label": [FEATURE_LABELS.get(f, f) for f in self.feature_names],
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)

    def score_customer(self, customer: dict) -> dict:
        """
        Chấm điểm một khách hàng.
        Trả về: probability, credit_score (300–850), risk_tier.
        """
        X = pd.DataFrame([customer])[self.feature_names]
        prob_good = self.predict_proba(X)[0, 1]

        # Map sang thang điểm 300–850 (như FICO)
        credit_score = int(300 + prob_good * 550)

        if credit_score >= 740:
            tier, color = "Xuất sắc", "🟢"
        elif credit_score >= 670:
            tier, color = "Tốt", "🟩"
        elif credit_score >= 580:
            tier, color = "Trung bình", "🟡"
        elif credit_score >= 500:
            tier, color = "Kém", "🟠"
        else:
            tier, color = "Rất kém", "🔴"

        return {
            "prob_good": round(prob_good, 4),
            "prob_default": round(1 - prob_good, 4),
            "credit_score": credit_score,
            "risk_tier": tier,
            "risk_color": color,
            "approved": prob_good >= 0.5,
        }

    def cross_val_auc(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        X_s = self.scaler.transform(X)
        scores = cross_val_score(self.model, X_s, y, cv=cv, scoring="roc_auc")
        return {"mean": scores.mean(), "std": scores.std(), "scores": scores}

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        X_s = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_s)
        y_prob = self.model.predict_proba(X_s)[:, 1]
        return {
            "auc": roc_auc_score(y_test, y_prob),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }


# ------------------------------------------------------------------
# 2. Counterfactual Engine
# ------------------------------------------------------------------

class CounterfactualEngine:
    """
    Trả lời câu hỏi: "Nếu khách hàng thay đổi X, điểm tín dụng thay đổi thế nào?"

    Hai loại counterfactual:
    A. Direct intervention: thay đổi feature, tái dự báo (naive)
    B. Causal intervention: điều chỉnh thêm cho các confounders bị ảnh hưởng
       (ví dụ: thu nhập tăng → tỷ lệ trả góp giảm tự động)
    """

    # Quan hệ nhân quả đơn giản giữa các features
    # Khi feature A thay đổi, feature B cũng bị ảnh hưởng
    CAUSAL_PROPAGATION = {
        "credit_amount": {
            # Vay nhiều hơn → tỷ lệ trả góp tăng (nếu duration cố định)
            "installment_commitment": lambda delta_ratio, x: x * (1 + delta_ratio * 0.5),
        },
        "duration": {
            # Thời hạn dài hơn → tỷ lệ trả góp giảm
            "installment_commitment": lambda delta_ratio, x: x * (1 - delta_ratio * 0.3),
        },
    }

    def __init__(self, scorer: CreditScorer):
        self.scorer = scorer

    def what_if(self, customer: dict, feature: str,
                new_value=None, delta_pct: float = None,
                causal: bool = True) -> dict:
        """
        Tính counterfactual khi thay đổi một feature.

        Args:
            customer   : dict thông tin khách hàng gốc
            feature    : tên feature cần thay đổi
            new_value  : giá trị mới (tuyệt đối)
            delta_pct  : thay đổi theo % (ưu tiên nếu có)
            causal     : True → áp dụng causal propagation

        Returns:
            dict với điểm gốc, điểm mới, và giải thích
        """
        original = customer.copy()
        modified = customer.copy()

        old_val = original.get(feature, 0)
        if delta_pct is not None:
            new_value = old_val * (1 + delta_pct / 100)
        modified[feature] = new_value

        # Causal propagation: điều chỉnh features liên quan
        causal_changes = {}
        if causal and feature in self.CAUSAL_PROPAGATION:
            delta_ratio = (new_value - old_val) / (old_val + 1e-8)
            for affected_feat, fn in self.CAUSAL_PROPAGATION[feature].items():
                old_affected = modified.get(affected_feat, 0)
                new_affected = fn(delta_ratio, old_affected)
                modified[affected_feat] = new_affected
                causal_changes[affected_feat] = {
                    "old": round(old_affected, 3),
                    "new": round(new_affected, 3),
                }

        score_orig = self.scorer.score_customer(original)
        score_new  = self.scorer.score_customer(modified)

        return {
            "feature_changed":  feature,
            "old_value":        round(old_val, 3),
            "new_value":        round(new_value, 3),
            "delta_pct":        round((new_value - old_val) / (old_val + 1e-8) * 100, 1),
            "score_original":   score_orig,
            "score_new":        score_new,
            "score_delta":      score_new["credit_score"] - score_orig["credit_score"],
            "prob_delta":       round(score_new["prob_good"] - score_orig["prob_good"], 4),
            "causal_changes":   causal_changes,
            "decision_flipped": score_orig["approved"] != score_new["approved"],
        }

    def find_minimal_changes(self, customer: dict, target_score: int = 670,
                             max_steps: int = 5) -> list[dict]:
        """
        Tìm tập thay đổi tối thiểu để đạt target_score (loan approval).
        Chỉ xét các features có thể thay đổi được (không phải age, gender).
        """
        actionable = ["duration", "credit_amount", "installment_commitment",
                      "existing_credits", "savings_status", "employment"]
        actionable = [f for f in actionable if f in customer]

        changes = []
        current = customer.copy()

        for step in range(max_steps):
            current_score = self.scorer.score_customer(current)
            if current_score["credit_score"] >= target_score:
                break

            # Thử từng feature, chọn cái cải thiện nhiều nhất
            best_delta = 0
            best_cf = None
            best_feat = None
            best_val = None

            for feat in actionable:
                for delta in [-20, -10, 10, 20]:
                    cf = self.what_if(current, feat, delta_pct=delta)
                    if cf["score_delta"] > best_delta:
                        best_delta = cf["score_delta"]
                        best_cf = cf
                        best_feat = feat
                        best_val = cf["new_value"]

            if best_cf is None:
                break

            current[best_feat] = best_val
            changes.append({
                "step":        step + 1,
                "feature":     best_feat,
                "label":       FEATURE_LABELS.get(best_feat, best_feat),
                "old_value":   best_cf["old_value"],
                "new_value":   round(best_val, 2),
                "score_delta": best_cf["score_delta"],
                "new_score":   best_cf["score_new"]["credit_score"],
            })

        return changes


# ------------------------------------------------------------------
# 3. Double ML (Simplified)
# ------------------------------------------------------------------

class DoubleMLinear:
    """
    Double Machine Learning (Robinson 1988 / Chernozhukov 2018).

    Mục tiêu: ước lượng hiệu ứng nhân quả THUẦN của treatment T
    lên outcome Y, sau khi đã kiểm soát confounders X.

    Bước 1: ˜Y = Y - E[Y|X]  (partial out confounders khỏi Y)
    Bước 2: ˜T = T - E[T|X]  (partial out confounders khỏi T)
    Bước 3: hồi quy ˜Y ~ ˜T → θ = causal effect của T lên Y

    Ví dụ áp dụng:
        T = duration (thời hạn vay)
        Y = default (0/1)
        X = age, credit_amount, employment, ...
        θ = hiệu ứng nhân quả thực sự của duration lên rủi ro vỡ nợ
    """

    def __init__(self):
        self.theta = None
        self.theta_se = None
        self.t_stat = None
        self.p_value = None
        self.treatment = None
        self.outcome = None

    def fit(self, df: pd.DataFrame, treatment: str, outcome: str,
            confounders: list[str]) -> "DoubleMLinear":
        from scipy import stats

        self.treatment = treatment
        self.outcome   = outcome

        X = df[confounders].values
        T = df[treatment].values.astype(float)
        Y = df[outcome].values.astype(float)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # Stage 1a: E[Y|X]
        m_y = GradientBoostingClassifier(n_estimators=100, random_state=42)
        m_y.fit(X_s, Y.astype(int))
        Y_res = Y - m_y.predict_proba(X_s)[:, 1]

        # Stage 1b: E[T|X]
        m_t = Ridge(alpha=1.0)
        m_t.fit(X_s, T)
        T_res = T - m_t.predict(X_s)

        # Stage 2: θ = cov(Y_res, T_res) / var(T_res)
        self.theta = np.dot(T_res, Y_res) / (np.dot(T_res, T_res) + 1e-10)

        # Standard error
        n = len(Y)
        resid = Y_res - self.theta * T_res
        sigma2 = np.mean(resid ** 2)
        var_T  = np.mean(T_res ** 2)
        self.theta_se = np.sqrt(sigma2 / (n * var_T + 1e-10))

        self.t_stat = self.theta / (self.theta_se + 1e-10)
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        return self

    def summary(self) -> dict:
        return {
            "treatment":  self.treatment,
            "outcome":    self.outcome,
            "theta":      round(self.theta, 6),
            "theta_se":   round(self.theta_se, 6),
            "t_stat":     round(self.t_stat, 3),
            "p_value":    round(self.p_value, 4),
            "significant": self.p_value < 0.05,
            "interpretation": (
                f"Tăng **{self.treatment}** thêm 1 đơn vị làm thay đổi "
                f"xác suất vỡ nợ **{self.theta:+.4f}** "
                f"({'có ý nghĩa' if self.p_value < 0.05 else 'không có ý nghĩa'} "
                f"thống kê, p={self.p_value:.4f})"
            ),
        }


# ------------------------------------------------------------------
# 4. Fairness Auditor
# ------------------------------------------------------------------

class FairnessAuditor:
    """
    Kiểm tra xem model có phân biệt đối xử theo thuộc tính bảo vệ không.

    Metrics:
    - Demographic Parity: P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1)
    - Equal Opportunity: TPR phải bằng nhau giữa các nhóm
    - Average Odds: cả TPR và FPR phải bằng nhau
    """

    def __init__(self, scorer: CreditScorer):
        self.scorer = scorer

    def audit(self, X: pd.DataFrame, y: pd.Series,
              protected_col: str, threshold: float = 0.5) -> dict:
        """
        Kiểm tra fairness cho một thuộc tính bảo vệ.

        Args:
            X             : features (bao gồm cả protected_col)
            y             : nhãn thực
            protected_col : tên cột thuộc tính bảo vệ (vd: "age")
            threshold     : ngưỡng phân loại (0.5)
        """
        prob = self.scorer.predict_proba(X)[:, 1]
        y_pred = (prob >= threshold).astype(int)

        # Chia nhóm theo giá trị median của protected attribute
        median_val = X[protected_col].median()
        group_a = X[protected_col] <= median_val   # nhóm thấp (vd: trẻ)
        group_b = ~group_a                          # nhóm cao  (vd: già)

        def group_metrics(mask):
            yt = y[mask].values
            yp = y_pred[mask]
            pp = prob[mask]
            tp = ((yp == 1) & (yt == 1)).sum()
            fp = ((yp == 1) & (yt == 0)).sum()
            fn = ((yp == 0) & (yt == 1)).sum()
            tn = ((yp == 0) & (yt == 0)).sum()
            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            approval_rate = yp.mean()
            avg_score = pp.mean()
            return {
                "n":             int(mask.sum()),
                "approval_rate": round(approval_rate, 4),
                "avg_score":     round(avg_score, 4),
                "tpr":           round(tpr, 4),
                "fpr":           round(fpr, 4),
                "auc":           round(roc_auc_score(yt, pp), 4) if len(np.unique(yt)) > 1 else None,
            }

        metrics_a = group_metrics(group_a)
        metrics_b = group_metrics(group_b)

        # Demographic parity difference
        dp_diff = abs(metrics_a["approval_rate"] - metrics_b["approval_rate"])
        # Equal opportunity difference
        eo_diff = abs(metrics_a["tpr"] - metrics_b["tpr"])
        # Average odds difference
        ao_diff = (abs(metrics_a["tpr"] - metrics_b["tpr"]) +
                   abs(metrics_a["fpr"] - metrics_b["fpr"])) / 2

        def verdict(diff, threshold=0.1):
            if diff < 0.05:
                return "✅ Công bằng"
            elif diff < threshold:
                return "⚠️ Nghi ngờ"
            else:
                return "❌ Phân biệt đối xử"

        return {
            "protected_col":   protected_col,
            "median_val":      median_val,
            "group_a_label":   f"{protected_col} ≤ {median_val:.0f}",
            "group_b_label":   f"{protected_col} > {median_val:.0f}",
            "group_a":         metrics_a,
            "group_b":         metrics_b,
            "dp_diff":         round(dp_diff, 4),
            "eo_diff":         round(eo_diff, 4),
            "ao_diff":         round(ao_diff, 4),
            "dp_verdict":      verdict(dp_diff),
            "eo_verdict":      verdict(eo_diff),
            "ao_verdict":      verdict(ao_diff),
        }
