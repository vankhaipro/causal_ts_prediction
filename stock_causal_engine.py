"""
Stock Causal Engine
-------------------
Áp dụng causal ML vào dữ liệu chứng khoán VN:

1. CounterfactualStock  — "Nếu SP500 hôm qua +3%, VN-Index hôm nay thay đổi bao nhiêu?"
2. DoubleMLinear        — Hiệu ứng nhân quả thuần của macro lên VNINDEX_Return
3. ScenarioAnalyzer     — Mô phỏng nhiều kịch bản thị trường cùng lúc
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy import stats


# ------------------------------------------------------------------
# Nhãn tiếng Việt cho từng biến
# ------------------------------------------------------------------
FEATURE_LABELS = {
    "SP500":          "S&P 500 (thị trường Mỹ)",
    "VIX":            "VIX (chỉ số sợ hãi toàn cầu)",
    "OIL":            "Dầu Brent",
    "GOLD":           "Giá vàng",
    "DXY":            "US Dollar Index",
    "VCB":            "Vietcombank",
    "BID":            "BIDV",
    "VIC":            "Vingroup",
    "HPG":            "Hoa Phát Group",
    "MSN":            "Masan Group",
    "MWG":            "Mobile World",
    "GAS":            "PetroVietnam Gas",
    "VNINDEX_Return": "VN-Index Return",
}

MACRO_COLS = ["SP500", "VIX", "OIL", "GOLD", "DXY"]
STOCK_COLS = ["VCB", "BID", "VIC", "HPG", "MSN", "MWG", "GAS"]

# Quan hệ nhân quả có thể lan truyền khi can thiệp
# Khi SP500 thay đổi → VIX thường thay đổi ngược chiều
CAUSAL_PROPAGATION = {
    "SP500": {
        "VIX":  lambda d, x: x * (1 - d * 0.5),   # SP500 tăng → VIX giảm
        "DXY":  lambda d, x: x * (1 + d * 0.1),   # SP500 tăng nhẹ → DXY tăng
    },
    "OIL": {
        "GAS":  lambda d, x: x * (1 + d * 0.6),   # Dầu tăng → GAS tăng
        "HPG":  lambda d, x: x * (1 - d * 0.2),   # Dầu tăng → HPG giảm nhẹ (chi phí)
    },
    "VIX": {
        "SP500": lambda d, x: x * (1 - d * 0.4),  # VIX tăng → SP500 giảm
        "GOLD":  lambda d, x: x * (1 + d * 0.3),  # VIX tăng → vàng tăng (trú ẩn)
    },
    "DXY": {
        "GOLD":  lambda d, x: x * (1 - d * 0.5),  # DXY tăng → vàng giảm
        "OIL":   lambda d, x: x * (1 - d * 0.3),  # DXY tăng → dầu giảm
    },
}


# ------------------------------------------------------------------
# Model dự báo VN-Index
# ------------------------------------------------------------------

class VNIndexForecaster:
    """
    GradientBoosting Regressor dự báo VNINDEX_Return.
    Dùng lag-1 của features (X_{t-1} → Y_t).
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance_df = None
        self.is_fitted = False

    def _prepare(self, df: pd.DataFrame):
        feat_cols = [c for c in df.columns if c != "VNINDEX_Return"]
        df_lag = df[feat_cols].shift(1).copy()
        df_lag["VNINDEX_Return"] = df["VNINDEX_Return"]
        df_lag = df_lag.dropna()
        X = df_lag[feat_cols].values
        y = df_lag["VNINDEX_Return"].values
        return X, y, feat_cols

    def fit(self, df: pd.DataFrame):
        X, y, feat_cols = self._prepare(df)
        self.feature_names = feat_cols
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)
        self.is_fitted = True

        result = permutation_importance(
            self.model, X_s, y, n_repeats=10, random_state=42
        )
        self.feature_importance_df = pd.DataFrame({
            "Feature":    self.feature_names,
            "Importance": result.importances_mean,
            "Std":        result.importances_std,
            "Label":      [FEATURE_LABELS.get(f, f) for f in self.feature_names],
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return self

    def predict(self, X_row: pd.DataFrame) -> float:
        X_s = self.scaler.transform(X_row[self.feature_names])
        return float(self.model.predict(X_s)[0])

    def evaluate(self, df: pd.DataFrame) -> dict:
        X, y, _ = self._prepare(df)
        X_s = self.scaler.transform(X)
        y_pred = self.model.predict(X_s)
        da = float(np.mean(np.sign(y) == np.sign(y_pred)))
        return {
            "mse": mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "da":  da,
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        }

    def predict_from_dict(self, feature_dict: dict) -> float:
        row = pd.DataFrame([{f: feature_dict.get(f, 0) for f in self.feature_names}])
        return self.predict(row)


# ------------------------------------------------------------------
# 1. Counterfactual Stock
# ------------------------------------------------------------------

class CounterfactualStock:
    """
    Trả lời: "Nếu [macro variable] thay đổi X%, VN-Index hôm nay thay đổi bao nhiêu?"

    Cách hoạt động:
    - Lấy dữ liệu ngày T-1 làm baseline (features lag-1)
    - Thay đổi giá trị của một feature
    - Áp dụng causal propagation (SP500 tăng → VIX giảm, v.v.)
    - Tái dự báo VNINDEX_Return
    """

    def __init__(self, forecaster: VNIndexForecaster):
        self.forecaster = forecaster

    def what_if(self, features_yesterday: dict,
                feature: str,
                delta_pct: float,
                causal: bool = True) -> dict:
        """
        Args:
            features_yesterday : dict giá trị log return của features hôm qua
            feature            : tên biến cần thay đổi
            delta_pct          : thay đổi thêm bao nhiêu % (vd: +3.0 = tăng 3%)
            causal             : áp dụng causal propagation hay không

        Returns:
            dict kết quả counterfactual
        """
        original = features_yesterday.copy()
        modified = features_yesterday.copy()

        old_val = original.get(feature, 0.0)
        # Với log return: delta_pct là điểm % thêm vào
        delta = delta_pct / 100.0
        new_val = old_val + delta
        modified[feature] = new_val

        # Causal propagation
        causal_changes = {}
        if causal and feature in CAUSAL_PROPAGATION:
            for affected, fn in CAUSAL_PROPAGATION[feature].items():
                if affected in modified:
                    old_aff = modified[affected]
                    new_aff = fn(delta, old_aff)
                    modified[affected] = new_aff
                    causal_changes[affected] = {
                        "old": round(old_aff, 5),
                        "new": round(new_aff, 5),
                        "delta": round(new_aff - old_aff, 5),
                    }

        pred_original = self.forecaster.predict_from_dict(original)
        pred_modified = self.forecaster.predict_from_dict(modified)

        return {
            "feature":          feature,
            "feature_label":    FEATURE_LABELS.get(feature, feature),
            "old_value":        round(old_val, 5),
            "new_value":        round(new_val, 5),
            "delta_pct":        delta_pct,
            "pred_original":    round(pred_original, 5),
            "pred_modified":    round(pred_modified, 5),
            "pred_delta":       round(pred_modified - pred_original, 5),
            "pred_delta_pct":   round((pred_modified - pred_original) * 100, 3),
            "direction_changed": np.sign(pred_original) != np.sign(pred_modified),
            "causal_changes":   causal_changes,
        }

    def sensitivity_sweep(self, features_yesterday: dict,
                           feature: str,
                           deltas: list = None) -> pd.DataFrame:
        """
        Quét nhiều mức delta để vẽ sensitivity curve.
        Trả về DataFrame: delta_pct → pred_vnindex
        """
        if deltas is None:
            deltas = np.arange(-5, 5.5, 0.5).tolist()
        rows = []
        for d in deltas:
            r = self.what_if(features_yesterday, feature, d, causal=True)
            rows.append({
                "delta_pct":    d,
                "pred_vnindex": r["pred_modified"],
            })
        return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 2. Double ML (continuous outcome)
# ------------------------------------------------------------------

class DoubleMLinear:
    """
    Double Machine Learning cho outcome liên tục (log return).

    Bước 1: ˜Y = Y - E[Y|X]  (partial out confounders khỏi Y)
    Bước 2: ˜T = T - E[T|X]  (partial out confounders khỏi T)
    Bước 3: θ = cov(˜Y, ˜T) / var(˜T)  → hiệu ứng nhân quả thuần

    Ví dụ:
        T = SP500_Return (treatment)
        Y = VNINDEX_Return (outcome)
        X = VIX, OIL, GOLD, DXY (confounders)
        θ = "1% tăng SP500 → VN-Index thay đổi θ%"
    """

    def __init__(self):
        self.theta    = None
        self.theta_se = None
        self.p_value  = None
        self.t_stat   = None
        self.treatment = None
        self.outcome   = None
        self.confounders = None
        self.r2_T = None
        self.r2_Y = None

    def fit(self, df: pd.DataFrame,
            treatment: str,
            outcome: str,
            confounders: list) -> "DoubleMLinear":

        self.treatment   = treatment
        self.outcome     = outcome
        self.confounders = confounders

        sub = df[[treatment, outcome] + confounders].dropna()
        X = sub[confounders].values
        T = sub[treatment].values.astype(float)
        Y = sub[outcome].values.astype(float)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # Stage 1a: E[Y|X] — dùng GBM
        m_y = GradientBoostingRegressor(n_estimators=100, random_state=42)
        m_y.fit(X_s, Y)
        Y_res = Y - m_y.predict(X_s)
        self.r2_Y = m_y.score(X_s, Y)

        # Stage 1b: E[T|X] — dùng Ridge
        m_t = Ridge(alpha=1.0)
        m_t.fit(X_s, T)
        T_res = T - m_t.predict(X_s)
        self.r2_T = m_t.score(X_s, T)

        # Stage 2: theta = cov / var
        n = len(Y)
        self.theta = np.dot(T_res, Y_res) / (np.dot(T_res, T_res) + 1e-12)

        resid  = Y_res - self.theta * T_res
        sigma2 = np.mean(resid ** 2)
        var_T  = np.mean(T_res ** 2)
        self.theta_se = np.sqrt(sigma2 / (n * var_T + 1e-12))
        self.t_stat   = self.theta / (self.theta_se + 1e-12)
        self.p_value  = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        return self

    def summary(self) -> dict:
        ci_lo = self.theta - 1.96 * self.theta_se
        ci_hi = self.theta + 1.96 * self.theta_se
        return {
            "treatment":     self.treatment,
            "treatment_label": FEATURE_LABELS.get(self.treatment, self.treatment),
            "outcome":       self.outcome,
            "theta":         round(self.theta, 6),
            "theta_se":      round(self.theta_se, 6),
            "t_stat":        round(self.t_stat, 3),
            "p_value":       round(self.p_value, 5),
            "ci_lo":         round(ci_lo, 6),
            "ci_hi":         round(ci_hi, 6),
            "significant":   self.p_value < 0.05,
            "r2_T":          round(self.r2_T, 4),
            "r2_Y":          round(self.r2_Y, 4),
            "interpretation": (
                f"Tăng **{FEATURE_LABELS.get(self.treatment, self.treatment)}** thêm **1%** "
                f"làm VNINDEX_Return thay đổi **{self.theta*100:+.4f}%** "
                f"({'có ý nghĩa thống kê' if self.p_value < 0.05 else 'không có ý nghĩa'}, "
                f"p={self.p_value:.4f})"
            ),
        }


# ------------------------------------------------------------------
# 3. Scenario Analyzer
# ------------------------------------------------------------------

class ScenarioAnalyzer:
    """
    Mô phỏng kịch bản thị trường: thay đổi nhiều biến cùng lúc.

    Ví dụ:
        Kịch bản "Khủng hoảng": SP500 -3%, VIX +20%, OIL -5%
        Kịch bản "Tăng trưởng": SP500 +2%, VIX -10%, GOLD -1%
    """

    PRESET_SCENARIOS = {
        "📈 Thị trường tích cực": {
            "SP500": +2.0, "VIX": -15.0, "OIL": +1.0, "GOLD": -0.5, "DXY": +0.5
        },
        "📉 Thị trường tiêu cực": {
            "SP500": -2.0, "VIX": +20.0, "OIL": -2.0, "GOLD": +1.5, "DXY": +1.0
        },
        "💥 Khủng hoảng": {
            "SP500": -5.0, "VIX": +50.0, "OIL": -10.0, "GOLD": +3.0, "DXY": +2.0
        },
        "🚀 Bull run": {
            "SP500": +4.0, "VIX": -25.0, "OIL": +3.0, "GOLD": +0.5, "DXY": -1.0
        },
        "🛢️ Dầu tăng mạnh": {
            "SP500": +0.5, "VIX": +5.0, "OIL": +8.0, "GOLD": +1.0, "DXY": +0.3
        },
        "💵 USD mạnh": {
            "SP500": +0.5, "VIX": +5.0, "OIL": -3.0, "GOLD": -2.0, "DXY": +2.0
        },
    }

    def __init__(self, forecaster: VNIndexForecaster,
                 cf_engine: CounterfactualStock):
        self.forecaster = forecaster
        self.cf_engine  = cf_engine

    def run_scenario(self, features_yesterday: dict,
                     scenario_deltas: dict) -> dict:
        """
        Áp dụng nhiều can thiệp cùng lúc (không causal propagation để tránh double-counting).
        """
        modified = features_yesterday.copy()
        changes = {}
        for feat, delta_pct in scenario_deltas.items():
            if feat in modified:
                old = modified[feat]
                modified[feat] = old + delta_pct / 100.0
                changes[feat] = {
                    "delta_pct": delta_pct,
                    "old": round(old, 5),
                    "new": round(modified[feat], 5),
                    "label": FEATURE_LABELS.get(feat, feat),
                }

        pred_orig = self.forecaster.predict_from_dict(features_yesterday)
        pred_new  = self.forecaster.predict_from_dict(modified)

        direction = "📈 TĂNG" if pred_new >= 0 else "📉 GIẢM"

        return {
            "pred_original":    round(pred_orig, 5),
            "pred_modified":    round(pred_new, 5),
            "pred_delta":       round(pred_new - pred_orig, 5),
            "pred_delta_pct":   round((pred_new - pred_orig) * 100, 3),
            "direction":        direction,
            "changes":          changes,
        }

    def compare_all_presets(self, features_yesterday: dict) -> pd.DataFrame:
        rows = []
        for name, deltas in self.PRESET_SCENARIOS.items():
            r = self.run_scenario(features_yesterday, deltas)
            rows.append({
                "Kịch bản": name,
                "VN-Index dự báo": f"{r['pred_modified']*100:+.3f}%",
                "Thay đổi so với baseline": f"{r['pred_delta']*100:+.3f}%",
                "Xu hướng": r["direction"],
                "_pred": r["pred_modified"],
            })
        return pd.DataFrame(rows).sort_values("_pred", ascending=False).drop(columns="_pred")
