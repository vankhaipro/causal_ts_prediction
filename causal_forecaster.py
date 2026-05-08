import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ------------------------------------------------------------------
# LSTM Model (PyTorch)
# ------------------------------------------------------------------

class _LSTMNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ------------------------------------------------------------------
# CausalForecaster
# ------------------------------------------------------------------

class CausalForecaster:
    def __init__(self, target_col='VNINDEX_Return'):
        self.target_col = target_col
        self.pcmci_results = None
        self.selected_features = []
        self.feature_names = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def check_stationarity(self, df, significance=0.05):
        """ADF test — returns list of non-stationary column names."""
        non_stationary = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                continue
            result = adfuller(df[col].dropna())
            if result[1] > significance:
                non_stationary.append(col)
        return non_stationary

    def reduce_dimensions(self, df, method='lasso', n_components=10):
        """Feature selection (LASSO) or compression (PCA)."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == 'lasso':
            lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
            coef = pd.Series(lasso.coef_, index=X.columns)
            selected = coef[coef != 0].index.tolist()
            print(f"  LASSO selected {len(selected)}/{len(X.columns)} features: {selected}")
            if not selected:
                print("  LASSO chọn 0 features, giữ nguyên tất cả.")
                return df
            return df[selected + [self.target_col]]

        elif method == 'pca':
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)
            cols = [f'PC{i+1}' for i in range(n_components)]
            reduced_df = pd.DataFrame(X_reduced, columns=cols, index=df.index)
            reduced_df[self.target_col] = y.values
            return reduced_df

        return df

    # ------------------------------------------------------------------
    # Causal Discovery — VAR-LiNGAM (Paper 2)
    # ------------------------------------------------------------------

    def perform_causal_discovery_lingam(self, df, lags=3):
        """
        Chạy VAR-LiNGAM để tìm causal parents của target.

        Khác PCMCI+ (conditional independence, Gaussian):
        - LiNGAM giả định nhiễu Non-Gaussian → identify được cả
          contemporaneous causal order
        - Phù hợp hơn với ít data points (monthly ~100 điểm)
        - Prune=True loại bỏ các link yếu tự động

        Paper 2: Oliveira et al. 2024 — union với PCMCI+ features
        cho feature set ổn định hơn qua các regime thị trường.
        """
        from lingam import VARLiNGAM

        feature_names = df.columns.tolist()
        target_idx    = feature_names.index(self.target_col)

        print(f"  Chạy VAR-LiNGAM (lags={lags}, prune=True)...")
        model = VARLiNGAM(lags=lags, criterion='bic', prune=True)
        model.fit(df.values.astype(float))

        # adjacency_matrices_: list độ dài lags, mỗi phần tử shape (n, n)
        # A[i][j] != 0 → biến j là causal parent của biến i tại lag đó
        lingam_parents = set()
        for lag_idx, A in enumerate(model.adjacency_matrices_):
            lag = lag_idx + 1
            row = A[target_idx]          # hàng target
            for j, coef in enumerate(row):
                if abs(coef) > 1e-10 and feature_names[j] != self.target_col:
                    lingam_parents.add(feature_names[j])
                    print(f"    LiNGAM: {feature_names[j]}(t-{lag}) "
                          f"→ {self.target_col}  coef={coef:.4f}")

        if not lingam_parents:
            print("  VAR-LiNGAM không tìm được causal parents.")

        return list(lingam_parents)

    # ------------------------------------------------------------------
    # Causal Discovery — PCMCI+
    # ------------------------------------------------------------------

    def perform_causal_discovery(self, df, tau_max=3, pc_alpha=0.05):
        """
        Chạy PCMCI+ để học cấu trúc nhân quả có độ trễ.
        tau_min=1 đảm bảo chỉ tìm X(t-tau) → Y(t), tránh look-ahead bias.
        """
        self.feature_names = df.columns.tolist()
        dataframe = pp.DataFrame(
            df.values.astype(float),
            var_names=self.feature_names
        )
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ParCorr(significance='analytic'),
            verbosity=1
        )
        self.pcmci_results = pcmci.run_pcmciplus(
            tau_min=1,
            tau_max=tau_max,
            pc_alpha=pc_alpha
        )
        print(f"\nPCMCI+ hoàn thành. Graph shape: {self.pcmci_results['graph'].shape}")
        return self.pcmci_results

    def extract_features(self):
        """Trích xuất features có cạnh nhân quả --> target."""
        if self.pcmci_results is None:
            raise ValueError("Chạy perform_causal_discovery trước.")

        graph = self.pcmci_results['graph']
        target_idx = self.feature_names.index(self.target_col)

        causal_pairs = []
        seen = set()
        for i, feat in enumerate(self.feature_names):
            if feat == self.target_col:
                continue
            for tau in range(graph.shape[2]):
                if graph[i, target_idx, tau] == '-->':
                    causal_pairs.append((feat, tau))
                    if feat not in seen:
                        seen.add(feat)

        self.selected_features = list(seen)
        print(f"Causal features → {self.target_col}: {causal_pairs}")
        return causal_pairs

    def build_causal_graph(self):
        """Trả về NetworkX DiGraph từ kết quả PCMCI+."""
        if self.pcmci_results is None:
            return None
        graph = self.pcmci_results['graph']
        G = nx.DiGraph()
        for feat in self.feature_names:
            G.add_node(feat)
        for i, src in enumerate(self.feature_names):
            for j, dst in enumerate(self.feature_names):
                for tau in range(graph.shape[2]):
                    if graph[i, j, tau] == '-->':
                        if G.has_edge(src, dst):
                            G[src][dst]['label'] += f',τ={tau}'
                        else:
                            G.add_edge(src, dst, label=f'τ={tau}')
        return G

    def visualize_graph(self, freq: str = "monthly"):
        """Vẽ causal graph với nhãn độ trễ (τ)."""
        G = self.build_causal_graph()
        if G is None:
            return

        unit = "ngày" if freq == "daily" else "tháng"
        fig, ax = plt.subplots(figsize=(14, 9))
        pos = nx.spring_layout(G, k=0.8, seed=42)
        colors = ['#FF4B4B' if n == self.target_col else '#1E90FF' for n in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=colors,
                node_size=2500, font_size=9, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray', alpha=0.85, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, 'label'),
                                     font_size=7, ax=ax)
        ax.set_title(f"PCMCI+ Causal Graph — {self.target_col} ({unit})", fontsize=14)
        plt.tight_layout()
        suffix = "_daily" if freq == "daily" else ""
        fname = f"causal_graph{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Đã lưu: {fname}")
        return fig

    # ------------------------------------------------------------------
    # Phase 5 — Interpretation & Visualization
    # ------------------------------------------------------------------

    def plot_effect_heatmap(self, freq: str = "monthly") -> None:
        """
        Heatmap MCI coefficient: (biến nguồn × biến đích).
        Ô có màu = tồn tại liên kết nhân quả có ý nghĩa.
        """
        if self.pcmci_results is None:
            print("Chưa chạy causal discovery.")
            return

        graph   = self.pcmci_results["graph"]
        val_mat = self.pcmci_results["val_matrix"]   # shape (N, N, tau+1)
        names   = self.feature_names
        N       = len(names)

        # Lấy giá trị mạnh nhất trên tất cả độ trễ cho mỗi cặp (i→j)
        effect = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                for tau in range(graph.shape[2]):
                    if graph[i, j, tau] == "-->":
                        if abs(val_mat[i, j, tau]) > abs(effect[i, j]):
                            effect[i, j] = val_mat[i, j, tau]

        unit = "ngày" if freq == "daily" else "tháng"
        fig, ax = plt.subplots(figsize=(max(10, N * 0.7), max(8, N * 0.6)))
        im = ax.imshow(effect.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(N)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Biến nguồn (nguyên nhân)", fontsize=10)
        ax.set_ylabel("Biến đích (kết quả)", fontsize=10)
        ax.set_title(f"Causal Effect Heatmap — MCI Coefficient ({unit})", fontsize=13)
        plt.colorbar(im, ax=ax, label="MCI coefficient")

        # Đánh dấu ô liên quan đến target
        tgt = names.index(self.target_col)
        ax.axhline(tgt - 0.5, color="gold", linewidth=2)
        ax.axhline(tgt + 0.5, color="gold", linewidth=2)

        plt.tight_layout()
        suffix = "_daily" if freq == "daily" else ""
        fname = f"causal_heatmap{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Đã lưu: {fname}")

    def plot_lag_effects(self, freq: str = "monthly") -> None:
        """
        Bar chart hệ số MCI theo từng độ trễ τ cho các liên kết
        X → VNINDEX_Return được tìm thấy bởi PCMCI+.
        """
        if self.pcmci_results is None:
            return

        graph   = self.pcmci_results["graph"]
        val_mat = self.pcmci_results["val_matrix"]
        pval    = self.pcmci_results["p_matrix"]
        names   = self.feature_names
        tgt     = names.index(self.target_col)
        tau_max = graph.shape[2] - 1
        unit    = "ngày" if freq == "daily" else "tháng"

        # Thu thập tất cả nguồn có liên kết → target
        sources = []
        for i, name in enumerate(names):
            if name == self.target_col:
                continue
            for tau in range(graph.shape[2]):
                if graph[i, tgt, tau] == "-->":
                    sources.append(name)
                    break

        if not sources:
            print("Không có causal links → target.")
            return

        fig, axes = plt.subplots(
            len(sources), 1,
            figsize=(8, 3 * len(sources)),
            squeeze=False,
        )

        for row, src in enumerate(sources):
            ax  = axes[row][0]
            idx = names.index(src)
            taus  = list(range(1, tau_max + 1))
            coefs = [val_mat[idx, tgt, t] for t in taus]
            pvals = [pval[idx, tgt, t]    for t in taus]
            colors = ["#C44E52" if p < 0.05 else "#AAAAAA" for p in pvals]

            bars = ax.bar(taus, coefs, color=colors, edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(taus)
            ax.set_xticklabels([f"τ={t}" for t in taus])
            ax.set_ylabel("MCI coef", fontsize=9)
            ax.set_title(
                f"{src} → {self.target_col}   "
                f"(đỏ = p<0.05)",
                fontsize=10,
            )
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        fig.suptitle(
            f"Hệ số nhân quả theo độ trễ τ ({unit}) — PCMCI+",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        suffix = "_daily" if freq == "daily" else ""
        fname = f"causal_lag_effects{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Đã lưu: {fname}")

    def news_impact_report(self) -> 'pd.DataFrame':
        """
        Bảng tóm tắt kết quả nhân quả dạng readable (Phase 5).
        Trả về DataFrame và in ra terminal.

        Columns: Nguyên nhân | Biến đích | Độ trễ | Hệ số MCI | p-value | Chiều tác động
        """
        import pandas as pd

        if self.pcmci_results is None:
            return pd.DataFrame()

        graph   = self.pcmci_results["graph"]
        val_mat = self.pcmci_results["val_matrix"]
        pval    = self.pcmci_results["p_matrix"]
        names   = self.feature_names

        rows = []
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                for tau in range(graph.shape[2]):
                    if graph[i, j, tau] == "-->":
                        coef = val_mat[i, j, tau]
                        p    = pval[i, j, tau]
                        direction = "↑ tăng" if coef > 0 else "↓ giảm"
                        rows.append({
                            "Nguyên nhân":      src,
                            "Biến đích":        dst,
                            "Độ trễ (τ)":       tau,
                            "Hệ số MCI":        round(coef, 4),
                            "p-value":          round(p, 4),
                            "Chiều tác động":   direction,
                            "→ Target":         dst == self.target_col,
                        })

        if not rows:
            print("Không tìm thấy liên kết nhân quả nào.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(
            ["→ Target", "p-value"], ascending=[False, True]
        ).reset_index(drop=True)

        print("\n" + "=" * 70)
        print(f"  NEWS IMPACT REPORT — Kết quả PCMCI+")
        print("=" * 70)

        target_links = df[df["→ Target"]]
        if target_links.empty:
            print(f"  Không có biến nào nhân quả trực tiếp với {self.target_col}.")
        else:
            print(f"\n  Liên kết trực tiếp → {self.target_col}:")
            cols = ["Nguyên nhân", "Độ trễ (τ)", "Hệ số MCI", "p-value", "Chiều tác động"]
            print(target_links[cols].to_string(index=False))

        print(f"\n  Tổng cộng: {len(df)} liên kết nhân quả trong hệ thống.")
        print("=" * 70)
        return df

    # ------------------------------------------------------------------
    # Forecasting — Ridge (Rolling Window)
    # ------------------------------------------------------------------

    def _rolling_forecast(self, df, feature_cols, window_size=60):
        """Rolling window Ridge Regression. Shift-1 được áp dụng nội bộ."""
        df_lag = df[feature_cols].shift(1).copy()
        df_lag[self.target_col] = df[self.target_col]
        df_lag = df_lag.dropna()

        y = df_lag[self.target_col].values
        X = df_lag[feature_cols].values

        predictions, actuals = [], []
        for i in range(window_size, len(df_lag)):
            X_train = X[i - window_size:i]
            y_train = y[i - window_size:i]
            X_test  = X[i:i + 1]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            pred = Ridge(alpha=1.0).fit(X_train_s, y_train).predict(X_test_s)[0]
            predictions.append(pred)
            actuals.append(y[i])

        return np.array(actuals), np.array(predictions)

    # ------------------------------------------------------------------
    # Forecasting — LSTM (Rolling Window, PyTorch)
    # ------------------------------------------------------------------

    def _rolling_forecast_lstm(self, df, feature_cols, window_size=60,
                                seq_len=10, epochs=30, hidden_size=64,
                                num_layers=2, lr=1e-3, verbose=False):
        """
        Rolling window LSTM Regression.

        Mỗi bước: dùng window_size quan sát gần nhất để train LSTM,
        dự báo bước tiếp theo bằng seq_len observations cuối.
        """
        if not TORCH_AVAILABLE:
            print("  [WARNING] torch không khả dụng, bỏ qua LSTM.")
            return None, None

        df_lag = df[feature_cols].shift(1).copy()
        df_lag[self.target_col] = df[self.target_col]
        df_lag = df_lag.dropna()

        y = df_lag[self.target_col].values.astype(np.float32)
        X = df_lag[feature_cols].values.astype(np.float32)
        n_features = X.shape[1]

        def make_sequences(X_w, y_w, seq_len):
            """Tạo sequences (X_seq, y_seq) từ window."""
            xs, ys = [], []
            for k in range(seq_len, len(X_w)):
                xs.append(X_w[k - seq_len:k])
                ys.append(y_w[k])
            return np.array(xs), np.array(ys)

        predictions, actuals = [], []
        device = torch.device('cpu')

        for i in range(window_size, len(df_lag)):
            X_train_raw = X[i - window_size:i]
            y_train_raw = y[i - window_size:i]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_raw).astype(np.float32)
            X_test_s  = scaler.transform(X[i:i + 1]).astype(np.float32)

            # Scale target
            y_mean, y_std = y_train_raw.mean(), y_train_raw.std() + 1e-8
            y_train_s = ((y_train_raw - y_mean) / y_std).astype(np.float32)

            X_seq, y_seq = make_sequences(X_train_s, y_train_s, seq_len)
            if len(X_seq) == 0:
                actuals.append(y[i])
                predictions.append(0.0)
                continue

            # Build last sequence for prediction
            last_seq = np.vstack([X_train_s[-(seq_len - 1):], X_test_s])
            last_seq = torch.tensor(last_seq[np.newaxis], dtype=torch.float32)

            X_t = torch.tensor(X_seq, dtype=torch.float32)
            y_t = torch.tensor(y_seq, dtype=torch.float32)

            model = _LSTMNet(n_features, hidden_size, num_layers).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss_fn(model(X_t), y_t).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                pred_s = model(last_seq).item()

            pred = pred_s * y_std + y_mean
            predictions.append(pred)
            actuals.append(y[i])

            if verbose and (i - window_size) % 50 == 0:
                print(f"    LSTM step {i - window_size}/{len(df_lag) - window_size}")

        return np.array(actuals), np.array(predictions)

    def forecast_next(self, df, feature_cols, window_size=60):
        """Dự báo kỳ tiếp theo bằng Ridge dùng toàn bộ dữ liệu cuối."""
        df_lag = df[feature_cols].shift(1).copy()
        df_lag[self.target_col] = df[self.target_col]
        df_lag = df_lag.dropna()

        X = df_lag[feature_cols].values
        y = df_lag[self.target_col].values

        X_train = X[-window_size:]
        y_train = y[-window_size:]

        # Features cho kỳ tiếp theo = giá trị hiện tại (chưa shift)
        X_next = df[feature_cols].values[-1:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_next_s  = scaler.transform(X_next)

        pred = Ridge(alpha=1.0).fit(X_train_s, y_train).predict(X_next_s)[0]
        return float(pred)

    def predict_all_stocks(self, df: pd.DataFrame, latest_prices: pd.Series,
                           window_size: int = 60,
                           buy_threshold: float = 0.003,
                           sell_threshold: float = -0.003) -> pd.DataFrame:
        """
        Dự báo giá kỳ tiếp theo cho từng cổ phiếu trong dataset.

        Với mỗi cổ phiếu S:
          1. Tìm causal parents của S trong PCMCI+ graph (nếu đã chạy)
             Fallback: dùng tất cả features còn lại
          2. Fit Ridge trên window_size kỳ gần nhất
          3. Dự báo predicted_return → predicted_price = latest_price × exp(ret)
          4. Đưa ra khuyến nghị MUA / BÁN / GIỮ

        Returns: DataFrame với các cột:
            Mã, Giá hiện tại, Giá dự báo, Thay đổi (%), Khuyến nghị, Độ tin cậy
        """
        all_symbols = [c for c in df.columns if c != "VNINDEX_Return"]
        rows = []

        for symbol in all_symbols:
            if symbol not in df.columns:
                continue

            # Tìm causal parents từ graph (nếu có)
            feature_cols = self._get_causal_parents(symbol, df)

            # Dự báo return kỳ tiếp theo
            try:
                pred_return = self._forecast_single(df, symbol, feature_cols, window_size)
            except Exception:
                pred_return = 0.0

            # Giá hiện tại
            current_price = latest_prices.get(symbol, np.nan)
            if np.isnan(current_price) or current_price <= 0:
                continue

            # Giá dự báo
            predicted_price = current_price * np.exp(pred_return)
            change_pct = (predicted_price - current_price) / current_price * 100

            # Khuyến nghị
            if pred_return >= buy_threshold:
                signal = "🟢 MUA"
                confidence = min(abs(pred_return) / (buy_threshold * 3), 1.0)
            elif pred_return <= sell_threshold:
                signal = "🔴 BÁN"
                confidence = min(abs(pred_return) / (abs(sell_threshold) * 3), 1.0)
            else:
                signal = "🟡 GIỮ"
                confidence = 1.0 - abs(pred_return) / buy_threshold

            rows.append({
                "Mã": symbol,
                "Giá hiện tại": round(current_price, 2),
                "Giá dự báo": round(predicted_price, 2),
                "Thay đổi (%)": round(change_pct, 2),
                "Predicted Return": round(pred_return, 5),
                "Khuyến nghị": signal,
                "Độ tin cậy": round(confidence * 100, 1),
            })

        return pd.DataFrame(rows)

    def _get_causal_parents(self, symbol: str, df: pd.DataFrame) -> list:
        """Trả về danh sách causal parents của symbol từ PCMCI+ graph."""
        if self.pcmci_results is None or symbol not in self.feature_names:
            # Fallback: tất cả features trừ chính nó
            return [c for c in df.columns if c != symbol]

        graph = self.pcmci_results['graph']
        target_idx = self.feature_names.index(symbol)
        parents = set()
        for i, feat in enumerate(self.feature_names):
            if feat == symbol:
                continue
            for tau in range(graph.shape[2]):
                if graph[i, target_idx, tau] == '-->':
                    parents.add(feat)
        # Fallback nếu không có causal parents
        return list(parents) if parents else [c for c in df.columns if c != symbol]

    def _forecast_single(self, df: pd.DataFrame, target: str,
                         feature_cols: list, window_size: int) -> float:
        """Ridge forecast kỳ tiếp theo cho một target bất kỳ."""
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            return 0.0

        df_lag = df[available].shift(1).copy()
        df_lag[target] = df[target]
        df_lag = df_lag.dropna()

        if len(df_lag) < window_size:
            window_size = len(df_lag) // 2

        X = df_lag[available].values
        y = df_lag[target].values
        X_train = X[-window_size:]
        y_train = y[-window_size:]
        X_next  = df[available].values[-1:]

        scaler = StandardScaler()
        pred = Ridge(alpha=1.0).fit(
            scaler.fit_transform(X_train), y_train
        ).predict(scaler.transform(X_next))[0]
        return float(pred)

    # ------------------------------------------------------------------
    # Predict Next Month — Ridge (chiều) + GBM (độ lớn) + Logistic (xác nhận)
    # ------------------------------------------------------------------

    def predict_next_month(self, df: pd.DataFrame,
                           window_size: int = 60,
                           current_price: float = None) -> dict:
        """
        Kết hợp 3 model:
          - All-features Ridge    → chiều tăng/giảm (DA=60.66%)
          - All-features GBM      → độ lớn return   (MAE=5.0%)
          - Causal Logistic       → xác nhận chiều  (DA=57.38%)

        Final return = sign(Ridge) × |GBM|
        Signal MUA/BÁN khi Ridge + Logistic đồng thuận và confidence ≥ 60%.
        """
        from sklearn.linear_model import LogisticRegressionCV, Ridge
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit

        # All-features dùng toàn bộ cột
        all_cols     = [c for c in df.columns if c != self.target_col]
        # Causal features cho Logistic
        causal_cols  = [c for c in (self.selected_features or all_cols) if c in df.columns]

        def _prep(cols):
            df_lag = df[cols].shift(1).copy()
            df_lag[self.target_col] = df[self.target_col]
            df_lag = df_lag.dropna()
            ws = min(window_size, len(df_lag) - 1)
            X_tr = df_lag[cols].values[-ws:]
            y_tr = df_lag[self.target_col].values[-ws:]
            X_nx = df[cols].values[-1:]
            sc   = StandardScaler()
            return sc.fit_transform(X_tr), y_tr, sc.transform(X_nx)

        # ── 1. All-features Ridge → chiều ──────────────────────────────
        X_tr, y_tr, X_nx = _prep(all_cols)
        ridge_pred = float(Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_nx)[0])
        ridge_dir  = 'TĂNG' if ridge_pred > 0 else 'GIẢM'

        # ── 2. All-features GBM → độ lớn ──────────────────────────────
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=42,
        )
        gbm.fit(X_tr, y_tr)
        gbm_pred = float(gbm.predict(X_nx)[0])
        gbm_magnitude = abs(gbm_pred)

        # Final return = chiều từ Ridge × độ lớn từ GBM
        final_return = (1 if ridge_dir == 'TĂNG' else -1) * gbm_magnitude

        # ── 3. Causal Logistic → xác nhận chiều ───────────────────────
        direction, confidence = ridge_dir, 0.5
        X_tr_c, y_tr_c, X_nx_c = _prep(causal_cols)
        y_dir = (y_tr_c > 0).astype(int)
        if len(np.unique(y_dir)) >= 2:
            tscv = TimeSeriesSplit(n_splits=3)
            clf  = LogisticRegressionCV(
                Cs=np.logspace(-3, 2, 10), cv=tscv,
                scoring='accuracy', penalty='l2',
                solver='lbfgs', max_iter=500, random_state=42,
            )
            clf.fit(X_tr_c, y_dir)
            proba      = clf.predict_proba(X_nx_c)[0]
            pred_dir   = clf.predict(X_nx_c)[0]
            direction  = 'TĂNG' if pred_dir == 1 else 'GIẢM'
            confidence = float(proba[pred_dir])

        # ── 4. Signal logic ────────────────────────────────────────────
        consistent = (ridge_dir == direction)
        if not consistent:
            signal = 'CHỜ'
        elif confidence >= 0.60:
            signal = 'MUA' if direction == 'TĂNG' else 'BÁN'
        else:
            signal = 'CHỜ'

        # ── 5. Giá dự báo ──────────────────────────────────────────────
        pred_price = None
        if current_price and current_price > 0:
            pred_price = round(current_price * np.exp(final_return), 2)
            change_pct = (np.exp(final_return) - 1) * 100
        else:
            change_pct = (np.exp(final_return) - 1) * 100

        result = {
            'direction':    direction,
            'confidence':   round(confidence, 4),
            'ridge_dir':    ridge_dir,
            'gbm_magnitude': round(gbm_magnitude * 100, 2),
            'final_return': round(final_return * 100, 2),
            'change_pct':   round(change_pct, 2),
            'pred_price':   pred_price,
            'signal':       signal,
            'consistent':   consistent,
        }

        # ── In kết quả ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  DỰ BÁO THÁNG TIẾP THEO — VN-Index")
        print(f"{'='*60}")
        print(f"  Ridge   (chiều)   : {ridge_dir}")
        print(f"  GBM     (độ lớn)  : {gbm_magnitude*100:.2f}%")
        print(f"  Logistic(xác nhận): {direction}  (confidence: {confidence:.1%})")
        print(f"  ──────────────────────────────────────────────────────")
        print(f"  Đồng thuận        : {'✓ Nhất quán' if consistent else '✗ Mâu thuẫn (Ridge vs Logistic)'}")
        print(f"  Return dự báo     : {final_return*100:+.2f}%")
        if current_price:
            print(f"  Giá hiện tại      : {current_price:,.2f}")
            print(f"  Giá dự báo        : {pred_price:,.2f}  ({change_pct:+.2f}%)")
        print(f"  Khuyến nghị       : "
              f"{'🟢 MUA' if signal=='MUA' else '🔴 BÁN' if signal=='BÁN' else '🟡 CHỜ'}")
        print(f"{'='*60}")

        return result

    # ------------------------------------------------------------------
    # Backtest Strategy — so sánh model vs Buy & Hold
    # ------------------------------------------------------------------

    def backtest_strategy(self, actuals: np.ndarray,
                          predictions: np.ndarray,
                          freq: str = "monthly") -> dict:
        """
        Mô phỏng chiến lược giao dịch dựa trên tín hiệu model.

        Chiến lược:
          - Dự báo TĂNG → mua (long), nhận actual_return tháng đó
          - Dự báo GIẢM → đứng ngoài (cash), return = 0
        So sánh với Buy & Hold (luôn nắm giữ).

        Metrics:
          total_return      : lợi nhuận tích lũy cuối kỳ (%)
          annual_return     : lợi nhuận trung bình mỗi năm (%)
          sharpe_ratio      : return / risk (> 1 là tốt)
          max_drawdown      : mức giảm tối đa từ đỉnh (%)
          win_rate          : % tháng có lợi nhuận dương
        """
        unit   = "tháng" if freq == "monthly" else "ngày"
        n_year = 12 if freq == "monthly" else 252

        # ── Chiến lược model: chỉ nắm giữ khi dự báo TĂNG ──────────
        model_returns = np.where(predictions > 0, actuals, 0.0)

        # ── Buy & Hold: luôn nắm giữ ──────────────────────────────
        bh_returns = actuals.copy()

        def _metrics(rets, name):
            cum   = np.cumprod(1 + rets) - 1          # tích lũy
            total = cum[-1] * 100
            ann   = ((1 + cum[-1]) ** (n_year / len(rets)) - 1) * 100
            std   = np.std(rets) * np.sqrt(n_year)
            sharpe = ann / (std * 100 + 1e-8)
            # Max drawdown
            wealth   = np.cumprod(1 + rets)
            peak     = np.maximum.accumulate(wealth)
            drawdown = (wealth - peak) / (peak + 1e-8)
            mdd      = drawdown.min() * 100
            win_rate = np.mean(rets > 0) * 100
            return {
                'name': name, 'cum_returns': cum,
                'total_return': round(total, 2),
                'annual_return': round(ann, 2),
                'sharpe_ratio': round(sharpe, 3),
                'max_drawdown': round(mdd, 2),
                'win_rate': round(win_rate, 2),
            }

        m_stats = _metrics(model_returns, 'Model Strategy')
        b_stats = _metrics(bh_returns,    'Buy & Hold')

        # ── In kết quả ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  BACKTEST — Model Strategy vs Buy & Hold")
        print(f"  (Dựa trên {len(actuals)} {unit} out-of-sample)")
        print(f"{'='*60}")
        print(f"  {'Chỉ số':<22} {'Model Strategy':>16} {'Buy & Hold':>12}")
        print(f"  {'-'*52}")
        metrics = [
            ('Tổng lợi nhuận (%)',  'total_return'),
            ('Lợi nhuận/năm (%)',   'annual_return'),
            ('Sharpe Ratio',        'sharpe_ratio'),
            ('Max Drawdown (%)',    'max_drawdown'),
            ('Win Rate (%)',        'win_rate'),
        ]
        for label, key in metrics:
            mv, bv = m_stats[key], b_stats[key]
            better = '✓' if (key == 'max_drawdown' and mv > bv) is False and (
                (key == 'max_drawdown' and mv > bv) or
                (key != 'max_drawdown' and mv >= bv)
            ) else ''
            print(f"  {label:<22} {mv:>14.2f}   {bv:>10.2f}  {better}")
        print(f"{'='*60}")

        # ── Vẽ biểu đồ so sánh ─────────────────────────────────────
        self._plot_backtest(m_stats, b_stats, freq)

        return {'model': m_stats, 'buyhold': b_stats}

    def _plot_backtest(self, m_stats: dict, b_stats: dict,
                       freq: str = "monthly") -> None:
        unit = "Tháng" if freq == "monthly" else "Ngày"
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Biểu đồ 1: tăng trưởng vốn
        ax = axes[0]
        ax.plot((m_stats['cum_returns'] + 1) * 100,
                label='Model Strategy', color='#C44E52', linewidth=2)
        ax.plot((b_stats['cum_returns'] + 1) * 100,
                label='Buy & Hold', color='#4C72B0',
                linewidth=2, linestyle='--')
        ax.axhline(100, color='gray', linewidth=0.8, linestyle=':')
        ax.set_ylabel('Giá trị danh mục (vốn gốc = 100)')
        ax.set_title(f'Tăng trưởng vốn — Model vs Buy & Hold ({unit}ly)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

        # Biểu đồ 2: so sánh metrics
        ax2 = axes[1]
        labels  = ['Lợi nhuận/năm\n(%)', 'Sharpe\nRatio', 'Win Rate\n(%)']
        m_vals  = [m_stats['annual_return'],
                   m_stats['sharpe_ratio'],
                   m_stats['win_rate']]
        b_vals  = [b_stats['annual_return'],
                   b_stats['sharpe_ratio'],
                   b_stats['win_rate']]
        x = np.arange(len(labels))
        w = 0.35
        ax2.bar(x - w/2, m_vals, w, label='Model Strategy', color='#C44E52', alpha=0.8)
        ax2.bar(x + w/2, b_vals, w, label='Buy & Hold',     color='#4C72B0', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_title('So sánh hiệu suất')
        ax2.legend()
        ax2.grid(axis='y', linestyle=':', alpha=0.5)

        plt.tight_layout()
        suffix = "_daily" if freq == "daily" else ""
        fname  = f"backtest{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Biểu đồ lưu tại: {fname}")

    # ------------------------------------------------------------------
    # Evaluation & Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def directional_accuracy(actuals, predictions):
        """% dự báo đúng chiều tăng/giảm."""
        return float(np.mean(np.sign(actuals) == np.sign(predictions)))

    def evaluate(self, actuals, predictions, model_name='Model'):
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        da  = self.directional_accuracy(actuals, predictions)
        print(f"  [{model_name}]  MSE={mse:.6f}  MAE={mae:.6f}  DA={da:.2%}")
        return {'model': model_name, 'mse': mse, 'mae': mae, 'da': da,
                'actuals': actuals, 'predictions': predictions}

    def _rolling_forecast_clf(self, df, feature_cols, window_size=60):
        """Rolling window Logistic Regression — dự báo trực tiếp chiều tăng/giảm.

        Paper 4 (Yang et al. 2022): dùng LogisticRegressionCV + TimeSeriesSplit
        để tune C đúng cách trên time series, tránh data leakage.
        """
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import TimeSeriesSplit

        df_lag = df[feature_cols].shift(1).copy()
        df_lag[self.target_col] = df[self.target_col]
        df_lag = df_lag.dropna()

        y_cont = df_lag[self.target_col].values
        y      = (y_cont > 0).astype(int)   # 1=tăng, 0=giảm
        X      = df_lag[feature_cols].values

        predictions, actuals = [], []
        tscv = TimeSeriesSplit(n_splits=3)

        for i in range(window_size, len(df_lag)):
            X_train, y_train = X[i - window_size:i], y[i - window_size:i]
            X_test = X[i:i + 1]

            if len(np.unique(y_train)) < 2:
                predictions.append(0)
                actuals.append(y_cont[i])
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            clf = LogisticRegressionCV(
                Cs=np.logspace(-3, 2, 10),
                cv=tscv,
                scoring="accuracy",
                penalty="l2",
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            )
            clf.fit(X_train_s, y_train)
            pred_dir = clf.predict(X_test_s)[0]
            predictions.append(1.0 if pred_dir == 1 else -1.0)
            actuals.append(y_cont[i])

        return np.array(actuals), np.array(predictions)

    def _rolling_forecast_xgb(self, df, feature_cols, window_size=60):
        """Rolling window Gradient Boosting — bắt quan hệ phi tuyến (Nhóm 1).
        Dùng sklearn GradientBoostingRegressor (không cần libomp).
        """
        from sklearn.ensemble import GradientBoostingRegressor

        df_lag = df[feature_cols].shift(1).copy()
        df_lag[self.target_col] = df[self.target_col]
        df_lag = df_lag.dropna()

        y = df_lag[self.target_col].values
        X = df_lag[feature_cols].values

        predictions, actuals = [], []
        for i in range(window_size, len(df_lag)):
            X_train, y_train = X[i - window_size:i], y[i - window_size:i]
            X_test = X[i:i + 1]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X_train_s, y_train)
            predictions.append(model.predict(X_test_s)[0])
            actuals.append(y[i])

        return np.array(actuals), np.array(predictions)

    def compare_models(self, df, window_size=60, freq: str = "monthly",
                       use_lstm: bool = False, lstm_epochs: int = 20):
        """
        So sánh các baseline:
          1. All-features Ridge
          2. LASSO Ridge
          3. Causal Ridge (PCMCI+)
          4. Causal Logistic (PCMCI+ features) — tối ưu trực tiếp DA
          5. Causal XGBoost — phi tuyến, bắt momentum
          5. Causal LSTM (PCMCI+ features, nếu use_lstm=True)
        """
        all_cols = [c for c in df.columns if c != self.target_col]

        print("\n[1/5] All-features Ridge")
        act, pred_all = self._rolling_forecast(df, all_cols, window_size)
        r_all = self.evaluate(act, pred_all, 'All-features Ridge')

        print("\n[2/5] All-features XGBoost (phi tuyến, tất cả features)")
        r_all_xgb = None
        act_axgb, pred_axgb = self._rolling_forecast_xgb(df, all_cols, window_size)
        if act_axgb is not None:
            r_all_xgb = self.evaluate(act_axgb, pred_axgb, 'All-features GBM')

        print("\n[3/6] LASSO Ridge")
        df_lasso  = self.reduce_dimensions(df, method='lasso')
        lasso_cols = [c for c in df_lasso.columns if c != self.target_col]
        _, pred_lasso = self._rolling_forecast(df, lasso_cols, window_size)
        r_lasso = self.evaluate(act, pred_lasso, 'LASSO Ridge')

        print("\n[4/6] Causal Ridge (PCMCI+)")
        r_causal = None
        if self.selected_features:
            _, pred_causal = self._rolling_forecast(df, self.selected_features, window_size)
            r_causal = self.evaluate(act, pred_causal, 'Causal Ridge (PCMCI+)')
        else:
            print("  Không tìm được causal features.")

        print("\n[5/6] Causal Logistic (PCMCI+ — tối ưu DA trực tiếp)")
        r_clf = None
        clf_cols = self.selected_features if self.selected_features else lasso_cols
        if clf_cols:
            _, pred_clf = self._rolling_forecast_clf(df, clf_cols, window_size)
            r_clf = self.evaluate(act, pred_clf, 'Causal Logistic')

        print("\n[6/6] Causal XGBoost (phi tuyến + momentum)")
        r_xgb = None
        xgb_cols = self.selected_features if self.selected_features else lasso_cols
        if xgb_cols:
            act_xgb, pred_xgb = self._rolling_forecast_xgb(df, xgb_cols, window_size)
            if act_xgb is not None:
                r_xgb = self.evaluate(act_xgb, pred_xgb, 'Causal GBM')

        r_lstm = None
        if use_lstm and TORCH_AVAILABLE and self.selected_features:
            print(f"\n[4/4] Causal LSTM (PCMCI+, epochs={lstm_epochs})")
            seq_len = min(10, window_size // 6)
            act_lstm, pred_lstm = self._rolling_forecast_lstm(
                df, self.selected_features,
                window_size=window_size,
                seq_len=seq_len,
                epochs=lstm_epochs,
            )
            if act_lstm is not None:
                r_lstm = self.evaluate(act_lstm, pred_lstm, 'Causal LSTM (PCMCI+)')

        results = {'all': r_all, 'all_xgb': r_all_xgb, 'lasso': r_lasso, 'causal': r_causal, 'clf': r_clf, 'xgb': r_xgb, 'lstm': r_lstm}
        self._plot_comparison(results, act, freq=freq)
        return results

    def _plot_comparison(self, results, actuals, freq: str = "monthly"):
        unit = "Ngày" if freq == "daily" else "Tháng"
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        ax = axes[0]
        ax.plot(actuals, label='Actual', color='black', linewidth=1.5)
        ax.plot(results['all']['predictions'],
                label='All-features Ridge', color='#4C72B0', linestyle='--', alpha=0.8)
        ax.plot(results['lasso']['predictions'],
                label='LASSO Ridge', color='#55A868', linestyle='--', alpha=0.8)
        if results['causal']:
            ax.plot(results['causal']['predictions'],
                    label='Causal Ridge (PCMCI+)', color='#C44E52', linewidth=2)
        if results.get('lstm'):
            ax.plot(results['lstm']['predictions'],
                    label='Causal LSTM (PCMCI+)', color='#FF7F0E', linewidth=2, linestyle='-.')
        ax.set_title(f'VN-Index {unit}ly Return: Model Comparison', fontsize=13)
        ax.set_ylabel('Log Return')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

        ax2 = axes[1]
        models_data = [('All-features', results['all']), ('LASSO', results['lasso'])]
        if results['causal']:
            models_data.append(('Causal\n(PCMCI+)', results['causal']))
        if results.get('lstm'):
            models_data.append(('Causal\nLSTM', results['lstm']))

        names = [m[0] for m in models_data]
        mses  = [m[1]['mse'] for m in models_data]
        das   = [m[1]['da']  for m in models_data]

        x, w = np.arange(len(names)), 0.35
        ax2b = ax2.twinx()
        ax2.bar(x - w / 2, mses, w, label='MSE ↓', color='#4C72B0', alpha=0.75)
        ax2b.bar(x + w / 2, das,  w, label='Dir. Acc. ↑', color='#C44E52', alpha=0.75)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, fontsize=11)
        ax2.set_ylabel('MSE (thấp hơn = tốt hơn)', color='#4C72B0')
        ax2b.set_ylabel('Directional Accuracy (cao hơn = tốt hơn)', color='#C44E52')
        ax2b.set_ylim(0, 1)
        ax2.set_title('So sánh hiệu suất các mô hình', fontsize=13)
        ax2.legend(loc='upper left')
        ax2b.legend(loc='upper right')

        plt.tight_layout()
        suffix = "_daily" if freq == "daily" else ""
        fname = f"model_comparison{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Đã lưu: {fname}")
        return fig
