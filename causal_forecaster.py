import numpy as np
import pandas as pd
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


class CausalForecaster:
    def __init__(self, target_col='VNINDEX_Return'):
        self.target_col = target_col
        self.pcmci_results = None
        self.selected_features = []   # feature names confirmed causal by PCMCI+
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
    # Causal Discovery — PCMCI+
    # ------------------------------------------------------------------

    def perform_causal_discovery(self, df, tau_max=3, pc_alpha=0.05):
        """
        Chạy PCMCI+ để học cấu trúc nhân quả có độ trễ.

        tau_min=1 đảm bảo chỉ tìm X(t-tau) → Y(t), tránh look-ahead bias
        hoàn toàn bằng cách thiết kế (không cần publication lag thủ công).
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
        """
        Trích xuất features có cạnh nhân quả --> target.
        Trả về list (feature_name, lag) và lưu feature names vào self.selected_features.
        """
        if self.pcmci_results is None:
            raise ValueError("Chạy perform_causal_discovery trước.")

        graph = self.pcmci_results['graph']   # shape: (n_vars, n_vars, tau_max+1)
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

    def visualize_graph(self):
        """Vẽ causal graph với nhãn độ trễ (τ)."""
        if self.pcmci_results is None:
            return

        graph = self.pcmci_results['graph']
        G = nx.DiGraph()
        for feat in self.feature_names:
            G.add_node(feat)

        for i, src in enumerate(self.feature_names):
            for j, dst in enumerate(self.feature_names):
                for tau in range(graph.shape[2]):
                    if graph[i, j, tau] == '-->':
                        # Nếu đã có cạnh, gộp nhãn lag
                        if G.has_edge(src, dst):
                            G[src][dst]['label'] += f',τ={tau}'
                        else:
                            G.add_edge(src, dst, label=f'τ={tau}')

        plt.figure(figsize=(14, 9))
        pos = nx.spring_layout(G, k=0.8, seed=42)
        colors = ['#FF4B4B' if n == self.target_col else '#1E90FF' for n in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=colors,
                node_size=2500, font_size=9, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray', alpha=0.85)
        nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, 'label'), font_size=7)
        plt.title(f"PCMCI+ Causal Graph — {self.target_col}", fontsize=14)
        plt.tight_layout()
        plt.savefig('causal_graph.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Đã lưu: causal_graph.png")

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def _rolling_forecast(self, df, feature_cols, window_size=60):
        """
        Rolling window Ridge Regression.

        Dùng lag-1 của feature_cols (shift 1 tháng) để dự báo target(t).
        Việc shift được thực hiện ở đây — KHÔNG áp dụng publication lag
        bên ngoài trước khi gọi hàm này.
        """
        df_lag = df[feature_cols].shift(1).copy()
        df_lag[self.target_col] = df[self.target_col]
        df_lag = df_lag.dropna()

        y = df_lag[self.target_col].values
        X = df_lag[feature_cols].values

        predictions, actuals = [], []
        for i in range(window_size, len(df_lag)):
            X_train, y_train = X[i - window_size:i], y[i - window_size:i]
            X_test, y_test   = X[i:i + 1],           y[i:i + 1]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            pred = Ridge(alpha=1.0).fit(X_train_s, y_train).predict(X_test_s)[0]
            predictions.append(pred)
            actuals.append(y_test[0])

        return np.array(actuals), np.array(predictions)

    # ------------------------------------------------------------------
    # Evaluation & Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def directional_accuracy(actuals, predictions):
        """Tỷ lệ dự báo đúng chiều (tăng/giảm) — metric thực tế trong finance."""
        return float(np.mean(np.sign(actuals) == np.sign(predictions)))

    def evaluate(self, actuals, predictions, model_name='Model'):
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        da  = self.directional_accuracy(actuals, predictions)
        print(f"  [{model_name}]  MSE={mse:.6f}  MAE={mae:.6f}  DA={da:.2%}")
        return {'model': model_name, 'mse': mse, 'mae': mae, 'da': da,
                'actuals': actuals, 'predictions': predictions}

    def compare_models(self, df, window_size=60):
        """
        So sánh 3 baseline:
          1. All-features Ridge   — tất cả features
          2. LASSO Ridge          — LASSO-selected features
          3. Causal Ridge         — PCMCI+-selected features

        Lưu ý: LASSO được fit trên toàn bộ dataset (ngoài rolling window),
        đây là cách tiêu chuẩn cho feature selection nhưng cần ghi chú trong báo cáo.
        """
        all_cols = [c for c in df.columns if c != self.target_col]

        print("\n[1/3] All-features Ridge")
        act, pred_all = self._rolling_forecast(df, all_cols, window_size)
        r_all = self.evaluate(act, pred_all, 'All-features Ridge')

        print("\n[2/3] LASSO Ridge")
        df_lasso  = self.reduce_dimensions(df, method='lasso')
        lasso_cols = [c for c in df_lasso.columns if c != self.target_col]
        _, pred_lasso = self._rolling_forecast(df, lasso_cols, window_size)
        r_lasso = self.evaluate(act, pred_lasso, 'LASSO Ridge')

        print("\n[3/3] Causal Ridge (PCMCI+)")
        if self.selected_features:
            _, pred_causal = self._rolling_forecast(df, self.selected_features, window_size)
            r_causal = self.evaluate(act, pred_causal, 'Causal Ridge (PCMCI+)')
        else:
            print("  Không tìm được causal features.")
            r_causal = None

        results = {'all': r_all, 'lasso': r_lasso, 'causal': r_causal}
        self._plot_comparison(results, act)
        return results

    def _plot_comparison(self, results, actuals):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # --- Time series ---
        ax = axes[0]
        ax.plot(actuals, label='Actual', color='black', linewidth=1.5)
        ax.plot(results['all']['predictions'],
                label='All-features Ridge', color='#4C72B0', linestyle='--', alpha=0.8)
        ax.plot(results['lasso']['predictions'],
                label='LASSO Ridge', color='#55A868', linestyle='--', alpha=0.8)
        if results['causal']:
            ax.plot(results['causal']['predictions'],
                    label='Causal Ridge (PCMCI+)', color='#C44E52', linewidth=2)
        ax.set_title('VN-Index Monthly Return: Model Comparison', fontsize=13)
        ax.set_ylabel('Log Return')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

        # --- Metrics bar chart ---
        ax2 = axes[1]
        models_data = [('All-features', results['all']), ('LASSO', results['lasso'])]
        if results['causal']:
            models_data.append(('Causal\n(PCMCI+)', results['causal']))

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
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Đã lưu: model_comparison.png")
