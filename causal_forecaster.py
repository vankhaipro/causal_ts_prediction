import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from castle.algorithms import PC
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

class CausalForecaster:
    def __init__(self, target_col='Target_Return'):
        self.target_col = target_col
        self.causal_model = None
        self.selected_features = []
        self.adjacency_matrix = None
        self.feature_names = None
        self.scaler = StandardScaler()

    def check_stationarity(self, df, significance=0.05):
        """
        Perform Augmented Dickey-Fuller (ADF) test to ensure stationarity.
        Returns a list of non-stationary columns.
        """
        non_stationary = []
        for col in df.columns:
            # Skip checking if series is constant
            if df[col].nunique() <= 1:
                continue
            res = adfuller(df[col].dropna())
            if res[1] > significance:
                non_stationary.append(col)
        return non_stationary

    def apply_publication_lag(self, df, macro_cols, lag=1):
        """
        Shift macro variables by 'lag' to account for publication delay.
        Avoids Look-ahead bias.
        """
        df_shifted = df.copy()
        for col in macro_cols:
            if col != self.target_col:
                df_shifted[col] = df_shifted[col].shift(lag)
        return df_shifted.dropna()

    def reduce_dimensions(self, df, method='lasso', n_components=10):
        """
        Reduce 128 indicator dimensions to a manageable set (Super-factors).
        Methods: 'lasso' (Feature Selection) or 'pca' (Dimensionality Reduction).
        """
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'pca':
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)
            cols = [f'PC{i+1}' for i in range(n_components)]
            reduced_df = pd.DataFrame(X_reduced, columns=cols, index=df.index)
            reduced_df[self.target_col] = y.values
            return reduced_df
        
        elif method == 'lasso':
            lasso = LassoCV(cv=5).fit(X_scaled, y)
            coef = pd.Series(lasso.coef_, index=X.columns)
            selected = coef[coef != 0].index.tolist()
            # If LASSO selects too many or too few, handle it here
            print(f"LASSO selected {len(selected)} features.")
            return df[selected + [self.target_col]]
            
        return df

    def perform_causal_discovery(self, df):
        """
        Identify causal relationships using PC algorithm from gcastle.
        To handle time series causality, we ensures the input data 
        includes lagged variables if necessary or uses the contemporaneous structure.
        """
        from castle.algorithms import PC
        
        self.feature_names = df.columns.tolist()
        data_scaled = self.scaler.fit_transform(df)
        
        # PC algorithm for structure learning
        model = PC()
        model.learn(data_scaled)
        
        # model.causal_matrix is the adjacency matrix
        # For PC, it's often a DAG or CPDAG represented as adjacency matrix
        self.adjacency_matrix = model.causal_matrix
        self.causal_model = model
        
        return self.adjacency_matrix

    def extract_features(self):
        """
        Extract features that have a directed edge to the target variable.
        """
        if self.adjacency_matrix is None:
            raise ValueError("Run perform_causal_discovery first.")
        
        target_idx = self.feature_names.index(self.target_col)
        
        # Identify nodes that have an edge to target_idx
        # In gcastle adjacency matrix: [i, j] = 1 means i -> j
        edges_to_target = np.where(self.adjacency_matrix[:, target_idx] != 0)[0]
        
        selected = []
        for idx in edges_to_target:
            feat = self.feature_names[idx]
            if feat != self.target_col:
                selected.append(feat)
        
        self.selected_features = list(set(selected))
        return self.selected_features

    def visualize_graph(self):
        """
        Visualize the causal graph using NetworkX.
        """
        if self.adjacency_matrix is None:
            return

        G = nx.DiGraph()
        d = len(self.feature_names)
        
        for feat in self.feature_names:
            G.add_node(feat)

        # Contemporaneous edges
        for i in range(d):
            for j in range(d):
                if self.adjacency_matrix[i, j] != 0:
                    G.add_edge(self.feature_names[i], self.feature_names[j])

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.6)
        
        node_colors = ['#FF4B4B' if node == self.target_col else '#1E90FF' for node in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=2500, font_size=9, font_weight='bold', 
                arrows=True, arrowsize=20, edge_color='gray', alpha=0.8)
        
        plt.title(f"Causal Dependencies for {self.target_col}", fontsize=15)
        plt.show()

    def forecast_rolling_window(self, df, window_size=150):
        """
        Perform rolling window validation with Ridge Regression.
        """
        # Prepare feature matrix
        # If we have lag features, we need to create them manually for the regression
        feature_names_reg = []
        for feat in self.selected_features:
            if feat.endswith('_lag'):
                orig_feat = feat.replace('_lag', '')
                df[feat] = df[orig_feat].shift(1)
                feature_names_reg.append(feat)
            else:
                feature_names_reg.append(feat)
        
        df_clean = df.dropna()
        y = df_clean[self.target_col].values
        X = df_clean[feature_names_reg].values
        
        predictions = []
        actuals = []
        
        for i in range(window_size, len(df_clean)):
            train_X, train_y = X[i-window_size:i], y[i-window_size:i]
            test_X, test_y = X[i:i+1], y[i:i+1]
            
            # Local scaling to avoid data leakage
            loc_scaler = StandardScaler()
            train_X_scaled = loc_scaler.fit_transform(train_X)
            test_X_scaled = loc_scaler.transform(test_X)
            
            model = Ridge(alpha=1.0)
            model.fit(train_X_scaled, train_y)
            
            predictions.append(model.predict(test_X_scaled)[0])
            actuals.append(test_y[0])
            
        return np.array(actuals), np.array(predictions)

    def evaluate(self, actuals, predictions):
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        print(f"\nEvaluation Results:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        
        plt.figure(figsize=(14, 7))
        plt.plot(actuals, label='Actual Returns', color='#2F4F4F', linewidth=1.5)
        plt.plot(predictions, label='Causal Forecast', color='#FF6347', linestyle='--', linewidth=1.5)
        plt.fill_between(range(len(actuals)), actuals, predictions, color='gray', alpha=0.2)
        plt.title("S&P 500 Return Prediction vs Actual (Causal Framework)", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()
        
        return mse, mae
