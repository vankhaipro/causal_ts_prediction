# Dự Báo VN-Index bằng Causal Discovery

**Khoá luận tốt nghiệp** — Ứng dụng thuật toán nhân quả (PCMCI+) vào dự báo chỉ số chứng khoán Việt Nam.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Cấu trúc dự án](#2-cấu-trúc-dự-án)
3. [Dữ liệu — Trường quan trọng & Cách tính](#3-dữ-liệu--trường-quan-trọng--cách-tính)
4. [Pipeline & Mô hình](#4-pipeline--mô-hình)
5. [Giải thích chi tiết từng bước](#5-giải-thích-chi-tiết-từng-bước)
6. [Cài đặt & Chạy](#6-cài-đặt--chạy)
7. [Web App](#7-web-app)
8. [Kết quả thực nghiệm](#8-kết-quả-thực-nghiệm)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

Hầu hết mô hình dự báo chứng khoán chọn features dựa trên **tương quan (correlation)** — không phân biệt được nguyên nhân thật sự và biến nhiễu, dẫn đến overfitting khi thị trường thay đổi cấu trúc.

Khoá luận này đề xuất pipeline **Causal Discovery → Forecasting**:

1. Dùng **PCMCI+** học cấu trúc nhân quả có độ trễ giữa các biến tài chính
2. Chỉ dùng **causal features** (nguyên nhân thật sự) để xây dựng mô hình dự báo
3. So sánh định lượng với baseline dùng correlation: All-features Ridge và LASSO Ridge

**Giả thuyết nghiên cứu:** Causal features cho Directional Accuracy cao hơn correlation features vì loại bỏ được spurious correlations.

---

## 2. Cấu trúc dự án

```
causal_ts_prediction/
│
├── data_loader.py        # Tải dữ liệu vnstock + yfinance, lưu CSV
├── causal_forecaster.py  # PCMCI+, Ridge, LSTM, đánh giá mô hình
├── main.py               # Chạy pipeline từ terminal
├── app.py                # Streamlit web app
├── run.sh                # Script chạy nhanh
│
└── Data/
    ├── dataset.csv           # Dataset monthly (dùng cho train)
    ├── dataset_daily.csv     # Dataset daily
    ├── vn_stocks.csv         # VN-Index + blue-chip log return (monthly)
    ├── vn_stocks_daily.csv   # VN-Index + blue-chip log return (daily)
    ├── macro.csv             # Macro global log return (monthly)
    ├── macro_daily.csv       # Macro global log return (daily)
    ├── prices.csv            # Giá đóng cửa thực monthly
    └── prices_daily.csv      # Giá đóng cửa thực daily
```

---

## 3. Dữ liệu — Trường quan trọng & Cách tính

### 3.1 Nguồn dữ liệu

| Nguồn | Nội dung | Thư viện |
|-------|---------|---------|
| vnstock (KBS) | VN-Index + 7 blue-chip VN | `vnstock` |
| Yahoo Finance | S&P500, VIX, Dầu, Vàng, DXY | `yfinance` |

### 3.2 Trường dữ liệu quan trọng

Tất cả giá trị trong dataset đều là **log return** — không phải giá tuyệt đối.

#### Log Return — Công thức tính

```
log_return(t) = ln( Close(t) / Close(t-1) )
```

**Tại sao dùng log return thay vì giá tuyệt đối?**
- **Stationarity:** Giá cổ phiếu có xu hướng tăng theo thời gian (non-stationary), log return thường stationary → phù hợp với PCMCI+ và ADF test
- **Tính cộng:** Log return cộng được qua các kỳ: `r(t1→t3) = r(t1→t2) + r(t2→t3)`
- **Chuẩn hoá:** Cho phép so sánh các tài sản có giá trị tuyệt đối khác nhau (VN-Index ~1200, S&P500 ~5000)
- **Phân phối:** Xấp xỉ phân phối chuẩn — phù hợp với giả định của các test thống kê

#### Bảng trường dữ liệu

| Cột | Loại | Công thức | Ý nghĩa |
|-----|------|-----------|---------|
| `VNINDEX_Return` | Target | `ln(VNINDEX_t / VNINDEX_{t-1})` | Log return VN-Index — **biến cần dự báo** |
| `VCB` | Feature | `ln(VCB_close_t / VCB_close_{t-1})` | Log return Vietcombank |
| `BID` | Feature | `ln(BID_close_t / BID_close_{t-1})` | Log return BIDV |
| `VIC` | Feature | `ln(VIC_close_t / VIC_close_{t-1})` | Log return Vingroup |
| `HPG` | Feature | `ln(HPG_close_t / HPG_close_{t-1})` | Log return Hoa Phát |
| `MSN` | Feature | `ln(MSN_close_t / MSN_close_{t-1})` | Log return Masan |
| `MWG` | Feature | `ln(MWG_close_t / MWG_close_{t-1})` | Log return Mobile World |
| `GAS` | Feature | `ln(GAS_close_t / GAS_close_{t-1})` | Log return PetroVN Gas |
| `SP500` | Macro | `ln(SP500_t / SP500_{t-1})` | Log return S&P 500 |
| `VIX` | Macro | `ln(VIX_t / VIX_{t-1})` | Log return VIX (chỉ số sợ hãi) |
| `OIL` | Macro | `ln(Brent_t / Brent_{t-1})` | Log return dầu Brent |
| `GOLD` | Macro | `ln(Gold_t / Gold_{t-1})` | Log return giá vàng |
| `DXY` | Macro | `ln(DXY_t / DXY_{t-1})` | Log return US Dollar Index |
| `VNINDEX_Vol5` | Derived | `std(VNINDEX_Return, window=5)` | Độ biến động ngắn hạn VN-Index (5 kỳ) |
| `VNINDEX_Vol20` | Derived | `std(VNINDEX_Return, window=20)` | Độ biến động dài hạn VN-Index (20 kỳ) |

**Ý nghĩa kinh tế của Volatility Features:**

- `VNINDEX_Vol5` cao → thị trường đang trong giai đoạn bất ổn ngắn hạn (panic/fomo)
- `VNINDEX_Vol20` cao → chế độ biến động kéo dài (khủng hoảng, điều chỉnh lớn)
- Nếu PCMCI+ tìm `VNINDEX_Vol5(t-1) → VNINDEX_Return(t)`, đó là bằng chứng **nhân quả** của hiệu ứng volatility clustering trên TTCK VN

#### Ví dụ đọc giá trị

```
SP500 = 0.03   → S&P 500 tăng ~3% so với kỳ trước
VIX   = -0.15  → VIX giảm 15% (thị trường bớt sợ hãi)
VNINDEX_Return = 0.02 → VN-Index tăng ~2%
```

### 3.3 Cổ phiếu được chọn và loại

**Giữ lại** (IPO trước 2015, có đủ lịch sử):

| Mã | Công ty | IPO | Lý do chọn |
|----|---------|-----|------------|
| VCB | Vietcombank | 2009 | Ngân hàng lớn nhất VN theo vốn hoá |
| BID | BIDV | 2013 | Ngân hàng quốc doanh, phản ánh chính sách tín dụng |
| VIC | Vingroup | 2007 | Tập đoàn tư nhân lớn nhất VN |
| HPG | Hoa Phát | 2007 | Thép — chỉ báo chu kỳ công nghiệp, nhạy với giá dầu |
| MSN | Masan | 2010 | Tiêu dùng thiết yếu — chỉ báo nhu cầu nội địa |
| MWG | Mobile World | 2014 | Bán lẻ — chỉ báo tiêu dùng điện tử |
| GAS | PetroVN Gas | 2012 | Năng lượng — tương quan trực tiếp với giá dầu |

**Loại khỏi danh sách:**
- **TCB** (Techcombank, IPO 2018) — chỉ có dữ liệu từ 2018, cắt ngắn dataset ~4 năm
- **VHM** (Vinhomes, IPO 2018) — cùng lý do

### 3.4 Xử lý missing data

```python
# VN stocks: bỏ cột thiếu > 30%, bỏ hàng thiếu > 20%
df = df.dropna(axis=1, thresh=int(len(df) * 0.7))
df = df.dropna(thresh=int(len(df.columns) * 0.8))

# Macro (yfinance): forward-fill tối đa 3 ngày (do lệch ngày nghỉ lễ VN/US)
df = df.ffill(limit=3).dropna()
```

**Lý do ffill:** Thị trường VN và Mỹ có ngày nghỉ lễ không trùng nhau. Macro data
(SP500, VIX) không có giá ngày VN nghỉ → dùng giá ngày trước đó (ffill tối đa 3 ngày).

---

## 4. Pipeline & Mô hình

```
Dữ liệu (vnstock + yfinance)
        │
        ▼
  Tính log return
  Align trading days (inner join + ffill ≤ 3 ngày)
        │
        ▼
  ADF Test — kiểm tra stationarity
  (nếu p > 0.05 → non-stationary → cần diff thêm)
        │
        ▼
  PCMCI+ (tau_min=1, tau_max=3 hoặc 5)
  Independence test: Partial Correlation (ParCorr)
  Output: Causal graph shape (13 × 13 × tau_max+1)
        │
        ▼
  Trích xuất causal features
  graph[i, target_idx, τ] == '-->' → X_i(t-τ) gây ra Y(t)
        │
        ▼
  Rolling Window Forecast (window = 60 tháng / 252 ngày)
  ├── All-features Ridge   → dùng tất cả 12 features
  ├── LASSO Ridge          → LassoCV chọn features
  └── Causal Ridge         → PCMCI+ causal features
        │
        ▼
  Đánh giá: MSE | MAE | Directional Accuracy
```

---

## 5. Giải thích chi tiết từng bước

### Bước 1 — Tính Log Return

```python
# data_loader.py
ret = np.log(close / close.shift(1)).dropna()
```

Giá trị `close` là giá đóng cửa. Log return của kỳ t được tính bằng ln(giá kỳ t / giá kỳ t-1).

---

### Bước 2 — ADF Test (Augmented Dickey-Fuller)

```python
# causal_forecaster.py — check_stationarity()
result = adfuller(df[col].dropna())
# result[1] là p-value
# p < 0.05 → stationary (bác bỏ giả thuyết có unit root)
```

**Ý nghĩa:** PCMCI+ giả định dữ liệu stationary. Log return thường đã stationary nhờ
loại bỏ trend của giá. Nếu p > 0.05 → cần lấy diff thêm một lần.

---

### Bước 3 — PCMCI+ (Causal Discovery)

```python
# causal_forecaster.py — perform_causal_discovery()
dataframe = pp.DataFrame(df.values.astype(float), var_names=feature_names)
pcmci = PCMCI(dataframe=dataframe,
              cond_ind_test=ParCorr(significance='analytic'),
              verbosity=1)
results = pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
```

**Các tham số quan trọng:**

| Tham số | Monthly | Daily | Ý nghĩa |
|---------|---------|-------|---------|
| `tau_min=1` | 1 | 1 | Lag tối thiểu = 1 → chỉ tìm X(t-1) trở về trước, **tránh look-ahead bias** |
| `tau_max` | 3 | 5 | Lag tối đa xét đến (3 tháng / 5 ngày giao dịch = 1 tuần) |
| `pc_alpha` | 0.05 | 0.01 | Ngưỡng ý nghĩa thống kê. Daily chặt hơn vì nhiều quan sát hơn |
| `ParCorr` | — | — | Partial Correlation — kiểm tra độc lập có điều kiện |

**Output — Causal graph:**
```
graph.shape = (n_vars, n_vars, tau_max+1)
graph[i, j, tau] == '-->'   →  X_i(t-tau) là nguyên nhân của X_j(t)
graph[i, j, 0]   == 'o-o'  →  X_i(t) và X_j(t) có quan hệ contemporaneous
```

**Tại sao `tau_min=1` quan trọng:**
Nếu `tau_min=0` thì PCMCI+ có thể tìm ra quan hệ đồng thời X(t) → Y(t), tức là
dùng thông tin cùng ngày để dự báo cùng ngày — **look-ahead bias**. Đặt `tau_min=1`
đảm bảo chỉ dùng thông tin **quá khứ** để dự báo **tương lai**.

---

### Bước 4 — Trích xuất Causal Features

```python
# causal_forecaster.py — extract_features()
for i, feat in enumerate(feature_names):
    for tau in range(graph.shape[2]):
        if graph[i, target_idx, tau] == '-->':
            causal_pairs.append((feat, tau))   # (tên feature, độ trễ)
```

Kết quả là danh sách `(feature, lag)` — ví dụ:
```
[('SP500', 1), ('SP500', 2), ('MWG', 3)]
→ SP500(t-1), SP500(t-2), MWG(t-3) là nguyên nhân của VNINDEX(t)
```

---

### Bước 5 — Rolling Window Ridge Regression

```python
# causal_forecaster.py — _rolling_forecast()
df_lag = df[feature_cols].shift(1)     # Shift thêm 1 bước: X(t-1) → dự báo Y(t)
                                        # Cộng với tau của PCMCI+ → đảm bảo không leak

for i in range(window_size, len(df_lag)):
    X_train = X[i - window_size : i]   # window_size kỳ gần nhất để train
    X_test  = X[i : i+1]              # kỳ tiếp theo cần dự báo

    scaler = StandardScaler()          # Chuẩn hoá trong mỗi window
    pred = Ridge(alpha=1.0).fit(
        scaler.fit_transform(X_train), y_train
    ).predict(scaler.transform(X_test))[0]
```

**Lý do shift thêm 1 bước trong `_rolling_forecast`:**
PCMCI+ đã tìm X(t-τ) → Y(t) với τ ≥ 1. Trong forecast, ta dùng X(t-1) làm feature
để dự báo Y(t). Shift(1) đảm bảo tại thời điểm dự báo Y(t), chỉ có dữ liệu đến t-1.

**Rolling window:** Mỗi bước chỉ dùng `window_size` kỳ gần nhất để fit model → 
mô phỏng thực tế: nhà đầu tư chỉ biết dữ liệu quá khứ gần đây nhất, không biết tương lai.

**StandardScaler trong mỗi window:** Fit scaler trên tập train, transform tập test.
Không fit trên toàn dataset để tránh data leakage.

---

### Bước 6 — LASSO Feature Selection

```python
# causal_forecaster.py — reduce_dimensions()
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
coef = pd.Series(lasso.coef_, index=X.columns)
selected = coef[coef != 0].index.tolist()
```

LASSO fit trên **toàn bộ dataset** (không phải rolling window) để chọn features.
Đây là cách tiêu chuẩn cho feature selection nhưng có **data leakage nhỏ** (dùng dữ liệu
tương lai để chọn features) — cần ghi chú khi so sánh với Causal Ridge trong báo cáo.

---

### Bước 7 — Đánh giá mô hình

```python
# causal_forecaster.py — evaluate()
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
da  = float(np.mean(np.sign(actuals) == np.sign(predictions)))  # Directional Accuracy
```

| Metric | Công thức | Ý nghĩa trong finance |
|--------|-----------|----------------------|
| MSE | mean((y - ŷ)²) | Phạt nặng sai số lớn |
| MAE | mean(\|y - ŷ\|) | Sai số trung bình tuyệt đối |
| **DA** | **mean(sign(y) == sign(ŷ))** | **% dự đúng chiều tăng/giảm — quan trọng nhất với nhà đầu tư** |

**Tại sao Directional Accuracy quan trọng nhất:**
Nhà đầu tư không cần biết chính xác VN-Index tăng 1.3% hay 1.5%. Điều quan trọng là
biết **hướng**: tăng hay giảm để ra quyết định mua/bán. DA = 60% nghĩa là cứ 10 ngày
dự đúng chiều 6 ngày — đủ để tạo lợi nhuận nếu quản lý rủi ro tốt.

---

## 6. Cài đặt & Chạy

### Cài đặt

```bash
git clone https://github.com/<username>/causal_ts_prediction
cd causal_ts_prediction

python -m venv .venv
source .venv/bin/activate

pip install vnstock yfinance tigramite scikit-learn statsmodels \
            networkx matplotlib pandas "numpy<2" streamlit torch
```

> **Quan trọng:** Cài `numpy<2` để tránh conflict với tigramite và torch.

### Tải dữ liệu

```bash
source .venv/bin/activate

python data_loader.py          # monthly
python data_loader.py daily    # daily
```

### Chạy pipeline terminal

```bash
python main.py                 # monthly (tau_max=3, window=60)
python main.py daily           # daily   (tau_max=5, window=252)
```

### Chạy Web App

```bash
# Luôn dùng .venv/bin/streamlit (không dùng system streamlit)
.venv/bin/streamlit run app.py

# Hoặc dùng script
./run.sh app.py
```

---

## 7. Web App

| Tab | Nội dung |
|-----|---------|
| 📊 Dữ liệu | Load data, thống kê mô tả, ADF test, correlation heatmap, biểu đồ return |
| 🔗 Causal Discovery | Chạy PCMCI+, xem causal graph tương tác, bảng causal pairs + p-value |
| 📈 So sánh mô hình | Rolling window 3 mô hình, bảng MSE/MAE/DA, biểu đồ dự báo vs thực tế |
| 🎯 Dự báo kỳ tiếp theo | Dự báo return kỳ tiếp theo, tín hiệu tăng/giảm kèm lịch sử 20 kỳ |
| 💹 Khuyến nghị cổ phiếu | Dự báo giá từng mã, tính `Giá dự báo = Giá hiện tại × exp(predicted_return)`, khuyến nghị MUA/GIỮ/BÁN |

---

## 8. Kết quả thực nghiệm

**Cấu hình:** Daily, tau_max=5, pc_alpha=0.01, rolling window=252 ngày

### Causal features tìm được → VNINDEX_Return

| Feature | Lag (τ) | p-value | Causal Effect |
|---------|---------|---------|---------------|
| SP500 | t-1 | ≈ 0.000 | +0.244 |
| SP500 | t-2 | < 0.001 | +0.101 |
| MWG   | t-3 | 0.003   | -0.060 |

**Giải thích:** S&P 500 ngày hôm qua và 2 ngày trước là nguyên nhân nhân quả mạnh nhất
của VN-Index hôm nay — phù hợp với thực tế thị trường VN phụ thuộc nhiều vào xu hướng
thị trường Mỹ (do múi giờ: thị trường Mỹ đóng cửa sau khi thị trường VN đã đóng).

### So sánh mô hình

| Mô hình | MSE ↓ | MAE ↓ | Directional Accuracy ↑ |
|---------|-------|-------|------------------------|
| All-features Ridge | 0.000163 | 0.008860 | 34.97% |
| LASSO Ridge | 0.000168 | 0.008887 | 31.67% |
| **Causal Ridge (PCMCI+)** | **0.000154** | **0.008562** | **36.04%** |

→ **Causal Ridge đạt MSE thấp nhất và DA cao nhất**, xác nhận giả thuyết: causal features
cho kết quả tốt hơn correlation features.

---

## 9. Tài liệu tham khảo

- Runge, J. et al. (2019). *Detecting and quantifying causal associations in large nonlinear time series datasets.* **Science Advances**, 5(11).
- Zhang, X. et al. (2014). *A causal feature selection algorithm for stock prediction modeling.* **Neurocomputing**, 142, 48–59.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* Cambridge University Press.
- [tigramite](https://github.com/jakobrunge/tigramite) — PCMCI+ implementation (Runge et al.)
- [vnstock](https://github.com/thinh-vu/vnstock) — Dữ liệu chứng khoán Việt Nam
