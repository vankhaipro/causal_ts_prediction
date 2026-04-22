# Khoá Luận Tốt Nghiệp — Dự Báo VN-Index bằng Causal Inference

> **Tên đề tài:** Ứng dụng thuật toán nhân quả trong dự báo chỉ số chứng khoán Việt Nam (VN-Index)
> **Sinh viên:** [Tên sinh viên] — Ngành Trí Tuệ Nhân Tạo

---

## Mục Lục

1. [Tổng Quan Đề Tài](#1-tổng-quan-đề-tài)
2. [Luồng Pipeline](#2-luồng-pipeline)
3. [Cấu Trúc Thư Mục](#3-cấu-trúc-thư-mục)
4. [Dữ Liệu](#4-dữ-liệu)
5. [Phương Pháp](#5-phương-pháp)
6. [Kết Quả & Đánh Giá](#6-kết-quả--đánh-giá)
7. [Cài Đặt & Chạy](#7-cài-đặt--chạy)
8. [Lộ Trình Phát Triển](#8-lộ-trình-phát-triển)

---

## 1. Tổng Quan Đề Tài

### Vấn đề
Hầu hết các mô hình dự báo chứng khoán truyền thống (ARIMA, LSTM, v.v.) chọn features dựa trên **tương quan (correlation)** — phương pháp này không phân biệt được nguyên nhân thật sự và biến nhiễu, dẫn đến overfitting và kết quả kém ổn định khi thị trường thay đổi cấu trúc.

### Giải pháp
Khoá luận đề xuất pipeline **Causal Discovery → Forecasting**:
1. Dùng **PCMCI+** (tigramite) để học cấu trúc nhân quả có độ trễ giữa các biến tài chính
2. Chỉ dùng các biến được chứng minh là **nguyên nhân thật sự** (causal features) để xây dựng mô hình dự báo
3. So sánh với các baseline dùng correlation (All-features Ridge, LASSO Ridge)

### Đóng góp chính
- Áp dụng PCMCI+ vào thị trường chứng khoán Việt Nam — thị trường mới nổi còn ít nghiên cứu
- Kết hợp dữ liệu nội địa (VN blue-chip stocks) với macro toàn cầu (S&P500, VIX, Oil, Gold, DXY)
- Chứng minh định lượng: causal features cho Directional Accuracy cao hơn correlation features

---

## 2. Luồng Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LUỒNG PIPELINE                              │
└─────────────────────────────────────────────────────────────────────┘

  [Nguồn dữ liệu]
       │
       ├── vnstock (KBS) ──→ VN-Index + 7 blue-chip (VCB, BID, VIC,
       │                     HPG, MSN, MWG, GAS)  [daily / monthly]
       │
       └── yfinance ────────→ S&P500, VIX, Brent Oil, Gold, DXY
                              [daily / monthly]
       │
       ▼
  [Bước 1: Tiền xử lý]
  ┌─────────────────────────────────┐
  │  • Tính log return              │
  │  • Align ngày giao dịch         │
  │  • Forward-fill gaps nhỏ        │
  │  • Lưu Data/dataset[_daily].csv │
  └─────────────────────────────────┘
       │
       ▼
  [Bước 2: Kiểm tra Stationarity]
  ┌─────────────────────────────────┐
  │  ADF Test (Augmented Dickey-    │
  │  Fuller) cho từng cột           │
  │  → Log returns thường stationary│
  └─────────────────────────────────┘
       │
       ▼
  [Bước 3: Causal Discovery — PCMCI+]
  ┌─────────────────────────────────────────────────────┐
  │  Input : ma trận T × N (T quan sát, N biến)         │
  │  Method: PCMCI+ với ParCorr independence test       │
  │  tau_min=1 → chỉ học X(t-τ) → Y(t)                 │
  │            → tránh look-ahead bias by design        │
  │  tau_max = 3 (monthly) / 5 (daily)                  │
  │  Output: Causal graph (n_vars × n_vars × tau_max+1) │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  [Bước 4: Trích xuất Causal Features]
  ┌─────────────────────────────────────────────────────┐
  │  Tìm các cạnh graph[i, target, τ] == '-->'          │
  │  → Danh sách (feature_name, lag) đã được xác nhận   │
  │    là nguyên nhân của VNINDEX_Return                 │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  [Bước 5: So sánh 3 mô hình — Rolling Window Ridge]
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  Baseline 1: All-features Ridge                                │
  │  → Dùng tất cả 12 features (correlation-based)                 │
  │                                                                │
  │  Baseline 2: LASSO Ridge                                       │
  │  → LassoCV chọn features quan trọng nhất (shrinkage)           │
  │                                                                │
  │  Model đề xuất: Causal Ridge (PCMCI+)                          │
  │  → Chỉ dùng features được PCMCI+ xác nhận là nhân quả         │
  │                                                                │
  │  Rolling window: 60 tháng / 252 ngày giao dịch                 │
  │  Lag-1 shift: X(t-1) → Y(t) trong forecast                    │
  └────────────────────────────────────────────────────────────────┘
       │
       ▼
  [Bước 6: Đánh giá]
  ┌─────────────────────────────────────────────────────┐
  │  • MSE  — Mean Squared Error (thấp hơn = tốt hơn)   │
  │  • MAE  — Mean Absolute Error                        │
  │  • DA   — Directional Accuracy: % dự đúng tăng/giảm │
  │           (metric quan trọng nhất trong finance)     │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  [Output]
  ┌─────────────────────────────────┐
  │  causal_graph[_daily].png       │
  │  model_comparison[_daily].png   │
  └─────────────────────────────────┘
```

---

## 3. Cấu Trúc Thư Mục

```
causal_ts_prediction/
│
├── data_loader.py          # Tải & xử lý dữ liệu (vnstock + yfinance)
├── causal_forecaster.py    # Class CausalForecaster: ADF, PCMCI+, Ridge, plot
├── main.py                 # Pipeline orchestrator
│
├── Data/
│   ├── vn_stocks.csv           # VN-Index + blue-chip (monthly log return)
│   ├── macro.csv               # S&P500, VIX, OIL, GOLD, DXY (monthly)
│   ├── dataset.csv             # Dataset tổng hợp monthly (dùng cho train)
│   ├── vn_stocks_daily.csv     # VN stocks (daily log return)
│   ├── macro_daily.csv         # Macro (daily)
│   └── dataset_daily.csv       # Dataset tổng hợp daily
│
├── causal_graph.png            # Causal graph monthly
├── causal_graph_daily.png      # Causal graph daily
├── model_comparison.png        # So sánh mô hình monthly
├── model_comparison_daily.png  # So sánh mô hình daily
│
└── my-stock-causal.ipynb       # Notebook thực nghiệm
```

---

## 4. Dữ Liệu

### 4.1 Biến mục tiêu (Target)
| Biến | Mô tả | Nguồn |
|------|-------|-------|
| `VNINDEX_Return` | Log return của VN-Index | vnstock (KBS) |

### 4.2 Features — VN Blue-chip Stocks
| Mã | Công ty | IPO | Lý do chọn |
|----|---------|-----|------------|
| VCB | Vietcombank | 2009 | Ngân hàng lớn nhất VN theo vốn hoá |
| BID | BIDV | 2013 | Ngân hàng quốc doanh, ảnh hưởng chính sách |
| VIC | Vingroup | 2007 | Tập đoàn tư nhân lớn nhất VN |
| HPG | Hoa Phát Group | 2007 | Thép — chỉ báo chu kỳ công nghiệp |
| MSN | Masan Group | 2010 | Tiêu dùng — chỉ báo nhu cầu nội địa |
| MWG | Mobile World | 2014 | Bán lẻ — chỉ báo tiêu dùng |
| GAS | PetroVietnam Gas | 2012 | Năng lượng — liên quan giá dầu |

> **Loại khỏi danh sách:** TCB (Techcombank, IPO 2018) và VHM (Vinhomes, IPO 2018) vì dữ liệu chỉ có từ 2018, làm cắt ngắn toàn bộ dataset.

### 4.3 Features — Macro Toàn Cầu
| Tên | Ticker | Mô tả |
|-----|--------|-------|
| SP500 | ^GSPC | S&P 500 — thị trường Mỹ dẫn dắt xu hướng toàn cầu |
| VIX | ^VIX | CBOE VIX — chỉ số sợ hãi, đo lường rủi ro thị trường |
| OIL | BZ=F | Dầu Brent — ảnh hưởng trực tiếp GAS, HPG, chi phí sản xuất |
| GOLD | GC=F | Vàng — tài sản trú ẩn khi thị trường bất ổn |
| DXY | DX-Y.NYB | US Dollar Index — áp lực tỷ giá, dòng vốn ngoại |

### 4.4 Thống kê dataset
| Tần suất | Số quan sát | Thời gian | Số features |
|----------|-------------|-----------|-------------|
| Monthly | ~123 tháng | 2016-01 → 2026-03 | 12 features + 1 target |
| Daily | ~2,498 ngày | 2015-12 → 2026-03 | 12 features + 1 target |

---

## 5. Phương Pháp

### 5.1 PCMCI+ (Causal Discovery)
PCMCI+ là thuật toán học cấu trúc nhân quả có độ trễ cho time series, được phát triển bởi Runge et al. (2019).

**Ưu điểm so với PC Algorithm:**
- PC Algorithm thiết kế cho dữ liệu i.i.d. (độc lập, phân phối giống nhau) — không phù hợp với time series
- PCMCI+ xử lý đúng cấu trúc temporal: tìm X(t-τ) → Y(t) với τ = 1, 2, ..., tau_max
- `tau_min=1` đảm bảo không có look-ahead bias theo thiết kế

**Tham số:**
| Tham số | Monthly | Daily | Ý nghĩa |
|---------|---------|-------|---------|
| tau_min | 1 | 1 | Lag tối thiểu — tránh look-ahead bias |
| tau_max | 3 | 5 | Lag tối đa xét đến |
| pc_alpha | 0.05 | 0.01 | Ngưỡng ý nghĩa thống kê |
| cond_ind_test | ParCorr | ParCorr | Partial Correlation test |

### 5.2 Rolling Window Ridge Regression (Forecasting)
- Mỗi bước dự báo: dùng `window_size` quan sát gần nhất để fit Ridge, dự báo bước tiếp theo
- Features được shift 1 bước (X(t-1) → Y(t)) để đảm bảo không dùng thông tin tương lai
- StandardScaler được fit mới trong mỗi window (tránh data leakage)

**Tham số:**
| Tham số | Monthly | Daily |
|---------|---------|-------|
| window_size | 60 tháng | 252 ngày |
| Ridge alpha | 1.0 | 1.0 |

### 5.3 Baseline Comparison
| Model | Feature Selection | Mục đích |
|-------|------------------|---------|
| All-features Ridge | Không chọn — dùng hết | Baseline đơn giản nhất |
| LASSO Ridge | LassoCV — shrinkage | Baseline correlation-based tốt nhất |
| **Causal Ridge (PCMCI+)** | **PCMCI+ — causal** | **Model đề xuất** |

### 5.4 Metrics Đánh Giá
| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| MSE | mean((y - ŷ)²) | Sai số bình phương trung bình |
| MAE | mean(\|y - ŷ\|) | Sai số tuyệt đối trung bình |
| **DA** | **mean(sign(y) == sign(ŷ))** | **% dự đúng chiều tăng/giảm — quan trọng nhất trong trading** |

---

## 6. Kết Quả & Đánh Giá

> *Phần này sẽ được cập nhật sau khi chạy thực nghiệm đầy đủ.*

**Hypothesis:** Causal Ridge (PCMCI+) sẽ đạt Directional Accuracy cao hơn 2 baseline vì:
1. Loại bỏ spurious correlations — các biến tương quan nhưng không phải nguyên nhân thật
2. Giảm số chiều (ít features hơn) → giảm overfitting trong rolling window nhỏ
3. Features được chọn ổn định hơn theo thời gian

---

## 7. Cài Đặt & Chạy

### Yêu cầu
```
Python 3.11
numpy < 2.0   (tương thích torch dependencies)
```

### Cài đặt
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install vnstock yfinance tigramite scikit-learn statsmodels networkx matplotlib pandas numpy
```

### Tải dữ liệu
```bash
# Tải dữ liệu monthly (mặc định)
python data_loader.py

# Tải dữ liệu daily
python data_loader.py daily
```

### Chạy pipeline
```bash
# Pipeline với dữ liệu monthly
python main.py

# Pipeline với dữ liệu daily (chạy lâu hơn ~10-30 phút)
python main.py daily
```

### Output
| File | Mô tả |
|------|-------|
| `causal_graph.png` | Đồ thị nhân quả (monthly) |
| `causal_graph_daily.png` | Đồ thị nhân quả (daily) |
| `model_comparison.png` | So sánh 3 mô hình (monthly) |
| `model_comparison_daily.png` | So sánh 3 mô hình (daily) |

---

## 8. Lộ Trình Phát Triển

| Tuần | Nội dung | Trạng thái |
|------|----------|------------|
| 1–2 | Thu thập dữ liệu, thiết lập pipeline cơ bản | ✅ Hoàn thành |
| 3–4 | Implement PCMCI+, baseline comparison, Directional Accuracy | ✅ Hoàn thành |
| 5–6 | Thêm LSTM (PyTorch, rolling window, causal features) | ✅ Hoàn thành |
| 7–8 | Web app Streamlit — 4 tab: Data / Causal / Models / Forecast | ✅ Hoàn thành |
| 9–10 | Viết báo cáo khoá luận, tổng hợp kết quả | 🔲 Chưa làm |

### TODO tiếp theo
- [ ] Chạy pipeline đầy đủ và ghi lại kết quả thực nghiệm (MSE, MAE, DA)
- [ ] So sánh kết quả monthly vs daily trong báo cáo
- [ ] Thêm Granger Causality làm baseline causal thứ 2 (optional)
- [ ] Viết báo cáo khoá luận (chương 3: Phương pháp, chương 4: Thực nghiệm)

---

## Tài Liệu Tham Khảo

- Runge, J., et al. (2019). *Detecting and quantifying causal associations in large nonlinear time series datasets.* Science Advances.
- Zhang, X., et al. (2014). *A causal feature selection algorithm for stock prediction modeling.* Neurocomputing.
- tigramite library: https://github.com/jakobrunge/tigramite
- vnstock: https://github.com/thinh-vu/vnstock
