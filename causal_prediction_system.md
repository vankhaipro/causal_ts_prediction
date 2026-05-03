# Causal TS Prediction: VN-Index Integration (Technical Specs)

Tài liệu này chi tiết hoá các thành phần kỹ thuật của hệ thống dự báo nhân quả VN-Index tích hợp tin tức VnExpress.

## 1. Dữ liệu (Data Sources)

| Thành phần | Nguồn | Phạm vi | Mô tả |
| :--- | :--- | :--- | :--- |
| **Tin tức** | VnExpress | 2016 – Nay | Chuyên mục Kinh doanh, Chứng khoán, Tài chính, Vĩ mô, Quốc tế. |
| **Thị trường VN** | vnstock (KBS) | 2015 – Nay | VN-Index, VCB, BID, VIC, HPG, MSN, MWG, GAS. |
| **Thế giới/Macro** | yfinance | 2015 – Nay | S&P 500, VIX, Dầu Brent, Vàng, DXY. |

## 2. Đặc trưng dữ liệu (Features)

### NLP Features (từ news_processor.py)
*   **Sentiment**: 
    *   `sentiment_mean`: Điểm tâm trạng trung bình [-1, 1].
    *   `sentiment_std`: Độ phân tán tâm trạng (ngày có nhiều ý kiến trái chiều).
    *   `sentiment_pos_pct` / `sentiment_neg_pct`: % bài báo tích cực/tiêu cực.
*   **Topics (LDA)**: Trọng số xác suất của 10-20 chủ đề.
    *   *Ví dụ chủ đề*: Chính sách tiền tệ, Xuất nhập khẩu, FDI, Chứng khoán.
*   **Volume**: `article_count` - Số lượng bài báo mỗi ngày.

### Financial Features (từ data_loader.py)
*   **Returns**: Log-returns của tất cả mã cổ phiếu và chỉ số macro.
*   **Volatility**: `VNINDEX_Vol5`, `VNINDEX_Vol20` (độ lệch chuẩn lăn).

## 3. Đầu vào & Đầu ra (I/O)

*   **Input**:
    *   Raw HTML/JSON từ VnExpress.
    *   Lịch sử giá CSV/API (OHLCV).
*   **Output**:
    *   `dataset_daily.csv` / `dataset.csv`: Dataset tổng hợp đã stationary.
    *   `causal_graph.png`: Đồ thị nhân quả với độ trễ τ.
    *   `model_comparison.png`: Biểu đồ so sánh MSE và Directional Accuracy.
    *   **Báo cáo**: Causal Impact Report (Bảng các nhân tố tin tức gây ra giá).

## 4. Thuật toán & Mô hình (Algorithms)

| Bước | Thuật toán | Mục tiêu |
| :--- | :--- | :--- |
| **Sentiment** | **PhoBERT** (wonrax/phobert-base-vietnamese-sentiment) | Phân loại cảm xúc văn bản tiếng Việt. |
| **Topic Modeling** | **LDA** (Latent Dirichlet Allocation) | Tự động phân loại tin tức theo chủ đề kinh tế. |
| **Causal Discovery** | **PCMCI+** | Tìm quan hệ nhân quả trong chuỗi thời gian, loại bỏ các mối quan hệ ảo do độ trễ hoặc biến trung gian. |
| **Nonlinear Test** | **CMIknn** (Conditional Mutual Information) | (Optional) Bắt các quan hệ phi tuyến giữa tin tức và giá. |
| **Forecasting** | **Ridge Regression / LSTM** | Dự báo giá dựa trên các "Causal Parents" đã tìm được. |

## 5. Thư viện sử dụng (Libraries)

*   **Dữ liệu**: `requests`, `beautifulsoup4`, `vnstock`, `yfinance`.
*   **NLP**: `transformers`, `torch`, `underthesea`, `gensim`.
*   **Causal**: `tigramite` (framework chính cho PCMCI+).
*   **Phân tích/ML**: `pandas`, `numpy`, `scikit-learn`, `statsmodels`.
*   **Visualization**: `matplotlib`, `networkx`, `seaborn`.

## 6. Tài liệu tham khảo (References / Papers)

1.  **PCMCI+**: *"Causal discovery from multivariate time series with conditional independence tests"* (Runge et al., 2019/2020).
2.  **Tigramite**: Python package for causal time series analysis.
3.  **PhoBERT**: *"PhoBERT: Pre-trained language models for Vietnamese"* (Nguyen & Tuan Nguyen, 2020).
4.  **Directional Accuracy**: Tiêu chuẩn đánh giá trong dự báo tài chính (quan trọng hơn MSE).

## 7. Tên các chủ đề (Topic Examples)

Dựa trên dữ liệu VnExpress, các chủ đề thường gặp:
1.  **Chính sách**: "lãi suất, ngân hàng nhà nước, điều hành, tiền tệ".
2.  **Đầu tư**: "fdi, vốn ngoại, dự án, khu công nghiệp".
3.  **Chứng khoán**: "vnindex, khối ngoại, tự doanh, thanh khoản".
4.  **Vĩ mô**: "gdp, lạm phát, cpi, tăng trưởng".
