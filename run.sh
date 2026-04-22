#!/bin/bash
# Chạy Streamlit apps với venv đúng

APP=${1:-app.py}
echo "Chạy: $APP"
.venv/bin/streamlit run "$APP"
