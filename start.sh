#!/bin/bash
set -e

echo "Поднимаем Streamlit"
uv run streamlit run app.py --server.port=8501 --server.address=0.0.0.0