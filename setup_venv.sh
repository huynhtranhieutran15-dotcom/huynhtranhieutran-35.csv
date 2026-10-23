#!/usr/bin/env bash
# Tạo môi trường ảo .venv và cài thư viện trong requirements.txt
# Cách dùng (macOS/Linux):  bash setup_venv.sh
set -e
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Xong! Chạy web bằng:"
echo "  source .venv/bin/activate"
echo "  uvicorn main:app --reload"
