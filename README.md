# VIN Group — Mô phỏng biến động giá cổ phiếu ABC-8

Cổ phiếu giả định của VIN Group, dữ liệu 365 ngày × 7 phiên. Phân tích bằng numpy, pandas, matplotlib; web dùng FastAPI + uvicorn.

## Cấu trúc
```
main.py            web (FastAPI) và lệnh xuất video
charts.py          biểu đồ và video 1 phút 30 giây
data_service.py    đọc dữ liệu, tính thống kê, viết kết luận
data/abc-8.csv     dữ liệu mẫu (dấu tab)
templates/         giao diện web (dashboard.html)
static/            CSS
setup_venv.sh      tạo môi trường ảo .venv (macOS/Linux)
setup_venv.bat     tạo môi trường ảo .venv (Windows)
colab_run.ipynb    chạy toàn bộ trên Google Colab
```

## 1. Tạo môi trường ảo (venv)

**macOS / Linux**
```bash
bash setup_venv.sh
source .venv/bin/activate
```

**Windows** (double-click `setup_venv.bat`, hoặc trong Command Prompt)
```bat
setup_venv.bat
.venv\Scripts\activate.bat
```

Sau khi kích hoạt, dấu `(.venv)` hiện ở đầu dòng lệnh.

## 2. Chạy web bằng uvicorn
```bash
uvicorn main:app --reload
```
M�� http://127.0.0.1:8000 rồi bấm **▶ Xuất video** (video mất vài phút).

## 3. Xuất video bằng dòng lệnh (không cần web)
```bash
python main.py render          # tạo output/abc.mp4, cần ffmpeg
```

## 4. Đưa lên GitHub
```bash
git init
git add .
git commit -m "VIN Group ABC-8: web + video"
git branch -M main
git remote add origin https://github.com/<ten-tai-khoan>/<ten-repo>.git
git push -u origin main
```
`.gitignore` đã loại trừ `.venv/`, `output/` và file upload, nên repo chỉ chứa code và dữ liệu mẫu.

## 5. Chạy trên Google Colab
M�� `colab_run.ipynb` và chạy từ trên xuống.

## 6. Triển khai Render
Tạo Web Service từ repo GitHub. `render.yaml` đã có lệnh cài ffmpeg và chạy `uvicorn main:app`.
