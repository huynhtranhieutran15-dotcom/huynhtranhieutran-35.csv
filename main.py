import matplotlib
matplotlib.use("Agg")                      

import os
import sys
from functools import lru_cache
from pathlib import Path

import charts
import data_service as ds

BASE = Path(__file__).parent
DATA_PATH = Path(os.getenv("DATA_PATH", BASE / "data" / "abc-8.csv"))
OUTPUT = BASE / "output"
VIDEO_MP4 = OUTPUT / "abc.mp4"
VIDEO_GIF = OUTPUT / "abc.gif"             
status = {"rendering": False, "error": ""}


def video_file():
    for p in (VIDEO_MP4, VIDEO_GIF):
        if p.exists():
            return p
    return None


def clear_video():
    for p in (VIDEO_MP4, VIDEO_GIF):
        p.unlink(missing_ok=True)


@lru_cache(maxsize=2)
def _compute(path: str, mtime: float) -> dict:
    
    df = ds.load_prices(path)
    anom = ds.find_anomalies(df)
    html = lambda t: t.to_html(classes="tbl", float_format=lambda v: f"{v:,.2f}", border=0, index_names=False)
    ms = ds.monthly_stats(df)                      # dữ liệu cho bản đồ nhiệt theo tháng
    lo_m, hi_m = ms["Trung bình"].min(), ms["Trung bình"].max()
    heat = [{"label": f"T{i}", "mean": r["Trung bình"], "median": r["Trung vị"],
             "alpha": 0.2 + 0.8 * (r["Trung bình"] - lo_m) / (hi_m - lo_m),
             "pct": 100 * r["Trung bình"] / hi_m, "med_pct": 100 * r["Trung vị"] / hi_m}
            for i, r in ms.iterrows()]
    return {
        "heat": heat,
        "candle_img": charts.fig_to_b64(charts.fig_candles(df)),
        "df": df, "info": ds.build_insights(df),
        "monthly": html(ds.monthly_stats(df)), "quarterly": html(ds.quarter_stats(df)),
        "anom": anom.to_html(classes="tbl", index=False, border=0), "n_anom": len(anom),
        "figs": {
            "Giá trung bình & trung vị theo tháng": charts.fig_to_b64(charts.fig_monthly_mean(df)),
            "Boxplot theo tháng": charts.fig_to_b64(charts.fig_boxplot(df)),
            "Biến động theo quý": charts.fig_to_b64(charts.fig_quarters(df)),
            "Phân bố giá: Q1+Q2 vs Q3+Q4": charts.fig_to_b64(charts.fig_hist_halves(df)),
            "Xu hướng cả năm": charts.fig_to_b64(charts.fig_trend(df)),
            "Ngày bất thường": charts.fig_to_b64(charts.fig_anomalies(df, anom)),
        },
    }


def render_video(path=None) -> str:
    
    df = ds.load_prices(Path(path) if path else DATA_PATH)
    clear_video()
    return charts.build_animation(df, VIDEO_MP4)


def _render_job():
    status.update(rendering=True, error="")
    try:
        render_video()
    except Exception as e:                
        status["error"] = f"{type(e).__name__}: {e}"
    finally:
        status["rendering"] = False


def create_app():
    from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    app = FastAPI(title="VIN Group - Mô phỏng biến động giá ABC-8")
    app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
    templates = Jinja2Templates(directory=str(BASE / "templates"))

    @app.get("/")
    def dashboard(request: Request):
        data = _compute(str(DATA_PATH), DATA_PATH.stat().st_mtime)
        ctx = {k: v for k, v in data.items() if k != "df"}
        vf = video_file()
        ctx.update(has_video=vf is not None, is_gif=bool(vf and vf.suffix == ".gif"),
                   rendering=status["rendering"], render_error=status["error"],
                   file_name=DATA_PATH.name)
        return templates.TemplateResponse(request, "dashboard.html", ctx)

    @app.get("/api/monthly")
    def api_monthly():
        df = ds.load_prices(DATA_PATH)
        return JSONResponse(ds.monthly_stats(df).reset_index().to_dict(orient="records"))

    @app.get("/video")
    def video():
        vf = video_file()
        if vf is None:
            return JSONResponse({"error": "chưa có video"}, status_code=404)
        return FileResponse(vf, media_type="video/mp4" if vf.suffix == ".mp4" else "image/gif")

    @app.post("/render")
    def render(background: BackgroundTasks):
        if not status["rendering"]:
            status["rendering"] = True     
            background.add_task(_render_job)
        return RedirectResponse("/", status_code=303)

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        global DATA_PATH
        target = BASE / "data" / "uploaded.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await file.read())
        try:
            ds.load_prices(target)
        except Exception as e:
            target.unlink(missing_ok=True)
            return JSONResponse({"error": f"File không đúng định dạng: {e}"}, status_code=400)
        DATA_PATH = target
        clear_video()
        return RedirectResponse("/", status_code=303)

    return app


try:
    app = create_app()
except ImportError:                        
    app = None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        print(render_video(sys.argv[2] if len(sys.argv) > 2 else None))
