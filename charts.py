from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Patch

from data_service import PRICE_COLS, long_prices, price_status, monthly_stats, quarter_stats, build_insights

COLORS = {"up": "#00c000", "down": "#ff2020", "ref": "#ffd400", "floor": "#1e90ff", "ceil": "#b050ff"}
LABELS = {"up": "Tăng (> tham chiếu)", "down": "Giảm (< tham chiếu)", "ref": "Tham chiếu",
          "floor": "Giá sàn", "ceil": "Giá trần"}
MONTHS = [f"T{m}" for m in range(1, 13)]

# Video 1 phút 30 giây = 90 giây = mở đầu 5s + chuyển động 73s + tổng kết 12s
FPS = 12
INTRO_SEC = 5
MOTION_SEC = 73       # 365 ngày, mỗi ngày 1 nến; 876 khung
OUTRO_SEC = 12
CANDLE_DAYS = 3                       # mỗi nến gom 3 ngày giao dịch (như video đầu tiên)
CANDLE_BODY = CANDLE_DAYS * 0.70     # độ rộng thân nến (đơn vị: ngày)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_monthly_mean(df):
    ms = monthly_stats(df)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, 12))
    bars = ax.bar(MONTHS, ms["Trung bình"], yerr=ms["Độ lệch chuẩn"], capsize=4, color=cols,
                  error_kw={"alpha": .6}, label="Trung bình")
    ax.scatter(MONTHS, ms["Trung vị"], marker="D", s=30, color="white", edgecolor="black",
               zorder=3, label="Trung vị")
    best = int(ms["Biên độ"].idxmax()) - 1
    bars[best].set_edgecolor("red"); bars[best].set_linewidth(2.5)
    for b, v in zip(bars, ms["Trung bình"]):
        ax.text(b.get_x() + b.get_width() / 2, 0, f"{v/1000:.1f}k", ha="center", va="bottom",
                color="white", fontsize=8, fontweight="bold")
    ax.set_title(f"Giá trung bình mỗi tháng (thanh = TB ± 1 độ lệch chuẩn; ◆ = trung vị; "
                 f"viền đỏ = biên độ lớn nhất, T{best+1})", fontsize=10)
    ax.set_ylabel("VND"); ax.grid(axis="y", alpha=.3); ax.legend(loc="lower right")
    return fig


def fig_boxplot(df):
    lp = long_prices(df)
    data = [lp.loc[lp.month == m, "price"] for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bp = ax.boxplot(data, tick_labels=MONTHS, patch_artist=True, showmeans=True,
                    flierprops={"marker": "o", "markersize": 3, "alpha": .5})
    for patch, c in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.15, 0.9, 12))):
        patch.set_facecolor(c); patch.set_alpha(.75)
    ax.set_title("Boxplot giá từng tháng (▲ = trung bình, vạch giữa hộp = trung vị)")
    ax.set_ylabel("VND"); ax.grid(axis="y", alpha=.3)
    return fig


def fig_quarters(df):
    qs = quarter_stats(df)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar([f"Q{q}" for q in qs.index], qs["Độ lệch chuẩn"], color="#4c78a8")
    axs[0].set_title("Độ lệch chuẩn giá theo quý (VND)")
    axs[1].bar([f"Q{q}" for q in qs.index], qs["Hệ số biến thiên (%)"], color="#f58518")
    axs[1].set_title("Hệ số biến thiên theo quý (%)")
    for a in axs: a.grid(axis="y", alpha=.3)
    fig.tight_layout()
    return fig


def fig_hist_halves(df):
    lp = long_prices(df)
    h1, h2 = lp[lp.quarter <= 2]["price"], lp[lp.quarter >= 3]["price"]
    bins = np.linspace(lp["price"].min(), lp["price"].max(), 35)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s, name, c in zip(axs, (h1, h2), ("Quý 1 + Quý 2", "Quý 3 + Quý 4"), ("#4c78a8", "#e45756")):
        ax.hist(s, bins=bins, color=c, alpha=.85, edgecolor="white")
        ax.axvline(s.mean(), color="k", ls="--", label=f"TB = {s.mean():,.0f}")
        ax.axvline(s.median(), color="orange", ls=":", lw=2, label=f"TV = {s.median():,.0f}")
        ax.set_title(name); ax.set_xlabel("Giá (VND)"); ax.legend()
    axs[0].set_ylabel("Số quan sát")
    fig.suptitle("Phân bố giá: nửa đầu năm so với nửa cuối năm")
    fig.tight_layout()
    return fig


def fig_trend(df):
    k, b = np.polyfit(df["day"], df["mean"], 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(df["date"], df["low"], df["high"], color="#9ecae1", alpha=.5, label="Khoảng thấp–cao trong ngày")
    ax.plot(df["date"], df["close"], lw=1, color="#08519c", label="Giá đóng cửa")
    ax.plot(df["date"], df["close"].rolling(14).mean(), color="orange", lw=2, label="MA 14 ngày")
    ax.plot(df["date"], k * df["day"] + b, "r--", lw=2, label=f"Xu hướng ({k:+.0f} VND/ngày)")
    ax.set_title("Xu hướng giá cả năm"); ax.legend(); ax.grid(alpha=.3); ax.set_ylabel("VND")
    return fig


def fig_anomalies(df, tbl):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df["date"], df["close"], color="grey", lw=.8)
    for key in ("floor", "ceil"):
        d = tbl[tbl["Đóng cửa ở"] == key]
        m = df["date"].dt.date.isin(d["Ngày"])
        ax.scatter(df.loc[m, "date"], df.loc[m, "close"], s=14, color=COLORS[key],
                   label=f"Đóng cửa tại {LABELS[key].lower()}")
    ax.set_title(f"Các ngày bất thường ({len(tbl)} ngày)"); ax.legend(); ax.grid(alpha=.3)
    return fig


def build_frames(n, fps=FPS, intro_sec=INTRO_SEC, motion_sec=MOTION_SEC, outro_sec=OUTRO_SEC):
    """Danh sách khung: mở đầu (mờ dần) -> 365 ngày chuyển động -> tổng kết (mờ dần)."""
    n_in, n_mo, n_out = int(intro_sec * fps), int(motion_sec * fps), int(outro_sec * fps)
    fade = fps                                     # mờ dần trong 1 giây
    intro = [("intro", min(1.0, (i + 1) / fade)) for i in range(n_in)]
    motion = [("motion", t) for t in np.linspace(1, n, n_mo)]
    outro = [("outro", min(1.0, (i + 1) / fade)) for i in range(n_out)]
    return intro + motion + outro


def fig_candles(df):
    """Biểu đồ nến lớn cả năm, mỗi nến gom 3 ngày (dùng cho trang web)."""
    fig, ax = plt.subplots(figsize=(16, 7.5))
    fig.patch.set_facecolor("#0b0f1a"); ax.set_facecolor("#0b0f1a")
    P = df[PRICE_COLS].to_numpy()
    for s in range(0, len(df), CANDLE_DAYS):
        e = min(s + CANDLE_DAYS, len(df)); last = e - 1
        o, c = P[s, 0], P[last, -1]
        hi, lo = P[s:e].max(), P[s:e].min()
        col = _status_color(c, df["ref"].iloc[last], df["floor"].iloc[last], df["ceil"].iloc[last])
        x = (s + e) / 2
        ax.plot([x, x], [lo, hi], color=col, lw=2.6)
        ax.add_patch(Rectangle((x - CANDLE_BODY / 2, min(o, c)), CANDLE_BODY, max(abs(c - o), 200), color=col))
    starts = df.groupby("month")["day"].min()
    ax.set_xticks(starts.to_numpy(), labels=[f"T{m}" for m in starts.index], color="white")
    ax.set_xlim(-2, len(df) + 2)
    ax.set_ylim(0, np.ceil(df["high"].max() * 1.1 / 10_000) * 10_000)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.tick_params(colors="white"); ax.set_ylabel("VND", color="white")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(axis="y", color="#222")
    ax.legend(handles=[Patch(color=COLORS[k_], label=LABELS[k_]) for k_ in COLORS], loc="upper right",
              facecolor="#111", edgecolor="#333", labelcolor="white", fontsize=11)
    fig.tight_layout()
    return fig


def _status_color(close, ref, floor, ceil):
    return COLORS[price_status(np.array([close]), np.array([ref]), np.array([floor]), np.array([ceil]))[0]]


class StockScene:
    """Bố cục: nến (mỗi nến gom 3 ngày) ở trên, đường 7 phiên ở dưới, số liệu cả năm ở mép phải."""

    def __init__(self, df: pd.DataFrame, figsize=(19.2, 10.8), dpi=100):
        self.df, self.n = df, len(df)
        self.k = k = len(PRICE_COLS)
        self.G = -(-self.n // CANDLE_DAYS)                 # số nến = làm tròn lên (n / 3)
        self.ys = build_insights(df)
        self.P = P = df[PRICE_COLS].to_numpy()
        self.ref, self.fl, self.ce = df["ref"].to_numpy(), df["floor"].to_numpy(), df["ceil"].to_numpy()
        self.op, self.cl = P[:, 0], P[:, -1]
        self.hi_d, self.lo_d = P.max(axis=1), P.min(axis=1)

        # đường 7 phiên: mỗi phiên một điểm, đoạn nối có màu theo quy ước HOSE
        self.px = (np.arange(self.n)[:, None] + np.arange(k)[None, :] / k).ravel()
        self.py = P.ravel()
        ref_t, fl_t, ce_t = (np.repeat(a, k) for a in (self.ref, self.fl, self.ce))
        self.pstat = price_status(self.py, ref_t, fl_t, ce_t)
        self.seg_colors = np.array([COLORS[s] for s in self.pstat[1:]])
        self.segs = np.stack([np.column_stack([self.px[:-1], self.py[:-1]]),
                              np.column_stack([self.px[1:], self.py[1:]])], axis=1)
        self.cm = np.cumsum(self.py) / np.arange(1, len(self.py) + 1)   # trung bình lũy kế theo từng phiên

        self.fig, (self.a1, self.a2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True,
                                                    gridspec_kw={"height_ratios": [1.1, 1.0]})
        self.fig.patch.set_facecolor("black")
        self.ymax = ymax = np.ceil(df["high"].max() * 1.12 / 10_000) * 10_000
        for a in (self.a1, self.a2):
            a.set_facecolor("black"); a.set_ylim(0, ymax); a.set_xlim(-2, self.n + 26)
            for sp in a.spines.values(): sp.set_color("white")
            a.tick_params(colors="white"); a.set_ylabel("VND", color="white")
            a.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        starts = df.groupby("month")["day"].min()          # trục x: đầu mỗi tháng
        self.a2.set_xticks(starts.to_numpy(), labels=[f"T{m}" for m in starts.index], color="white")
        self.a2.set_xlabel("Năm 2025", color="#888")
        self.a1.tick_params(labelbottom=False)
        self.fig.subplots_adjust(left=.07, right=.975, top=.90, bottom=.10, hspace=.10)

        # --- nến: mỗi nến gom CANDLE_DAYS ngày; OHLC của cả nến (dùng cho nến đã đủ ngày) ---
        self.g_first = np.arange(self.G) * CANDLE_DAYS
        self.g_last = np.minimum(self.g_first + CANDLE_DAYS - 1, self.n - 1)
        self.g_x = self.g_first + (self.g_last - self.g_first + 1) / 2
        self.g_open = self.op[self.g_first]
        self.g_close = self.cl[self.g_last]
        self.g_high = np.array([self.hi_d[f:l + 1].max() for f, l in zip(self.g_first, self.g_last)])
        self.g_low = np.array([self.lo_d[f:l + 1].min() for f, l in zip(self.g_first, self.g_last)])
        self.g_col = [_status_color(self.g_close[j], self.ref[self.g_last[j]], self.fl[self.g_last[j]],
                                    self.ce[self.g_last[j]]) for j in range(self.G)]
        self.wick, self.body = [], []
        for j in range(self.G):
            wk, = self.a1.plot([], [], color=self.g_col[j], lw=2.4, visible=False)
            bd = Rectangle((0, 0), CANDLE_BODY, 0, color=self.g_col[j], visible=False)
            self.a1.add_patch(bd)
            self.wick.append(wk); self.body.append(bd)

        # --- đường giá, trung bình lũy kế, tham chiếu / sàn / trần của ngày đang chạy ---
        self.line = LineCollection([], linewidths=2.8)
        self.a2.add_collection(self.line)
        self.rmean, = self.a2.plot([], [], color="#F59E0B", lw=2.2, ls="--", label="Trung bình lũy kế")
        self.ref_l = self.a2.axhline(0, color="#EAB308", lw=1.6, ls="--", label="Tham chiếu")
        self.fl_l = self.a2.axhline(0, color="#06B6D4", lw=1.6, ls=":", label="Giá sàn")
        self.ce_l = self.a2.axhline(0, color="#A855F7", lw=1.6, ls=":", label="Giá trần")
        self.spec = self.a2.scatter([], [], s=34, zorder=5)
        self.dot, = self.a2.plot([], [], "o", color="white", ms=9, zorder=6)

        # --- cao nhất / thấp nhất cả năm: cố định từ đầu video, ghi ở mép phải ---
        hi_y, lo_y = df["high"].max(), df["low"].min()
        self.a2.axhline(hi_y, color="#22C55E", lw=1.3, ls=":")
        self.a2.axhline(lo_y, color="#EF4444", lw=1.3, ls=":")
        self.a2.text(self.n + 3, hi_y, f"CAO NHẤT\n{hi_y:,.0f}", color="#22C55E", fontsize=14,
                     fontweight="bold", va="center")
        self.a2.text(self.n + 3, lo_y, f"THẤP NHẤT\n{lo_y:,.0f}", color="#EF4444", fontsize=14,
                     fontweight="bold", va="center")
        self.tcur = self.a2.text(self.n + 3, 0, "", color="white", fontsize=15, fontweight="bold", va="center")
        # --- chú thích: màu nến ở góc phải trên, đường ở góc trái trên của hình trên ---
        leg1 = self.a1.legend(handles=[Patch(color=COLORS[k_], label=LABELS[k_]) for k_ in COLORS],
                              loc="upper right", facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=12)
        self.a1.add_artist(leg1)
        self.a1.legend(handles=[self.rmean, self.ref_l, self.fl_l, self.ce_l], loc="upper left",
                       facecolor="#111", edgecolor="#555", labelcolor="white", fontsize=12)

        # --- tiêu đề, dòng thống kê lũy kế, nhãn góc, thanh tiến trình ---
        self.title = self.fig.text(.5, .955, "", color="white", ha="center", fontsize=22, fontweight="bold")
        self.sub = self.fig.text(.5, .925, "", color="#bbb", ha="center", fontsize=15)
        self.badge = self.fig.text(.03, .975, "ABC-8  •  VIN GROUP (GIẢ ĐỊNH)", color="#ffd400",
                                   fontsize=13, fontweight="bold", va="top")
        self.footer = self.fig.text(.5, .035, "", color="#FDE047", ha="center", fontsize=16, fontweight="bold")
        self.prog_ax = self.fig.add_axes([.09, .012, .88, .008])
        self.prog_ax.set_axis_off(); self.prog_ax.set_xlim(0, 1); self.prog_ax.set_ylim(0, 1)
        self.prog_ax.add_patch(Rectangle((0, 0), 1, 1, color="#222"))
        self.prog = Rectangle((0, 0), 0, 1, color="#ffd400")
        self.prog_ax.add_patch(self.prog)
        self.chrome = [self.title, self.sub, self.badge, self.footer, self.prog_ax]

        # --- mở đầu: tên công ty + vài nến đầu tiên của năm ---
        self.intro_ax = self.fig.add_axes([.12, .08, .76, .34])
        self.intro_ax.set_axis_off()
        self._draw_intro_candles()
        self.intro_texts = [
            self.fig.text(.5, .76, "VIN GROUP", color="#ffd400", ha="center", fontsize=60, fontweight="bold"),
            self.fig.text(.5, .68, "Mô phỏng biến động giá cổ phiếu ABC-8", color="white", ha="center", fontsize=28),
            self.fig.text(.5, .63, "Dữ liệu giả định  •  365 ngày giao dịch  •  7 phiên mỗi ngày",
                          color="#bbb", ha="center", fontsize=16),
        ]

        # --- tổng kết cuối video ---
        self.summary = self.fig.text(
            .5, .5, self._summary_text(), color="white", ha="center", va="center", fontsize=18,
            linespacing=1.9, visible=False,
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#0d1117", edgecolor="#FFD166", alpha=.92))

        for j in range(self.G):                     # đặt sẵn hình dạng đầy đủ của từng nến
            self._set_full(j)

    # ---------- nội dung chữ ----------
    def _summary_text(self):
        i = self.ys
        return (
            "TỔNG KẾT 365 NGÀY GIAO DỊCH\n\n"
            f"Trung bình: {i['year_mean']:,.0f} VND        Trung vị: {i['year_median']:,.0f} VND\n"
            f"Thay đổi cả năm: {i['year_change_pct']:+.2f}%        Xu hướng: {i['trend']}\n"
            f"Đỉnh: {i['peak_price']:,.0f} ({i['peak_date']:%d/%m/%Y})        "
            f"Đáy: {i['trough_price']:,.0f} ({i['trough_date']:%d/%m/%Y})\n"
            f"Biên độ ngày lớn nhất: {i['max_amp_value']:,.0f} VND ({i['max_amp_day']:%d/%m/%Y})\n"
            f"Ngày có yếu tố bất thường: {i['n_anom']} ngày"
        )

    # ---------- nến ----------
    def _put_candle(self, j, hi, lo, o, c, col):
        x = self.g_x[j]
        self.wick[j].set_data([x, x], [lo, hi]); self.wick[j].set_color(col)
        self.body[j].set_bounds(x - CANDLE_BODY / 2, min(o, c), CANDLE_BODY, max(abs(c - o), self.ymax * .002))
        self.body[j].set_color(col)

    def _set_full(self, j):
        """Nến j khi đã đủ ngày: OHLC của cả nến."""
        self._put_candle(j, self.g_high[j], self.g_low[j], self.g_open[j], self.g_close[j], self.g_col[j])

    def _forming(self, j, n_full, rem):
        """Nến j đang hình thành: OHLC tính trên các phiên đã hiện."""
        f, l = int(self.g_first[j]), int(self.g_last[j])
        done = np.arange(f, min(n_full, l + 1))              # ngày đã xong trong nến
        vals = [self.P[done].ravel()] if len(done) else []
        cur = n_full if rem and n_full <= l else None       # ngày đang giao dịch trong nến
        if cur is not None:
            vals.append(self.P[cur, :rem])
        allp = np.concatenate(vals)
        day = cur if cur is not None else n_full - 1
        col = _status_color(allp[-1], self.ref[day], self.fl[day], self.ce[day])
        return allp.max(), allp.min(), self.op[f], allp[-1], col

    def _draw_candles(self, n_full, rem):
        """n_full ngày đã xong; rem phiên của ngày đang chạy."""
        cur_day = n_full if rem else n_full - 1
        j_last = (n_full - 1) // CANDLE_DAYS if n_full >= 1 else -1
        for j in range(self.G):
            f, l = int(self.g_first[j]), int(self.g_last[j])
            if l < n_full:                                   # nến đã đủ ngày
                if j == j_last:
                    self._set_full(j)
                self.wick[j].set_visible(True); self.body[j].set_visible(True)
            elif f <= cur_day <= l:                          # nến đang hình thành
                self._put_candle(j, *self._forming(j, n_full, rem))
                self.wick[j].set_visible(True); self.body[j].set_visible(True)
            else:
                self.wick[j].set_visible(False); self.body[j].set_visible(False)

    def _draw_intro_candles(self):
        G = 13                                               # 13 nến đầu = 39 ngày
        for j in range(G):
            x = self.g_x[j]
            self.intro_ax.plot([x, x], [self.g_low[j], self.g_high[j]], color=self.g_col[j], lw=2.6)
            self.intro_ax.add_patch(Rectangle((x - CANDLE_BODY / 2, min(self.g_open[j], self.g_close[j])),
                                              CANDLE_BODY, max(abs(self.g_close[j] - self.g_open[j]), 1),
                                              color=self.g_col[j]))
        self.intro_ax.set_xlim(0, CANDLE_DAYS * G)
        self.intro_ax.set_ylim(self.g_low[:G].min() * .995, self.g_high[:G].max() * 1.005)

    # ---------- khung hình ----------
    def render(self, item):
        """Vẽ một khung. item = ("intro", độ_mờ) | ("motion", t) | ("outro", độ_mờ)."""
        kind, val = item
        chart_on = kind != "intro"
        for a in (self.a1, self.a2):
            a.set_visible(chart_on)
        for art in self.chrome:
            art.set_visible(chart_on)
        self.intro_ax.set_visible(not chart_on)
        for txt in self.intro_texts:
            txt.set_visible(not chart_on)
            txt.set_alpha(val if kind == "intro" else 1.0)

        if kind == "intro":
            self.summary.set_visible(False)
            return self.fig
        if kind == "motion":
            self.summary.set_visible(False)
            return self.draw(val)

        self.draw(self.n)                                   # giữ khung cuối, phủ bảng tổng kết
        self.summary.set_visible(True)
        self.summary.set_alpha(val)
        self.summary.get_bbox_patch().set_alpha(.92 * val)
        return self.fig

    def draw(self, t: float):
        """Vẽ trạng thái sau t ngày (1 <= t <= n). t là số lẻ thì phiên giá hiện dần trong ngày."""
        t = float(np.clip(t, 1, self.n))
        m = int(np.clip(round(t * self.k), self.k, self.n * self.k))   # số mức giá đã hiện
        n_full, rem = divmod(m, self.k)                                 # ngày đã xong, phiên của ngày đang chạy
        self._draw_candles(n_full, rem)

        self.line.set_segments(self.segs[:m - 1]); self.line.set_color(self.seg_colors[:m - 1])
        self.rmean.set_data(self.px[:m], self.cm[:m])
        day = min(n_full if rem else n_full - 1, self.n - 1)           # ngày đang giao dịch
        r = self.df.iloc[day]
        for ln, v in ((self.ref_l, r["ref"]), (self.fl_l, r["floor"]), (self.ce_l, r["ceil"])):
            ln.set_ydata([v, v])

        sp = np.where(np.isin(self.pstat[:m], ["floor", "ceil", "ref"]) & (np.arange(m) % self.k == self.k - 1))[0]
        self.spec.set_offsets(np.column_stack([self.px[sp], self.py[sp]]) if len(sp) else np.empty((0, 2)))
        self.spec.set_color([COLORS[s] for s in self.pstat[sp]])

        cur = self.py[m - 1]
        self.dot.set_data([self.px[m - 1]], [cur])
        self.tcur.set_position((self.n + 3, cur)); self.tcur.set_text(f"{cur:,.0f}")

        pct = (cur / r["ref"] - 1) * 100
        self.title.set_text(f"{r['date']:%d/%m/%Y}  •  Ngày {day + 1}/{self.n}")
        self.sub.set_text(f"Giá hiện tại {cur:,.0f}  |  Tham chiếu {r['ref']:,.0f}  |  {pct:+.1f}%  |  "
                          f"Cao nhất đến nay {self.py[:m].max():,.0f}  •  Thấp nhất đến nay {self.py[:m].min():,.0f}")
        self.sub.set_color(COLORS[self.pstat[m - 1]])

        seen = self.py[:m]
        self.footer.set_text(f"Đến ngày {day + 1}:   trung bình lũy kế {seen.mean():,.0f}   •   "
                             f"trung vị lũy kế {np.median(seen):,.0f}")
        self.prog.set_width(t / self.n)
        return self.fig


def build_animation(df, out_path="output/abc.mp4", fps=FPS, intro_sec=INTRO_SEC, motion_sec=MOTION_SEC,
                    outro_sec=OUTRO_SEC, dpi=100, verbose=True) -> str:
    """Xuất video 1 phút 30 giây. Có ffmpeg thì ra MP4; không có thì tự chuyển sang GIF."""
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = StockScene(df, dpi=dpi)
    frames = build_frames(scene.n, fps, intro_sec, motion_sec, outro_sec)
    anim = animation.FuncAnimation(scene.fig, scene.render, frames=frames, repeat=False)
    if animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    else:
        writer = animation.PillowWriter(fps=min(fps, 15))
        out_path = out_path.with_suffix(".gif")
    cb = (lambda i, n: print(f"\r  khung {i + 1}/{n}", end="") if i % 50 == 0 else None) if verbose else None
    anim.save(str(out_path), writer=writer, progress_callback=cb, savefig_kwargs={"facecolor": "black"})
    plt.close(scene.fig)
    if verbose: print(f"\nĐã lưu: {out_path} ({len(frames) / fps:.0f} giây)")
    return str(out_path)


def save_frame(df, item, out_png="output/preview.png") -> str:
    """Lưu 1 khung hình tĩnh để kiểm tra giao diện. item như trong build_frames."""
    scene = StockScene(df)
    scene.render(item).savefig(out_png, facecolor="black")
    plt.close(scene.fig)
    return out_png
