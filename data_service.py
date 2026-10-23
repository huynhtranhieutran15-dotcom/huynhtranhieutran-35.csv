import numpy as np
import pandas as pd

PRICE_COLS = [f"p{i}" for i in range(1, 8)]   # 7 phiên giao dịch mỗi ngày
BASE_PRICE = 10000
START_DATE = "2025-01-01"


def _read_raw(path):
    """Đọc file CSV, tự nhận dấu tab hoặc dấu phẩy (file ABC-8 dùng tab)."""
    with open(path, encoding="utf-8-sig") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    return pd.read_csv(path, header=None, sep=sep)


def load_prices(path, start=START_DATE, base_price=BASE_PRICE):
    df = _read_raw(path).iloc[:, :7].dropna().astype(float)
    df.columns = PRICE_COLS
    df = df.reset_index(drop=True)

    df["day"] = np.arange(len(df))
    df["date"] = pd.date_range(start, periods=len(df), freq="D")
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["open"] = df["p1"]
    df["close"] = df["p7"]
    df["high"] = df[PRICE_COLS].max(axis=1)
    df["low"] = df[PRICE_COLS].min(axis=1)
    df["mean"] = df[PRICE_COLS].mean(axis=1)
    df["ref"] = df["close"].shift(1).fillna(base_price)
    df["floor"] = df["ref"] * 0.9
    df["ceil"] = df["ref"] * 1.1
    df["ret"] = (df["close"] / df["ref"] - 1) * 100
    df["amp"] = df["high"] - df["low"]
    return df


def price_status(price, ref, floor, ceil):
    price, ref, floor, ceil = map(np.asarray, (price, ref, floor, ceil))
    s = np.where(price > ref, "up", "down").astype(object)
    s[np.isclose(price, ref)] = "ref"
    s[np.isclose(price, floor)] = "floor"
    s[np.isclose(price, ceil)] = "ceil"
    return s


def long_prices(df):
    return df[["date", "month", "quarter"] + PRICE_COLS].melt(
        id_vars=["date", "month", "quarter"], value_name="price"
    ).drop(columns="variable")


def _stats(s):
    t = pd.DataFrame({
        "Trung bình": s.mean(),
        "Trung vị": s.median(),
        "Độ lệch chuẩn": s.std(),
        "Thấp nhất": s.min(),
        "Cao nhất": s.max(),
    })
    t["Biên độ"] = t["Cao nhất"] - t["Thấp nhất"]
    t["Biên độ / TB (%)"] = t["Biên độ"] / t["Trung bình"] * 100
    t["Hệ số biến thiên (%)"] = t["Độ lệch chuẩn"] / t["Trung bình"] * 100
    return t.round(2)


def monthly_stats(df):
    t = _stats(long_prices(df).groupby("month")["price"])
    t.index.name = "Tháng"
    return t


def quarter_stats(df):
    t = _stats(long_prices(df).groupby("quarter")["price"])
    t["Biến động ngày (%)"] = df.groupby("quarter")["ret"].std().round(2)
    t.index.name = "Quý"
    return t


def year_stats(df):
    p = long_prices(df)["price"]
    return {"mean": p.mean(), "median": p.median()}


def find_anomalies(df, min_touch=6):
    rows = []
    for _, r in df.iterrows():
        p = r[PRICE_COLS].to_numpy()
        s = price_status(p, r["ref"], r["floor"], r["ceil"])
        nf, nc, nr = (s == "floor").sum(), (s == "ceil").sum(), (s == "ref").sum()
        no = ((p < r["floor"] - 1e-6) | (p > r["ceil"] + 1e-6)).sum()

        if nf + nc >= min_touch or nr or no:
            note = []
            if nf: note.append(f"chạm sàn {nf} lần")
            if nc: note.append(f"chạm trần {nc} lần")
            if nr: note.append(f"bằng tham chiếu {nr} lần")
            if nf == 7: note.append("KỊCH SÀN cả ngày")
            if nc == 7: note.append("KỊCH TRẦN cả ngày")
            if no: note.append(f"vượt khung {no} lần")
            rows.append({
                "Ngày": r["date"].date(),
                "Tham chiếu": r["ref"], "Sàn": r["floor"],
                "Trần": r["ceil"], "Đóng cửa": r["close"],
                "Đóng cửa ở": s[-1], "Số lần sàn": nf,
                "Số lần trần": nc, "Số lần = TC": nr,
                "Ghi chú": "; ".join(note),
            })
    return pd.DataFrame(rows)


def build_insights(df):
    m, q = monthly_stats(df), quarter_stats(df)
    ys = year_stats(df)
    anom = find_anomalies(df)
    hi, lo = df["high"].idxmax(), df["low"].idxmin()
    amp = df["amp"].idxmax()
    slope = np.polyfit(df["day"], df["mean"], 1)[0]
    first, last = df["open"].iloc[0], df["close"].iloc[-1]
    change = (last / first - 1) * 100
    trend = "TĂNG" if slope > 0 else "GIẢM"

    gap = ys["mean"] - ys["median"]
    if abs(gap) < 0.005 * ys["median"]:
        gap_txt = "hai số này gần bằng nhau, giá phân bố khá đều quanh mức giữa."
    elif gap > 0:
        gap_txt = "trung bình cao hơn trung vị, nghĩa là một số ngày giá cao đã kéo trung bình lên."
    else:
        gap_txt = "trung bình thấp hơn trung vị, nghĩa là một số ngày giá thấp đã kéo trung bình xuống."

    n_full = int(((anom["Số lần sàn"] == 7) | (anom["Số lần trần"] == 7)).sum()) if len(anom) else 0
    best, worst = int(m["Trung bình"].idxmax()), int(m["Trung bình"].idxmin())
    wide = int(m["Biên độ"].idxmax())
    q_hi, q_lo = int(q["Độ lệch chuẩn"].idxmax()), int(q["Độ lệch chuẩn"].idxmin())

    conclusions = [
        f"Xu hướng: đường xu hướng {trend}, độ dốc {slope:+,.0f} VND mỗi ngày. "
        f"Giá đóng cửa ngày cuối năm so với ngày đầu năm thay đổi {change:+.2f}%.",
        f"Trung bình và trung vị cả năm: {ys['mean']:,.0f} VND và {ys['median']:,.0f} VND; {gap_txt}",
        f"Theo tháng: T{best} có giá trung bình cao nhất ({m.loc[best, 'Trung bình']:,.0f} VND), "
        f"T{worst} thấp nhất ({m.loc[worst, 'Trung bình']:,.0f} VND). T{wide} có biên độ lớn nhất "
        f"({m.loc[wide, 'Biên độ']:,.0f} VND), đây là tháng dao động mạnh nhất.",
        f"Theo quý: Q{q_hi} có độ lệch chuẩn lớn nhất ({q.loc[q_hi, 'Độ lệch chuẩn']:,.0f} VND), "
        f"Q{q_lo} nhỏ nhất ({q.loc[q_lo, 'Độ lệch chuẩn']:,.0f} VND). Độ lệch chuẩn càng lớn thì giá càng phân tán.",
        f"Ngày bất thường: {len(anom)} ngày có giá chạm sàn/trần từ 6 phiên trở lên, hoặc có phiên bằng giá "
        f"tham chiếu, hoặc vượt khung sàn–trần. Trong đó {n_full} ngày kịch sàn hoặc kịch trần cả ngày.",
        f"Đỉnh giá {df.loc[hi, 'high']:,.0f} VND vào ngày {df.loc[hi, 'date']:%d/%m/%Y}; "
        f"đáy giá {df.loc[lo, 'low']:,.0f} VND vào ngày {df.loc[lo, 'date']:%d/%m/%Y}.",
        f"Biên độ dao động lớn nhất trong một ngày là {df.loc[amp, 'amp']:,.0f} VND, "
        f"xảy ra vào ngày {df.loc[amp, 'date']:%d/%m/%Y}.",
        "Vì sao đường đi xuống nhưng màu xanh: màu của mỗi đoạn không phụ thuộc hướng lên hay xuống, "
        "mà do giá tại thời điểm đó so với giá tham chiếu của chính ngày đó. Xanh lá nghĩa là cao hơn "
        "tham chiếu, đỏ là thấp hơn. Đường có thể đi xuống nhưng vẫn ở trên giá tham chiếu nên vẫn xanh.",
    ]

    lo_all, hi_all = df["low"].min(), df["high"].max()
    span = hi_all - lo_all
    pos = lambda v: (v - lo_all) / span * 100          # vị trí (%) trên thanh phạm vi giá năm
    return {
        "year_change_pct": change,
        "trend": trend,
        "year_mean": ys["mean"],
        "year_median": ys["median"],
        "peak_price": df.loc[hi, "high"],
        "peak_date": df.loc[hi, "date"].date(),
        "trough_price": df.loc[lo, "low"],
        "trough_date": df.loc[lo, "date"].date(),
        "max_amp_day": df.loc[amp, "date"].date(),
        "max_amp_value": df.loc[amp, "amp"],
        "n_anom": len(anom),
        "range_lo": lo_all, "range_hi": hi_all,
        "pos_mean": pos(ys["mean"]), "pos_median": pos(ys["median"]),
        "pos_peak": pos(df.loc[hi, "high"]), "pos_trough": pos(df.loc[lo, "low"]),
        "conclusions": conclusions,
    }
