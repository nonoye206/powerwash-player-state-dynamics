from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "step5_6_recovery_threshold_scan"
OUT_DIR = ROOT / "step5_6_recovery_threshold_scan"
BIN_PATH = IN_DIR / "recovery_probability_binned_trends.csv"
BEST_PATH = IN_DIR / "recovery_probability_best_thresholds.csv"
BASE_PATH = IN_DIR / "s4_recovery_threshold_scan_base.csv"


WIDTH, HEIGHT = 1500, 1100
PANEL_W, PANEL_H = 650, 320
MARGIN_X, MARGIN_Y = 90, 180
GAP_X, GAP_Y = 90, 95


COLORS = {
    "line": (32, 92, 126),
    "point": (25, 112, 105),
    "threshold": (184, 60, 60),
    "axis": (70, 70, 70),
    "grid": (220, 225, 228),
    "text": (34, 34, 34),
    "muted": (102, 112, 122),
    "bg": (250, 250, 248),
}


def font(size: int, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_text(draw, xy, text, size=24, bold=False, fill=None, anchor=None):
    draw.text(xy, text, font=font(size, bold), fill=fill or COLORS["text"], anchor=anchor)


def nice_num(x: float) -> str:
    if abs(x) >= 10:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.1f}"
    return f"{x:.2f}"


def panel_coords(panel_idx: int):
    row = panel_idx // 2
    col = panel_idx % 2
    x0 = MARGIN_X + col * (PANEL_W + GAP_X)
    y0 = MARGIN_Y + row * (PANEL_H + GAP_Y)
    return x0, y0, x0 + PANEL_W, y0 + PANEL_H


def draw_panel(draw, df, best, variable, title, subtitle, panel_idx):
    x0, y0, x1, y1 = panel_coords(panel_idx)
    left, right = x0 + 70, x1 - 20
    top, bottom = y0 + 55, y1 - 55
    plot_w, plot_h = right - left, bottom - top

    d = df[(df["threshold_days"].eq(30)) & (df["variable"].eq(variable))].copy()
    d = d.sort_values("variable_median")
    xs = d["variable_median"].astype(float).tolist()
    ys = d["recovery_rate"].astype(float).tolist()
    if not xs:
        return

    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        xmax = xmin + 1
    ymin, ymax = 0.0, max(0.75, max(ys) * 1.1)
    ymax = min(1.0, ymax)

    def sx(v):
        return left + (v - xmin) / (xmax - xmin) * plot_w

    def sy(v):
        return bottom - (v - ymin) / (ymax - ymin) * plot_h

    draw_text(draw, (x0, y0 - 38), title, size=25, bold=True)
    draw_text(draw, (x0, y0 - 12), subtitle, size=18, fill=COLORS["muted"])

    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = bottom - frac * plot_h
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        label = f"{int((ymin + frac * (ymax - ymin)) * 100)}%"
        draw_text(draw, (left - 12, yy), label, size=16, fill=COLORS["muted"], anchor="rm")

    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)
    draw_text(draw, (left, top - 26), "Recovery probability", size=16, fill=COLORS["muted"])

    pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
    if len(pts) > 1:
        draw.line(pts, fill=COLORS["line"], width=4)
    for px, py in pts:
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=COLORS["point"])

    b = best[(best["threshold_days"].eq(30)) & (best["variable"].eq(variable))]
    if not b.empty:
        threshold = float(b.iloc[0]["threshold"])
        if xmin <= threshold <= xmax:
            tx = sx(threshold)
            draw.line((tx, top, tx, bottom), fill=COLORS["threshold"], width=2)
            draw_text(draw, (tx + 8, top + 8), f"threshold {nice_num(threshold)}", size=16, fill=COLORS["threshold"])

    tick_vals = [xmin, (xmin + xmax) / 2, xmax]
    for tv in tick_vals:
        xx = sx(tv)
        draw.line((xx, bottom, xx, bottom + 6), fill=COLORS["axis"], width=1)
        draw_text(draw, (xx, bottom + 12), nice_num(tv), size=15, fill=COLORS["muted"], anchor="ma")

    for x, y, n in zip(xs, ys, d["sessions"].astype(int).tolist()):
        px, py = sx(x), sy(y)
        draw_text(draw, (px, py - 22), f"{y*100:.0f}%", size=15, fill=COLORS["text"], anchor="ma")
        draw_text(draw, (px, bottom + 34), f"n={n}", size=13, fill=COLORS["muted"], anchor="ma")


def make_main_plot(df, best):
    img = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_text(draw, (70, 35), "Recovery Probability After S4", size=36, bold=True)
    draw_text(draw, (70, 82), "30-day recovery to S1/S2 by binned tipping-point variables", size=22, fill=COLORS["muted"])

    panels = [
        ("recent_s4_density_prev5", "Recent S4 density, previous 5", "Ex-ante signal. Collapse point near 0.5."),
        ("gap_before_s4_days", "Gap before S4", "Ex-ante, but adjusted signal is weaker."),
        ("prior_s4_share", "Prior S4 share", "Historical exposure, descriptive drop."),
        ("observed_lifecycle_position", "Observed lifecycle position", "Hindsight descriptor, not early warning."),
    ]
    for i, (var, title, subtitle) in enumerate(panels):
        draw_panel(draw, df, best, var, title, subtitle, i)

    footer = "Outcome: next observed session recovers to S1/S2 within 30 days after S4. Points are quantile bins; red line is best threshold scan."
    draw_text(draw, (70, HEIGHT - 45), footer, size=18, fill=COLORS["muted"])
    out = OUT_DIR / "recovery_probability_binned_plot_30d.png"
    img.save(out)
    return out


def make_density_curve(df, best):
    img = Image.new("RGB", (1200, 760), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_text(draw, (70, 35), "Recent S4 Concentration and Recovery Probability", size=34, bold=True)
    draw_text(draw, (70, 80), "30-day recovery to S1/S2 after an S4 session", size=21, fill=COLORS["muted"])

    d = df[(df["threshold_days"].eq(30)) & (df["variable"].eq("recent_s4_density_prev5"))].copy()
    d = d.sort_values("variable_median")
    left, right = 115, 1080
    top, bottom = 165, 610
    ymin, ymax = 0.0, 0.42
    xmin, xmax = float(d["variable_median"].min()), float(d["variable_median"].max())

    def sx(v):
        return left + (float(v) - xmin) / (xmax - xmin) * (right - left)

    def sy(v):
        return bottom - (float(v) - ymin) / (ymax - ymin) * (bottom - top)

    for frac in [0, 0.25, 0.50, 0.75, 1.0]:
        yy = bottom - frac * (bottom - top)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw_text(draw, (left - 14, yy), f"{int((ymin + frac * (ymax - ymin))*100)}%", size=16, fill=COLORS["muted"], anchor="rm")
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)
    draw_text(draw, (left, top - 30), "Recovery probability", size=17, fill=COLORS["muted"])

    pts = [(sx(r.variable_median), sy(r.recovery_rate)) for r in d.itertuples(index=False)]
    draw.line(pts, fill=COLORS["line"], width=4)

    max_n = d["sessions"].max()
    for r in d.itertuples(index=False):
        x = sx(r.variable_median)
        y = sy(r.recovery_rate)
        radius = 5 + int(13 * (r.sessions / max_n) ** 0.5)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS["point"])
        draw_text(draw, (x, y - radius - 16), f"{r.recovery_rate*100:.0f}%", size=16, fill=COLORS["text"], anchor="ma")
        draw_text(draw, (x, bottom + 40), f"n={int(r.sessions)}", size=14, fill=COLORS["muted"], anchor="ma")

    tx = sx(0.5)
    draw.line((tx, top, tx, bottom), fill=COLORS["threshold"], width=2)
    draw_text(draw, (tx + 10, top + 12), "Empirical split around density >= 0.5", size=18, fill=COLORS["threshold"])

    for tv in [xmin, (xmin + xmax) / 2, xmax]:
        xx = sx(tv)
        draw.line((xx, bottom, xx, bottom + 7), fill=COLORS["axis"], width=1)
        draw_text(draw, (xx, bottom + 14), f"{tv:.2f}", size=16, fill=COLORS["muted"], anchor="ma")
    draw_text(draw, ((left + right) / 2, bottom + 76), "Binned recent S4 density (bin median)", size=18, fill=COLORS["muted"], anchor="ma")

    draw_text(draw, (70, 705), "Recovery probability is substantially lower when S4 becomes concentrated in recent play history.", size=18, fill=COLORS["muted"])
    out = OUT_DIR / "recovery_probability_curve_recent_s4_density_prev5_30d.png"
    img.save(out)
    return out


def main():
    df = pd.read_csv(BIN_PATH)
    best = pd.read_csv(BEST_PATH)
    p1 = make_main_plot(df, best)
    p2 = make_density_curve(df, best)
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
