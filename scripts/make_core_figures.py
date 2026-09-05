from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

RECOVERY_RISK = ROOT / "step5_3_adjusted_recovery_path_model" / "adjusted_recovery_path_unadjusted_risk.csv"
RECOVERY_COEF = ROOT / "step5_3_adjusted_recovery_path_model" / "adjusted_recovery_path_coefficients.csv"
SPLINE_SRC = ROOT / "step5_7_recovery_spline_model" / "recovery_spline_curve_recent_s4_density_prev5_30d.png"


COLORS = {
    "bg": (250, 250, 248),
    "text": (34, 34, 34),
    "muted": (98, 108, 118),
    "axis": (64, 64, 64),
    "grid": (218, 224, 228),
    "s4": (184, 60, 60),
    "recovery": (25, 112, 105),
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


def make_recovery_vs_persistence():
    risk = pd.read_csv(RECOVERY_RISK)
    coef = pd.read_csv(RECOVERY_COEF)
    risk30 = risk[risk["analysis"].eq("main_30d")].copy()
    recovery = risk30[risk30["path_label"].eq("S4->S1/S2")].iloc[0]
    persistence = risk30[risk30["path_label"].eq("S4->S4")].iloc[0]
    adjusted = coef[(coef["analysis"].eq("main_30d")) & (coef["term"].eq("recovery_path"))].iloc[0]
    robust = coef[(coef["analysis"].eq("exclude_update_window_30d")) & (coef["term"].eq("recovery_path"))].iloc[0]

    width, height = 1200, 760
    img = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_text(draw, (70, 35), "Recovery Path After S4 and Subsequent Disengagement", size=33, bold=True)
    draw_text(draw, (70, 80), "Destination-session 30-day observed disengagement risk", size=21, fill=COLORS["muted"])

    left, right = 190, 1030
    top, bottom = 175, 540
    ymax = 0.50

    def sy(v):
        return bottom - v / ymax * (bottom - top)

    for frac in [0, 0.25, 0.50, 0.75, 1.0]:
        y = frac * ymax
        yy = sy(y)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw_text(draw, (left - 15, yy), f"{int(y*100)}%", size=17, fill=COLORS["muted"], anchor="rm")
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)
    draw_text(draw, (left, top - 32), "30-day disengagement risk", size=17, fill=COLORS["muted"])

    bars = [
        ("S4->S1/S2\nrecovery", float(recovery["risk"]), int(recovery["sessions"]), COLORS["recovery"]),
        ("S4->S4\npersistence", float(persistence["risk"]), int(persistence["sessions"]), COLORS["s4"]),
    ]
    x_positions = [430, 790]
    bar_w = 170
    for (label, value, n, color), x in zip(bars, x_positions):
        y = sy(value)
        draw.rounded_rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), radius=4, fill=color)
        draw_text(draw, (x, y - 38), f"{value*100:.1f}%", size=30, bold=True, fill=COLORS["text"], anchor="ma")
        lines = label.split("\n")
        draw_text(draw, (x, bottom + 24), lines[0], size=21, bold=True, anchor="ma")
        draw_text(draw, (x, bottom + 52), lines[1], size=18, fill=COLORS["muted"], anchor="ma")
        draw_text(draw, (x, bottom + 82), f"n={n:,}", size=17, fill=COLORS["muted"], anchor="ma")

    callout = (
        f"Adjusted OR for recovery: {adjusted['adjusted_odds_ratio']:.3f} "
        f"({adjusted['or_ci_low_95']:.3f}-{adjusted['or_ci_high_95']:.3f})"
    )
    draw.rounded_rectangle((300, 625, 900, 690), radius=6, outline=(205, 212, 216), width=1, fill=(255, 255, 253))
    draw_text(draw, (600, 646), callout, size=20, bold=True, anchor="ma")
    draw_text(
        draw,
        (600, 674),
        f"Excluding Jan31-Feb7: OR {robust['adjusted_odds_ratio']:.3f}",
        size=17,
        fill=COLORS["muted"],
        anchor="ma",
    )

    draw_text(
        draw,
        (70, 720),
        "Reference path in adjusted model: S4->S4. Standard errors clustered by player.",
        size=17,
        fill=COLORS["muted"],
    )
    out = FIG_DIR / "figure_1_recovery_vs_persistence.png"
    img.save(out)
    return out


def copy_spline():
    out = FIG_DIR / "figure_2_recoverability_curve.png"
    shutil.copy2(SPLINE_SRC, out)
    return out


def main():
    print(make_recovery_vs_persistence())
    print(copy_spline())


if __name__ == "__main__":
    main()
