from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PCA_DIR = Path("step3_pca")


def font(size):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_axes(draw, box, title, xlabel, ylabel, x_range=None, y_range=None):
    left, top, right, bottom = box
    draw.rectangle((0, 0, 1100, 820), fill="white")
    draw.text((left, 24), title, fill="#1f2933", font=font(28))
    draw.line((left, bottom, right, bottom), fill="#9aa5b1", width=2)
    draw.line((left, bottom, left, top), fill="#9aa5b1", width=2)
    draw.text(((left + right) // 2 - 45, bottom + 42), xlabel, fill="#394b59", font=font(18))
    draw.text((16, (top + bottom) // 2 - 10), ylabel, fill="#394b59", font=font(18))
    if x_range:
        draw.text((left - 10, bottom + 14), f"{x_range[0]:.2f}", fill="#52616b", font=font(14), anchor="ra")
        draw.text((right, bottom + 14), f"{x_range[1]:.2f}", fill="#52616b", font=font(14), anchor="ma")
    if y_range:
        draw.text((left - 12, bottom), f"{y_range[0]:.2f}", fill="#52616b", font=font(14), anchor="ra")
        draw.text((left - 12, top), f"{y_range[1]:.2f}", fill="#52616b", font=font(14), anchor="ra")


def save_scree():
    df = pd.read_csv(PCA_DIR / "pca_explained_variance.csv").head(12)
    values = df["explained_variance_ratio"].to_numpy() * 100
    width, height = 1100, 720
    box = (90, 70, 1060, 620)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw_axes(draw, box, "PCA explained variance", "principal component", "variance %")
    left, top, right, bottom = box
    ymax = np.ceil(values.max() / 5) * 5
    gap = (right - left) / len(values)
    bar_w = gap * 0.68
    for i, v in enumerate(values):
        x0 = left + i * gap + gap * 0.16
        h = v / ymax * (bottom - top)
        y0 = bottom - h
        draw.rectangle((x0, y0, x0 + bar_w, bottom), fill="#5b8def")
        draw.text((x0 + bar_w / 2, bottom + 16), f"PC{i+1}", fill="#52616b", font=font(13), anchor="ma")
        draw.text((x0 + bar_w / 2, y0 - 20), f"{v:.1f}%", fill="#394b59", font=font(13), anchor="ma")
    img.save(PCA_DIR / "pca_scree_plot.png")


def save_scatter(x_col, y_col, out_name):
    df = pd.read_csv(PCA_DIR / "pca_session_scores.csv", usecols=[x_col, y_col])
    if len(df) > 30000:
        df = df.sample(30000, random_state=42)
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    width, height = 1100, 820
    box = (95, 70, 1060, 705)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    x_lo, x_hi = np.quantile(x, [0.005, 0.995])
    y_lo, y_hi = np.quantile(y, [0.005, 0.995])
    draw_axes(draw, box, f"PCA session scores: {x_col} vs {y_col}", x_col, y_col, (x_lo, x_hi), (y_lo, y_hi))
    left, top, right, bottom = box

    xs = left + (np.clip(x, x_lo, x_hi) - x_lo) / (x_hi - x_lo) * (right - left)
    ys = bottom - (np.clip(y, y_lo, y_hi) - y_lo) / (y_hi - y_lo) * (bottom - top)
    for px, py in zip(xs, ys):
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(47, 111, 159, 38))
    img.save(PCA_DIR / out_name)


def main():
    shutil.copyfile(PCA_DIR / "pca_feature_loadings.csv", PCA_DIR / "pca_loadings.csv")
    shutil.copyfile(PCA_DIR / "step3_0_1_report.md", PCA_DIR / "pca_report.md")
    save_scree()
    save_scatter("PC1", "PC2", "pca_pc1_pc2.png")
    save_scatter("PC1", "PC3", "pca_pc1_pc3.png")
    for name in [
        "pca_loadings.csv",
        "pca_report.md",
        "pca_scree_plot.png",
        "pca_pc1_pc2.png",
        "pca_pc1_pc3.png",
    ]:
        path = PCA_DIR / name
        print(f"{path} {path.stat().st_size}")


if __name__ == "__main__":
    main()
