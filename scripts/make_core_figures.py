from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TEAL, RED, MUTED = "#137F78", "#B95562", "#667078"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": "#24282C", "axes.labelcolor": "#24282C",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#C6CDD1", "axes.linewidth": .8,
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def header(fig, number, title, subtitle):
    fig.text(.08, .95, f"FIGURE {number}  /  PLAYER-STATE DYNAMICS", fontsize=9,
             color=MUTED, weight="bold")
    fig.text(.08, .895, title, fontsize=20, weight="bold")
    fig.text(.08, .851, subtitle, fontsize=11, color=MUTED)


def style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E8EBED", linewidth=.8)
    ax.tick_params(length=0, pad=9)
    ax.set_ylim(0, .55)
    ax.set_yticks([0, .1, .2, .3, .4, .5])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", dpi=240)
    plt.close(fig)
    return FIG_DIR / f"{name}.png"


def make_recovery_vs_persistence():
    folder = ROOT / "step5_3_adjusted_recovery_path_model"
    risk = pd.read_csv(folder / "adjusted_recovery_path_unadjusted_risk.csv")
    coef = pd.read_csv(folder / "adjusted_recovery_path_coefficients.csv")
    rows = risk[risk.analysis.eq("main_30d")].set_index("path_label")
    effects = coef[coef.term.eq("recovery_path")].set_index("analysis")
    main = effects.loc["main_30d"]
    robust = effects.loc["exclude_update_window_30d"]
    fig = plt.figure(figsize=(11, 8))
    header(fig, 1, "Recovery paths and subsequent disengagement",
           "30-day observed disengagement after the destination session following S4")
    ax = fig.add_axes([.12, .34, .79, .43])
    style(ax)
    ax.set_xlim(-.65, 1.65)
    ax.set_ylabel("Observed disengagement risk", labelpad=14)
    for i, (path, color) in enumerate(zip(["S4->S1/S2", "S4->S4"], [TEAL, RED])):
        row = rows.loc[path]
        ax.bar(i, row.risk, width=.46, color=color, zorder=3)
        ax.text(i, row.risk + .018, f"{row.risk:.2%}", ha="center",
                fontsize=20, weight="bold", color=color)
    ax.set_xticks([0, 1], [
        f"Recovery: S4 → S1/S2\nn = {int(rows.loc['S4->S1/S2', 'sessions']):,} sessions",
        f"Persistence: S4 → S4\nn = {int(rows.loc['S4->S4', 'sessions']):,} sessions",
    ])
    fig.text(.12, .207, "ADJUSTED ASSOCIATION", fontsize=9, weight="bold", color=MUTED)
    fig.text(.12, .165,
             f"OR {main.adjusted_odds_ratio:.3f}  "
             f"(95% CI {main.or_ci_low_95:.3f}–{main.or_ci_high_95:.3f})",
             fontsize=16, weight="bold", color=TEAL)
    fig.text(.12, .126,
             f"Excluding Jan 31–Feb 7: OR {robust.adjusted_odds_ratio:.3f}  "
             f"(95% CI {robust.or_ci_low_95:.3f}–{robust.or_ci_high_95:.3f})",
             fontsize=10, color=MUTED)
    fig.text(.12, .064,
             "Bars show unadjusted risks. Adjusted OR compares recovery with persistence.\n"
             "Logistic regression with controls; standard errors clustered by player.",
             fontsize=9, color=MUTED, linespacing=1.6)
    return save(fig, "figure_1_recovery_vs_persistence")


def make_recoverability_curve():
    folder = ROOT / "step5_7_recovery_spline_model"
    curve = pd.read_csv(folder / "recovery_spline_predicted_curve.csv")
    bins = pd.read_csv(folder / "recovery_spline_observed_bins.csv")
    fig = plt.figure(figsize=(11, 8))
    header(fig, 2, "Recent S4 exposure and recoverability",
           "30-day recovery to S1/S2 after an S4 session")
    ax = fig.add_axes([.12, .40, .79, .38])
    style(ax)
    x = curve.recent_s4_density_prev5
    ax.fill_between(x, curve.predicted_recovery_low, curve.predicted_recovery_high,
                    color=TEAL, alpha=.16, linewidth=0, label="95% confidence interval")
    ax.plot(x, curve.predicted_recovery_fit, color=TEAL, linewidth=2.5,
            label="Adjusted spline prediction")
    ax.set(xlim=(0, 1), ylabel="Adjusted recovery probability")
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel("S4 share in the previous 5 sessions", labelpad=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", frameon=False, fontsize=9)
    for density, offset in [(0, (12, 10)), (1, (-12, 12))]:
        row = curve.loc[x.eq(density)].iloc[0]
        val = row.predicted_recovery_fit
        ax.plot(density, val, "o", color=TEAL, markersize=5, clip_on=False)
        ax.annotate(f"{val:.1%}", (density, val), xytext=offset,
                    textcoords="offset points", ha="left" if density == 0 else "right",
                    fontsize=12, weight="bold", color=TEAL)
    fig.text(.12, .269, "OBSERVED RECOVERY BY DENSITY BIN", fontsize=9,
             weight="bold", color=MUTED)
    for xpos, label, (_, row) in zip([.12, .40, .68], ["0.00–0.20", ">0.20–0.40", ">0.40–1.00"], bins.iterrows()):
        fig.text(xpos, .226, f"{row.observed_recovery_rate:.1%}", fontsize=18, weight="bold")
        fig.text(xpos, .192, f"Density {label}", fontsize=10, color=MUTED)
        fig.text(xpos, .165, f"n = {int(row.sessions):,} sessions", fontsize=9, color=MUTED)
    fig.text(.12, .065,
             "Restricted cubic spline with controls; shaded band: player-clustered 95% CI.\n"
             "Observed bin rates are unadjusted. The smooth curve does not establish a unique breakpoint.",
             fontsize=9, color=MUTED, linespacing=1.6)
    return save(fig, "figure_2_recoverability_curve")


def main():
    print(make_recovery_vs_persistence())
    print(make_recoverability_curve())


if __name__ == "__main__":
    main()
