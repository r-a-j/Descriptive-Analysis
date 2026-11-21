"""End-to-end Descriptive Demographic Analysis of U.S. Census Bureau IDB (2001 vs. 2021).

Analyzes data/census2001_2021.csv (227 countries across 5 regions and 21 subregions):
    Task 1 (2021): Frequency distributions of Life Expectancy (Both, Male, Female),
                   Infant Mortality Rate, and the national sex longevity gap.
    Task 2 (2021): Bivariate Pearson r and Spearman rho correlations and non-linear
                   convex saturation between Infant Mortality and Life Expectancy.
    Task 3 (2021): Homogeneity within vs. heterogeneity between all 21 subregions via
                   exact variance decomposition (SS_B / SS_T = eta^2) and IQR ranking.
    Task 4 (2001 vs. 2021): Paired 20-year trajectories across 221 matched nations
                   (6 missing in 2001 audited explicitly), sigma-convergence, and
                   beta-convergence catch-up dynamics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.demographic_stats import (
    correlation_matrix_dual,
    load_census_data,
    paired_longitudinal_changes,
    summarize_distribution,
    variance_decomposition,
)

REGION_COLORS = {
    "Africa": "#dc2626",
    "Asia": "#d97706",
    "Americas": "#0f766e",
    "Europe": "#1d4ed8",
    "Oceania": "#7c3aed",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 12.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.3,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.75,
        }
    )


def plot_figures(
    d21: pd.DataFrame,
    matched: pd.DataFrame,
    pear: pd.DataFrame,
    spear: pd.DataFrame,
    le_vd: dict[str, float],
    le_sub_tbl: pd.DataFrame,
    imr_vd: dict[str, float],
    imr_sub_tbl: pd.DataFrame,
    conv_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    _apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2021)

    # 1. Task 1A: 2021 Male vs Female Life Expectancy Distribution & Sex Gap Callout
    fig, ax = plt.subplots(figsize=(8.6, 5.9), dpi=220)
    m_vals = d21["le_male"].to_numpy()
    f_vals = d21["le_female"].to_numpy()
    bins = np.linspace(50, 95, 24)
    ax.hist(m_vals, bins=bins, density=True, alpha=0.34, color="#1d4ed8", edgecolor="white", lw=0.7, label=f"Males (median {np.median(m_vals):.1f} yr, skew -0.64)")
    ax.hist(f_vals, bins=bins, density=True, alpha=0.34, color="#e11d48", edgecolor="white", lw=0.7, label=f"Females (median {np.median(f_vals):.1f} yr, skew -0.77)")

    xs = np.linspace(50, 95, 300)
    ax.plot(xs, gaussian_kde(m_vals)(xs), color="#1e3a8a", lw=2.4)
    ax.plot(xs, gaussian_kde(f_vals)(xs), color="#be123c", lw=2.4)
    ax.axvline(np.median(m_vals), color="#1e3a8a", ls="--", lw=1.6)
    ax.axvline(np.median(f_vals), color="#be123c", ls="--", lw=1.6)

    # Highlight the +5.4 yr median gap arrow cleanly above the histogram peaks
    ax.annotate(
        "",
        xy=(np.median(f_vals), 0.0835),
        xytext=(np.median(m_vals), 0.0835),
        arrowprops=dict(arrowstyle="<->", color="#0f172a", lw=1.8),
    )
    ax.text((np.median(m_vals) + np.median(f_vals)) / 2.0, 0.0855, "+5.4 yr\nmedian gap", ha="center", va="bottom", fontsize=9.1, fontweight="bold", color="#0f172a")

    gap = d21["sex_gap"]
    n_fem_higher = int((gap > 0).sum())
    ax.text(
        0.025,
        0.965,
        f"Sex Gap (Female - Male, 2021)\n"
        f"Female > Male: {n_fem_higher}/{len(d21)} (99.6%)\n"
        f"Mean: +{gap.mean():.2f} yr | Med: +{gap.median():.2f} yr\n"
        f"Max: Russia (+11.4 yr)\n"
        f"Sole reversal: Montserrat (-2.1 yr)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#cbd5e1", alpha=0.97),
    )
    ax.set_xlim(50, 95)
    ax.set_ylim(0, 0.098)
    ax.set_xlabel("Life Expectancy at Birth in 2021 (years)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Global Life Expectancy Distributions by Sex (2021, n = 227 Countries)\n(Left-Skewed Distributions With Universal Female Longevity Advantage)",
        loc="left",
        pad=10,
        fontsize=11.6,
    )
    ax.grid(True, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.45), frameon=True, facecolor="white", framealpha=0.96, fontsize=8.8)
    fig.savefig(fig_dir / "distributions_2021_le_sex_gap.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    # 2. Task 1B: 2021 Infant Mortality Rate Distribution (Strong Right Skew)
    fig, ax = plt.subplots(figsize=(8.6, 5.9), dpi=220)
    imr = d21["imr_both"].to_numpy()
    q25, med, q75 = np.percentile(imr, [25.0, 50.0, 75.0])
    mean_v = float(np.mean(imr))

    ax.axvspan(q25, q75, color="#fef3c7", alpha=0.65, zorder=1, label=f"IQR: {q25:.1f} to {q75:.1f} (span {q75 - q25:.1f})")
    bins_imr = np.linspace(0, 110, 28)
    ax.hist(imr, bins=bins_imr, color="#0f766e", alpha=0.72, edgecolor="white", lw=0.8, zorder=2)
    ax.axvline(med, color="#0f172a", ls="-", lw=2.2, zorder=4, label=f"Median: {med:.2f} per 1,000")
    ax.axvline(mean_v, color="#dc2626", ls="--", lw=2.2, zorder=4, label=f"Mean: {mean_v:.2f} (+61% above median)")

    ax.annotate(
        "Lowest IMR\nSlovenia (1.5), Singapore (1.6)",
        xy=(2.6, 39.5),
        xytext=(24, 48),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.5, connectionstyle="arc3,rad=0.10"),
        fontsize=9.0,
        fontweight="bold",
        color="#065f46",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#99f6e4", alpha=0.97),
    )
    ax.annotate(
        "Extreme Right Tail\nAfghanistan (106.8)\nSomalia (88.0), CAR (84.2)",
        xy=(106.75, 2.0),
        xytext=(63, 24),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.4, connectionstyle="arc3,rad=0.14"),
        fontsize=9.0,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#fecaca", alpha=0.97),
    )
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 62)
    ax.set_xlabel("Infant Mortality Rate in 2021 (deaths per 1,000 live births)")
    ax.set_ylabel("Number of Countries (n = 227)")
    ax.set_title(
        "Global Infant Mortality Rate Distribution (2021, Skewness = +1.56)\n(Heavy Right Tail Pulls Mean 61% Above Median; Robust Median/IQR Preferred)",
        loc="left",
        pad=10,
        fontsize=11.4,
    )
    ax.grid(True, axis="y", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.97, fontsize=9.0)
    fig.savefig(fig_dir / "infant_mortality_distribution_2021.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    # 3. Task 2: Bivariate Correlation: 2021 Infant Mortality vs Life Expectancy (Convex Curve)
    fig, ax = plt.subplots(figsize=(8.6, 5.9), dpi=220)
    for reg, col in REGION_COLORS.items():
        sub = d21[d21["region"] == reg]
        ax.scatter(
            sub["imr_both"],
            sub["le_both"],
            s=42,
            color=col,
            alpha=0.80,
            edgecolors="white",
            lw=0.45,
            zorder=3,
            label=f"{reg} (n = {len(sub)})",
        )

    # Fit smooth log-linear curve LE = a + b * log(IMR + 5) to illustrate convex relationship
    imr_all = d21["imr_both"].to_numpy()
    le_all = d21["le_both"].to_numpy()
    x_grid = np.linspace(1.5, 107.0, 200)
    coef_log = np.polyfit(np.log(imr_all + 8.0), le_all, 1)
    y_curve = coef_log[0] * np.log(x_grid + 8.0) + coef_log[1]
    ax.plot(x_grid, y_curve, "-", color="#0f172a", lw=2.2, alpha=0.88, zorder=4, label="Log-linear trend fit")

    # Annotate key landmarks cleanly
    anchors = [
        ("Monaco", (8, -2), "left"),
        ("Singapore", (10, -4), "left"),
        ("Afghanistan", (-8, 22), "right"),
        ("Somalia", (-10, -18), "right"),
        ("Russia", (-2, -26), "left"),
    ]
    for cname, (dx, dy), ha in anchors:
        row = d21[d21["country"] == cname].iloc[0]
        ax.annotate(
            f"{cname} ({row['imr_both']:.1f}, {row['le_both']:.1f}y)",
            xy=(float(row["imr_both"]), float(row["le_both"])),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=8.3,
            fontweight="bold",
            color="#1e293b",
            arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.9),
        )

    r_p = float(pear.loc["le_both", "imr_both"])
    r_s = float(spear.loc["le_both", "imr_both"])
    r_mf = float(pear.loc["le_male", "le_female"])
    ax.text(
        0.97,
        0.96,
        f"2021 Bivariate Correlations (n = 227)\n"
        f"LE vs. IMR Pearson r = {r_p:.3f}\n"
        f"LE vs. IMR Spearman rho = {r_s:.3f}\n"
        f"Male vs. Female LE r = {r_mf:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.1,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1", alpha=0.97),
    )
    ax.set_xlim(-2, 112)
    ax.set_ylim(50, 92)
    ax.set_xlabel("Infant Mortality Rate (deaths per 1,000 live births, 2021)")
    ax.set_ylabel("Life Expectancy at Birth, Both Sexes (years, 2021)")
    ax.set_title(
        "Infant Mortality vs. Life Expectancy Across 227 Nations (2021)\n(Strong Negative Convex Coupling: Steep Early Gains, Saturation Below IMR 10)",
        loc="left",
        pad=10,
        fontsize=11.4,
    )
    ax.grid(True, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.96, fontsize=8.8)
    fig.savefig(fig_dir / "le_vs_imr_correlation_2021.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    # 4. Task 3A: All 21 Subregions Ranked by 2021 Median Life Expectancy (Homogeneity vs Level)
    fig, ax = plt.subplots(figsize=(9.2, 7.6), dpi=220)
    g_q25, g_med, g_q75 = np.percentile(d21["le_both"], [25.0, 50.0, 75.0])
    ax.axvspan(g_q25, g_q75, color="#e2e8f0", alpha=0.55, zorder=0, label=f"Global IQR ({g_q25:.1f}-{g_q75:.1f} yr, span {g_q75 - g_q25:.1f})")
    ax.axvline(g_med, color="#475569", ls="--", lw=1.4, zorder=1, label=f"Global Median ({g_med:.1f} yr)")

    for idx, row in le_sub_tbl.iterrows():
        sr = str(row["subregion"])
        reg = str(row["region"])
        col = REGION_COLORS[reg]
        vals = d21.loc[d21["subregion"] == sr, "le_both"].to_numpy()
        # Full min-max whisker + thick IQR bar + median dot + individual countries
        ax.plot([row["min"], row["max"]], [idx, idx], "-", color=col, lw=1.3, alpha=0.45, zorder=2)
        ax.plot([row["q25"], row["q75"]], [idx, idx], "-", color=col, lw=5.2, alpha=0.85, solid_capstyle="round", zorder=3)
        jit = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(vals, np.full(len(vals), idx) + jit, s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
        ax.scatter([row["median"]], [idx], s=62, color="white", edgecolors=col, lw=2.0, zorder=5)
        ax.text(
            float(row["max"]) + 0.7,
            idx,
            f"{row['median']:.1f}y  (IQR {row['iqr']:.1f})",
            va="center",
            ha="left",
            fontsize=8.1,
            fontweight="bold",
            color="#1e293b",
        )

    ax.set_yticks(np.arange(len(le_sub_tbl)))
    ax.set_yticklabels([f"{r['subregion']} (n={int(r['n'])})" for _, r in le_sub_tbl.iterrows()], fontsize=8.8, fontweight="bold")
    ax.set_xlim(51, 99.5)
    ax.set_xlabel("2021 Life Expectancy at Birth (years; thick bar = IQR, thin line = min-max, circle = median)")
    ax.set_title(
        f"21 Subregions Ranked by 2021 Median Life Expectancy (Between-Subregion η² = {100*le_vd['eta_sq_between']:.1f}%)\n"
        f"(Tight Within-Subregion Spread in Europe/Oceania vs. Wide Eastern Asia & Western Africa)",
        loc="left",
        pad=10,
        fontsize=11.3,
    )
    ax.grid(True, axis="x", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    reg_handles = [
        plt.Line2D([0], [0], marker="o", color=c, lw=3.0, markersize=6.5, label=r)
        for r, c in REGION_COLORS.items()
    ]
    h_base, l_base = ax.get_legend_handles_labels()
    ax.legend(handles=reg_handles + h_base, loc="upper left", frameon=True, facecolor="white", framealpha=0.97, fontsize=8.4)
    fig.savefig(fig_dir / "subregion_homogeneity_2021.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    # 5. Task 3B: 21-Subregion Homogeneity vs Central Level Quadrant (IMR Median vs IQR)
    fig, ax = plt.subplots(figsize=(8.6, 5.9), dpi=220)
    g_imr_iqr = float(imr_vd["global_iqr"])
    ax.axhline(g_imr_iqr, color="#dc2626", ls="--", lw=1.4, alpha=0.85, label=f"Global IMR IQR ({g_imr_iqr:.1f})")

    for _, row in imr_sub_tbl.iterrows():
        col = REGION_COLORS[str(row["region"])]
        ax.scatter(
            float(row["median"]),
            float(row["iqr"]),
            s=45 + 9 * float(row["n"]),
            color=col,
            alpha=0.82,
            edgecolors="#0f172a",
            lw=0.7,
            zorder=4,
        )

    label_specs = {
        "Western Europe": (10, -4, "left"),
        "Northern Europe": (20, -2, "left"),
        "Eastern Asia": (9, -2, "left"),
        "Eastern Europe": (22, 8, "left"),
        "South-Eastern Asia": (10, 7, "left"),
        "Northern Africa": (9, 4, "left"),
        "South-Central Asia": (9, -8, "left"),
        "Southern Africa": (9, 4, "left"),
        "Eastern Africa": (9, -6, "left"),
        "Western Africa": (-9, 6, "right"),
        "Middle Africa": (-9, -12, "right"),
    }
    for sr, (dx, dy, ha) in label_specs.items():
        r = imr_sub_tbl[imr_sub_tbl["subregion"] == sr].iloc[0]
        ax.annotate(
            f"{sr}",
            xy=(float(r["median"]), float(r["iqr"])),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=8.3,
            fontweight="bold",
            color="#1e293b",
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.8),
        )

    ax.text(
        0.97,
        0.10,
        f"IMR Variance Decomposition (n = 227)\n"
        f"Between 21 Subregions: η² = {100*imr_vd['eta_sq_between']:.1f}%\n"
        f"Within Subregions: {100*imr_vd['within_share']:.1f}%\n"
        f"(Bubble size proportional to country count)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.9,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#cbd5e1", alpha=0.97),
    )
    reg_h = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="#0f172a", markersize=7.5, label=r) for r, c in REGION_COLORS.items()]
    h_l, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h_l + reg_h, loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.96, fontsize=8.3)
    ax.set_xlim(-2, 68)
    ax.set_ylim(-1.5, 26.8)
    ax.set_xlabel("Subregion Median Infant Mortality Rate (per 1,000, 2021)")
    ax.set_ylabel("Within-Subregion Spread: IMR IQR (per 1,000)")
    ax.set_title(
        "Subregion Central Tendency vs. Internal Dispersion (2021 IMR)\n(Low-IMR Europe Tightly Clustered; SE Asia Exceeds Global IQR)",
        loc="left",
        pad=10,
        fontsize=11.5,
    )
    ax.grid(True, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "subregion_imr_dispersion_2021.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    # 6. Task 4: 20-Year Beta-Convergence (2001 Life Expectancy vs 2001 -> 2021 Gain, n = 221)
    fig, ax = plt.subplots(figsize=(8.6, 5.9), dpi=220)
    pair = matched[["country", "region", "le_both_2001", "le_both_2021", "delta_le_both"]].dropna().copy()
    ax.axhline(0.0, color="#0f172a", ls="--", lw=1.5, zorder=2)

    for reg, col in REGION_COLORS.items():
        sub = pair[pair["region"] == reg]
        ax.scatter(
            sub["le_both_2001"],
            sub["delta_le_both"],
            s=42,
            color=col,
            alpha=0.80,
            edgecolors="white",
            lw=0.45,
            zorder=3,
            label=f"{reg} (n = {len(sub)})",
        )

    # OLS convergence slope
    x0 = pair["le_both_2001"].to_numpy()
    dy_v = pair["delta_le_both"].to_numpy()
    slope, intercept = np.polyfit(x0, dy_v, 1)
    xg = np.linspace(42, 83, 100)
    ax.plot(xg, slope * xg + intercept, "-", color="#0f172a", lw=2.1, alpha=0.85, zorder=4, label=f"Beta-convergence slope ({slope:.2f} yr/yr)")

    conv_le = conv_df[conv_df["variable"] == "le_both"].iloc[0]
    callouts = [
        ("Zambia", (9, 4), "left"),
        ("Uganda", (9, -3), "left"),
        ("Zimbabwe", (9, -11), "left"),
        ("Mexico", (-10, -14), "right"),
        ("Peru", (-10, 6), "right"),
    ]
    for cname, (ox, oy), ha in callouts:
        r = pair[pair["country"] == cname].iloc[0]
        ax.annotate(
            f"{cname} ({r['delta_le_both']:+.1f}y)",
            xy=(float(r["le_both_2001"]), float(r["delta_le_both"])),
            xytext=(ox, oy),
            textcoords="offset points",
            ha=ha,
            fontsize=8.4,
            fontweight="bold",
            color="#1e293b",
            arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.9),
        )

    ax.text(
        0.97,
        0.96,
        f"20-Year Convergence (n = {int(conv_le['n_matched'])} Matched)\n"
        f"Improved: {int(conv_le['n_improved'])}/{int(conv_le['n_matched'])} (98.6%)\n"
        f"Mean Gain: +{conv_le['mean_delta']:.2f} yr  |  Med: +{conv_le['median_delta']:.2f} yr\n"
        f"Global SD: {conv_le['sd_start']:.2f} -> {conv_le['sd_end']:.2f} yr ({conv_le['sd_pct_change']:.1f}%)\n"
        f"Beta-Convergence: r = {conv_le['beta_convergence_r']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1", alpha=0.97),
    )
    ax.set_xlim(41, 84)
    ax.set_ylim(-4.8, 22.5)
    ax.set_xlabel("Baseline 2001 Life Expectancy at Birth (years)")
    ax.set_ylabel("20-Year Change in Life Expectancy (2021 - 2001, years)")
    ax.set_title(
        "Global Longevity Convergence Over 20 Years (2001 -> 2021, n = 221)\n(Catch-Up Gains in Sub-Saharan Africa Cut Global SD by 28%; Only 3 Declines)",
        loc="left",
        pad=10,
        fontsize=11.3,
    )
    ax.grid(True, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.96, fontsize=8.6)
    fig.savefig(fig_dir / "convergence_2001_2021.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run descriptive demographic analysis on census2001_2021.csv.")
    parser.add_argument("--data", type=Path, default=Path("data/census2001_2021.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    res_dir = args.out_dir / "results"
    fig_dir = args.out_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_census_data(args.data)
    d21 = df[df["year"] == 2021].copy()
    missing_2001 = sorted(df[(df["year"] == 2001) & (df["le_both"].isna())]["country"].tolist())

    # Task 1: 2021 Descriptive summaries
    desc_rows = [
        summarize_distribution(d21["le_both"], "Life Expectancy (Both Sexes, yr)"),
        summarize_distribution(d21["le_male"], "Life Expectancy (Males, yr)"),
        summarize_distribution(d21["le_female"], "Life Expectancy (Females, yr)"),
        summarize_distribution(d21["sex_gap"], "Sex Longevity Gap (Female - Male, yr)"),
        summarize_distribution(d21["imr_both"], "Infant Mortality Rate (per 1,000)"),
    ]
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(res_dir / "descriptive_2021.csv", index=False)

    # Task 2: Dual correlation matrices (Pearson & Spearman)
    corr_cols = ["le_both", "le_male", "le_female", "imr_both"]
    pear, spear = correlation_matrix_dual(d21, corr_cols)
    pear.to_csv(res_dir / "correlations_pearson_2021.csv")
    spear.to_csv(res_dir / "correlations_spearman_2021.csv")

    # Task 3: Subregion and Region variance decomposition (2021)
    le_vd_sub, le_sub_tbl = variance_decomposition(d21, "subregion", "le_both", extra_cols=("region",))
    imr_vd_sub, imr_sub_tbl = variance_decomposition(d21, "subregion", "imr_both", extra_cols=("region",))
    le_vd_reg, _ = variance_decomposition(d21, "region", "le_both")
    imr_vd_reg, _ = variance_decomposition(d21, "region", "imr_both")

    sub_merged = pd.merge(
        le_sub_tbl[["region", "subregion", "n", "median", "iqr", "sd", "min", "max", "iqr_ratio_to_global"]],
        imr_sub_tbl[["subregion", "median", "iqr", "sd", "min", "max", "iqr_ratio_to_global"]],
        on="subregion",
        suffixes=("_le", "_imr"),
    )
    sub_merged.to_csv(res_dir / "subregion_summary_2021.csv", index=False)

    # Task 4: Paired 2001 vs 2021 longitudinal convergence
    matched, conv_df = paired_longitudinal_changes(df, ["le_both", "le_male", "le_female", "imr_both"], 2001, 2021)
    conv_df.to_csv(res_dir / "longitudinal_2001_2021_summary.csv", index=False)
    matched.to_csv(res_dir / "matched_countries_2001_2021.csv", index=False)

    summary = {
        "n_countries_2021": int(len(d21)),
        "n_regions": int(d21["region"].nunique()),
        "n_subregions": int(d21["subregion"].nunique()),
        "missing_2001_countries": missing_2001,
        "n_matched_2001_2021": int(conv_df.loc[conv_df["variable"] == "le_both", "n_matched"].iloc[0]),
        "correlations_2021": {
            "le_imr_pearson": float(pear.loc["le_both", "imr_both"]),
            "le_imr_spearman": float(spear.loc["le_both", "imr_both"]),
            "male_female_le_pearson": float(pear.loc["le_male", "le_female"]),
        },
        "variance_decomposition_2021": {
            "le_subregion_eta_sq": le_vd_sub["eta_sq_between"],
            "le_region_eta_sq": le_vd_reg["eta_sq_between"],
            "imr_subregion_eta_sq": imr_vd_sub["eta_sq_between"],
            "imr_region_eta_sq": imr_vd_reg["eta_sq_between"],
        },
    }
    (res_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_figures(d21, matched, pear, spear, le_vd_sub, le_sub_tbl, imr_vd_sub, imr_sub_tbl, conv_df, fig_dir)

    print("=== TASK 1: 2021 DESCRIPTIVE SUMMARY (n = 227) ===")
    print(desc_df[["variable", "n", "mean", "sd", "median", "iqr", "min", "max", "skewness"]].to_string(index=False))
    print("\n=== TASK 3: VARIANCE DECOMPOSITION (2021) ===")
    print(f"Life Expectancy: Subregion eta^2 = {le_vd_sub['eta_sq_between']:.4f} | Region eta^2 = {le_vd_reg['eta_sq_between']:.4f}")
    print(f"Infant Mortality: Subregion eta^2 = {imr_vd_sub['eta_sq_between']:.4f} | Region eta^2 = {imr_vd_reg['eta_sq_between']:.4f}")
    print("\n=== TASK 4: PAIRED 2001 -> 2021 CONVERGENCE (n = 221) ===")
    print(conv_df[["variable", "n_matched", "median_start", "median_end", "mean_delta", "sd_start", "sd_end", "sd_pct_change", "n_improved", "beta_convergence_r"]].to_string(index=False))


if __name__ == "__main__":
    main()
