"""Descriptive Demographic Statistics, Variance Decomposition, and Longitudinal Convergence.

Implements clean loading and missingness auditing for the U.S. Census Bureau IDB dataset
(227 countries across 5 regions and 21 subregions, 2001 vs. 2021):
    - Distributional summaries (mean, SD, median, IQR, skewness, kurtosis, sex longevity gap)
    - Dual Pearson r and Spearman rho bivariate correlation matrices
    - Exact between- vs. within-group variance decomposition (SS_B + SS_W = SS_T, eta^2)
    - Paired 2001 -> 2021 convergence analytics (sigma-convergence SD drop & beta-convergence)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


COLUMN_MAP = {
    "Country.Name": "country",
    "Subregion": "subregion",
    "Region": "region",
    "Year": "year",
    "Life.Expectancy..Both.Sexes": "le_both",
    "Life.Expectancy..Males": "le_male",
    "Life.Expectancy..Females": "le_female",
    "Infant.Mortality.Rate..Both.Sexes": "imr_both",
}


def load_census_data(csv_path: Path | str) -> pd.DataFrame:
    """Load census2001_2021.csv safely across UTF-8 / Latin-1 encodings with clean schema."""
    path = Path(csv_path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    df = df.rename(columns=COLUMN_MAP).copy()
    df["sex_gap"] = df["le_female"] - df["le_male"]
    return df


def summarize_distribution(values: Sequence[float] | pd.Series, name: str = "") -> dict[str, object]:
    """Compute exact parametric and robust descriptive statistics on non-null values."""
    arr = np.asarray(pd.Series(values).dropna(), dtype=np.float64)
    if len(arr) < 2:
        raise ValueError(f"Need at least 2 non-null observations for {name}.")
    q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    mean_v = float(np.mean(arr))
    sd_v = float(np.std(arr, ddof=1))
    return {
        "variable": name,
        "n": int(len(arr)),
        "mean": mean_v,
        "sd": sd_v,
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "skewness": float(stats.skew(arr, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(arr, bias=False)),
        "cv": float(sd_v / mean_v) if abs(mean_v) > 1e-12 else np.nan,
    }


def correlation_matrix_dual(df: pd.DataFrame, cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute both Pearson r and Spearman rho correlation matrices."""
    sub = df[list(cols)].dropna()
    return sub.corr(method="pearson"), sub.corr(method="spearman")


def variance_decomposition(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    extra_cols: Sequence[str] = (),
) -> tuple[dict[str, float], pd.DataFrame]:
    """Decompose total variance into Between-Group (eta^2) and Within-Group (1 - eta^2) shares."""
    keep = list(dict.fromkeys([group_col, value_col, *extra_cols]))
    sub = df[keep].dropna(subset=[group_col, value_col]).copy()
    y = sub[value_col].to_numpy(dtype=np.float64)
    grand_mean = float(np.mean(y))
    global_iqr = float(np.percentile(y, 75.0) - np.percentile(y, 25.0))
    global_sd = float(np.std(y, ddof=1))

    ss_total = float(np.sum((y - grand_mean) ** 2))
    ss_between = 0.0
    ss_within = 0.0
    rows: list[dict[str, object]] = []

    for grp_name, gdf in sub.groupby(group_col, sort=True):
        vals = gdf[value_col].to_numpy(dtype=np.float64)
        n_g = len(vals)
        m_g = float(np.mean(vals))
        ss_between += n_g * (m_g - grand_mean) ** 2
        ss_within += float(np.sum((vals - m_g) ** 2))
        q25, med, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
        sd_g = float(np.std(vals, ddof=1)) if n_g > 1 else 0.0
        iqr_g = float(q75 - q25)
        rec: dict[str, object] = {
            group_col: str(grp_name),
            "n": int(n_g),
            "mean": m_g,
            "sd": sd_g,
            "median": float(med),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": iqr_g,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "iqr_ratio_to_global": float(iqr_g / global_iqr) if global_iqr > 0 else np.nan,
            "sd_ratio_to_global": float(sd_g / global_sd) if global_sd > 0 else np.nan,
        }
        for ec in extra_cols:
            rec[ec] = str(gdf[ec].iloc[0])
        rows.append(rec)

    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    summary = {
        "n_obs": float(len(sub)),
        "n_groups": float(len(rows)),
        "ss_total": ss_total,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "eta_sq_between": float(eta_sq),
        "within_share": float(1.0 - eta_sq),
        "global_sd": global_sd,
        "global_iqr": global_iqr,
    }
    return summary, pd.DataFrame(rows).sort_values("median").reset_index(drop=True)


def paired_longitudinal_changes(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    year_start: int = 2001,
    year_end: int = 2021,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match countries across year_start and year_end and compute exact convergence metrics."""
    d0 = df[df["year"] == year_start][["country", "region", "subregion", *value_cols]].copy()
    d1 = df[df["year"] == year_end][["country", *value_cols]].copy()
    merged = pd.merge(d0, d1, on="country", suffixes=(f"_{year_start}", f"_{year_end}"))

    summaries: list[dict[str, object]] = []
    for col in value_cols:
        c0 = f"{col}_{year_start}"
        c1 = f"{col}_{year_end}"
        delta_col = f"delta_{col}"
        merged[delta_col] = merged[c1] - merged[c0]

        pair = merged[[c0, c1, delta_col]].dropna()
        x0 = pair[c0].to_numpy(dtype=np.float64)
        x1 = pair[c1].to_numpy(dtype=np.float64)
        dx = pair[delta_col].to_numpy(dtype=np.float64)

        sd0 = float(np.std(x0, ddof=1))
        sd1 = float(np.std(x1, ddof=1))
        beta_r, beta_p = stats.pearsonr(x0, dx)
        improved = int(np.sum(dx < 0.0)) if "imr" in col else int(np.sum(dx > 0.0))

        summaries.append(
            {
                "variable": col,
                "n_matched": int(len(pair)),
                "median_start": float(np.median(x0)),
                "median_end": float(np.median(x1)),
                "mean_start": float(np.mean(x0)),
                "mean_end": float(np.mean(x1)),
                "sd_start": sd0,
                "sd_end": sd1,
                "sd_pct_change": float(100.0 * (sd1 - sd0) / sd0),
                "mean_delta": float(np.mean(dx)),
                "median_delta": float(np.median(dx)),
                "n_improved": improved,
                "n_declined": int(len(pair) - improved),
                "beta_convergence_r": float(beta_r),
                "beta_convergence_p": float(beta_p),
            }
        )

    return merged, pd.DataFrame(summaries)
