"""Unit tests for descriptive demographic statistics, variance decomposition, and convergence."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.demographic_stats import (
    correlation_matrix_dual,
    paired_longitudinal_changes,
    summarize_distribution,
    variance_decomposition,
)


class TestDemographicStats(unittest.TestCase):
    def test_summarize_distribution_and_correlations(self) -> None:
        vals = np.array([10.0, 12.0, 14.0, 16.0, 38.0])
        s = summarize_distribution(vals, "test")
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(float(s["mean"]), 18.0, places=10)
        self.assertAlmostEqual(float(s["median"]), 14.0, places=10)
        self.assertAlmostEqual(float(s["iqr"]), 4.0, places=10)
        self.assertGreater(float(s["skewness"]), 1.0)

        # Monotone non-linear inverse relationship: Spearman = -1.0, |Pearson| < 1.0
        x = np.linspace(1.0, 10.0, 30)
        df = pd.DataFrame({"x": x, "y": np.exp(-0.5 * x)})
        pear, spear = correlation_matrix_dual(df, ["x", "y"])
        self.assertAlmostEqual(float(spear.loc["x", "y"]), -1.0, places=10)
        self.assertGreater(float(pear.loc["x", "y"]), -0.99)
        self.assertLess(float(pear.loc["x", "y"]), -0.70)

    def test_variance_decomposition_exact_ss_identity(self) -> None:
        df = pd.DataFrame(
            {
                "subregion": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "region": ["R1", "R1", "R1", "R1", "R1", "R1", "R2", "R2", "R2"],
                "val": [60.0, 61.0, 62.0, 74.0, 75.0, 76.0, 82.0, 83.0, 84.0],
            }
        )
        summary, tbl = variance_decomposition(df, "subregion", "val", extra_cols=("region",))
        self.assertAlmostEqual(
            summary["ss_between"] + summary["ss_within"],
            summary["ss_total"],
            places=10,
        )
        self.assertGreater(summary["eta_sq_between"], 0.95)
        self.assertEqual(len(tbl), 3)
        self.assertListEqual(tbl["subregion"].tolist(), ["A", "B", "C"])

    def test_paired_longitudinal_convergence_and_missingness(self) -> None:
        df = pd.DataFrame(
            {
                "country": ["C1", "C2", "C3", "C4", "C1", "C2", "C3", "C4"],
                "region": ["Africa", "Asia", "Europe", "Americas"] * 2,
                "subregion": ["Sub1", "Sub2", "Sub3", "Sub4"] * 2,
                "year": [2001, 2001, 2001, 2001, 2021, 2021, 2021, 2021],
                "le_both": [50.0, 68.0, 78.0, np.nan, 68.0, 74.0, 82.0, 79.0],
                "imr_both": [90.0, 30.0, 5.0, np.nan, 35.0, 12.0, 3.0, 6.0],
            }
        )
        _, summary = paired_longitudinal_changes(df, ["le_both", "imr_both"], 2001, 2021)
        row_le = summary[summary["variable"] == "le_both"].iloc[0]
        row_imr = summary[summary["variable"] == "imr_both"].iloc[0]

        # C4 missing in 2001 -> 3 matched pairs
        self.assertEqual(int(row_le["n_matched"]), 3)
        self.assertEqual(int(row_le["n_improved"]), 3)
        self.assertEqual(int(row_imr["n_improved"]), 3)
        # Strong catch-up convergence: SD drops and beta_convergence_r < -0.9
        self.assertLess(float(row_le["sd_pct_change"]), -40.0)
        self.assertLess(float(row_le["beta_convergence_r"]), -0.9)


if __name__ == "__main__":
    unittest.main()
