"""
bornhuetter_ferguson.py
-----------------------
Bornhuetter-Ferguson (BF) reserve estimator.

The BF method addresses a key weakness of chain ladder: for immature
accident years, the CDF is large and small errors in early reported
losses get amplified into large reserve swings.

BF blends chain ladder with an a priori expected loss ratio (ELR):

    IBNR_BF  = Premium × ELR × (1 - 1/CDF)
    Ultimate_BF = LatestLoss + IBNR_BF

Where (1 - 1/CDF) is the "unreported factor" — the fraction of ultimate
losses not yet reported at the current development age.

Interpretation
--------------
    - For immature years (large CDF): BF leans heavily on the a priori ELR
    - For mature years (CDF → 1.0): BF converges to chain ladder
    - The ELR is typically set from pricing assumptions or industry benchmarks

This is directly analogous to Bayesian credibility:
    BF estimate = credibility × observed + (1 - credibility) × a priori
    where credibility = 1/CDF (fraction of losses already reported)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from chain_ladder import ChainLadder, load_cas_triangle


class BornhuetterFerguson:
    """
    Bornhuetter-Ferguson reserve estimator.

    Requires a fitted ChainLadder object for the CDFs, plus either:
        - An explicit ELR (scalar or dict by accident year)
        - Premium series (ELR computed as wtd avg chain ladder LR)

    Parameters
    ----------
    chain_ladder : fitted ChainLadder object
    elr          : expected loss ratio — scalar, or dict {AccidentYear: elr}
                   if None, uses weighted average ultimate LR from chain ladder
    """

    def __init__(self, chain_ladder: ChainLadder, elr=None):
        if not chain_ladder._fitted:
            raise RuntimeError("ChainLadder must be fitted before passing to BF")

        self.cl = chain_ladder
        self.triangle = chain_ladder.triangle
        self.premium = chain_ladder.premium
        self.cdfs = chain_ladder.cdfs
        self._elr_input = elr
        self._fitted = False

    def fit(self):
        """Compute BF IBNR and ultimate estimates."""

        # ── Determine ELR ───────────────────────────────────────────
        if self._elr_input is None:
            # Use weighted average ultimate loss ratio from chain ladder
            cl_results = self.cl.results
            if "UltimateLR" in cl_results.columns:
                elr = (
                    cl_results["Ultimate"].sum() / cl_results["Premium"].sum()
                )
                print(f"  A priori ELR (from chain ladder wtd avg): {elr:.4f} ({100*elr:.1f}%)")
            else:
                raise ValueError(
                    "No premium data available. Pass explicit elr= to BornhuetterFerguson."
                )
            self.elr = elr
        elif isinstance(self._elr_input, dict):
            self.elr = self._elr_input  # per accident year
        else:
            self.elr = float(self._elr_input)
            print(f"  A priori ELR (user-specified): {self.elr:.4f} ({100*self.elr:.1f}%)")

        # ── BF calculation ───────────────────────────────────────────
        results = []
        for acc_year in self.triangle.index:
            row = self.triangle.loc[acc_year].dropna()
            if len(row) == 0:
                continue

            latest_lag = row.index.max()
            latest_loss = row[latest_lag]
            cdf = self.cdfs[latest_lag]

            # Unreported factor = fraction of ultimate not yet reported
            unreported_factor = 1.0 - (1.0 / cdf)

            # ELR for this accident year
            if isinstance(self.elr, dict):
                elr_i = self.elr.get(acc_year, self.elr.get("default", 0.7))
            else:
                elr_i = self.elr

            # BF IBNR
            if self.premium is not None and acc_year in self.premium.index:
                prem = self.premium[acc_year]
                ibnr_bf = prem * elr_i * unreported_factor
                ultimate_bf = latest_loss + ibnr_bf
            else:
                # Without premium, use chain ladder ultimate as base
                cl_ult = self.cl.results.loc[acc_year, "Ultimate"]
                ibnr_bf = cl_ult * elr_i * unreported_factor
                ultimate_bf = latest_loss + ibnr_bf
                prem = np.nan

            # Chain ladder results for comparison
            cl_ibnr = self.cl.results.loc[acc_year, "IBNR"]
            cl_ult = self.cl.results.loc[acc_year, "Ultimate"]

            results.append({
                "AccidentYear": acc_year,
                "LatestLag": latest_lag,
                "LatestLoss": latest_loss,
                "CDF": cdf,
                "UnreportedFactor": unreported_factor,
                "ELR": elr_i,
                "Premium": prem,
                "IBNR_BF": ibnr_bf,
                "Ultimate_BF": ultimate_bf,
                "IBNR_CL": cl_ibnr,
                "Ultimate_CL": cl_ult,
                "IBNR_Diff": ibnr_bf - cl_ibnr,
            })

        self.results = pd.DataFrame(results).set_index("AccidentYear")

        # Compute BF loss ratios
        if self.premium is not None:
            self.results["UltimateLR_BF"] = (
                self.results["Ultimate_BF"] / self.results["Premium"]
            )
            self.results["UltimateLR_CL"] = (
                self.results["Ultimate_CL"] / self.results["Premium"]
            )

        self._fitted = True
        return self

    def summary(self):
        """Print BF vs chain ladder reserve comparison."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        r = self.results
        print("=" * 80)
        print("  BORNHUETTER-FERGUSON vs CHAIN LADDER RESERVE COMPARISON")
        print("=" * 80)

        display_cols = ["LatestLag", "LatestLoss", "UnreportedFactor",
                        "IBNR_BF", "Ultimate_BF", "IBNR_CL", "Ultimate_CL"]
        print(r[display_cols].round({
            "LatestLoss": 0, "UnreportedFactor": 4,
            "IBNR_BF": 0, "Ultimate_BF": 0,
            "IBNR_CL": 0, "Ultimate_CL": 0,
        }).to_string())

        print("-" * 80)
        print(f"  Total IBNR (BF)          : ${r['IBNR_BF'].sum():>15,.0f}")
        print(f"  Total IBNR (Chain Ladder): ${r['IBNR_CL'].sum():>15,.0f}")
        print(f"  Difference               : ${r['IBNR_Diff'].sum():>15,.0f}")

        if "UltimateLR_BF" in r.columns:
            wtd_lr_bf = r["Ultimate_BF"].sum() / r["Premium"].sum()
            wtd_lr_cl = r["Ultimate_CL"].sum() / r["Premium"].sum()
            print(f"  Wtd Ultimate LR (BF)     : {wtd_lr_bf:.4f}  ({100*wtd_lr_bf:.1f}%)")
            print(f"  Wtd Ultimate LR (CL)     : {wtd_lr_cl:.4f}  ({100*wtd_lr_cl:.1f}%)")
        print("=" * 80)

    def credibility_weights(self) -> pd.DataFrame:
        """
        Show the implicit credibility weights in the BF method.

        BF = z * CL_ultimate + (1-z) * a_priori_ultimate
        where z = 1/CDF = fraction of losses already reported

        This makes the Bayesian interpretation explicit.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        r = self.results.copy()
        r["Credibility (z=1/CDF)"] = 1.0 / r["CDF"]
        r["Weight on A Priori"] = 1.0 - (1.0 / r["CDF"])

        print("\n--- BF Credibility Weights (Bayesian Interpretation) ---")
        print("z = 1/CDF = fraction reported = weight on observed data")
        print()
        display = r[["LatestLag", "CDF", "Credibility (z=1/CDF)", "Weight on A Priori"]].round(4)
        print(display.to_string())
        print()
        print("Mature years (z → 1.0): BF converges to chain ladder")
        print("Immature years (z → 0): BF leans on a priori ELR")
        return display

    def plot(self, save_path: str = None, grname: str = ""):
        """
        4-panel comparison plot:
            1. IBNR: BF vs chain ladder by accident year
            2. Ultimate loss ratio: BF vs chain ladder
            3. Credibility weights by accident year
            4. IBNR difference (BF - CL)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        r = self.results
        years = r.index.astype(str)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(
            f"BF vs Chain Ladder{': ' + grname if grname else ''}  "
            f"(ELR={self.elr:.1%})",
            fontsize=13, fontweight="bold"
        )

        x = np.arange(len(years))
        width = 0.35

        # --- 1. IBNR comparison ---
        ax = axes[0, 0]
        ax.bar(x - width/2, r["IBNR_CL"]/1000, width,
               label="Chain Ladder", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, r["IBNR_BF"]/1000, width,
               label="BF", color="crimson", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, fontsize=8)
        ax.set_ylabel("IBNR ($000s)")
        ax.set_title("IBNR by Accident Year: BF vs Chain Ladder")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        # --- 2. Ultimate loss ratio ---
        ax2 = axes[0, 1]
        if "UltimateLR_BF" in r.columns:
            ax2.plot(r.index, r["UltimateLR_CL"], "o--",
                     color="steelblue", lw=2, label="Chain Ladder LR")
            ax2.plot(r.index, r["UltimateLR_BF"], "s-",
                     color="crimson", lw=2, label="BF LR")
            ax2.axhline(self.elr, color="gray", linestyle=":",
                        alpha=0.8, label=f"A Priori ELR ({self.elr:.1%})")
            ax2.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
            ax2.set_ylabel("Ultimate Loss Ratio")
        ax2.set_xlabel("Accident Year")
        ax2.set_title("Ultimate Loss Ratio: BF vs Chain Ladder")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # --- 3. Credibility weights ---
        ax3 = axes[1, 0]
        credibility = 1.0 / r["CDF"]
        ax3.bar(years, credibility, color="seagreen", alpha=0.8)
        ax3.set_xlabel("Accident Year")
        ax3.set_ylabel("Credibility z = 1/CDF")
        ax3.set_title("BF Credibility Weight by Accident Year\n(z=1 → chain ladder, z=0 → a priori)")
        ax3.set_ylim(0, 1.05)
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(alpha=0.3, axis="y")
        for i, (yr, z) in enumerate(zip(years, credibility)):
            ax3.text(i, z + 0.02, f"{z:.2f}", ha="center", fontsize=7)

        # --- 4. IBNR difference ---
        ax4 = axes[1, 1]
        colors = ["crimson" if d > 0 else "steelblue" for d in r["IBNR_Diff"]]
        ax4.bar(years, r["IBNR_Diff"]/1000, color=colors, alpha=0.8)
        ax4.axhline(0, color="black", lw=1)
        ax4.set_xlabel("Accident Year")
        ax4.set_ylabel("IBNR Difference ($000s)")
        ax4.set_title("IBNR Difference: BF minus Chain Ladder\n(red = BF higher, blue = BF lower)")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")
        plt.show()
        return fig


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------

def truncate_triangle(triangle: pd.DataFrame, valuation_lag: int) -> pd.DataFrame:
    """
    Truncate a fully-developed triangle to simulate a mid-development
    valuation date.

    For example, truncate at lag 6 means:
        - Earliest accident year has 6 years of development
        - Each subsequent year has one less lag observed
        - Most recent year has only 1 lag observed

    This mirrors a real actuarial valuation where recent years are immature.

    Parameters
    ----------
    triangle     : fully developed triangle (AccidentYear x DevelopmentLag)
    valuation_lag: maximum development lag for the earliest accident year

    Returns
    -------
    Truncated triangle with NaN in the lower-right corner
    """
    tri = triangle.copy().astype(float)
    acc_years = sorted(tri.index)
    lags = sorted(tri.columns)

    for i, acc_year in enumerate(acc_years):
        # Each subsequent year has one fewer lag observed
        max_lag = valuation_lag - i
        for lag in lags:
            if lag > max_lag:
                tri.loc[acc_year, lag] = np.nan

    # Drop columns that are entirely NaN
    tri = tri.dropna(axis=1, how="all")
    return tri


if __name__ == "__main__":
    import sys
    data_path = "data/cas_triangles/ppauto_pos98-07.csv"

    if not Path(data_path).exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    # Load data
    triangle, premium, grname = load_cas_triangle(data_path)

    # Truncate to simulate valuation at end of 2003
    # 1998 AY has 6 lags, 1999 has 5, ..., 2003 has 1
    print("\nTruncating triangle to simulate mid-development valuation (lag 6)...")
    triangle_truncated = truncate_triangle(triangle, valuation_lag=6)
    print(triangle_truncated.to_string())
    print()

    # Fit chain ladder on truncated triangle
    cl = ChainLadder(triangle_truncated, premium)
    cl.fit()

    print("\nFitting Bornhuetter-Ferguson...\n")

    # BF with ELR from chain ladder weighted average
    bf = BornhuetterFerguson(cl)
    bf.fit()
    bf.summary()
    bf.credibility_weights()
    bf.plot(save_path="bornhuetter_ferguson.png", grname=grname)

    # BF with user-specified ELR (e.g. from pricing department)
    print("\n--- BF with user-specified ELR = 65% ---\n")
    bf_manual = BornhuetterFerguson(cl, elr=0.65)
    bf_manual.fit()
    bf_manual.summary()
