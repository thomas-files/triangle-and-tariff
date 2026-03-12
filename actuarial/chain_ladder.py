"""
chain_ladder.py
---------------
Chain ladder (development) method for loss reserving.

The chain ladder method projects cumulative losses to ultimate by computing
age-to-age development factors from the observed triangle, then applying
those factors to fill in the lower-right (未развитый) portion.

Key outputs:
    - Age-to-age development factors (LDFs)
    - Cumulative development factors (CDFs) to ultimate
    - Projected ultimate losses by accident year
    - IBNR = Ultimate - Latest Diagonal
    - Loss ratio by accident year

Data
----
    CAS Schedule P — private passenger auto
    Columns used: AccidentYear, DevelopmentLag, CumPaidLoss, EarnedPremNet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_cas_triangle(
    path: str,
    grcode: int = None,
    loss_col: str = "CumPaidLoss",
) -> tuple:
    """
    Load CAS Schedule P data and pivot into a loss triangle.

    Parameters
    ----------
    path     : path to CAS CSV file
    grcode   : company code to filter on (None = use largest by premium)
    loss_col : column to use for triangle values (CumPaidLoss or IncurredLosses)

    Returns
    -------
    triangle : pd.DataFrame  (AccidentYear × DevelopmentLag, cumulative losses)
    premium  : pd.Series     (EarnedPremNet by AccidentYear)
    grname   : str           (company name)
    """
    df = pd.read_csv(path)

    if grcode is None:
        # Pick company with most earned premium
        grcode = (
            df.groupby("GRCODE")["EarnedPremNet"].sum().idxmax()
        )

    subset = df[df["GRCODE"] == grcode].copy()
    grname = subset["GRNAME"].iloc[0]

    # Pivot to triangle: rows = AccidentYear, cols = DevelopmentLag
    triangle = subset.pivot(
        index="AccidentYear",
        columns="DevelopmentLag",
        values=loss_col,
    ).sort_index()

    # Premium by accident year (constant across development lags)
    premium = (
        subset.groupby("AccidentYear")["EarnedPremNet"].first()
    )

    print(f"Loaded: {grname} (GRCODE={grcode})")
    print(f"  Accident years : {triangle.index.min()} – {triangle.index.max()}")
    print(f"  Development lags: {triangle.columns.min()} – {triangle.columns.max()}")
    print(f"  Total premium  : ${premium.sum():,.0f}")

    return triangle, premium, grname


def list_companies(path: str) -> pd.DataFrame:
    """List all companies in the CAS file with their total premium."""
    df = pd.read_csv(path)
    summary = (
        df.groupby(["GRCODE", "GRNAME"])["EarnedPremNet"]
        .sum()
        .reset_index()
        .sort_values("EarnedPremNet", ascending=False)
    )
    summary["EarnedPremNet"] = summary["EarnedPremNet"].round(0)
    return summary


# ---------------------------------------------------------------------
# Chain Ladder
# ---------------------------------------------------------------------

class ChainLadder:
    """
    Volume-weighted chain ladder reserve estimator.

    Steps:
        1. Compute age-to-age factors: f_k = Σ C_{i,k+1} / Σ C_{i,k}
        2. Compute tail factor (assume 1.0 — fully developed at lag 10)
        3. Compute CDFs to ultimate: F_k = f_k * f_{k+1} * ... * f_tail
        4. Project each accident year: Ultimate_i = C_{i, latest} * CDF_i
        5. IBNR_i = Ultimate_i - C_{i, latest}

    Parameters
    ----------
    triangle : pd.DataFrame (AccidentYear × DevelopmentLag, cumulative losses)
    premium  : pd.Series    (EarnedPremNet by AccidentYear)
    tail     : float        (tail factor beyond last development period, default 1.0)
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        premium: pd.Series = None,
        tail: float = 1.0,
    ):
        self.triangle = triangle.copy().astype(float)
        self.premium = premium
        self.tail = tail
        self._fitted = False

    def fit(self):
        """Compute development factors and project to ultimate."""
        tri = self.triangle
        lags = sorted(tri.columns)
        n_years = len(tri)

        # ── Step 1: Age-to-age factors ──────────────────────────────
        ldfs = {}
        for i, lag in enumerate(lags[:-1]):
            next_lag = lags[i + 1]
            # Only use rows where both columns are observed
            mask = tri[lag].notna() & tri[next_lag].notna()
            if mask.sum() == 0:
                ldfs[lag] = 1.0
            else:
                ldfs[lag] = tri.loc[mask, next_lag].sum() / tri.loc[mask, lag].sum()

        self.ldfs = pd.Series(ldfs, name="LDF")
        self.ldfs.index.name = "From Lag"

        # ── Step 2: CDFs to ultimate ────────────────────────────────
        cdfs = {}
        lags_sorted = sorted(ldfs.keys())
        for i, lag in enumerate(lags_sorted):
            # CDF = product of all future LDFs × tail
            cdf = self.tail
            for future_lag in lags_sorted[i:]:
                cdf *= ldfs[future_lag]
            cdfs[lag] = cdf
        # Last lag: just the tail
        cdfs[lags[-1]] = self.tail
        self.cdfs = pd.Series(cdfs, name="CDF to Ultimate")
        self.cdfs.index.name = "From Lag"

        # ── Step 3: Project each accident year to ultimate ──────────
        results = []
        for acc_year in tri.index:
            row = tri.loc[acc_year]
            # Latest non-null value
            observed = row.dropna()
            if len(observed) == 0:
                continue
            latest_lag = observed.index.max()
            latest_loss = observed[latest_lag]
            cdf = self.cdfs[latest_lag]
            ultimate = latest_loss * cdf
            ibnr = ultimate - latest_loss

            result = {
                "AccidentYear": acc_year,
                "LatestLag": latest_lag,
                "LatestLoss": latest_loss,
                "CDF": cdf,
                "Ultimate": ultimate,
                "IBNR": ibnr,
            }

            if self.premium is not None and acc_year in self.premium.index:
                prem = self.premium[acc_year]
                result["Premium"] = prem
                result["UltimateLR"] = ultimate / prem if prem > 0 else np.nan
                result["ReportedLR"] = latest_loss / prem if prem > 0 else np.nan

            results.append(result)

        self.results = pd.DataFrame(results).set_index("AccidentYear")
        self._fitted = True
        return self

    def summary(self):
        """Print reserve summary table."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        r = self.results
        print("=" * 75)
        print("  CHAIN LADDER RESERVE SUMMARY")
        print("=" * 75)
        cols = ["LatestLag", "LatestLoss", "CDF", "Ultimate", "IBNR"]
        if "UltimateLR" in r.columns:
            cols += ["UltimateLR"]
        print(r[cols].round({"LatestLoss": 0, "Ultimate": 0,
                              "IBNR": 0, "CDF": 4, "UltimateLR": 4}).to_string())
        print("-" * 75)
        print(f"  Total IBNR    : ${r['IBNR'].sum():>15,.0f}")
        print(f"  Total Ultimate: ${r['Ultimate'].sum():>15,.0f}")
        if "UltimateLR" in r.columns:
            wtd_lr = r["Ultimate"].sum() / r["Premium"].sum()
            print(f"  Wtd Ultimate LR: {wtd_lr:.4f}  ({100*wtd_lr:.1f}%)")
        print("=" * 75)

    def development_factors(self):
        """Print LDF and CDF table."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        df = pd.DataFrame({
            "LDF (age-to-age)": self.ldfs,
            "CDF (to ultimate)": self.cdfs.reindex(self.ldfs.index),
        }).round(5)
        print("\n--- Development Factors ---")
        print(df.to_string())
        print(f"  Tail factor: {self.tail}")

    def projected_triangle(self) -> pd.DataFrame:
        """
        Return the full projected triangle — observed values plus
        chain ladder projections in the lower-right corner.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        tri = self.triangle.copy()
        lags = sorted(tri.columns)

        for acc_year in tri.index:
            row = tri.loc[acc_year]
            observed = row.dropna()
            if len(observed) == 0:
                continue
            latest_lag = observed.index.max()
            latest_loss = observed[latest_lag]

            # Fill in future development
            current_loss = latest_loss
            lag_list = sorted(self.ldfs.index)
            for lag in lag_list:
                if lag >= latest_lag:
                    next_lag_candidates = [l for l in lags if l > lag]
                    if not next_lag_candidates:
                        break
                    next_lag = next_lag_candidates[0]
                    if pd.isna(tri.loc[acc_year, next_lag]):
                        current_loss = current_loss * self.ldfs[lag]
                        tri.loc[acc_year, next_lag] = current_loss

        return tri

    def plot(self, save_path: str = None, grname: str = ""):
        """
        4-panel diagnostic plot:
            1. Loss triangle heatmap
            2. Development factor chart
            3. IBNR by accident year
            4. Ultimate loss ratio by accident year
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Chain Ladder Reserve Analysis{': ' + grname if grname else ''}",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # --- 1. Triangle heatmap ---
        ax1 = fig.add_subplot(gs[0, 0])
        proj = self.projected_triangle()
        # Normalize by row max for color scale
        norm = proj.div(proj.max(axis=1), axis=0)
        im = ax1.imshow(norm.values, aspect="auto", cmap="YlOrRd")
        ax1.set_xticks(range(len(proj.columns)))
        ax1.set_xticklabels(proj.columns, fontsize=8)
        ax1.set_yticks(range(len(proj.index)))
        ax1.set_yticklabels(proj.index, fontsize=8)
        ax1.set_xlabel("Development Lag")
        ax1.set_ylabel("Accident Year")
        ax1.set_title("Loss Development Triangle\n(darker = more developed)")
        plt.colorbar(im, ax=ax1, fraction=0.046)

        # --- 2. Development factors ---
        ax2 = fig.add_subplot(gs[0, 1])
        ldf_vals = self.ldfs.values
        ldf_idx = [str(i) for i in self.ldfs.index]
        bars = ax2.bar(ldf_idx, ldf_vals, color="steelblue", alpha=0.8)
        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        ax2.set_xlabel("From Development Lag")
        ax2.set_ylabel("Age-to-Age Factor")
        ax2.set_title("Loss Development Factors (LDFs)")
        ax2.grid(alpha=0.3, axis="y")
        for bar, val in zip(bars, ldf_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        # --- 3. IBNR by accident year ---
        ax3 = fig.add_subplot(gs[1, 0])
        r = self.results
        ax3.bar(r.index.astype(str), r["LatestLoss"] / 1000,
                label="Reported", color="steelblue", alpha=0.8)
        ax3.bar(r.index.astype(str), r["IBNR"] / 1000,
                bottom=r["LatestLoss"] / 1000,
                label="IBNR", color="crimson", alpha=0.7)
        ax3.set_xlabel("Accident Year")
        ax3.set_ylabel("Losses ($000s)")
        ax3.set_title("Reported vs IBNR by Accident Year")
        ax3.legend(fontsize=9)
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(alpha=0.3, axis="y")

        # --- 4. Ultimate loss ratio ---
        ax4 = fig.add_subplot(gs[1, 1])
        if "UltimateLR" in r.columns:
            ax4.plot(r.index, r["ReportedLR"], "o--",
                     color="steelblue", lw=2, label="Reported LR")
            ax4.plot(r.index, r["UltimateLR"], "s-",
                     color="crimson", lw=2, label="Ultimate LR")
            ax4.set_ylabel("Loss Ratio")
            ax4.set_title("Reported vs Ultimate Loss Ratio")
            ax4.legend(fontsize=9)
            ax4.grid(alpha=0.3)
            ax4.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
        else:
            ax4.plot(r.index, r["IBNR"] / r["Ultimate"],
                     "o-", color="crimson", lw=2)
            ax4.set_ylabel("IBNR / Ultimate")
            ax4.set_title("IBNR as % of Ultimate by Accident Year")
            ax4.grid(alpha=0.3)

        ax4.set_xlabel("Accident Year")
        ax4.tick_params(axis="x", rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")
        plt.show()
        return fig


def truncate_triangle(triangle: pd.DataFrame, valuation_lag: int) -> pd.DataFrame:
    """
    Truncate a fully-developed triangle to simulate a mid-development
    valuation date. Useful for backtesting and demonstration.

    Parameters
    ----------
    triangle     : fully developed triangle
    valuation_lag: max lag for the earliest accident year

    Returns
    -------
    Truncated triangle with NaN in the lower-right corner
    """
    tri = triangle.copy().astype(float)
    acc_years = sorted(tri.index)
    lags = sorted(tri.columns)

    for i, acc_year in enumerate(acc_years):
        max_lag = valuation_lag - i
        for lag in lags:
            if lag > max_lag:
                tri.loc[acc_year, lag] = np.nan

    return tri.dropna(axis=1, how="all")


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_path = "data/cas_triangles/ppauto_pos98-07.csv"

    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        print("Download from: casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p")
        sys.exit(1)

    # Show available companies
    print("Top 10 companies by earned premium:")
    companies = list_companies(data_path)
    print(companies.head(10).to_string(index=False))
    print()

    # Load largest company
    triangle, premium, grname = load_cas_triangle(data_path)

    # Truncate to simulate mid-development valuation (lag 6)
    print("\nTruncating triangle to simulate valuation at lag 6...")
    triangle_truncated = truncate_triangle(triangle, valuation_lag=6)

    # Fit chain ladder
    cl = ChainLadder(triangle_truncated, premium)
    cl.fit()
    cl.development_factors()
    print()
    cl.summary()
    cl.plot(save_path="chain_ladder.png", grname=grname)
