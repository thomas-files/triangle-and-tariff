"""
report.py
---------
Combined actuarial summary report.

Ties together:
    - Reserving : IBNR by accident year (chain ladder + BF)
    - Pricing   : pure premium by risk segment (freq × severity GLM)
    - Summary   : indicated rate change, reserve adequacy, combined ratio

Usage
-----
    python actuarial/report.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from chain_ladder import ChainLadder, load_cas_triangle, truncate_triangle
from bornhuetter_ferguson import BornhuetterFerguson
from frequency import FrequencyModel, load_fremtpl
from severity import SeverityModel, load_severity_data


class ActuarialReport:
    """
    End-to-end actuarial report combining reserving and pricing.

    Reserving side (CAS Schedule P):
        - Chain ladder IBNR by accident year
        - BF IBNR with a priori ELR
        - Reserve comparison and adequacy metrics

    Pricing side (French MTPL):
        - Poisson GLM claim frequency
        - Gamma GLM claim severity
        - Pure premium = frequency x severity by risk segment

    Parameters
    ----------
    triangle_path : path to CAS Schedule P CSV
    freq_path     : path to freMTPL2freq.csv
    sev_path      : path to freMTPL2sev.csv
    valuation_lag : development lag to truncate triangle at (default 6)
    elr           : a priori expected loss ratio for BF (default: from CL)
    """

    def __init__(
        self,
        triangle_path: str,
        freq_path: str,
        sev_path: str,
        valuation_lag: int = 6,
        elr: float = None,
    ):
        self.triangle_path = triangle_path
        self.freq_path = freq_path
        self.sev_path = sev_path
        self.valuation_lag = valuation_lag
        self.elr = elr
        self._fitted = False

    def run(self, seed: int = 42):
        """Run full actuarial workflow."""
        print("=" * 65)
        print("  TRIANGLE AND TARIFF — ACTUARIAL REPORT")
        print("=" * 65)

        # Step 1-3: Reserving
        print("\n[1/6] Loading loss triangle...")
        triangle, premium, self.grname = load_cas_triangle(self.triangle_path)
        triangle_trunc = truncate_triangle(triangle, self.valuation_lag)

        print(f"\n[2/6] Fitting chain ladder (valuation lag={self.valuation_lag})...")
        self.cl = ChainLadder(triangle_trunc, premium)
        self.cl.fit()

        print("\n[3/6] Fitting Bornhuetter-Ferguson...")
        self.bf = BornhuetterFerguson(self.cl, elr=self.elr)
        self.bf.fit()

        # Step 4-6: Pricing
        print("\n[4/6] Loading pricing data...")
        self.freq_df = load_fremtpl(self.freq_path)
        self.sev_df = load_severity_data(self.freq_path, self.sev_path)

        print("\n[5/6] Fitting frequency model (Poisson GLM)...")
        self.freq_model = FrequencyModel()
        self.freq_model.fit(self.freq_df, seed=seed)

        print("\n[6/6] Fitting severity model (Gamma GLM)...")
        self.sev_model = SeverityModel()
        self.sev_model.fit(self.sev_df, seed=seed)

        # Pure premium
        df = self.freq_df.copy()
        df["pred_freq"] = self.freq_model.model.predict(
            df, offset=np.log(df["Exposure"])
        ).values / df["Exposure"].values
        df["pred_sev"] = self.sev_model.model.predict(df).values
        df["pure_premium"] = df["pred_freq"] * df["pred_sev"]
        self.pricing_df = df

        self._fitted = True
        print("\n✓ All models fitted successfully.\n")
        return self

    def summary(self):
        """Print combined actuarial summary report."""
        if not self._fitted:
            raise RuntimeError("Call .run() first")

        cl_r = self.cl.results
        bf_r = self.bf.results
        pp = self.pricing_df

        print("\n" + "=" * 65)
        print("  COMBINED ACTUARIAL SUMMARY")
        print("=" * 65)

        # Reserving
        print("\n-- RESERVING --")
        print(f"  Company         : {self.grname}")
        print(f"  Valuation lag   : {self.valuation_lag}")
        print(f"  {'Metric':<35} {'Chain Ladder':>15} {'BF':>15}")
        print(f"  {'-'*35} {'-'*15} {'-'*15}")
        print(f"  {'Total IBNR':<35} ${cl_r['IBNR'].sum():>14,.0f} ${bf_r['IBNR_BF'].sum():>14,.0f}")
        print(f"  {'Total Ultimate':<35} ${cl_r['Ultimate'].sum():>14,.0f} ${bf_r['Ultimate_BF'].sum():>14,.0f}")

        if "UltimateLR_CL" in bf_r.columns:
            wtd_lr_cl = cl_r["Ultimate"].sum() / cl_r["Premium"].sum()
            wtd_lr_bf = bf_r["Ultimate_BF"].sum() / bf_r["Premium"].sum()
            print(f"  {'Wtd Ultimate Loss Ratio':<35} {wtd_lr_cl:>14.1%} {wtd_lr_bf:>14.1%}")

        ibnr_diff = bf_r["IBNR_BF"].sum() - cl_r["IBNR"].sum()
        print(f"\n  BF vs CL IBNR difference: ${ibnr_diff:+,.0f}")
        if abs(ibnr_diff) / max(cl_r["IBNR"].sum(), 1) > 0.10:
            print("  Warning: Methods diverge >10% — review a priori ELR")
        else:
            print("  Methods broadly consistent")

        # Pricing
        print("\n-- PRICING --")
        print(f"  Dataset          : French MTPL ({len(self.freq_df):,} policies)")
        print(f"  Mean frequency   : {pp['pred_freq'].mean():.4f} claims/year")
        print(f"  Mean severity    : ${pp['pred_sev'].mean():,.2f} per claim")
        print(f"  Mean pure premium: ${pp['pure_premium'].mean():,.2f} per policy-year")

        # Indicated rate change
        if "UltimateLR_BF" in bf_r.columns:
            target_lr = 1.0 / 1.25
            current_lr = bf_r["Ultimate_BF"].sum() / bf_r["Premium"].sum()
            indicated_change = (current_lr / target_lr) - 1
            print(f"\n-- INDICATED RATE CHANGE --")
            print(f"  Target loss ratio : {target_lr:.1%}  (assuming 25% expense ratio)")
            print(f"  Current loss ratio: {current_lr:.1%}  (BF ultimate)")
            print(f"  Indicated change  : {indicated_change:+.1%}")
            direction = "increase" if indicated_change > 0 else "decrease"
            print(f"  Rate {direction} of {abs(indicated_change):.1%} indicated")

        # Risk segments
        print("\n-- TOP RISK SEGMENTS (by pure premium) --")
        age_pp = pp.groupby("DrivAgeBand")["pure_premium"].mean()
        age_pp = age_pp.reindex(["18-25", "26-35", "36-50", "51-65", "66+"])
        overall_pp = pp["pure_premium"].mean()
        for age, pp_val in age_pp.items():
            rel = pp_val / overall_pp
            bar = "█" * int(rel * 10)
            print(f"  {age:<8} ${pp_val:>8,.2f}  {rel:.2f}x  {bar}")

        print("\n" + "=" * 65)

    def plot(self, save_path: str = None):
        """6-panel combined actuarial dashboard."""
        if not self._fitted:
            raise RuntimeError("Call .run() first")

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Triangle and Tariff — Actuarial Dashboard",
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.4)

        cl_r = self.cl.results
        bf_r = self.bf.results
        pp = self.pricing_df

        # Panel 1: IBNR comparison
        ax1 = fig.add_subplot(gs[0, 0])
        years = cl_r.index.astype(str)
        x = np.arange(len(years))
        w = 0.35
        ax1.bar(x - w/2, cl_r["IBNR"]/1000, w,
                label="Chain Ladder", color="steelblue", alpha=0.8)
        ax1.bar(x + w/2, bf_r["IBNR_BF"]/1000, w,
                label="BF", color="crimson", alpha=0.8)
        ax1.set_title("IBNR by Accident Year")
        ax1.set_xlabel("Accident Year")
        ax1.set_ylabel("IBNR ($000s)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(years, rotation=45, fontsize=7)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3, axis="y")

        # Panel 2: Credibility weights
        ax2 = fig.add_subplot(gs[0, 1])
        cred = 1.0 / bf_r["CDF"]
        ax2.bar(years, cred, color="seagreen", alpha=0.8)
        ax2.set_title("BF Credibility Weight\n(z = 1/CDF)")
        ax2.set_xlabel("Accident Year")
        ax2.set_ylabel("Credibility z")
        ax2.set_ylim(0, 1.1)
        ax2.set_xticklabels(years, rotation=45, fontsize=7)
        ax2.grid(alpha=0.3, axis="y")

        # Panel 3: Ultimate loss ratio
        ax3 = fig.add_subplot(gs[0, 2])
        if "UltimateLR_BF" in bf_r.columns:
            ax3.plot(cl_r.index, bf_r["UltimateLR_CL"],
                     "o--", color="steelblue", lw=2, label="CL")
            ax3.plot(cl_r.index, bf_r["UltimateLR_BF"],
                     "s-", color="crimson", lw=2, label="BF")
            ax3.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
            ax3.legend(fontsize=8)
        ax3.set_title("Ultimate Loss Ratio")
        ax3.set_xlabel("Accident Year")
        ax3.tick_params(axis="x", rotation=45, labelsize=7)
        ax3.grid(alpha=0.3)

        # Panel 4: Frequency by driver age
        ax4 = fig.add_subplot(gs[1, 0])
        age_order = ["18-25", "26-35", "36-50", "51-65", "66+"]
        freq_by_age = pp.groupby("DrivAgeBand")["pred_freq"].mean().reindex(age_order)
        ax4.bar(age_order, freq_by_age, color="steelblue", alpha=0.8)
        ax4.set_title("Predicted Frequency\nby Driver Age")
        ax4.set_xlabel("Driver Age Band")
        ax4.set_ylabel("Claims / Policy-Year")
        ax4.tick_params(axis="x", rotation=30, labelsize=8)
        ax4.grid(alpha=0.3, axis="y")

        # Panel 5: Severity by driver age
        ax5 = fig.add_subplot(gs[1, 1])
        sev_by_age = pp.groupby("DrivAgeBand")["pred_sev"].mean().reindex(age_order)
        ax5.bar(age_order, sev_by_age, color="seagreen", alpha=0.8)
        ax5.set_title("Predicted Severity\nby Driver Age")
        ax5.set_xlabel("Driver Age Band")
        ax5.set_ylabel("Cost per Claim ($)")
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax5.tick_params(axis="x", rotation=30, labelsize=8)
        ax5.grid(alpha=0.3, axis="y")

        # Panel 6: Pure premium by driver age
        ax6 = fig.add_subplot(gs[1, 2])
        pp_by_age = pp.groupby("DrivAgeBand")["pure_premium"].mean().reindex(age_order)
        colors = ["crimson" if v > pp["pure_premium"].mean() else "steelblue"
                  for v in pp_by_age]
        ax6.bar(age_order, pp_by_age, color=colors, alpha=0.8)
        ax6.axhline(pp["pure_premium"].mean(), color="black",
                    linestyle="--", lw=1.5, label="Portfolio avg")
        ax6.set_title("Pure Premium\nby Driver Age")
        ax6.set_xlabel("Driver Age Band")
        ax6.set_ylabel("Pure Premium ($/yr)")
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax6.tick_params(axis="x", rotation=30, labelsize=8)
        ax6.legend(fontsize=8)
        ax6.grid(alpha=0.3, axis="y")

        # Panel 7: Full bottom row — pure premium by vehicle age
        ax7 = fig.add_subplot(gs[2, :])
        veh_order = ["0-1yr", "2-5yr", "6-10yr", "11-15yr", "15+yr"]
        pp_by_veh = pp.groupby("VehAgeBand")["pure_premium"].mean().reindex(veh_order)
        freq_by_veh = pp.groupby("VehAgeBand")["pred_freq"].mean().reindex(veh_order)
        sev_by_veh = pp.groupby("VehAgeBand")["pred_sev"].mean().reindex(veh_order)

        x = np.arange(len(veh_order))
        w = 0.25
        ax7.bar(x - w, freq_by_veh * 1000, w,
                label="Frequency (x1000)", color="steelblue", alpha=0.8)
        ax7.bar(x, sev_by_veh / 10, w,
                label="Severity (÷10)", color="seagreen", alpha=0.8)
        ax7.bar(x + w, pp_by_veh, w,
                label="Pure Premium", color="crimson", alpha=0.8)
        ax7.set_xticks(x)
        ax7.set_xticklabels(veh_order, fontsize=10)
        ax7.set_xlabel("Vehicle Age Band")
        ax7.set_ylabel("Value ($)")
        ax7.set_title("Frequency x Severity = Pure Premium by Vehicle Age\n"
                      "(frequency scaled x1000, severity scaled ÷10 for display)")
        ax7.legend(fontsize=9)
        ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax7.grid(alpha=0.3, axis="y")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")
        plt.show()
        return fig


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    triangle_path = "data/cas_triangles/ppauto_pos98-07.csv"
    freq_path = "data/fremtpl/freMTPL2freq.csv"
    sev_path = "data/fremtpl/freMTPL2sev.csv"

    for p in [triangle_path, freq_path, sev_path]:
        if not Path(p).exists():
            print(f"Data not found: {p}")
            sys.exit(1)

    report = ActuarialReport(
        triangle_path=triangle_path,
        freq_path=freq_path,
        sev_path=sev_path,
        valuation_lag=6,
    )
    report.run()
    report.summary()
    report.plot(save_path="actuarial_report.png")
