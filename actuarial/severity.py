"""
severity.py
-----------
Gamma GLM for claims severity (average cost per claim).

Models expected claim severity as:
    E[S_i | N_i > 0] = exp(X_i @ beta)

The Gamma distribution is appropriate because:
    - Claim costs are positive and continuous
    - Variance tends to increase with the mean (constant CV assumption)
    - Log link ensures positive predictions

We fit ONLY on policies with at least one claim (N_i > 0).
This is the standard frequency-severity split:
    Pure Premium = Frequency * Severity
    E[Total Cost] = E[N] * E[S | N > 0]

Key outputs:
    - Fitted Gamma GLM with coefficients and relativities
    - Pure premium = predicted frequency * predicted severity
    - Risk segmentation: pure premium by driver age and vehicle type

Data
----
    freMTPL2freq.csv  — policy-level features + claim counts
    freMTPL2sev.csv   — claim-level amounts (one row per claim)

    Join on IDpol: aggregate severity to policy level, then merge
    with frequency features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from frequency import load_fremtpl, FrequencyModel


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_severity_data(freq_path: str, sev_path: str) -> pd.DataFrame:
    """
    Load and join frequency and severity datasets.

    Steps:
        1. Load freMTPL2sev (one row per claim)
        2. Aggregate to policy level: sum and mean claim amount per IDpol
        3. Join to freMTPL2freq features
        4. Filter to policies with at least one claim

    Parameters
    ----------
    freq_path : path to freMTPL2freq.csv
    sev_path  : path to freMTPL2sev.csv

    Returns
    -------
    pd.DataFrame with one row per claimant policy, including features
    """
    # Load frequency features (already engineered)
    freq_df = load_fremtpl(freq_path)

    # Load severity
    sev_df = pd.read_csv(sev_path)
    print(f"\nLoaded freMTPL2sev: {len(sev_df):,} individual claims")
    print(f"  Claim amount range: ${sev_df['ClaimAmount'].min():,.2f} – ${sev_df['ClaimAmount'].max():,.2f}")
    print(f"  Mean claim amount : ${sev_df['ClaimAmount'].mean():,.2f}")
    print(f"  Median claim amount: ${sev_df['ClaimAmount'].median():,.2f}")

    # Aggregate to policy level
    sev_agg = sev_df.groupby("IDpol").agg(
        TotalClaimAmount=("ClaimAmount", "sum"),
        MeanClaimAmount=("ClaimAmount", "mean"),
        NClaimsInSev=("ClaimAmount", "count"),
    ).reset_index()

    # Join to frequency features
    merged = freq_df.merge(sev_agg, on="IDpol", how="inner")

    # Filter to policies with positive claim amount
    merged = merged[merged["TotalClaimAmount"] > 0].copy()

    print(f"\nAfter join: {len(merged):,} claimant policies")
    print(f"  Mean total claim  : ${merged['TotalClaimAmount'].mean():,.2f}")
    print(f"  Mean claim amount : ${merged['MeanClaimAmount'].mean():,.2f}")

    return merged


# ---------------------------------------------------------------------
# Gamma GLM
# ---------------------------------------------------------------------

class SeverityModel:
    """
    Gamma GLM for claim severity.

    Fits on claimant policies only (those with TotalClaimAmount > 0).
    Uses same rating factors as frequency model.

    Parameters
    ----------
    formula : statsmodels formula string
    target  : column to model ('MeanClaimAmount' or 'TotalClaimAmount')
    """

    DEFAULT_FORMULA = (
        "MeanClaimAmount ~ LogBonusMalus + DrivAgeBand + VehAgeBand + "
        "VehPowerBand + VehGas + Area + LogDensity"
    )

    def __init__(self, formula: str = None, target: str = "MeanClaimAmount"):
        self.formula = formula or self.DEFAULT_FORMULA
        self.target = target
        self._fitted = False

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        """
        Fit Gamma GLM on claimant policies.

        Parameters
        ----------
        df        : output of load_severity_data()
        test_size : test split fraction
        seed      : random seed
        """
        self.df = df.copy()

        self.train, self.test = train_test_split(
            df, test_size=test_size, random_state=seed
        )

        print(f"  Train: {len(self.train):,} claimant policies")
        print(f"  Test : {len(self.test):,} claimant policies")

        self.model = smf.glm(
            formula=self.formula,
            data=self.train,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        ).fit()

        self._fitted = True
        print(f"\n  Converged: {self.model.converged}")
        print(f"  Deviance : {self.model.deviance:,.1f}")
        print(f"  AIC      : {self.model.aic:,.1f}")

        return self

    def summary(self):
        """Print Gamma GLM coefficient table with relativities."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        print("\n" + "=" * 65)
        print("  GAMMA GLM — SEVERITY MODEL")
        print("=" * 65)

        params = self.model.params
        conf = self.model.conf_int()
        pvals = self.model.pvalues

        table = pd.DataFrame({
            "Coefficient": params,
            "Relativity": np.exp(params),
            "CI_Low": np.exp(conf[0]),
            "CI_High": np.exp(conf[1]),
            "p_value": pvals,
        }).round(4)

        table["Significant"] = table["p_value"] < 0.05
        print(table.to_string())
        print("=" * 65)
        print("Relativity = exp(coef) — multiplicative effect on severity")

    def predict(self, df: pd.DataFrame = None) -> np.ndarray:
        """Predict expected severity (cost per claim)."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")
        if df is None:
            df = self.test
        return self.model.predict(df).values

    def evaluate(self) -> dict:
        """Evaluate on test set — MAE and RMSE."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        y_true = self.test[self.target].values
        y_pred = self.predict(self.test)

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {"mae": mae, "rmse": rmse, "mape": mape}

        print("\n--- Severity Model Evaluation (Test Set) ---")
        print(f"  MAE  : ${mae:,.2f}")
        print(f"  RMSE : ${rmse:,.2f}")
        print(f"  MAPE : {mape:.1f}%")

        return metrics

    def plot(self, save_path: str = None):
        """
        4-panel diagnostic plot:
            1. Actual vs predicted severity by decile
            2. Severity relativity vs BonusMalus
            3. Severity distribution: actual vs fitted
            4. Severity by driver age band
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        test = self.test.copy()
        test["predicted_sev"] = self.predict(test)
        test["actual_sev"] = test[self.target]

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("Gamma GLM — Severity Model Diagnostics",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # --- 1. Actual vs predicted by decile ---
        ax1 = fig.add_subplot(gs[0, 0])
        test["pred_decile"] = pd.qcut(
            test["predicted_sev"], q=10, labels=False, duplicates="drop"
        )
        decile_summary = test.groupby("pred_decile").agg(
            actual=("actual_sev", "mean"),
            predicted=("predicted_sev", "mean"),
        )
        ax1.plot(decile_summary.index + 1, decile_summary["actual"],
                 "o-", color="steelblue", lw=2, label="Actual")
        ax1.plot(decile_summary.index + 1, decile_summary["predicted"],
                 "s--", color="crimson", lw=2, label="Predicted")
        ax1.set_xlabel("Predicted Severity Decile")
        ax1.set_ylabel("Mean Severity ($)")
        ax1.set_title("Actual vs Predicted Severity by Decile")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )

        # --- 2. BonusMalus vs severity relativity ---
        ax2 = fig.add_subplot(gs[0, 1])
        bm_range = np.linspace(
            self.df["LogBonusMalus"].min(),
            self.df["LogBonusMalus"].max(), 50
        )
        base_pred = self.model.params["Intercept"]
        bm_effect = base_pred + self.model.params["LogBonusMalus"] * bm_range
        ax2.plot(np.exp(bm_range), np.exp(bm_effect - base_pred),
                 color="steelblue", lw=2.5)
        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax2.set_xlabel("BonusMalus Score")
        ax2.set_ylabel("Severity Relativity")
        ax2.set_title("Severity Relativity vs BonusMalus")
        ax2.grid(alpha=0.3)

        # --- 3. Claim amount distribution ---
        ax3 = fig.add_subplot(gs[1, 0])
        clip_val = test["actual_sev"].quantile(0.99)
        actual_clipped = test["actual_sev"].clip(upper=clip_val)
        pred_clipped = test["predicted_sev"].clip(upper=clip_val)
        ax3.hist(actual_clipped, bins=50, alpha=0.6,
                 color="steelblue", label="Actual", density=True)
        ax3.hist(pred_clipped, bins=50, alpha=0.6,
                 color="crimson", label="Predicted", density=True)
        ax3.set_xlabel("Claim Amount ($)")
        ax3.set_ylabel("Density")
        ax3.set_title("Claim Amount Distribution\n(clipped at 99th percentile)")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        # --- 4. Severity by driver age band ---
        ax4 = fig.add_subplot(gs[1, 1])
        age_summary = test.groupby("DrivAgeBand").agg(
            actual=("actual_sev", "mean"),
            predicted=("predicted_sev", "mean"),
        ).reindex(["18-25", "26-35", "36-50", "51-65", "66+"])

        x = np.arange(len(age_summary))
        width = 0.35
        ax4.bar(x - width/2, age_summary["actual"],
                width, label="Actual", color="steelblue", alpha=0.8)
        ax4.bar(x + width/2, age_summary["predicted"],
                width, label="Predicted", color="crimson", alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(age_summary.index, fontsize=9)
        ax4.set_xlabel("Driver Age Band")
        ax4.set_ylabel("Mean Severity ($)")
        ax4.set_title("Actual vs Predicted Severity by Driver Age")
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3, axis="y")
        ax4.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")
        plt.show()
        return fig


# ---------------------------------------------------------------------
# Pure premium = frequency × severity
# ---------------------------------------------------------------------

def pure_premium_analysis(
    freq_model: FrequencyModel,
    sev_model: SeverityModel,
    df_full: pd.DataFrame,
    save_path: str = None,
):
    """
    Compute pure premium by risk segment.

    Pure Premium = Expected Frequency × Expected Severity
    = E[N/Exposure] × E[S | N > 0]

    This is the fundamental pricing equation: the expected cost
    to insure one policy for one year.

    Parameters
    ----------
    freq_model : fitted FrequencyModel
    sev_model  : fitted SeverityModel
    df_full    : full frequency dataset (all policies)
    save_path  : optional path to save plot
    """
    df = df_full.copy()

    # Predicted frequency (claims per policy-year)
    df["pred_freq"] = freq_model.model.predict(
        df, offset=np.log(df["Exposure"])
    ).values / df["Exposure"].values

    # Predicted severity (cost per claim)
    df["pred_sev"] = sev_model.model.predict(df).values

    # Pure premium
    df["pure_premium"] = df["pred_freq"] * df["pred_sev"]

    print("\n" + "=" * 65)
    print("  PURE PREMIUM ANALYSIS")
    print("=" * 65)
    print(f"  Mean predicted frequency : {df['pred_freq'].mean():.4f}")
    print(f"  Mean predicted severity  : ${df['pred_sev'].mean():,.2f}")
    print(f"  Mean pure premium        : ${df['pure_premium'].mean():,.2f}")

    # Pure premium by driver age band
    pp_by_age = df.groupby("DrivAgeBand").agg(
        n_policies=("pure_premium", "count"),
        mean_freq=("pred_freq", "mean"),
        mean_sev=("pred_sev", "mean"),
        mean_pp=("pure_premium", "mean"),
    ).reindex(["18-25", "26-35", "36-50", "51-65", "66+"])
    pp_by_age["relativity"] = pp_by_age["mean_pp"] / df["pure_premium"].mean()

    print("\n--- Pure Premium by Driver Age Band ---")
    print(pp_by_age.round({"mean_freq": 4, "mean_sev": 2,
                            "mean_pp": 2, "relativity": 3}).to_string())

    # Pure premium by vehicle age band
    pp_by_veh = df.groupby("VehAgeBand").agg(
        n_policies=("pure_premium", "count"),
        mean_freq=("pred_freq", "mean"),
        mean_sev=("pred_sev", "mean"),
        mean_pp=("pure_premium", "mean"),
    )
    pp_by_veh["relativity"] = pp_by_veh["mean_pp"] / df["pure_premium"].mean()

    print("\n--- Pure Premium by Vehicle Age Band ---")
    print(pp_by_veh.round({"mean_freq": 4, "mean_sev": 2,
                            "mean_pp": 2, "relativity": 3}).to_string())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Pure Premium Analysis: Frequency × Severity",
                 fontsize=13, fontweight="bold")

    age_order = ["18-25", "26-35", "36-50", "51-65", "66+"]

    # Frequency by age
    ax1 = axes[0]
    ax1.bar(age_order,
            pp_by_age.loc[age_order, "mean_freq"],
            color="steelblue", alpha=0.8)
    ax1.set_title("Predicted Frequency")
    ax1.set_ylabel("Claims per Policy-Year")
    ax1.set_xlabel("Driver Age Band")
    ax1.grid(alpha=0.3, axis="y")

    # Severity by age
    ax2 = axes[1]
    ax2.bar(age_order,
            pp_by_age.loc[age_order, "mean_sev"],
            color="seagreen", alpha=0.8)
    ax2.set_title("Predicted Severity")
    ax2.set_ylabel("Cost per Claim ($)")
    ax2.set_xlabel("Driver Age Band")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax2.grid(alpha=0.3, axis="y")

    # Pure premium by age
    ax3 = axes[2]
    ax3.bar(age_order,
            pp_by_age.loc[age_order, "mean_pp"],
            color="crimson", alpha=0.8)
    ax3.set_title("Pure Premium (Freq × Sev)")
    ax3.set_ylabel("Pure Premium ($/year)")
    ax3.set_xlabel("Driver Age Band")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax3.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved: {save_path}")
    plt.show()

    return df[["IDpol", "pred_freq", "pred_sev", "pure_premium",
               "DrivAgeBand", "VehAgeBand", "BonusMalus"]]


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    freq_path = "data/fremtpl/freMTPL2freq.csv"
    sev_path = "data/fremtpl/freMTPL2sev.csv"

    for p in [freq_path, sev_path]:
        if not Path(p).exists():
            print(f"Data not found: {p}")
            sys.exit(1)

    # Load data
    df_sev = load_severity_data(freq_path, sev_path)

    # Fit severity model
    print("\nFitting Gamma GLM...\n")
    sev_model = SeverityModel()
    sev_model.fit(df_sev)
    sev_model.summary()
    sev_model.evaluate()
    sev_model.plot(save_path="severity_model.png")

    # Pure premium
    freq_df = load_fremtpl(freq_path)
    freq_model = FrequencyModel()
    freq_model.fit(freq_df)
    pure_premium_analysis(
        freq_model, sev_model, freq_df,
        save_path="pure_premium.png"
    )
