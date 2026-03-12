"""
frequency.py
------------
Poisson GLM for claims frequency modeling.

Models expected claim count as:
    E[N_i] = Exposure_i * exp(X_i @ beta)

Using a log link with exposure offset:
    log(E[N_i] / Exposure_i) = X_i @ beta
    log(mu_i) = log(Exposure_i) + X_i @ beta

The Poisson distribution is appropriate because:
    - Claims are counts (non-negative integers)
    - Mean = Variance assumption is reasonable for frequency
    - Log link ensures positive predictions

Key outputs:
    - Fitted GLM with coefficients and standard errors
    - Relativities by rating factor (exp(beta) = multiplicative effect)
    - Lift curve and Gini coefficient for model evaluation
    - Predicted frequency by risk segment

Data
----
    French Motor Third-Party Liability dataset (freMTPL2freq)
    ~700,000 policies with vehicle and driver characteristics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------

def load_fremtpl(path: str) -> pd.DataFrame:
    """
    Load and prepare the French MTPL frequency dataset.

    Feature engineering:
        - DrivAge bands (young/adult/senior)
        - VehAge bands
        - VehPower bands
        - BonusMalus log-transform (highly skewed)
        - Area and VehGas as categories

    Parameters
    ----------
    path : path to freMTPL2freq.csv

    Returns
    -------
    pd.DataFrame with engineered features
    """
    df = pd.read_csv(path, index_col=0)

    print(f"Loaded freMTPL2freq: {len(df):,} policies")
    print(f"  Total claims    : {df['ClaimNb'].sum():,}")
    print(f"  Total exposure  : {df['Exposure'].sum():,.1f} policy-years")
    print(f"  Overall freq    : {df['ClaimNb'].sum() / df['Exposure'].sum():.4f} claims/year")

    # ── Feature engineering ──────────────────────────────────────────

    # Driver age bands
    df["DrivAgeBand"] = pd.cut(
        df["DrivAge"],
        bins=[17, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "66+"],
        right=True,
    ).astype(str)

    # Vehicle age bands
    df["VehAgeBand"] = pd.cut(
        df["VehAge"],
        bins=[-1, 1, 5, 10, 15, 100],
        labels=["0-1yr", "2-5yr", "6-10yr", "11-15yr", "15+yr"],
        right=True,
    ).astype(str)

    # Vehicle power bands
    df["VehPowerBand"] = pd.cut(
        df["VehPower"],
        bins=[0, 5, 7, 9, 15],
        labels=["low", "medium", "high", "very_high"],
        right=True,
    ).astype(str)

    # BonusMalus log transform (ranges from 50 to 350, highly right-skewed)
    df["LogBonusMalus"] = np.log(df["BonusMalus"])

    # Log density (population density per km²)
    df["LogDensity"] = np.log1p(df["Density"])

    # Categorical types
    for col in ["Area", "VehGas", "VehBrand", "Region"]:
        df[col] = df[col].astype(str)

    # Clip exposure to (0, 1] — some policies have >1 year due to data quirks
    df["Exposure"] = df["Exposure"].clip(0.001, 1.0)

    # Claim frequency (for diagnostics)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    return df


def eda_summary(df: pd.DataFrame):
    """
    Print exploratory data analysis summary.
    Shows claim frequency by key rating factors.
    """
    print("\n=== EDA: Claim Frequency by Rating Factor ===\n")

    factors = {
        "DrivAgeBand": "Driver Age Band",
        "VehAgeBand": "Vehicle Age Band",
        "VehPowerBand": "Vehicle Power Band",
        "VehGas": "Fuel Type",
        "Area": "Area",
    }

    for col, label in factors.items():
        if col not in df.columns:
            continue
        summary = df.groupby(col).agg(
            n_policies=("Exposure", "count"),
            exposure=("Exposure", "sum"),
            claims=("ClaimNb", "sum"),
        )
        summary["frequency"] = summary["claims"] / summary["exposure"]
        summary["relativity"] = summary["frequency"] / (df["ClaimNb"].sum() / df["Exposure"].sum())
        print(f"--- {label} ---")
        print(summary.round({"exposure": 0, "frequency": 4, "relativity": 3}).to_string())
        print()


# ---------------------------------------------------------------------
# Poisson GLM
# ---------------------------------------------------------------------

class FrequencyModel:
    """
    Poisson GLM for claim frequency.

    Formula uses statsmodels formula API for interpretability.
    Exposure is included as an offset (log scale).

    Parameters
    ----------
    formula : statsmodels formula string
              default uses key rating factors
    """

    DEFAULT_FORMULA = (
        "ClaimNb ~ LogBonusMalus + DrivAgeBand + VehAgeBand + "
        "VehPowerBand + VehGas + Area + LogDensity"
    )

    def __init__(self, formula: str = None):
        self.formula = formula or self.DEFAULT_FORMULA
        self._fitted = False

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        """
        Fit Poisson GLM on training split.

        Parameters
        ----------
        df        : prepared DataFrame from load_fremtpl()
        test_size : fraction held out for evaluation
        seed      : random seed
        """
        self.df = df.copy()

        # Train/test split
        self.train, self.test = train_test_split(
            df, test_size=test_size, random_state=seed
        )
        print(f"  Train: {len(self.train):,} policies")
        print(f"  Test : {len(self.test):,} policies")

        # Fit Poisson GLM with log exposure offset
        self.model = smf.glm(
            formula=self.formula,
            data=self.train,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=np.log(self.train["Exposure"]),
        ).fit()

        self._fitted = True
        print(f"\n  Converged: {self.model.converged}")
        print(f"  Deviance : {self.model.deviance:,.1f}")
        print(f"  AIC      : {self.model.aic:,.1f}")

        return self

    def summary(self):
        """Print GLM coefficient table with relativities."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        print("\n" + "=" * 65)
        print("  POISSON GLM — FREQUENCY MODEL")
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
        print("Relativity = exp(coef) — multiplicative effect on frequency")
        print("Relativity > 1: higher frequency than base; < 1: lower")

    def relativities(self) -> pd.DataFrame:
        """
        Extract and format relativities by rating factor.
        Relativity = exp(coefficient) = multiplicative effect on base rate.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        params = self.model.params
        rel = np.exp(params).reset_index()
        rel.columns = ["Variable", "Relativity"]
        rel = rel[rel["Variable"] != "Intercept"].copy()
        rel["Direction"] = rel["Relativity"].apply(
            lambda x: "Higher risk" if x > 1 else "Lower risk"
        )
        return rel.sort_values("Relativity", ascending=False)

    def predict(self, df: pd.DataFrame = None) -> np.ndarray:
        """
        Predict expected claim frequency (claims per policy-year).

        Parameters
        ----------
        df : DataFrame to predict on (default: test set)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        if df is None:
            df = self.test

        # Predict with unit exposure to get per-policy-year frequency
        df_pred = df.copy()
        df_pred["_unit_exposure"] = 1.0

        mu = self.model.predict(
            df_pred,
            offset=np.log(df_pred["Exposure"]),
        )
        return mu.values / df_pred["Exposure"].values

    def evaluate(self) -> dict:
        """
        Evaluate model on test set.
        Returns MAE, Poisson deviance, and Gini coefficient.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        y_true = self.test["ClaimNb"].values
        exposure = self.test["Exposure"].values

        # Predicted counts
        y_pred_counts = self.model.predict(
            self.test,
            offset=np.log(exposure),
        ).values

        # Predicted frequency
        y_pred_freq = y_pred_counts / exposure
        y_true_freq = y_true / exposure

        mae = mean_absolute_error(y_true_freq, y_pred_freq)

        # Poisson deviance
        eps = 1e-10
        deviance = 2 * np.sum(
            y_true * np.log((y_true + eps) / (y_pred_counts + eps))
            - (y_true - y_pred_counts)
        )

        # Gini coefficient (ordered by predicted frequency)
        gini = self._gini(y_true_freq, y_pred_freq, exposure)

        metrics = {
            "mae": mae,
            "poisson_deviance": deviance,
            "gini": gini,
            "n_test": len(self.test),
        }

        print("\n--- Model Evaluation (Test Set) ---")
        print(f"  MAE (frequency)  : {mae:.6f}")
        print(f"  Poisson deviance : {deviance:,.1f}")
        print(f"  Gini coefficient : {gini:.4f}  ({100*gini:.1f}%)")
        print(f"  N test policies  : {metrics['n_test']:,}")

        return metrics

    def _gini(self, y_true, y_pred, exposure) -> float:
        """
        Compute Gini coefficient — standard lift metric in insurance pricing.
        Measures how well the model separates high and low risk.
        Gini = 0: no discrimination. Gini = 1: perfect discrimination.
        """
        order = np.argsort(-y_pred)  # descending: highest risk first
        y_sorted = y_true[order]
        exp_sorted = exposure[order]

        cum_exp = np.cumsum(exp_sorted) / exp_sorted.sum()
        cum_loss = np.cumsum(y_sorted * exp_sorted) / (y_sorted * exp_sorted).sum()

        # Area under Lorenz curve
        auc = np.trapezoid(cum_loss, cum_exp)
        return 2 * auc - 1

    def plot(self, save_path: str = None):
        """
        4-panel diagnostic plot:
            1. Actual vs predicted frequency by decile
            2. Relativities by key rating factor
            3. Lift curve (Lorenz curve)
            4. Residuals
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        test = self.test.copy()
        test["predicted_freq"] = self.model.predict(
            test, offset=np.log(test["Exposure"])
        ).values / test["Exposure"].values
        test["actual_freq"] = test["ClaimNb"] / test["Exposure"]

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("Poisson GLM — Frequency Model Diagnostics",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # --- 1. Actual vs predicted by decile ---
        ax1 = fig.add_subplot(gs[0, 0])
        test["pred_decile"] = pd.qcut(
            test["predicted_freq"], q=10, labels=False, duplicates="drop"
        )
        decile_summary = test.groupby("pred_decile").agg(
            actual=("actual_freq", "mean"),
            predicted=("predicted_freq", "mean"),
        )
        ax1.plot(decile_summary.index + 1, decile_summary["actual"],
                 "o-", color="steelblue", lw=2, label="Actual")
        ax1.plot(decile_summary.index + 1, decile_summary["predicted"],
                 "s--", color="crimson", lw=2, label="Predicted")
        ax1.set_xlabel("Predicted Frequency Decile")
        ax1.set_ylabel("Mean Frequency")
        ax1.set_title("Actual vs Predicted by Decile\n(good model: lines overlap)")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # --- 2. Relativities for BonusMalus and DrivAge ---
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
        ax2.set_ylabel("Relativity")
        ax2.set_title("Frequency Relativity vs BonusMalus\n(claims history score)")
        ax2.grid(alpha=0.3)

        # --- 3. Lift curve ---
        ax3 = fig.add_subplot(gs[1, 0])
        order = np.argsort(-test["predicted_freq"].values)  # descending
        exp_sorted = test["Exposure"].values[order]
        loss_sorted = (test["actual_freq"] * test["Exposure"]).values[order]
        cum_exp = np.cumsum(exp_sorted) / exp_sorted.sum()
        cum_loss = np.cumsum(loss_sorted) / loss_sorted.sum()
        ax3.plot(cum_exp, cum_loss, color="steelblue", lw=2,
                 label=f"Model (Gini={self._gini(test['actual_freq'].values, test['predicted_freq'].values, test['Exposure'].values):.3f})")
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax3.set_xlabel("Cumulative Exposure (sorted by predicted risk)")
        ax3.set_ylabel("Cumulative Claims")
        ax3.set_title("Lift Curve (Lorenz Curve)\n(higher bow = better discrimination)")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        # --- 4. Frequency by driver age band ---
        ax4 = fig.add_subplot(gs[1, 1])
        age_summary = test.groupby("DrivAgeBand").agg(
            actual=("actual_freq", "mean"),
            predicted=("predicted_freq", "mean"),
            exposure=("Exposure", "sum"),
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
        ax4.set_ylabel("Mean Claim Frequency")
        ax4.set_title("Actual vs Predicted by Driver Age Band")
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3, axis="y")

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
    data_path = "data/fremtpl/freMTPL2freq.csv"

    if not Path(data_path).exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    # Load and prepare data
    df = load_fremtpl(data_path)

    # EDA
    eda_summary(df)

    # Fit model
    print("\nFitting Poisson GLM...\n")
    model = FrequencyModel()
    model.fit(df)

    # Results
    model.summary()
    model.evaluate()
    model.plot(save_path="frequency_model.png")
