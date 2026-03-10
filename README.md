# pc-actuarial-toolkit

**A full P&C actuarial workflow: GLM pricing and loss reserving on real industry data.**

This toolkit answers the two core questions every P&C actuarial analyst faces:

- **What should we charge?** — GLM frequency/severity pricing on the French Motor MTPL dataset
- **How much should we reserve?** — Chain ladder and Bornhuetter-Ferguson reserving on CAS Schedule P triangle data

Built as a portfolio project targeting P&C actuarial analyst roles.

---

## The Problem

A P&C insurer needs to simultaneously:

1. **Price new business** — estimate the expected cost of insuring each risk, broken into claim frequency (how often) and severity (how much). Mispricing leads to adverse selection or lost business.

2. **Reserve for incurred losses** — claims take years to fully develop. At any point in time, the insurer must estimate how much it will ultimately pay on claims already incurred (IBNR — Incurred But Not Reported). Underreserving leads to insolvency; overreserving distorts profitability.

These two problems are deeply connected: the same underlying loss process drives both the pricing model and the development patterns in the reserve triangles.

---

## Datasets

### French Motor Third-Party Liability (`freMTPL2`)
- ~700,000 policies with vehicle and driver characteristics
- Claim counts (`freMTPL2freq`) and claim amounts (`freMTPL2sev`)
- Standard GLM pricing benchmark in actuarial literature
- Source: R `CASdatasets` package / Kaggle

### CAS Loss Reserve Database (Schedule P)
- Real U.S. P&C insurer loss triangles, accident years 1988–1997
- 10-year development lag, 6 lines of business
- Private passenger auto, workers' comp, medical malpractice, and more
- Source: Casualty Actuarial Society

---

## Project Structure

```
pc-actuarial-toolkit/
├── actuarial/
│   ├── chain_ladder.py          # Development factors, CDF, IBNR
│   ├── bornhuetter_ferguson.py  # A priori loss ratio + credibility blend
│   ├── frequency.py             # Poisson GLM — claims per exposure
│   ├── severity.py              # Gamma GLM — cost per claim
│   └── report.py                # Combined actuarial summary
├── notebooks/
│   ├── 01_reserving.ipynb       # Chain ladder + BF walkthrough
│   ├── 02_pricing.ipynb         # GLM frequency/severity modeling
│   └── 03_combined_analysis.ipynb  # Full actuarial workflow
├── data/
│   ├── cas_triangles/           # CAS Schedule P CSVs
│   └── fremtpl/                 # French auto pricing data
└── requirements.txt
```

---

## Key Concepts

### Loss Reserving

At any valuation date, a loss triangle shows cumulative paid losses by accident year (rows) and development age (columns):

```
Accident Year | 12 months | 24 months | 36 months | ... | 120 months
    1988      |   10,234  |   18,412  |   24,301  | ... |   31,200
    1989      |   11,801  |   20,144  |   26,891  | ... |      ?
    ...       |    ...    |    ...    |    ...    | ... |      ?
    1997      |   14,322  |      ?    |      ?    | ... |      ?
```

The **chain ladder method** fills in the lower-right triangle by computing age-to-age development factors:

```
f_k = Σ C_{i,k+1} / Σ C_{i,k}    (volume-weighted average)
```

**Bornhuetter-Ferguson** improves on chain ladder for immature accident years by blending the development pattern with an *a priori* expected loss ratio — exactly analogous to Bayesian credibility theory.

### GLM Pricing

Claims frequency is modeled as:

```
E[N_i] = exposure_i · exp(β₀ + β₁·age_i + β₂·vehicle_i + ...)
```

using a **Poisson GLM** with log link and exposure offset. Claims severity uses a **Gamma GLM**. Pure premium = frequency × severity.

The log link ensures predicted values are always positive, and the Poisson/Gamma distributions match the actual data-generating process for insurance claims.

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/pc-actuarial-toolkit.git
cd pc-actuarial-toolkit
pip install -r requirements.txt
jupyter notebook notebooks/01_reserving.ipynb
```

### Data Setup

```bash
# CAS triangles — download from CAS website
# Place CSVs in data/cas_triangles/

# French MTPL — download from Kaggle or R CASdatasets
# Place CSVs in data/fremtpl/
```

---

## Notebooks

### `01_reserving.ipynb` — Loss Reserving
- Load and visualize CAS Schedule P triangles
- Compute age-to-age development factors
- Chain ladder IBNR by accident year
- Bornhuetter-Ferguson with a priori loss ratios
- Mack variance — uncertainty bands around reserve estimates
- Compare methods: where do they agree? Where do they diverge?

### `02_pricing.ipynb` — GLM Pricing
- Exploratory analysis of French MTPL dataset
- Feature engineering: driver age bands, vehicle power categories, bonus-malus
- Poisson GLM for claim frequency
- Gamma GLM for claim severity
- Pure premium = frequency × severity by risk segment
- Lift curves and Gini coefficient for model evaluation

### `03_combined_analysis.ipynb` — Full Actuarial Workflow
- Connect pricing and reserving: same loss process, different views
- Indicated rate change from pricing model
- Reserve adequacy: are current reserves consistent with pricing assumptions?
- Combined ratio analysis

---

## Roadmap

- [ ] Mack (1993) variance formula for reserve uncertainty
- [ ] Over-dispersed Poisson GLM for frequency
- [ ] Tweedie compound Poisson-Gamma for pure premium directly
- [ ] Bootstrap reserve distribution (stochastic reserving)
- [ ] Clark LDF method (parametric development curves)

---

## References

- Mack, T. (1993). *Distribution-free calculation of the standard error of chain ladder reserve estimates.* ASTIN Bulletin.
- Bornhuetter, R.L. & Ferguson, R.E. (1972). *The actuary and IBNR.* PCAS.
- Frees, E.W. (2014). *Predictive Modeling Applications in Actuarial Science.* Cambridge.
- CAS Loss Reserve Database: casact.org/research/index.cfm?fa=loss_reserves_data
- freMTPL2 dataset: openml.org / R CASdatasets package

---

## License

MIT
