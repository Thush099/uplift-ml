# Uplift ML

Causal uplift modeling toolkit implementing **S-Learner**, **T-Learner**, and **X-Learner** meta-learners with **Qini coefficient**, **AUUC**, **uplift-by-decile**, and **PEHE** evaluation.

## Causal Meta-Learner Architecture

```text
                       +---------------------------+
                       |  Data Loader & Preproc    |
                       | (preserves T=0 & T=1)     |
                       +-------------+-------------+
                                     |
          +--------------------------+--------------------------+
          |                          |                          |
          v                          v                          v
  +---------------+          +---------------+          +--------------------+
  |   S-Learner   |          |   T-Learner   |          |     X-Learner      |
  |  mu(X, T)     |          | mu_0(X) T=0   |          | Stage 1: mu_0,mu_1 |
  |               |          | mu_1(X) T=1   |          | Stage 2: D_1, D_0  |
  | tau =         |          | tau =         |          |   -> tau_1, tau_0  |
  | mu(X,1)-mu(X,0)|         | mu_1 - mu_0   |          | Stage 3: g(X)*tau_0|
  +-------+-------+          +-------+-------+          |   + (1-g(X))*tau_1 |
          |                          |                  +---------+----------+
          +--------------------------+----------------------------+
                                     |
                                     v
                       +---------------------------+
                       |     UpliftEvaluator       |
                       | ATE, Deciles, Qini, AUUC, |
                       | and PEHE (RMSE vs CATE)   |
                       +---------------------------+
```

### Implemented Meta-Learners

- **`SLearner`**: Trains a single outcome estimator $\hat{\mu}(X, T)$ with binary treatment indicator $T$ appended as a feature and estimates $\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$.
- **`TLearner`**: Fits separate control ($\hat{\mu}_0$ on $T=0$) and treated ($\hat{\mu}_1$ on $T=1$) response estimators and estimates $\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$.
- **`XLearner`**: Three-stage estimator (Kunzel et al., 2019):
  1. Fit $\hat{\mu}_0(x)$ on control and $\hat{\mu}_1(x)$ on treated.
  2. Impute counterfactual treatment effects $D_1 = Y_1 - \hat{\mu}_0(X_1)$ and $D_0 = \hat{\mu}_1(X_0) - Y_0$, and fit CATE models $\hat{\tau}_1(x)$ and $\hat{\tau}_0(x)$.
  3. Blend predictions via propensity $\hat{g}(x) = P(T=1 \mid X)$ estimated with standardized `LogisticRegression` clipped to $[0.01, 0.99]$:
     $$\hat{\tau}(x) = \hat{g}(x)\,\hat{\tau}_0(x) + (1 - \hat{g}(x))\,\hat{\tau}_1(x)$$

### Evaluation Metrics (`UpliftEvaluator`)

- **Predicted & Observed ATE**: Mean predicted CATE vs difference-in-means $\bar{Y}_{T=1} - \bar{Y}_{T=0}$.
- **Uplift-by-Decile**: Ranked decile breakdown from highest predicted uplift (Decile 1) to lowest (Decile 10).
- **Qini Coefficient**: Area above random targeting divided by optimal-minus-random Qini area.
- **AUUC**: Area Under the Uplift Curve (normalized per sample, with gain over random baseline).
- **PEHE**: Precision in Estimation of Heterogeneous Effect ($\sqrt{\text{mean}((\hat{\tau} - \tau)^2)}$) when ground-truth CATE is available.

## Installation

```bash
pip install -r requirements.txt
```

## CLI Usage

Run the out-of-the-box synthetic heterogeneous CATE benchmark (comparing `S-learner`, `T-learner`, and `X-learner` across `persuadables`, `sure_things`, `sleeping_dogs`, and `lost_causes` subgroups):

```bash
python main.py
```

Run on a custom CSV dataset:

```bash
python main.py \
  --data-path path/to/experiment.csv \
  --model-type X-learner \
  --outcome-var outcome \
  --treatment-var treatment
```

Use `--model-type all` to benchmark all three meta-learners on your CSV.

### Benchmark Output (`python main.py`)

```text
=== Causal Uplift Meta-Learner Benchmark (Synthetic Heterogeneous CATE) ===
Model          Pred ATE   True ATE    Obs ATE     PEHE     Qini     AUUC   Top Decile   Bot Decile
------------------------------------------------------------------------------------------------
S-learner       +0.1084    +0.0978    +0.1005   0.0666   0.2714   0.1343      +0.5415      -0.5413
T-learner       +0.1260    +0.0978    +0.1005   0.1280   0.2663   0.1343      +0.6670      -0.3950
X-learner       +0.1231    +0.0978    +0.1005   0.0986   0.2629   0.1329      +0.4550      -0.3519

Synthetic Subgroup Ground Truth:
  - persuadables   n= 861 | true_cate=+0.5540
  - lost_causes    n= 889 | true_cate=+0.0200
  - sure_things    n= 639 | true_cate=+0.0200
  - sleeping_dogs  n= 611 | true_cate=-0.3400

Uplift-by-Decile (X-learner, test n=900):
Decile      N  Treated  Control  Pred Uplift   Obs Uplift
----------------------------------------------------------
     1     90       40       50      +0.5980      +0.4550
     2     90       44       46      +0.5164      +0.5958
     3     90       48       42      +0.4062      +0.4137
     4     90       48       42      +0.1583      +0.1399
     5     90       42       48      +0.0880      -0.0030
     6     90       44       46      +0.0472      +0.0158
     7     90       48       42      +0.0029      +0.0595
     8     90       47       43      -0.0645      +0.0752
     9     90       49       41      -0.2059      -0.3569
    10     90       41       49      -0.3159      -0.3519
```

## Running Tests

```bash
pytest -v
```
