# CSI_6_ARI — Predictive Maintenance for SMI Vehicle Fleet (Coursework 2)

Machine learning coursework building a cost-aware predictive maintenance system for
"SMI" (Sensor Maintenance Insights) fleet vehicles. Given 170 anonymised sensor
readings per vehicle, the goal is to predict component failure ahead of time and
drive a lightweight AI agent that turns model predictions into maintenance actions,
under a strongly asymmetric cost structure (a missed failure costs far more than an
unnecessary inspection).

**Student ID:** 4214293

## Repository structure

```
.
├── datasets/
│   ├── SMI_Train_4214293.csv           # 8,000 labelled training records (170 sensors + class)
│   ├── SMI_Test_800.csv                # 800-record held-out common test set (labelled)
│   ├── SMI_Operational_200.csv         # 200-record unlabelled operational feed
│   └── CSI_6_ARI_2526_CW2_AGENT_4214293.csv   # Agent output (generated deliverable)
├── notebook/
│   ├── CSI_6_ARI_2526_CW2_CODE_4214293.ipynb  # Full analysis, modelling and agent notebook
│   ├── fig7_kde_class_overlay.png
│   └── fig8_violin_log_scale.png
└── README.md
```

## Dataset

Each record has an anonymised `class` label (`neg` = normal, `pos` = failure) and
170 `Sensor_XXX` readings. Missing values are encoded as the string `"na"`.

Key characteristics identified during EDA (Section 1 of the notebook):

- **Severe class imbalance** — only 1.66% of training records are failures (~60:1 ratio).
- **~8% missing data** across 169 of 170 sensor features.
- **Informative missingness** — failures show 18.33% missing data vs. 8.13% for
  normal records, i.e. the *absence* of a reading is itself predictive.
- **High dimensionality and multicollinearity** — 171 feature pairs with |ρ| > 0.95
  and 1,855 pairs with |ρ| > 0.80.

## Methodology

The notebook (`notebook/CSI_6_ARI_2526_CW2_CODE_4214293.ipynb`) is organised into
four sections:

### Section 1 — Dataset Description and Experimental Design
Exploratory analysis of dataset characteristics, missingness patterns, feature
correlation structure, and the resulting experimental design decisions.

### Section 2 — Preprocessing and Model Development
A `scikit-learn` `Pipeline` fitted only on training data:
- **Median imputation** with missingness indicators (`SimpleImputer(add_indicator=True)`)
  to preserve the informative-missingness signal.
- **Variance threshold filtering** to drop zero-variance/indicator columns.
- **Log1p transform + standard scaling** (MLP pipeline only — tree models are
  scale-invariant).
- **Class imbalance handling** via `class_weight='balanced'` (Random Forest) or
  inverse-frequency sample weights (Gradient Boosting, MLP).

Three models are tuned with `RandomizedSearchCV` (n_iter=20) under stratified
5-fold cross-validation, optimising F1 on the failure class:

1. Gradient Boosting Classifier
2. Multi-Layer Perceptron Classifier
3. Random Forest Classifier

### Section 3 — Benchmarking and Champion Model Selection
Models are compared on a held-out validation split (20%, stratified) using
precision, recall, F1, ROC-AUC, PR-AUC, and a **cost-aware** evaluation
(FP = £10, FN = £500) at both the default (0.5) and cost-optimal thresholds.

| Model | Val F1 | ROC-AUC | PR-AUC | Cost (θ=0.5) | Optimal θ | Cost (optimal θ) | Recall (optimal θ) |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 0.7347 | 0.9912 | 0.7775 | £4,540 | 0.044 | £1,230 | 0.9259 |
| **MLP (champion)** | 0.3852 | 0.9888 | 0.6228 | £1,320 | 0.419 | **£990** | **1.0000** |
| Random Forest | 0.6301 | 0.9896 | 0.6481 | £2,230 | 0.079 | £1,260 | 1.0000 |

**Champion: MLP**, selected on minimum operational cost at its cost-optimal
threshold (θ = 0.419), followed by recall and PR-AUC — accuracy is deliberately
excluded as a criterion, since a trivial "always normal" classifier would score
98.3% accuracy while catching zero failures.

### Section 4 — AI Maintenance Agent
A decision-support agent applies the champion MLP (fitted pipeline included) to the
unlabelled operational feed (`SMI_Operational_200.csv`) and maps failure
probabilities to a three-tier action space, using thresholds calibrated
**exclusively on validation data**:

| Action | Code | Probability band | Rationale |
|---|---|---|---|
| Continue monitoring | A0 | p < 0.419 | Below cost-optimal threshold |
| Schedule inspection | A1 | 0.419 ≤ p < 0.75 | Potential developing fault |
| Immediate intervention | A2 | p ≥ 0.75 | Expected non-action cost (£375+) far exceeds inspection cost |

The agent exports `datasets/CSI_6_ARI_2526_CW2_AGENT_4214293.csv` with columns
`record_id`, `model_score`, `recommended_action`, `decision_rationale`, and
`expected_cost_gbp` for each of the 200 operational records.

## Final test set results

The champion MLP (θ = 0.419) was evaluated once on the untouched common test set
(`SMI_Test_800.csv`) — never used in training, preprocessing, tuning, or threshold
selection:

| Metric | Value |
|---|---|
| Precision | 0.7876 |
| Recall | 0.9500 |
| F1-score | 0.8612 |
| ROC-AUC | 0.9702 |
| PR-AUC | 0.9084 |
| Total cost (FP×£10 + FN×£500) | £4,410 |

Confusion matrix (n=800): 599 TN, 41 FP, 8 FN, 152 TP.

## Insights

**Missingness is a signal, not just noise.** `Sensor_079` alone is missing in
82.11% of records, and 169 of 170 sensors have some missingness — but it isn't
random. Failure-class records are missing data at more than double the rate of
normal records (18.33% vs. 8.13%), and a handful of sensors show an even sharper
class-conditional gap. This points to sensors going silent, saturating, or being
skipped as a vehicle degrades, so simply dropping or mean-filling missing values
would throw away real predictive signal. Keeping `add_indicator=True` on the
imputer — turning "was this reading missing?" into its own binary feature — is
what lets models exploit that pattern.

**Multicollinearity means the sensor array is redundant by design.** With 171
feature pairs above |ρ| = 0.95 and 1,855 above |ρ| = 0.80 out of only 170
features, most sensors are not independent measurements — they're likely
correlated instruments on the same subsystems (e.g. clustered pressure/temperature
readings). This inflates the *apparent* dimensionality without adding much real
information, which is why variance filtering alone (rather than aggressive feature
selection) was sufficient: tree ensembles are naturally robust to redundant
correlated inputs, and removing them isn't necessary for performance.

**Accuracy is actively misleading on this dataset.** At a 60:1 class ratio, a
model that predicts "normal" for every vehicle scores 98.34% accuracy while
missing 100% of failures — worse than doing nothing, given a £500 cost per missed
failure. This is why F1/recall/PR-AUC on the failure class, not accuracy, drove
every modelling decision, and why accuracy is absent from the champion selection
criteria entirely.

**The best-scoring model on standard metrics was not the cheapest to operate.**
Gradient Boosting had the strongest ROC-AUC (0.9912), PR-AUC (0.7775), and by far
the best F1 (0.7347) at its own optimal threshold — on paper the strongest
classifier. Yet the MLP was crowned champion because, once thresholds are tuned to
the £10/£500 cost asymmetry, it reaches perfect recall (1.0) at a lower total cost
(£990 vs. £1,230), missing 0 of 27 validation failures at the expense of more false
alarms (82 FP vs. 9 FP for GBM). This is the central lesson of the cost-aware
framing: **which model "wins" depends entirely on the cost function**, not on
generic classification metrics — a model that is statistically weaker can be
operationally superior when false negatives are 50x more expensive than false
positives.

**The champion generalised well, and precision improved out-of-sample.**
At the cost-optimal threshold (0.419), validation recall was a perfect 1.0 (all
27 known failures caught, 0 missed) with precision of 0.21; on the untouched test
set, recall held at 0.95 (152/160 failures caught, 8 missed) while precision
nearly quadrupled to 0.79. Precision improving out-of-sample — rather than
collapsing, as would be expected from overfitting — is a strong sign the model
learned a genuine failure signature rather than memorising validation-set noise,
though the small drop in recall (27/27 → 152/160) is worth monitoring as more
operational data accumulates.

**The two-threshold agent design converts a probability into a graded, cost-
justified response instead of a binary alarm.** Rather than a single cut-off, θ_low
(0.419, cost-optimal) separates "do nothing" from "worth a look," while θ_high
(0.75) is derived directly from the cost ratio itself: at p ≥ 0.75, the expected
cost of inaction (0.75 × £500 = £375) is 37.5× the cost of an inspection, so
escalation is unambiguously cheaper than waiting. On the 200-vehicle operational
feed this produced 173 A0 / 10 A1 / 17 A2 — only 13.5% of vehicles were flagged
for any action at all, meaning the agent is conservative by construction: it only
escalates when the maths clearly favours doing so, avoiding alert fatigue while
still catching high-risk cases early.

**Total expected operational cost is dominated by residual failure risk, not
inspection cost.** Of the £15,955 total expected cost across the 200-vehicle feed,
the 17 A2 cases alone account for £8,123 (51%) even though they're only 8.5% of
records — because expected cost scales with failure probability itself (up to
£500 × p), not just the action taken. This highlights that no threshold policy can
drive cost to zero under this cost structure; the agent minimises *expected* cost
given uncertainty, it doesn't eliminate risk.

## Reproducibility

- Random seed fixed at `SEED = 42` for both `numpy` and `random`.
- Library versions used: `scikit-learn 1.8.0`, `numpy 2.4.3`, `pandas 3.0.1`.
- Data leakage is avoided throughout: all preprocessing is fitted on training data
  only, threshold selection uses validation data only, and the test set is touched
  exactly once, at the final evaluation step.

## Running the notebook

```
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook notebook/CSI_6_ARI_2526_CW2_CODE_4214293.ipynb
```

Run all cells top to bottom; the notebook reads CSVs via relative paths
(`../datasets/...`) so it must be executed from within the `notebook/` directory.
