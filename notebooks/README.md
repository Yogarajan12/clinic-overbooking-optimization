# Archived notebooks

The four stages with outputs
intact. This is the evidential record: every figure quoted in the top-level
README can be traced to a printed cell here without re-running anything.

| Notebook | Stage |
|---|---|
| `01_preprocess_and_features.ipynb` | cleaning, exploratory analysis, feature engineering |
| `02_train_and_audit_models.ipynb` | models, calibration, SHAP, fairness audit |
| `03_optimise_overbooking.ipynb` | cost model, convexity, constrained optimisation |
| `04_simulate_and_validate.ipynb` | Monte Carlo comparison, sensitivity, stress tests |

Platform metadata (workspace identifiers, per-cell execution IDs, content
hashes) has been stripped. Cell sources and outputs are untouched.

## Two markdown cells are stale

Stage 4 carries two hand-written narrative cells that describe an earlier run
and were never updated. Both now carry an inline editorial note pointing at the
executed output directly above them.

The first claims a simulated no-show rate of 0.260020. The cell above it prints
0.429460 theoretical against 0.429592 simulated. The second quotes a bootstrap
interval of $543.25 to $587.38 around $564.85 for `fixed_10pct`. The cell above
it prints $1,114.05 with an interval of $1,086.93 to $1,141.17.

Anyone reading the `.py` exports rather than these notebooks sees only the
stale prose, because notebook exports keep markdown cells as comments and drop
outputs entirely. That is worth knowing before drawing conclusions from
`pipeline/`.

## The number worth stopping on

Stage 3, cell 6:

```
✓ Loaded model type: stacking
✓ Generated 14,392 predictions
  Mean predicted P(no-show): 0.429
  Actual no-show rate: 0.261
```

The study's calibration argument does not survive that line. See defect 3 in
[`../docs/limitations.md`](../docs/limitations.md).
