# Limitations and known inconsistencies

This page exists because the headline number of this study, a 64.5% cost
reduction, is more fragile than the number alone suggests. Everything here was
found by reading the pipeline against its own outputs. None of it invalidates
the work; all of it changes how the result should be read.

## Known inconsistencies between stages

### 1. Stages 3 and 4 price the same outcome differently

Stage 3 prices an under-filled session as

```
idle = max(0, capacity - shows) * service_time * c_idle
```

and adds a waiting term that grows quadratically in overflow. Stage 4 prices
the same session as

```
idle = max(0, capacity - shows) * c_idle
```

with no service-time scaling and no waiting term at all. Since
`service_time = 0.333`, idle capacity is roughly three times more expensive in
the simulation than in the optimisation the simulation is meant to validate,
and overflow carries no congestion penalty. Both directions push the simulated
optimum towards more aggressive overbooking than stage 3's theory supports.

`src/noshow_overbooking/cost.py` implements the stage 3 model.
`src/noshow_overbooking/simulate.py` implements the stage 4 model. They are
kept separate and both are documented rather than quietly reconciled, because
the reported results come from the second one.

### 2. The winning policy is not the optimiser

The policy that carries the headline is called `cost_optimal`, which invites
the reading that it is the Poisson-Binomial argmin from stage 3. It is not. In
stage 4 it is a cost-ratio heuristic:

```python
k = round(expected_no_shows * 1.2) if c_idle >= c_overflow else round(expected_no_shows * 0.8)
```

The convexity proof, the KKT conditions and the shadow-price analysis all sit
in stage 3 and never touch the simulated result. The exact optimiser is
available as `policies.exact_cost_optimal_k` and, on the default cost
structure, lands near the same k for a typical session. That is a coincidence
of parameters, not a validation. Re-running the simulation with the exact
optimiser substituted for the heuristic is the first item in
`docs/future-work.md`.

### 3. The probabilities driving the simulation are not calibrated

This is the most serious defect in the study, and the pipeline printed it at
the time. Stage 3, when it generates predictions for the test split, reports:

```
✓ Loaded model type: stacking
✓ Generated 14,392 predictions
  Mean predicted P(no-show): 0.429
  Actual no-show rate: 0.261
```

The model claims an average no-show risk of 0.429 on a population whose true
rate is 0.261. That is a 17 percentage point over-prediction, about 65% in
relative terms, and it is the single quantity the entire decision layer
integrates over.

The likely mechanism is visible in stage 2. Every base learner is trained with
`class_weight='balanced'` or `scale_pos_weight=2.38`, which shifts the
effective prior from the observed base rate towards 0.5. Mapping a true rate of
0.261 through a weight ratio of 2.38 gives roughly 0.46, which brackets the
0.429 observed. Isotonic calibration was fitted and saved for each base model,
but the artefact stage 3 loads is the stacking ensemble, so the calibration
step the study relies on is bypassed at exactly the point where it matters.

The consequence runs straight into the headline. Under a never-overbook
baseline, idle slots scale directly with the no-show rate. At 0.429 the
baseline wastes about 8.6 of 20 slots, which produces the $1,289 baseline cost
and the 57% utilisation figure. At the true 0.261 it wastes about 5.2, and both
the baseline cost and the savings measured against it shrink substantially.
The optimal overbooking level moves with it, since k tracks expected no-shows.

So the reported 64.5% is inflated in the direction that flatters the
conclusion, by an amount that cannot be determined without re-running stages 3
and 4 on properly calibrated probabilities. The qualitative finding that
risk-aware policies beat flat-rate policies is not threatened by this, because
every policy is evaluated on the same inflated inputs. The magnitude is.

It also undercuts the study's own defence of a modest AUC. The argument was
that the decision layer needs calibration rather than ranking. That argument
holds, but the probabilities that reached the decision layer were not the
calibrated ones.

### 4. Three paths produce `pred_noshow_prob`, and none of them announce it

Stage 3 resolves the prediction column in three ways, in order of precedence:

1. a pre-existing `pred_noshow_prob` column in `data/processed/test_full.csv`,
   which short-circuits before inference and is silent;
2. inference from the loaded model, which is the path the archived run took;
3. a hand-tuned fallback heuristic, reached when the model fails to load or
   predict, built from base rate, age, lead time, prior no-show rate and
   scholarship status, wrapped in a bare `except` that lets the run continue.

The archived notebook records which path ran only because stage 3 happens to
print the model type. Path 1 leaves no trace at all, and path 3 would produce
a plausible-looking mean without any indication that no model was involved.

A re-run should delete the fallback and let a load failure raise, assert that
mean predicted risk tracks the observed rate within a stated tolerance, and
fail loudly when it does not. The check that would have caught defect 3 above
already exists as a print statement; it simply was not an assertion.

Any re-run should remove the fallback and let the failure surface. Silent
degradation to a heuristic is exactly the failure mode that makes deployed
clinical models untrustworthy.

## Methodological limitations

**Discrimination is weak.** Test AUC of 0.620 means the model separates
no-shows from attendees only modestly. It sits inside the 0.60 to 0.75 band
reported in the no-show literature, and a large share of attendance behaviour
is genuinely unpredictable, but it should not be described as accurate
prediction. The optimisation leans on calibration rather than ranking, which
is the weaker requirement and the one the model actually meets.

**Overbooked patients are modelled as clones.** Both the optimiser and the
simulator draw extra bookings from the session's own risk distribution. In a
real clinic those slots are filled from a wait list whose risk profile differs,
often systematically.

**Sessions are contiguous blocks.** Appointments are partitioned into groups of
twenty in chronological order. Real sessions are defined by clinician, room and
specialty, and their risk composition is far less homogeneous.

**Bootstrap p-values have a floor.** The reported p < 0.0001 means no bootstrap
replicate out of 10,000 crossed zero. It is a tail proportion, not an analytic
test statistic, and its resolution is bounded by the replicate count.

**Cost parameters are assumptions, not measurements.** $150 per idle clinician
hour, $75 per overflow patient and $30 per patient waiting hour come from the
literature and from judgement, not from a clinic's books. The sensitivity
analysis exists precisely because the ranking of policies inverts when these
change.

**Fairness scope is narrow.** The audit covers age band and scholarship status
only, because those are the attributes the dataset carries. Race, disability,
distance from the clinic and language are all absent and all plausibly relevant.

**The equal-opportunity gaps are large.** True-positive-rate disparity across
age groups is 0.60, and 0.21 across scholarship status. The model is accurate
about a real behavioural difference, which is exactly what makes the disparity
hard to dismiss: acting on an accurate prediction still concentrates the
burden of overflow on the patients least able to absorb it.

**Simulation is not deployment.** No policy in this study has been run in a
clinic. Everything reported is a claim about a model of a clinic.

## Not a medical device

Research code for a methodological study. Not validated, not approved, and not
suitable for clinical use.
