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

### 3. The test period is not representative

The chronological split puts the last 14,392 appointments in the test set, and
that period has a 43% no-show rate against 28.5% for the pooled cohort. The
simulation's own law-of-large-numbers check confirms it: theoretical rate
0.4295, simulated 0.4296.

A baseline that never overbooks therefore wastes about 8.6 of 20 slots per
session, which is what produces the $1,289 baseline cost and the 57%
utilisation figure. On a 28.5% cohort the baseline is much less wasteful and
the savings from overbooking shrink accordingly. The 64.5% figure is a
property of this test period, not a general estimate of what overbooking buys.

### 4. Stage 3 has a silent fallback

If the serialised model fails to load or predict, stage 3 catches the
exception and generates probabilities from a hand-tuned heuristic built out of
base rate, age, lead time, prior no-show rate and scholarship status. It then
proceeds as though nothing happened. The heuristic produces a mean near the
observed 0.43, so the printed calibration check passes either way, and the
archived notebook output does not record which path ran.

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
