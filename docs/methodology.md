# Methodology

## Problem

A clinic session holds twenty appointments. If patients fail to attend, the
clinician sits idle and the capacity is gone; it cannot be banked. If the
clinic books extra patients to compensate and too many arrive, someone waits
or is turned away. The decision variable is k, the number of patients booked
beyond capacity, and the question is how to choose it from what the clinic
knows about the people in the session.

Two framings are possible. One treats this as a prediction problem: forecast
who will attend. The other treats it as a decision problem: choose k to
minimise expected cost. The second subsumes the first, and it changes what the
model has to be good at. A decision layer that integrates over attendance
needs probabilities that are right on average at every risk level. It does not
particularly need the sharp ranking that AUC rewards. That distinction runs
through the whole study.

## Stage 1: cohort and features

The Kaggle cohort holds 110,527 appointments from Vitoria, Brazil, recorded
between April and June 2016. Cleaning removes 38,568 rows whose scheduled date
falls after the appointment date, which is a recording error rather than a
same-day booking, leaving 71,959 appointments at a 28.5% no-show rate.

The split is chronological rather than random: 43,175 train, 14,392
validation, 14,392 test. Random splitting would let a patient's later
behaviour inform the prediction of their earlier appointment, and it would
flatter the model by hiding drift. The test period's own no-show rate, 26.1%,
sits close to the pooled 28.5%, so the split is not itself distorting. What
does distort the downstream results is that the model predicts a mean risk of
42.9% on that same period; see defect 3 in `docs/limitations.md`.

Sixteen features survive. Lead time is the strongest single signal, rising
from 21.4% no-show on same-day appointments to 34.1% beyond a month.
`Patient_Prior_NoShow_Rate` is computed with a leakage guard so that a
patient's own outcome never enters their own feature. `Neighborhood_Risk` is a
target-encoded neighbourhood effect. Two interaction terms, age by lead time
and SMS by lead time, are built because the exploratory analysis showed the
lead-time gradient differs sharply by age. Gender, hypertension, diabetes,
alcoholism and handicap are dropped for carrying no signal.

## Stage 2: risk, calibration, explanation, audit

Four base learners are trained: random forest, XGBoost, logistic regression and
a small multilayer perceptron. A logistic meta-model stacks them, weighting
XGBoost heavily and logistic regression almost not at all.

Every model is then calibrated with both Platt scaling and isotonic
regression, and isotonic wins on Brier score for all four. This is the step
that makes the optimisation defensible. The decision layer integrates the
predicted probabilities directly, so a model that says 30% has to be wrong 30%
of the time; a model that ranks well but reports inflated probabilities would
produce systematically wrong values of k while looking fine on AUC.

Interpretation is done with SHAP on the XGBoost component, with logistic
coefficients kept as a linear cross-check. The two agree on direction for the
features they share, which is a cheap but real guard against reading structure
into an attribution method that is partly reporting its own artefacts.

The fairness audit measures true-positive-rate parity across age bands and
scholarship status, and finds large gaps. They are recorded rather than
patched, because the gaps come from genuine base-rate differences and the
right place to address them is the policy layer, where the burden is actually
allocated.

## Stage 3: the cost model and the optimiser

Attendance in a session is a sum of independent Bernoulli variables with
different success probabilities, so the number of arrivals follows a
Poisson-Binomial distribution, not a Binomial one. The exact PMF is built by a
dynamic program in O(n²), which is trivial at session scale, and expected cost
is the PMF-weighted average of the realised cost:

```
idle     = max(0, capacity - shows) * service_time * c_idle
overflow = max(0, shows - capacity) * c_overflow
waiting  = overflow * (overflow * service_time / 2) * c_wait
```

The waiting term is quadratic on purpose. The n-th overflow patient waits
behind the n-1 in front of them, so congestion hurts faster than headcount
suggests.

Expected cost is shown to be convex in k by checking that second differences
are non-negative, which means the enumerated minimum is global and a
stationary point of the continuous relaxation is the optimum rather than a
saddle. A normal approximation gives a continuous version, solved with
gradient descent and Newton's method as a cross-check, and then re-solved under
an overflow-probability constraint P(overflow) ≤ α using SLSQP. The Lagrange
multiplier on that constraint is a shadow price: it says, in dollars, what one
percentage point of service-level guarantee costs. At α = 0.2 the constrained
optimum moves from k* = 9.24 to 5.30 at a price of roughly $760.

That shadow price is the most useful object in the stage, because it converts
an ethical preference into a number a clinic manager can weigh.

## Stage 4: simulation and validation

Twelve policies are compared on 719 held-out sessions, 1,000 Monte Carlo
iterations each, with attendance sampled from the calibrated probabilities.
The policies span blind rules, risk-aware rules and equity-aware rules, so the
comparison isolates what the model contributes over a fixed percentage.

Validation runs at several levels. Deterministic tests pin the policy and cost
arithmetic to hand-computed values. A law-of-large-numbers check confirms the
sampler reproduces the probabilities it is given, to within 0.001. Confidence
intervals come from a percentile bootstrap; policy comparisons come from a
paired bootstrap that resamples session indices so shared session composition
cancels out. Sensitivity re-runs the whole comparison across five cost
structures. Stress tests degrade the model deliberately, adding noise and
shifting the no-show rate, to see which policies fail gracefully. The fairness
analysis measures overflow rate by age band and scholarship status under each
policy, asking not whether the model is fair but who ends up waiting.
