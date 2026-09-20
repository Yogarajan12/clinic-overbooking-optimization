# Future work

Ordered by how much they would change the conclusions.

**Reconcile the two cost models.** Stage 3 and stage 4 price the same session
differently (`docs/limitations.md`). Running the simulation against the stage 3
cost function, waiting term included, is the single change most likely to move
the headline number.

**Substitute the real optimiser for the heuristic.** Replace the cost-ratio
rule labelled `cost_optimal` with `policies.exact_cost_optimal_k` and re-run
the comparison. If the Poisson-Binomial optimum does not beat a one-line
heuristic, that is worth knowing and worth saying.

**Remove the silent fallback in stage 3.** Let a model-loading failure raise.
Add an assertion that mean predicted risk tracks the observed rate within a
stated tolerance, and fail the run when it does not.

**Re-weight or re-sample the test period.** The test window sits at a 43%
no-show rate. Reporting savings under a reweighted 28.5% cohort alongside the
raw figure would separate what overbooking buys from what an unusual period
buys.

**Move fairness into the objective.** At present fairness is a post-hoc policy
choice. The constrained solver already prices an overflow-probability cap, so
a bounded-disparity constraint solved alongside it is a modest extension with
a directly interpretable shadow price: the dollar cost of equalising overflow
across groups.

**Model the wait list properly.** Extra bookings currently inherit the
session's own risk profile. Drawing them from a separate wait-list distribution
would test how sensitive the savings are to that assumption.

**Strengthen the calibration reporting.** Add temperature scaling as a third
method, report expected calibration error with bootstrap intervals rather than
point estimates, and check calibration within demographic strata rather than
only in aggregate. A model calibrated on average can be badly miscalibrated for
a subgroup, which matters when the decision layer integrates those
probabilities.

**Probe what the model has learned, not just which features it used.** SHAP
answers attribution. Probing the learned representation for whether
socioeconomic proxies are encoded even after the explicit feature is removed
would answer a harder question about what the model is actually keying on.

**Shadow-mode pilot.** Wrap the calibrated model and the optimiser behind a
small service, log recommended k against actual attendance without acting on
it, and compare predicted with realised cost over a quarter before anyone's
appointment is affected.
