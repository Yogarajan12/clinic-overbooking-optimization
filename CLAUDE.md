# Working notes for Claude Code

Context for agentic sessions in this repository. Read before making changes.

## What this repository is

A four-stage study of clinic overbooking: calibrated no-show risk feeding a
cost-aware optimiser, validated by Monte Carlo simulation and audited for
fairness. It is a research artefact tied to published numbers, not an
application under active development.

## The rule that matters most

**Refactoring must not change numerical output.** The README, the report and
the figures all cite specific values. Before changing anything in `pipeline/`,
capture the current outputs under `results/` as a reference, make the change,
re-run, and diff. If a number moves, either the change is wrong or the change
is a finding that has to be documented, not absorbed silently.

## Layout

```
notebooks/    the archived run with outputs. The evidential record; do not
              re-execute or clear outputs.
pipeline/     the same stages as scripts, driven by the Makefile.
src/          library extracted from the pipeline: cost model, policies,
              simulator, metrics. Unit-tested. No I/O.
tests/        pytest suite covering src/ only.
config/       cost scenarios. Every headline number depends on these.
docs/         methodology, limitations, future work, README figures.
data/         gitignored. Stage 1 populates it.
results/      gitignored. Stages 1 to 4 populate it.
```

`pipeline/` and `src/` currently duplicate logic. That is deliberate and
temporary: the pipeline is preserved as-run for provenance, and the library is
the tested version. Migrating the pipeline onto the library is a planned task
and must be done stage by stage with output diffs at each step.

## Conventions

- British spelling in prose. Code identifiers follow the existing American
  spellings where they already exist (`utilization`, `optimal_k`); do not
  rename them, since serialised outputs use those keys.
- Line length 100. `ruff check src tests` must pass.
- No personal comments addressed to the author anywhere in the files.
- Type hints and docstrings on everything public in `src/`. Docstrings explain
  why a choice was made, not what the line does.
- Paths are never hardcoded relative to the caller's working directory. Stage
  scripts anchor to `REPO_ROOT` and chdir; the library takes no paths at all.
- Seeds are explicit. `np.random.default_rng(seed)` rather than the global
  legacy state.

## Known issues, already documented

`docs/limitations.md` records four defects: the two stages price cost
differently, the winning policy is a heuristic and not the optimiser it is
named after, the probabilities reaching the optimiser are uncalibrated (mean
0.429 against a true 0.261), and three separate code paths can produce the
prediction column without announcing which ran. Do not "fix" these quietly in
a refactor. They are findings with a documented status, and changing them
changes the reported results.

Two markdown cells in `notebooks/04_simulate_and_validate.ipynb` are stale
prose from an earlier run, annotated inline. The `.py` exports in `pipeline/`
preserve that stale prose and drop the outputs that contradict it, so verify
any claim against the notebooks rather than the scripts.

## What not to do

- Do not commit data, models or generated figures. `.gitignore` covers them;
  `docs/figures/` is the only committed image directory.
- Do not add a dependency without pinning it in `requirements.txt`.
- Do not soften the limitations section. It is the most valuable part of the
  repository.
