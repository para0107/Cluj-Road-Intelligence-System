# RDDS detector — experiment comparison

Metric: **mAP50-95** on the **test** split.

Measured seed-noise floor: **0.0022**. Any difference smaller than this is not a result.

## Leaderboard

| Experiment | Seeds | Mean | Std | Min | Max | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| E0-baseline ⚠dirty | 2 | 0.1971 | 0.0022 | 0.1955 | 0.1986 | [0.1955, 0.1986] |

*⚠dirty = at least one run was produced from an uncommitted working tree and is not reportable.*

## Method

Across-seed spread is reported as the standard deviation plus a 10,000-sample percentile bootstrap CI for the mean (Efron & Tibshirani, 1993). Two experiments are compared by a two-sided paired permutation test over per-class AP@50, generating the null by random sign flips of the paired differences (Ernst, 2004) — no normality assumption, which matters at 10 classes.

Limitation: the stronger test would bootstrap over test images rather than classes. That needs per-image predictions, which are not among the artefacts these runs currently emit.
