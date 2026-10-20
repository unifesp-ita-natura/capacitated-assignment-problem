# Hyndman — Another look at forecast-accuracy metrics for intermittent demand

**Link/DOI:** https://robjhyndman.com/papers/foresight.pdf
**Read by:** Claude (via WebSearch summary, not full text — flag for a human
deep-read before relying on this for a final metric choice)
**Date:** 2026-09-08

## Summary

Argues that MAPE breaks down on data with zero or near-zero actuals (division
by zero, or a single extreme ratio dominating the average) — exactly the
failure mode of intermittent/low-volume demand. Proposes MASE (Mean Absolute
Scaled Error), which scales the absolute error by the in-sample mean absolute
error of a naive seasonal benchmark, as the metric that works across series
of very different scales and that don't break on zeros.

## Relevant to us because

Sector volumes ($L_{i,c}$) span roughly 693-800 sectors of very different
sizes, and small sectors can plausibly have zero-order cycles. This is the
direct justification, in `docs/papers/forecasting-volume-spec.tex`
(Section 3.1), for choosing MAE + MASE as the primary comparison metric
instead of MAPE.

## Open questions / follow-ups

- Confirm on the real demand base how common zero-volume sector-cycles
  actually are — if rare, MAPE's failure mode matters less than assumed here.
- A human should read the full paper (this note is based on a search-engine
  summary) before citing it in the actual Overleaf submission.
