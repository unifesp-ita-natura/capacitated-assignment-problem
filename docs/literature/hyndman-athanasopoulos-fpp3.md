# Hyndman & Athanasopoulos — Forecasting: Principles and Practice (3rd ed.)

**Link/DOI:** https://otexts.com/fpp3/ (Chapter 11 — hierarchical/grouped
time series: https://otexts.com/fpp3/hierarchical.html; Section 5.10 — time
series cross-validation: https://otexts.com/fpp3/tscv.html)
**Read by:** Claude (via WebSearch summary, not full text — flag for a human
deep-read before relying on this for a final method choice)
**Date:** 2026-09-08

## Summary

Free online textbook, two chapters used here:

- **Ch. 11 (hierarchical/grouped forecasting):** the most disaggregated
  level of a hierarchy is usually the noisiest/hardest to forecast directly;
  aggregated series are smoother. Top-down (forecast aggregate, disaggregate
  by proportion) vs. bottom-up (forecast each bottom series, sum up) are the
  two classic strategies, with bottom-up ignoring cross-series relations and
  performing worse at the aggregate level.
- **§5.10 (time series cross-validation):** validation for time series must
  respect temporal order — rolling-origin (walk-forward) evaluation, either
  expanding-window or fixed rolling-window, simulates real forecasting by
  training only on the past relative to each test point.

## Relevant to us because

- Ch. 11 backs the recommendation in
  `docs/papers/forecasting-volume-spec.tex` (Section 2.3) to pool
  ~693-800 noisy per-sector series (by region, or via a single global
  gradient-boosting model) rather than fitting 800 fully independent local
  statistical models.
- §5.10 is the direct source for the rolling-origin/walk-forward validation
  scheme specified in Section 3.2 of the same document, as opposed to a
  single random or fixed train/test split.

## Open questions / follow-ups

- A human should read the actual chapters (this note is based on a
  search-engine summary) before citing them in the Overleaf submission.
- Decide, once real cycle history length is known, whether the walk-forward
  scheme should use an expanding or a fixed-size training window (see open
  item in `docs/papers/forecasting-volume-spec.tex`, Section 4).
