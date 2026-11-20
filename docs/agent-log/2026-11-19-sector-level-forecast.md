# Shared hierarchical forecasts through sector-cycle

**Date:** 2026-11-19

## Task

Make the selected shared-distribution hierarchy stop at sectors and produce
sector-cycle level totals rather than sector-city-cycle predictions.

## Outcome

The default production profile is region → management → sector: ETS regional
volumes and learned child shares. The existing daily shape consumes each sector
total directly. Optional city profiles require `target_level: city`.

## What changed

Added a sector projection path, output-grain configuration, sector validation
exports and effective manifest keys. Sector composition features remain based
on historical city summaries to preserve the previously evaluated sector model;
no city demand model is trained in the new path. Updated production YAMLs and
the combined forecast guide.

## Notes

Recursive history contains predicted region/management/sector volumes only.
Unknown future city composition carries forward the last closed training state;
no actual future purchases enter model inputs. Validation aggregates actual
city demand to sector-day before scoring. Optimization and SA code are unchanged.

Validation: the 89-test forecasting suite passed, as did 35 focused forecasting
and solver-input tests, Ruff and grade-A complexity checks. The additional
sector-leaf backtest test passed. The projection module has 100% coverage in
the focused run. Nine archived origins reproduced 5,506 known sector forecasts
within 1.82e-12 item; three original missing-prediction points retain their zero
scoring treatment in the equivalence audit.

The production daily replay for 202612 completed under
`experiments/forecast_combined/outputs/2026-11-19_sector_level_validation_v01/`:
MAE 259.99, WMAPE 122.19%, Bias -4.96%, volume coverage 99.32% (13 actual rows
without a forecast). These daily results do not demonstrate good shape accuracy
and are distinct from the archived sector-cycle level metrics.
