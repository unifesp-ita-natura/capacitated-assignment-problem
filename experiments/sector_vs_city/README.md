# sector_vs_city

Does forecasting each (sector, city) series and summing the cities beat
forecasting the sector directly?

The same candidates and the same rolling-origin folds run twice:

- **`[setor]`**: one series per sector (`build_item_panel(raw, by_city=False)`).
- **`[cidades→setor]`**: one series per (sector, city) (`build_item_panel(raw)`),
  each fold's predictions summed per (sector, cycle).

Both are scored against the sector's real total, so a city a candidate could
not predict (no history yet) counts as error instead of disappearing. Both use
the sector run's naive error as the MASE scale. The ranking runs on the
(sector, cycle, fold) points every version predicted (`comparison.compare`).

```bash
uv run python -m experiments.sector_vs_city.run configs/experiments/sector_vs_city/all.yaml
```

Outputs, in a fresh `outputs/run_<id>/` folder:

| File | Content |
|---|---|
| `comparison.csv` | one row per candidate × version, sector-level metrics |
| `predictions.csv` | sector-level actual vs predicted for both versions; opens in `dashboard/` |
| `city_predictions.csv` | the raw per-(sector, city) rows behind `[cidades→setor]` |
| `config.yaml`, `manifest.json` | the run record, as in `compare_forecasters` |

`n_missing` on a `[cidades→setor]` row counts city series without a
prediction, not sectors.

## The city panel has zero rows

The base only records cities that bought. Without filling the gaps, the
backtest would only ask about cities that are known to buy in the target
cycle, which a real forecast can't know. `build_item_panel` adds a zero row
for each cycle a city skipped after its first purchase, while its sector was
active (`dataset._fill_quiet_cycles`).
