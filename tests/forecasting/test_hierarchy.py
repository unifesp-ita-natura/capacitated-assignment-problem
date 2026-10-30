"""Verify closed-cycle features, cold cities, coherent forecasts and identical hierarchy scoring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from experiments.hierarchical_level.run import load_scenarios, run
from src.forecasting.dataset import build_item_panel, cycle_calendar
from src.forecasting.hierarchy import forecast_hierarchy
from src.forecasting.hierarchy_config import HierarchySettings
from src.forecasting.hierarchy_data import aggregate_nodes
from src.forecasting.hierarchy_evaluation import evaluate_hierarchy, origin_plan, summarize
from src.forecasting.hierarchy_features import BLOCKS, build_features, feature_columns
from src.forecasting.hierarchy_models import normalize_scores
from src.forecasting.hierarchy_reconciliation import assert_coherent, distribute, reconcile


@pytest.fixture
def raw():
    rows = []
    for cycle, opening in enumerate(pd.date_range("2026-01-01", periods=9, freq="7D"), 1):
        for sector, region, management in [
            ("01", "r1", "g1"),
            ("02", "r1", "g2"),
            ("03", "r2", "g3"),
        ]:
            for state, multiplier in [("MG", 10), ("GO", 3)]:
                if _quiet(state, cycle):
                    continue
                rows.extend(_rows(cycle, opening, sector, region, management, state, multiplier))
    rows.extend(_rows(9, pd.Timestamp("2026-02-26"), "01", "r1", "g1", "SP", 1))
    return pd.DataFrame(rows)


def _quiet(state, cycle):
    return state == "GO" and cycle % 2 == 0


def _rows(cycle, opening, sector, region, management, state, multiplier):
    return [
        dict(
            data_pedido=opening + pd.Timedelta(days=day),
            cd_setor=sector,
            CD_RE=region,
            CD_GV=management,
            estado=state,
            cidade="HOMONIMA",
            CICLOS=f"2026{cycle:02}",
            nm_ciclo="test",
            aa_ciclo=2026,
            dia_ciclo=day,
            total_itens_mascarado=multiplier * (cycle + 1),
            total_pedidos_mascarado=1,
            total_volumes_mascarado=1,
            **{"Dt Abertura": opening, "Dt Fechamento": opening + pd.Timedelta(days=2)},
        )
        for day in [0, 2]
    ]


@pytest.fixture
def settings():
    return HierarchySettings(
        comparison="historical",
        history_cycles=3,
        baseline_windows=[2],
        max_origins=1,
        learning={"n_estimators": 5, "min_child_samples": 1, "num_leaves": 3},
    )


def node_panel(raw):
    cal = cycle_calendar(raw)
    panel = build_item_panel(raw, cal).merge(cal[["CICLOS", "closing_date"]], on="CICLOS")
    return aggregate_nodes(panel, list(BLOCK_LEVELS))


BLOCK_LEVELS = ["region", "management", "sector", "city"]


def test_node_totals_and_homonyms(raw):
    nodes = node_panel(raw)
    totals = nodes.groupby(["level", "CICLOS"])["items"].sum().unstack("level")
    np.testing.assert_allclose(totals.to_numpy(), totals[["city"]].to_numpy().repeat(4, axis=1))
    cities = nodes[nodes.level.eq("city")]
    assert cities.node.nunique() == 7
    assert not nodes.duplicated(["node", "CICLOS"]).any()


def test_feature_targets_do_not_enter_their_own_inputs(raw):
    nodes = node_panel(raw)
    before = build_features(nodes)
    changed = nodes.copy()
    changed.loc[changed.CICLOS.eq("202609"), "items"] *= 1000
    after = build_features(changed)
    cols = feature_columns(list(BLOCKS))
    pd.testing.assert_frame_equal(
        before.loc[before.CICLOS.eq("202609"), cols], after.loc[after.CICLOS.eq("202609"), cols]
    )


def test_features_skip_unclosed_previous_cycles(raw):
    nodes = node_panel(raw)
    dates = nodes["opening_date"] + pd.Timedelta(days=9)
    nodes["closing_date"] = dates
    featured = build_features(nodes)
    root = featured[featured.level.eq("region") & featured.node.str.contains("r1")]
    target = root[root.CICLOS.eq("202604")].iloc[0]
    expected = root[root.CICLOS.eq("202602")].iloc[0]["items"]
    assert target.lag_1 == expected
    assert target.n_history == 2


@pytest.mark.parametrize("window", [1, 2, 3])
def test_city_recent_shares_use_closed_ratio_of_sums(raw, window):
    nodes = node_panel(raw)
    nodes["closing_date"] = nodes["opening_date"] + pd.Timedelta(days=9)
    featured = build_features(nodes)
    city = featured[featured.level.eq("city")].iloc[0].node
    target = featured[featured.node.eq(city) & featured.CICLOS.eq("202606")].iloc[0]
    closed = featured[featured.node.eq(city) & featured.closing_date.lt(target.opening_date)].tail(
        window
    )
    expected = closed["items"].sum() / closed["parent_actual"].sum()
    assert target[f"share_{window}"] == pytest.approx(expected)
    assert featured.loc[featured.level.ne("city"), f"share_{window}"].isna().all()


def test_city_recent_shares_handle_zero_parent_and_quiet_city(raw):
    nodes = node_panel(raw)
    nodes.loc[nodes.CICLOS.eq("202602"), "items"] = 0
    featured = build_features(nodes)
    target = featured[featured.level.eq("city") & featured.CICLOS.eq("202603")]
    assert target.share_1.isna().all()
    assert target.share_2.notna().all()
    featured = build_features(node_panel(raw))
    quiet = featured[
        featured.level.eq("city") & featured.node.str.contains("GO") & featured.CICLOS.eq("202605")
    ]
    assert quiet.share_1.eq(0).all()


def test_origin_policy_uses_closed_cycles(raw, settings):
    calendar = cycle_calendar(raw)
    calendar["closing_date"] = calendar["opening_date"] + pd.Timedelta(days=9)
    for origin in origin_plan(calendar, settings):
        closes = calendar.set_index("CICLOS").loc[origin["training_cycles"], "closing_date"]
        assert closes.lt(origin["as_of"]).all()
        assert len(closes) == 3


def test_distribution_and_reconciliation_are_distinct_and_coherent():
    query = pd.DataFrame(
        {"node": ["r", "a", "b"], "parent": ["", "r", "r"], "level": ["region", "city", "city"]}
    )
    history = pd.DataFrame({"node": ["r", "a", "b"], "items": [100, 20, 20]})
    base = pd.Series([100.0, 20.0, 20.0], index=query.node)
    distributed = distribute(
        query, np.array([100.0]), {"city": np.array([0.5, 0.5])}, ["region", "city"]
    )
    reconciled, info = reconcile(history, query, base, "city", 150)
    np.testing.assert_allclose(reconciled.to_numpy(), [80, 40, 40], rtol=1e-5)
    assert info["method"] == "nonnegative_wls"
    assert distributed["r"] == 100
    assert_coherent(query, distributed, "city")
    assert_coherent(query, reconciled, "city")


def test_uniform_share_fallback():
    query = pd.DataFrame({"node": ["a", "b"], "parent": ["r", "r"]})
    np.testing.assert_allclose(normalize_scores(query, np.array([0.0, 0.0])), [0.5, 0.5])


def test_real_hybrid_models_and_unknown_city_coverage(raw, settings):
    result = evaluate_hierarchy(raw, settings)
    names = set(result.comparison.candidate)
    assert names == {
        "direct_ma_2",
        "distribution_ma_2",
        "reconciliation_ma_2",
        "direct_ets_lgbm",
        "distribution_ets_lgbm",
        "reconciliation_ets_lgbm",
    }
    city = result.comparison[result.comparison.level.eq("city")]
    assert city.n_missing.eq(1).all()
    assert city.coverage_volume_pct.lt(100).all()
    assert np.isfinite(result.comparison[["mae", "wmape_pct", "bias_pct"]]).all().all()
    _assert_consistent_totals(result.predictions)


def _assert_consistent_totals(predictions):
    for name, frame in predictions.groupby("candidate"):
        if not name.startswith("direct_"):
            totals = frame.groupby("level").items_pred.sum()
            np.testing.assert_allclose(totals, totals.iloc[0], atol=1e-5)


def test_future_demand_does_not_change_predictions(raw, settings):
    before = evaluate_hierarchy(raw, settings).predictions
    changed = raw.copy()
    changed.loc[changed.CICLOS.eq("202609"), "total_itens_mascarado"] *= 100
    after = evaluate_hierarchy(changed, settings).predictions
    keys = ["candidate", "level", "node"]
    comparison = before.merge(after, on=keys, suffixes=("_before", "_after"))
    np.testing.assert_allclose(comparison.items_pred_before, comparison.items_pred_after)


def test_scenario_dates_do_not_create_fake_accuracy(raw, settings):
    result = evaluate_hierarchy(raw, settings)
    from src.forecasting.hierarchy_evaluation import _observed_windows

    windows = _observed_windows(raw, cycle_calendar(raw), "202609")
    observed = windows.assign(scenario_id="original")
    alternative = windows.assign(
        scenario_id="shifted", window_start=windows.window_start + pd.Timedelta(days=1)
    )
    both = evaluate_hierarchy(raw, settings, pd.concat([observed, alternative]))
    shifted = both.predictions[both.predictions.scenario_id.eq("shifted")]
    assert shifted.actual.isna().all()
    assert set(both.metrics_by_cycle.scenario_id) == {"original"}
    assert len(both.comparison) == len(result.comparison)


def test_configuration_validation():
    with pytest.raises(ValidationError):
        HierarchySettings(levels=["city", "sector"])
    with pytest.raises(ValidationError):
        HierarchySettings(baseline_windows=[5, 5])
    with pytest.raises(ValidationError):
        HierarchySettings(ets_alphas=[0])
    with pytest.raises(ValidationError, match="standard comparison"):
        HierarchySettings(baseline_windows=[3, 5])


@pytest.mark.parametrize("city_grouping", ["sector", "management"])
def test_standard_candidates_preserve_historical_predictions(raw, settings, city_grouping):
    config = settings.model_copy(
        update={"comparison": "standard", "history_cycles": 5, "baseline_windows": [5]}
    )
    if city_grouping == "management":
        config = config.model_copy(
            update={"levels": ["region", "management", "city"], "city_grouping": city_grouping}
        )
    standard = evaluate_hierarchy(raw, config)
    historical = evaluate_hierarchy(raw, config.model_copy(update={"comparison": "historical"}))
    candidates = {"direct_ma_5", "distribution_ets_lgbm", "reconciliation_ets_lgbm"}
    assert set(standard.comparison.candidate) == candidates
    expected = historical.predictions[historical.predictions.candidate.isin(candidates)]
    pd.testing.assert_frame_equal(standard.predictions, expected.reset_index(drop=True))
    metrics = historical.comparison[historical.comparison.candidate.isin(candidates)]
    pd.testing.assert_frame_equal(standard.comparison, metrics.reset_index(drop=True))
    assert "reconciliation_ma_5" not in {fit.get("candidate") for fit in standard.fitted}


def test_standard_moving_average_does_not_distribute_or_reconcile(raw, monkeypatch):
    from src.forecasting.hierarchy import HierarchyResult, _baselines

    def unexpected_adjustment(*args, **kwargs):
        raise AssertionError("pure moving average must not use a hierarchy adjustment")

    monkeypatch.setattr("src.forecasting.hierarchy.distribute", unexpected_adjustment)
    monkeypatch.setattr("src.forecasting.hierarchy.reconcile", unexpected_adjustment)
    nodes = node_panel(raw)
    history = nodes[nodes.CICLOS.lt("202609")]
    target = nodes[nodes.CICLOS.eq("202609")]
    result = HierarchyResult()
    _baselines(history, target, HierarchySettings(), result)
    expected = history.sort_values("opening_date").groupby("node").tail(5)
    means = expected.groupby("node")["items"].mean()
    np.testing.assert_allclose(result.predictions["direct_ma_5"], target.node.map(means).fillna(0))


@pytest.mark.parametrize(
    "filename", ["smoke.yaml", "full.yaml", "management_city.yaml", "recent_city_shares.yaml"]
)
def test_standard_yaml_settings(filename):
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "experiments" / "hierarchical_level" / filename
    config = HierarchySettings.model_validate(yaml.safe_load(path.read_text())["settings"])
    assert config.comparison == "standard"
    assert config.history_cycles == 20
    assert config.baseline_windows == [5]


def test_ablation_accepts_old_historical_settings_and_rejects_standard_reference(settings):
    from experiments.hierarchical_level.ablation import _check_reference_settings

    original = {
        "data": {"sha256": "same"},
        "status": "complete",
        "effective_settings": settings.model_dump(exclude={"comparison"}),
    }
    current = {**original, "effective_settings": settings.model_dump()}
    _check_reference_settings(original, current)
    current["effective_settings"] = {**settings.model_dump(), "comparison": "standard"}
    with pytest.raises(ValueError, match="settings differ"):
        _check_reference_settings(original, current)


@pytest.mark.parametrize("comparison", ["standard", "historical"])
def test_archives_complete_inputs_and_refuses_overwrite(tmp_path, raw, settings, comparison):
    updates = {
        "standard": {"comparison": "standard", "history_cycles": 5, "baseline_windows": [5]},
        "historical": {},
    }
    settings = settings.model_copy(update=updates[comparison])
    source = tmp_path / "demand.csv"
    raw.to_csv(source, index=False)
    folder = tmp_path / "result"
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"base_csv": str(source), "run_dir": str(folder)},
                "settings": settings.model_dump(),
            }
        ),
        encoding="utf-8",
    )
    run(config)
    assert {
        "comparison.csv",
        "predictions.csv",
        "metrics_by_cycle.csv",
        "comparison_all_points.csv",
        "metrics_by_cycle_all_points.csv",
        "folds.csv",
        "config.yaml",
        "features.json",
        "fitted_models.json",
        "manifest.json",
    } <= {p.name for p in folder.iterdir()}
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["effective_settings"]["comparison"] == comparison
    assert "actual > 0" in manifest["scoring"]
    with pytest.raises(FileExistsError):
        run(config)


def test_active_city_metrics_keep_unknown_buyers_and_full_parent_totals():
    frame = pd.DataFrame(
        {
            "candidate": ["model"] * 4,
            "level": ["city", "city", "city", "sector"],
            "node": ["buyer", "quiet", "new_buyer", "parent"],
            "actual": [100.0, 0.0, 20.0, 120.0],
            "items_pred": [80.0, 50.0, 0.0, 130.0],
            "abs_error": [20.0, 50.0, 20.0, 10.0],
            "has_prediction": [True, True, False, True],
            "is_observed": [True] * 4,
        }
    )
    primary = summarize(frame, ["level", "candidate"]).set_index("level")
    full = summarize(frame, ["level", "candidate"], city_active_only=False).set_index("level")
    city = primary.loc["city"]
    expected = {
        "wmape_pct": pytest.approx(100 * 40 / 120),
        "mae": 20,
        "bias_pct": pytest.approx(-100 * 40 / 120),
        "n_points": 2,
        "n_missing": 1,
        "n_excluded_points": 1,
        "n_total_points": 3,
        "scoring_scope": "city_actual_positive",
    }
    assert city[list(expected)].to_dict() == expected
    assert full.loc["city", "wmape_pct"] == 75
    pd.testing.assert_series_equal(primary.loc["sector"], full.loc["sector"])


def test_inactive_city_group_has_no_score_and_no_fake_scenario_metrics():
    frame = pd.DataFrame(
        {
            "candidate": ["model", "model"],
            "level": ["city", "city"],
            "node": ["quiet", "alternative"],
            "actual": [0.0, np.nan],
            "items_pred": [50.0, 70.0],
            "abs_error": [50.0, np.nan],
            "has_prediction": [True, True],
            "is_observed": [True, False],
        }
    )
    primary = summarize(frame, ["level", "candidate"]).iloc[0]
    assert primary[["n_points", "n_excluded_points"]].to_dict() == {
        "n_points": 0,
        "n_excluded_points": 1,
    }
    np.testing.assert_allclose(
        primary[["mae", "wmape_pct"]].to_numpy(dtype=float), [np.nan, np.nan], equal_nan=True
    )
    full = summarize(frame, ["level", "candidate"], city_active_only=False).iloc[0]
    assert full[["n_points", "mae"]].to_dict() == {"n_points": 1, "mae": 50}


def test_invalid_scenario_file(tmp_path, raw):
    path = tmp_path / "scenarios.csv"
    pd.DataFrame({"scenario_id": ["bad"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_scenarios(str(path), cycle_calendar(raw))
    assert load_scenarios(None, cycle_calendar(raw)) is None


def test_partial_scenario_is_rejected(raw, settings):
    from src.forecasting.hierarchy_evaluation import _observed_windows, _scenario_layouts

    windows = _observed_windows(raw, cycle_calendar(raw), "202609")
    partial = windows.head(1).assign(scenario_id="incomplete")
    with pytest.raises(ValueError, match="full planned sector roster"):
        _scenario_layouts(windows, partial, "202609")


def test_forecast_api_rejects_unclosed_history(raw, settings):
    nodes = node_panel(raw)
    query = nodes[nodes.CICLOS.eq("202609")].copy()
    history = nodes[nodes.CICLOS.lt("202609")].copy()
    history["closing_date"] = query["opening_date"].iloc[0]
    with pytest.raises(ValueError, match="closed before"):
        forecast_hierarchy(history, query, settings)
    with pytest.raises(ValueError, match="no known cities"):
        forecast_hierarchy(history, query.head(0), settings)


@pytest.fixture
def shared_management_raw(raw):
    frame = raw.copy()
    frame.loc[frame.cd_setor.eq("02"), "CD_GV"] = "g1"
    return frame


def test_management_city_aggregation_merges_sectors_and_preserves_homonyms(shared_management_raw):
    cal = cycle_calendar(shared_management_raw)
    panel = build_item_panel(shared_management_raw, cal).merge(
        cal[["CICLOS", "closing_date"]], on="CICLOS"
    )
    nodes = aggregate_nodes(panel, ["region", "management", "city"], "management")
    cities = nodes[nodes.level.eq("city")]
    assert cities.node.nunique() == 5
    selected = cities[cities.CICLOS.eq("202601") & cities.node.str.contains('"g1", "MG"')]
    assert selected["items"].iloc[0] == 80
    totals = nodes.groupby(["level", "CICLOS"])["items"].sum().unstack("level")
    np.testing.assert_allclose(totals.to_numpy(), totals[["city"]].to_numpy().repeat(3, axis=1))
    pd.testing.assert_frame_equal(
        aggregate_nodes(panel, BLOCK_LEVELS), aggregate_nodes(panel, BLOCK_LEVELS, "sector")
    )


def test_management_city_backtest_is_coherent_and_exports_identity(shared_management_raw, settings):
    config = settings.model_copy(
        update={"levels": ["region", "management", "city"], "city_grouping": "management"}
    )
    result = evaluate_hierarchy(shared_management_raw, config)
    _assert_consistent_totals(result.predictions)
    cities = result.predictions[result.predictions.level.eq("city")]
    assert "cd_setor" not in cities.columns
    assert cities[["CD_RE", "CD_GV", "estado", "cidade"]].notna().all().all()
    assert result.comparison.query('level == "city"').n_missing.eq(1).all()
    changed = shared_management_raw.copy()
    changed.loc[changed.CICLOS.eq("202609"), "total_itens_mascarado"] *= 100
    after = evaluate_hierarchy(changed, config)
    np.testing.assert_allclose(result.predictions.items_pred, after.predictions.items_pred)


def test_management_city_configuration_rejects_sector_parent():
    with pytest.raises(ValidationError, match="management directly before city"):
        HierarchySettings(city_grouping="management")
    accepted = HierarchySettings(
        levels=["region", "management", "city"], city_grouping="management"
    )
    assert accepted.city_grouping == "management"


def test_ablation_cached_model_path_matches_production_and_ignores_future(raw, settings):
    from experiments.hierarchical_level.ablation import fold_tables, learned_only

    calendar = cycle_calendar(raw)
    plan = origin_plan(calendar, settings)[0]
    tables = fold_tables(raw, calendar, plan, settings)
    cached = learned_only(tables, settings)
    standard = forecast_hierarchy(tables.training, tables.query, settings)
    for name, values in cached.predictions.items():
        pd.testing.assert_series_equal(values, standard.predictions[name])
    changed = raw.copy()
    changed.loc[changed.CICLOS.eq(plan["target_cycle"]), "total_itens_mascarado"] *= 100
    after = fold_tables(changed, calendar, plan, settings)
    pd.testing.assert_frame_equal(tables.training, after.training)
    pd.testing.assert_frame_equal(tables.target, after.target)


def test_ablation_archives_variants_and_reproduces_reference(tmp_path, raw, settings):
    from experiments.hierarchical_level.ablation import run as run_ablation

    base = tmp_path / "base.csv"
    raw.to_csv(base, index=False)
    reference = tmp_path / "reference"
    source = tmp_path / "config.yaml"
    config = {
        "settings": settings.model_dump(),
        "paths": {"base_csv": str(base), "run_dir": str(reference)},
    }
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    run(source)
    directory = tmp_path / "ablation"
    config["paths"].update(run_dir=str(directory), reference_run=str(reference))
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    comparison = run_ablation(source)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert comparison.variant.nunique() == len(settings.learning.features) + 1
    removed = json.loads(
        (directory / "variants" / "without_calendar" / "features.json").read_text()
    )
    assert "cycle_month" not in removed["selected_features"]
    assert (directory / "ablation_effects_by_cycle.csv").exists()
