"""Build the geo-sliceable dashboard data (national / per-state / per-city) from predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("experiments/compare_forecasters/outputs")
RUNS = {
    "Features + temporais": "2026-10-19_features_temporais",
    "Externas por UF": "2026-10-19_externos_uf",
    "Momentum + calendario": "2026-10-19_dynamics",
    "Prophet (amostra 60 setores)": "2026-10-19_prophet_amostra",
    "Croston / SBA / TSB": "2026-10-19_croston",
    "Objetivos Tweedie / Quantile": "2026-10-19_objective",
    "Quantile alto (margem de seguranca)": "2026-10-19_quantile_alto",
}
TOP_CITIES = 120  # cities with the most real volume (per experiment) kept for the city slice


def _metrics(g: pd.DataFrame) -> dict:
    actual = g["actual"].to_numpy(float)
    pred = g["items_pred"].to_numpy(float)
    err = pred - actual
    n = len(g)
    sum_actual = np.abs(actual).sum()
    return {
        "mae_common": round(float(np.abs(err).mean()), 3) if n else None,
        "wmape_common": round(float(np.abs(err).sum() / sum_actual), 3) if sum_actual else None,
        "bias_pct_common": round(float(100 * err.sum() / sum_actual), 3) if sum_actual else None,
        "rmse_common": round(float(np.sqrt((err**2).mean())), 3) if n else None,
        "n": int(n),
    }


def _rank(df: pd.DataFrame) -> list[dict]:
    rows = []
    for cand, g in df.groupby("candidate"):
        rows.append({"candidate": cand, **_metrics(g)})
    rows.sort(key=lambda r: (r["mae_common"] is None, r["mae_common"]))
    return rows


def build_experiment(run_dir: str) -> dict:
    p = pd.read_csv(
        BASE / run_dir / "predictions.csv",
        usecols=["candidate", "estado", "cidade", "actual", "items_pred"],
    )
    p = p.dropna(subset=["actual", "items_pred"])

    result = {"__all__": _rank(p)}

    # per state
    states = {}
    for uf, g in p.groupby("estado"):
        states[str(uf)] = _rank(g)
    result["states"] = states

    # per city, limited to the TOP_CITIES by real volume (keyed "ESTADO|CIDADE")
    p["city_key"] = p["estado"].astype(str) + "|" + p["cidade"].astype(str)
    top = p.groupby("city_key")["actual"].sum().sort_values(ascending=False).head(TOP_CITIES).index
    cities = {}
    for ck, g in p[p["city_key"].isin(set(top))].groupby("city_key"):
        cities[ck] = _rank(g)
    result["cities"] = cities
    return result


def _band_metrics(g: pd.DataFrame) -> dict:
    err = g["items_pred"].to_numpy(float) - g["actual"].to_numpy(float)
    sa = g["actual"].abs().sum()
    n = len(g)
    return {
        "mae": round(float(np.abs(err).mean()), 1) if n else None,
        "wmape": round(float(np.abs(err).sum() / sa), 3) if sa else None,
        "bias_pct": round(float(100 * err.sum() / sa), 1) if sa else None,
        "frac_under": round(float((g["items_pred"] < g["actual"]).mean()), 3) if n else None,
    }


def build_volume_bands(run_dir: str) -> dict:
    """Per-candidate error within small/medium/large series bands (by mean real demand)."""
    p = pd.read_csv(
        BASE / run_dir / "predictions.csv", usecols=["candidate", "serie", "actual", "items_pred"]
    ).dropna(subset=["actual", "items_pred"])
    serie_mean = p.groupby("serie")["actual"].mean()
    q33, q66 = serie_mean.quantile([0.33, 0.66])

    def band(v: float) -> str:
        return "pequena" if v <= q33 else ("media" if v <= q66 else "grande")

    p["faixa"] = p["serie"].map(serie_mean).map(band)
    rows = []
    for cand, gc in p.groupby("candidate"):
        entry = {"candidate": cand}
        for fx in ["pequena", "media", "grande"]:
            entry[fx] = _band_metrics(gc[gc["faixa"] == fx])
        entry["todas"] = _band_metrics(gc)
        rows.append(entry)
    return {"cuts": {"q33": round(float(q33), 1), "q66": round(float(q66), 1)}, "rows": rows}


def main() -> None:
    out = {name: build_experiment(rd) for name, rd in RUNS.items()}
    out["__bands__"] = build_volume_bands(RUNS["Quantile alto (margem de seguranca)"])
    Path("reports_dashboard/_data.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )
    sizes = {n: (len(v["states"]), len(v["cities"])) for n, v in out.items() if n != "__bands__"}
    print("built:", sizes)
    print("file KB:", round(Path("reports_dashboard/_data.json").stat().st_size / 1024, 1))


if __name__ == "__main__":
    main()
