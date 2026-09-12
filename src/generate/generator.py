"""
Synthetic generator for AS-IS orders
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

BLOCKS = [1, 2, 3]  # Start week 1 ... 3 (within one cycle)
SUBLOCKS = [1, 2, 3, 4, 5]  # Start day: Monday ... Friday
SLOTS = [(b, sb) for b in BLOCKS for sb in SUBLOCKS]

# TODO: we're initially pretending this is fixed, but we know that
# for different cycles this varies, make it real
WINDOW_LENGTH = 21
CYCLE_SPAN = len(SLOTS) + WINDOW_LENGTH - 1  # is this right? that's ignoring weekends
N_CYCLES = 7  # how many are we optimizing at a time

N_SECTORS = 800
SECTORS = [f"S{i:02d}" for i in range(1, N_SECTORS + 1)]

CURRENT_BLOCK_WEIGHTS = {
    1: 0.15,
    2: 0.65,
    3: 0.2,
}  # skewed toward block 2 (~45% real concentration)


# TODO: double check this works for weekends, feels wrong -> same as CYCLE_SPAN
def slot_start_day(block: int, sublock: int) -> int:
    """Day-in-cycle on which a (block, sublock) slot's sales window opens"""
    return (block - 1) * len(SUBLOCKS) + sublock


def uniform_block_distribution(blocks: list[int]) -> dict[int, float]:
    """Equally distributed blocks"""
    return {block: 1 / len(blocks) for block in blocks}


def build_current_assignment(
    rng, sectors: list[str] | None = None, block_weights: dict[int, float] = None
) -> dict[str, tuple[int, int]]:
    """As-is (block, sublock) per sector
    If block_weights is None, defaults to uniform between sectors
    """
    sectors = sectors if sectors is not None else SECTORS
    block_weights = (
        block_weights if block_weights is not None else uniform_block_distribution(BLOCKS)
    )
    block_assigned = rng.choice(BLOCKS, size=len(sectors), p=[block_weights[b] for b in BLOCKS])
    sublock_assigned = rng.choice(SUBLOCKS, size=len(sectors))
    return {
        sector: (int(block), int(sublock))
        for sector, block, sublock in zip(sectors, block_assigned, sublock_assigned)
    }


def build_sector_volume_parameters(
    rng,
    sectors: list[str],
    min_volume: int = 20,
    max_volume: int = 60,
    variance_percentage: float = 0.15,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Each sector's baseline order volume and its idiosyncratic
    scaling factor."""
    sectors = sectors if sectors is not None else SECTORS
    baseline = dict(zip(sectors, rng.uniform(min_volume, max_volume, size=len(sectors))))
    factor = dict(
        zip(
            sectors,
            rng.uniform(1 - variance_percentage, 1 + variance_percentage, size=len(sectors)),
        )
    )
    return baseline, factor


# TODO: placeholder ranges pending calibration against real ratios computable
# from demanda_level.csv (total_volumes/total_pedidos, total_itens/total_pedidos)
def build_sector_metric_ratios(
    rng,
    sectors: list[str],
    volume_range: tuple[float, float] = (1.0, 3.0),
    item_range: tuple[float, float] = (1.5, 5.0),
) -> dict[str, dict[str, float]]:
    """Each sector's volumes-per-order and itens-per-order ratios, used to
    derive correlated volume/item counts from its simulated order count."""
    return {
        sector: {
            "volumes_per_order": rng.uniform(*volume_range),
            "itens_per_order": rng.uniform(*item_range),
        }
        for sector in sectors
    }


def hump_alpha(window_length: int, concentration: int = 20, peak_frac: float = 0.5) -> float:
    """Hump-shaped Dirichlet alpha for a window of arbitrary length,
    peak at peak_frac of the window."""
    days = np.arange(window_length)
    peak = peak_frac * (window_length - 1)
    spread = max(window_length / 4, 0.75)  # avoid degenerate spread for length 1-2
    weights = np.exp(-0.5 * ((days - peak) / spread) ** 2)
    return weights / weights.sum() * concentration


def build_sector_shape_traits(
    rng,
    sectors: list[str],
    conc_range: tuple[int, int] = (10, 40),
    peak_range: tuple[float, float] = (0.3, 0.7),
) -> dict[str, dict]:
    """Each sector's stable within-cycle curve trait (peaky vs flat,
    front-loaded vs back-loaded), independent of any cycle's window
    length so it can be realized at whatever length a given cycle has."""
    return {
        sector: {
            "peak_frac": rng.uniform(*peak_range),
            "concentration": rng.uniform(*conc_range),
        }
        for sector in sectors
    }


def realize_window_shape(rng, trait: dict, window_length: int):
    """One sector-cycle's window shape: the sector's stable curve trait
    realized at this cycle's window length. Returns np.array(window_length,)
    summing to 1."""
    alpha = hump_alpha(window_length, trait["concentration"], trait["peak_frac"])
    return rng.dirichlet(alpha)


def build_cycle_window_shapes(
    rng, sectors: list[str], shape_traits: dict[str, dict], window_length: int
):
    """Every sector's window shape for a single cycle, at that
    cycle's window length."""
    return {
        sector: realize_window_shape(rng, shape_traits[sector], window_length) for sector in sectors
    }


def expected_orders(
    sector: str,
    window_day_share: float,
    sector_baseline: int,
    sector_factor: float,
    cycle_factor: float,
    window_length: int | None = None,
):
    """Poisson mean orders for one sector-day, given its cycle
    position and window-day share.
    """
    window_length = window_length if window_length is not None else WINDOW_LENGTH
    return sector_baseline * sector_factor * cycle_factor * window_day_share * window_length


def generate_sector_cycle_orders(
    rng,
    sector: str,
    cycle_id: int,
    cycle_start,
    assignment,
    cycle_factor: float,
    window_shape,
    baseline: float,
    factor: float,
    metric_ratios: dict[str, float],
):
    """
    One sector's simulated order rows for a cycle, following its fixed
    within-window curve.
    """
    block, sublock = assignment[sector]
    start_day = slot_start_day(block, sublock)
    window_start = cycle_start + pd.offsets.BDay(start_day - 1)
    rows = []
    for offset, share in enumerate(window_shape):
        orders = int(rng.poisson(expected_orders(sector, share, baseline, factor, cycle_factor)))
        rows.append(
            {
                "order_date": window_start + pd.Timedelta(days=offset),
                "cycle_id": cycle_id,
                "day_in_cycle": start_day + offset,
                "offset": offset,
                "block": block,
                "sublock": sublock,
                "sector": sector,
                "orders": orders,
                "volumes": int(rng.poisson(orders * metric_ratios["volumes_per_order"])),
                "itens": int(rng.poisson(orders * metric_ratios["itens_per_order"])),
            }
        )
    return rows


def generate_cycle_orders(
    rng,
    sectors,
    cycle_id,
    cycle_start,
    assignment,
    cycle_factor,
    window_shapes,
    baseline,
    factor,
    metric_ratios,
):
    """Every sector's simulated order rows for a single cycle."""
    return [
        row
        for sector in sectors
        for row in generate_sector_cycle_orders(
            rng,
            sector,
            cycle_id,
            cycle_start,
            assignment,
            cycle_factor,
            window_shapes[sector],
            baseline[sector],
            factor[sector],
            metric_ratios[sector],
        )
    ]


def generate_orders(
    rng,
    sectors,
    cycle_starts,
    assignment,
    cycle_factors=None,
    window_lengths: dict[int, int] | None = None,
):
    """Synthetic historical orders for every sector across
    all cycles, given an assignment. `window_lengths` maps cycle_id to
    that cycle's window length, defaulting to WINDOW_LENGTH for any
    cycle not listed."""
    baseline, factor = build_sector_volume_parameters(rng, sectors)
    metric_ratios = build_sector_metric_ratios(rng, sectors)
    shape_traits = build_sector_shape_traits(rng, sectors)
    cycle_factors = cycle_factors or {}
    window_lengths = window_lengths or {}
    rows = [
        row
        for cycle_id, cycle_start in enumerate(cycle_starts, start=1)
        for row in generate_cycle_orders(
            rng,
            sectors,
            cycle_id,
            cycle_start,
            assignment,
            cycle_factors.get(cycle_id, 1.0),
            build_cycle_window_shapes(
                rng, sectors, shape_traits, window_lengths.get(cycle_id, WINDOW_LENGTH)
            ),
            baseline,
            factor,
            metric_ratios,
        )
    ]
    return pd.DataFrame(rows)


def build_demand_level(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate synthetic orders to one row per (cd_setor, ciclo), matching
    the schema of data/processed/demanda_level.csv
    (src/processamento_dados/pipeline.py::build_demand_level)."""
    df = orders_df.rename(columns={"sector": "cd_setor"}).copy()
    df["ciclo"] = df["cycle_id"].astype(str)
    return (
        df.groupby(["cd_setor", "ciclo"], as_index=False)
        .agg(
            date=("order_date", "min"),
            cycle_duration=("offset", "max"),
            total_pedidos=("orders", "sum"),
            total_volumes=("volumes", "sum"),
            total_itens=("itens", "sum"),
        )
        .assign(cycle_duration=lambda d: d["cycle_duration"] + 1)
        .sort_values(by=["cd_setor", "date"])
        .reset_index(drop=True)
    )


def build_demand_shape(orders_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cd_setor, ciclo, data_pedido), matching the schema of
    data/processed/demanda_shape.csv
    (src/processamento_dados/pipeline.py::build_demand_shape)."""
    df = orders_df.rename(
        columns={
            "sector": "cd_setor",
            "order_date": "data_pedido",
            "orders": "total_pedidos",
            "volumes": "total_volumes",
            "itens": "total_itens",
        }
    ).copy()
    df["ciclo"] = df["cycle_id"].astype(str)
    cycle_length = df.groupby(["cd_setor", "ciclo"])["offset"].transform("max") + 1
    df["relative_date"] = df["offset"] / cycle_length
    df[["ciclo_total_pedidos", "ciclo_total_volumes", "ciclo_total_itens"]] = df.groupby(
        ["cd_setor", "ciclo"]
    )[["total_pedidos", "total_volumes", "total_itens"]].transform("sum")
    df["share_pedidos"] = df["total_pedidos"] / df["ciclo_total_pedidos"]
    df["share_volumes"] = df["total_volumes"] / df["ciclo_total_volumes"]
    df["share_itens"] = df["total_itens"] / df["ciclo_total_itens"]
    return df[
        [
            "cd_setor",
            "ciclo",
            "data_pedido",
            "relative_date",
            "total_pedidos",
            "total_volumes",
            "total_itens",
            "ciclo_total_pedidos",
            "ciclo_total_volumes",
            "ciclo_total_itens",
            "share_pedidos",
            "share_volumes",
            "share_itens",
        ]
    ]
