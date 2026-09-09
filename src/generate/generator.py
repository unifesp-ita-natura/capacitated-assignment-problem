"""
Synthetic generator for AS-IS orders
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

BLOCKS = [1, 2, 3]  # Start week 1 ... 3 (within one campanha)
SUBLOCKS = [1, 2, 3, 4, 5]  # Start day: Monday ... Friday
SLOTS = [(b, sb) for b in BLOCKS for sb in SUBLOCKS]

# TODO: we're initially pretending this is fixed, but we know that
# for different campanhas this varies, make it real
WINDOW_LENGTH = 21
CAMPANHA_SPAN = len(SLOTS) + WINDOW_LENGTH - 1  # is this right? that's ignoring weekends
N_CAMPANHAS = 7  # how many are we optimizing at a time

N_SECTORS = 800
SECTORS = [f"S{i:02d}" for i in range(1, N_SECTORS + 1)]

CURRENT_BLOCK_WEIGHTS = {
    1: 0.15,
    2: 0.65,
    3: 0.2,
}  # skewed toward block 2 (~45% real concentration)


# TODO: double check this works for weekends, feels wrong -> same as CAMPANHA_SPAN
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


def hump_alpha(window_length: int, concentration: int = 20, peak_frac: float = 0.5) -> float:
    """Hump-shaped Dirichlet alpha for a window of arbitrary length,
    peak at peak_frac of the window."""
    days = np.arange(window_length)
    peak = peak_frac * (window_length - 1)
    spread = max(window_length / 4, 0.75)  # avoid degenerate spread for length 1-2
    weights = np.exp(-0.5 * ((days - peak) / spread) ** 2)
    return weights / weights.sum() * concentration


def build_sector_window_shapes(
    rng,
    sector_meta: dict[str, dict],
    conc_range: tuple[int, int] = (10, 40),
    peak_range: tuple[float, float] = (0.3, 0.7),
):
    """
    sector_meta: dict sector_id -> {"window_length": int, ...}
    Returns dict sector_id -> np.array(window_length,) summing to 1.
    """
    shapes = {}
    for s, meta in sector_meta.items():
        length = meta["window_length"]
        peak_frac = rng.uniform(*peak_range)  # front-loaded vs back-loaded sectors
        concentration = rng.uniform(*conc_range)  # peaky vs flat sectors
        alpha = hump_alpha(length, concentration, peak_frac)
        shapes[s] = rng.dirichlet(alpha)
    return shapes


def expected_orders(
    sector: str,
    window_day_share: float,
    sector_baseline: int,
    sector_factor: float,
    campanha_factor: float,
    window_length: int | None = None,
):
    """Poisson mean orders for one sector-day, given its cycle
    position and window-day share.
    """
    window_length = window_length if window_length is not None else WINDOW_LENGTH
    return sector_baseline * sector_factor * campanha_factor * window_day_share * window_length


def generate_sector_campanha_orders(
    rng,
    sector: str,
    campanha_id: int,
    campanha_start,
    assignment,
    campanha_factor: float,
    window_shape,
    baseline: float,
    factor: float,
):
    """
    One sector's simulated order rows for a cycle, following its fixed
    within-window curve.
    """
    block, sublock = assignment[sector]
    start_day = slot_start_day(block, sublock)
    window_start = campanha_start + pd.offsets.BDay(start_day - 1)
    return [
        {
            "order_date": window_start + pd.Timedelta(days=offset),
            "campanha_id": campanha_id,
            "day_in_cycle": start_day + offset,
            "offset": offset,
            "block": block,
            "sublock": sublock,
            "sector": sector,
            "orders": int(
                rng.poisson(expected_orders(sector, share, baseline, factor, campanha_factor))
            ),
        }
        for offset, share in enumerate(window_shape)
    ]


def generate_campanha_orders(
    rng,
    sectors,
    campanha_id,
    campanha_start,
    assignment,
    campanha_factor,
    window_shapes,
    baseline,
    factor,
):
    """Every sector's simulated order rows for a single cycle."""
    return [
        row
        for sector in sectors
        for row in generate_sector_campanha_orders(
            rng,
            sector,
            campanha_id,
            campanha_start,
            assignment,
            campanha_factor,
            window_shapes[sector],
            baseline[sector],
            factor[sector],
        )
    ]


def generate_orders(rng, sectors, campanha_starts, assignment, campanha_factors=None):
    """Synthetic historical orders for every sector across
    all cycles, given an assignment"""
    baseline, factor = build_sector_volume_parameters(rng, sectors)
    window_shapes = build_sector_window_shapes(
        rng, {sector: {"window_length": WINDOW_LENGTH} for sector in sectors}
    )
    campanha_factors = campanha_factors or {}
    rows = [
        row
        for campanha_id, campanha_start in enumerate(campanha_starts, start=1)
        for row in generate_campanha_orders(
            rng,
            sectors,
            campanha_id,
            campanha_start,
            assignment,
            campanha_factors.get(campanha_id, 1.0),
            window_shapes,
            baseline,
            factor,
        )
    ]
    return pd.DataFrame(rows)
