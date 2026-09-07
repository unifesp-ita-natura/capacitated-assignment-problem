import numpy as np

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


def build_current_assignment(rng, sectors=None, block_weights=None):
    """As-is (block, sublock) per sector
    If block_weights is None, defaults to uniform between sectors
    """
    sectors = sectors if sectors is not None else SECTORS
    block_weights = block_weights if block_weights is not None else 1 / len(BLOCKS)
    block_assigned = rng.choice(BLOCKS, size=len(sectors), p=[block_weights[b] for b in BLOCKS])
    sublock_assigned = rng.choice(SUBLOCKS, size=len(sectors))
    return {
        sector: (int(block), int(sublock))
        for sector, block, sublock in zip(sectors, block_assigned, sublock_assigned)
    }


def build_sector_volume_parameters(
    rng, sectors, min_volume=20, max_volume=60, variance_percentage=0.15
):
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


def expected_orders(sector, campanha_id, weekday_share):
    pass
