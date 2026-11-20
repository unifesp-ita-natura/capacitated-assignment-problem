"""Validate isolated level hierarchy experiments and selectable feature blocks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LEVEL_KEYS = {
    "region": ["CD_RE"],
    "management": ["CD_RE", "CD_GV"],
    "sector": ["cd_setor"],
    "city": ["cd_setor", "estado", "cidade"],
}
BASELINE_WINDOW = 5
Level = Literal["region", "management", "sector", "city"]
FeatureBlock = Literal[
    "history", "calendar", "structure", "hierarchy", "intermittency", "city_recent_shares"
]


def level_keys(level: str, city_grouping: str = "sector") -> list[str]:
    """Keep city identity explicit when sectors are aggregated within a management."""
    if level == "city" and city_grouping == "management":
        return ["CD_RE", "CD_GV", "estado", "cidade"]
    return LEVEL_KEYS[level]


def _ordered_levels(levels: list[str]) -> list[str]:
    return [level for level in LEVEL_KEYS if level in levels]


class LearningSettings(BaseModel):
    """One pooled LightGBM per internal level, with regularized city predictions."""

    model_config = ConfigDict(extra="forbid")
    features: list[FeatureBlock] = Field(
        default_factory=lambda: ["history", "calendar", "structure", "hierarchy", "intermittency"]
    )
    n_estimators: int = Field(default=100, ge=1)
    num_leaves: int = Field(default=7, ge=2)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    min_child_samples: int = Field(default=20, ge=1)
    city_prior_cycles: float = Field(default=5, gt=0)
    volume_objective: Literal["tweedie", "regression", "regression_l1"] = "tweedie"
    random_state: int = 42


class HierarchyForecastParams(BaseModel):
    """Forecast shared hierarchy totals through sectors, with optional legacy city output."""

    model_config = ConfigDict(extra="forbid")
    model: Literal["hierarchical"] = "hierarchical"
    target_level: Literal["sector", "city"] = "sector"
    city_grouping: Literal["sector", "management"] = "sector"
    share_window: int = Field(default=5, ge=1)
    learning: LearningSettings = Field(default_factory=LearningSettings)

    @model_validator(mode="after")
    def preset_features(self):
        if "features" not in self.learning.model_fields_set:
            blocks = ["history", "calendar", "structure", "hierarchy", "intermittency"]
            if self.target_level == "city":
                blocks = [*blocks, "city_recent_shares"]
            if self.city_grouping == "management":
                blocks = blocks[1:]
            self.learning = self.learning.model_copy(update={"features": blocks})
        return self

    @model_validator(mode="after")
    def valid_target(self):
        if self.target_level == "sector" and self.city_grouping != "sector":
            raise ValueError(
                "sector target requires city_grouping=sector; "
                "use target_level=city for management-city"
            )
        return self


def forecast_entity_keys(params) -> list[str]:
    """Identify the output entity without changing the legacy city model contract."""
    if isinstance(params, HierarchyForecastParams) and params.target_level == "sector":
        return ["cd_setor"]
    return ["cd_setor", "estado", "cidade"]


class HierarchySettings(BaseModel):
    """Choose the hierarchy and rolling-origin scope with a fixed five-cycle baseline."""

    model_config = ConfigDict(extra="forbid")
    levels: list[Level] = Field(default_factory=lambda: list(LEVEL_KEYS))
    city_grouping: Literal["sector", "management"] = "sector"
    history_cycles: int = Field(default=20, ge=BASELINE_WINDOW)
    share_window: int = Field(default=5, ge=1)
    ets_alphas: list[float] = Field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4, 0.8])
    learning: LearningSettings = Field(default_factory=LearningSettings)
    max_origins: int | None = Field(default=None, ge=1)
    first_target_cycle: str | None = None
    last_target_cycle: str | None = None
    scenarios_csv: str | None = None
    reconciliation_max_iter: int = Field(default=150, ge=1)

    @model_validator(mode="after")
    def valid_structure(self):
        expected = _ordered_levels(self.levels)
        if (
            self.levels != expected
            or len(self.levels) < 2
            or self.levels[-1] not in {"sector", "city"}
        ):
            raise ValueError(
                "levels must be ordered, unique, end in sector or city and include a parent"
            )
        return self

    @model_validator(mode="after")
    def valid_city_grouping(self):
        if self.city_grouping == "management" and self.levels[-2:] != ["management", "city"]:
            raise ValueError("management city grouping requires management directly before city")
        return self

    @model_validator(mode="after")
    def valid_alphas(self):
        if not self.ets_alphas or not all(0 < alpha <= 1 for alpha in self.ets_alphas):
            raise ValueError("ets_alphas must be in (0, 1]")
        return self
