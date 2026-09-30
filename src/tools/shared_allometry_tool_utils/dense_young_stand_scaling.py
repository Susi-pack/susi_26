"""Dense young stand scaling: the rule, its default numbers and its config
table, shared by xml_to_allometry.py and metsakeskus_to_allometry.py.

A young stand can be recorded with so many stems that simulating it as
recorded is meaningless. When the option is on, such a stand's basal area and
stem count are multiplied, for every species, by one scaling factor before the
growth model runs, so its allometry starts from the scaled-down stand. See
CONTEXT.md, "Dense young stand scaling", and docs/dense_young_stand_scaling.md.

The rule reads everything off the strata (PerSpecies[TreeStratum]), never off
a source's own stand-level summary, so both tools judge a stand the same way.

This is not a cutting-management event: nothing is harvested in any run, and
it has nothing to do with a Thinning.
"""

from dataclasses import replace
from typing import Self

from pydantic import model_validator

from susi.io.extra_pydantic_types import PositiveFloat, StrictFrozenModel
from susi.io.load_output_data import StandID
from tools.shared_allometry_tool_utils.tree_stratum import PerSpecies, TreeStratum

# %% The default numbers
#
# Defined once, here. They are the defaults of DenseYoungStandScalingConfig
# (so what a config file gets when it leaves a key out), and the numbers the
# tools' warning judges a stand by whatever the config file says (see
# stands_the_default_rule_would_scale). They are the numbers
# xml_to_allometry.py used before #250.
#
# "Other" means pine- or deciduous-dominated (species code 1 or 3).

# Only stands with a mean diameter below this are scaled, cm.
DEFAULT_MAX_MEAN_DIAMETER = 8.0
# A stand is scaled when its stem count is above this, stems/ha.
DEFAULT_STEM_COUNT_THRESHOLD_SPRUCE = 2200.0
DEFAULT_STEM_COUNT_THRESHOLD_OTHER = 2500.0
# The stem count a scaled stand starts from, stems/ha.
DEFAULT_TARGET_STEM_COUNT_SPRUCE = 1800.0
DEFAULT_TARGET_STEM_COUNT_OTHER = 2000.0

# The docs page that explains the option: linked from each tool's
# default_config.toml and from the warning.
DOCS_URL = "https://susi-pack.github.io/susi_26/dense_young_stand_scaling/"

# Species codes, as in the inventory sources: 1 pine, 2 spruce, 3 deciduous.
SPRUCE = 2


# %% Config


class DenseYoungStandScalingConfig(StrictFrozenModel):
    """
    The [dense_young_stand_scaling] table of a generating tool's config file.
    Off by default; every number defaults to the module constants above.
    """

    enabled: bool = False
    max_mean_diameter: PositiveFloat = DEFAULT_MAX_MEAN_DIAMETER  # cm
    stem_count_threshold_spruce: PositiveFloat = DEFAULT_STEM_COUNT_THRESHOLD_SPRUCE
    stem_count_threshold_other: PositiveFloat = DEFAULT_STEM_COUNT_THRESHOLD_OTHER
    target_stem_count_spruce: PositiveFloat = DEFAULT_TARGET_STEM_COUNT_SPRUCE
    target_stem_count_other: PositiveFloat = DEFAULT_TARGET_STEM_COUNT_OTHER

    @model_validator(mode="after")
    def targets_do_not_exceed_thresholds(self) -> Self:
        # A stand is scaled to its target whenever it has more stems than its
        # threshold. With a target above the threshold, a stand with a stem
        # count between the two would be scaled UP, which is never the intent.
        for main_species_name, threshold, target in (
            ("spruce", self.stem_count_threshold_spruce, self.target_stem_count_spruce),
            ("other", self.stem_count_threshold_other, self.target_stem_count_other),
        ):
            if target > threshold:
                raise ValueError(
                    f"target_stem_count_{main_species_name} ({target:g}) is above "
                    f"stem_count_threshold_{main_species_name} ({threshold:g}). "
                    "That would give some stands more stems than the inventory "
                    "recorded: dense young stand scaling only ever scales a stand "
                    "down. Set the target at or below the threshold."
                )
        return self


# %% What the rule reads off a stand's strata


def total_stem_count(strata: PerSpecies[TreeStratum]) -> float:
    """The stand's stem count, stems/ha: the sum over the three species."""
    return (
        strata.pine.stem_count + strata.spruce.stem_count + strata.deciduous.stem_count
    )


def basal_area_weighted_mean_diameter(strata: PerSpecies[TreeStratum]) -> float:
    """The stand's mean diameter, cm: each species' mean diameter weighted by
    its basal area.

    Raises ValueError for a stand with no basal area at all: there is nothing
    to weigh by, and a quiet 0 (or nan) would read as a real diameter.
    """
    total_basal_area = (
        strata.pine.basal_area + strata.spruce.basal_area + strata.deciduous.basal_area
    )
    if total_basal_area == 0:
        raise ValueError(
            "A stand with no basal area has no basal-area-weighted mean diameter"
        )
    return (
        strata.pine.mean_diameter * strata.pine.basal_area
        + strata.spruce.mean_diameter * strata.spruce.basal_area
        + strata.deciduous.mean_diameter * strata.deciduous.basal_area
    ) / total_basal_area


def main_species(strata: PerSpecies[TreeStratum]) -> int:
    """The species code (1 pine, 2 spruce, 3 deciduous) of the species with
    the largest basal area. On a tie, the first of the tied species in pine,
    spruce, deciduous order."""
    basal_areas = [
        strata.pine.basal_area,
        strata.spruce.basal_area,
        strata.deciduous.basal_area,
    ]
    # index 0 -> species code 1 (pine), 1 -> 2 (spruce), 2 -> 3 (deciduous):
    # the +1 turns the index into the species code. list.index returns the
    # first match, which is what settles a tie.
    return basal_areas.index(max(basal_areas)) + 1


# %% The rule


def dense_young_stand_scaling_factor(
    strata: PerSpecies[TreeStratum], config: DenseYoungStandScalingConfig
) -> float | None:
    """
    The scaling factor of a stand the rule scales, None for every other stand.

    A stand is scaled when its mean diameter is below config.max_mean_diameter
    AND its stem count is above the threshold for its main species. Both
    comparisons are strict: a stand exactly at either number is left alone.
    The factor is the target stem count for its main species over its stem
    count, so it is always below 1.

    A stand with no basal area or no stems returns None: there is nothing to
    judge.

    config.enabled is ignored here. Whether the option is on is the caller's
    question; the warning, for one, asks this function about stands the
    option is off for.
    """
    stem_count = total_stem_count(strata)
    basal_area = (
        strata.pine.basal_area + strata.spruce.basal_area + strata.deciduous.basal_area
    )
    if stem_count == 0 or basal_area == 0:
        return None

    if main_species(strata) == SPRUCE:
        stem_count_threshold = config.stem_count_threshold_spruce
        target_stem_count = config.target_stem_count_spruce
    else:
        stem_count_threshold = config.stem_count_threshold_other
        target_stem_count = config.target_stem_count_other

    is_young = basal_area_weighted_mean_diameter(strata) < config.max_mean_diameter
    is_dense = stem_count > stem_count_threshold
    if is_young and is_dense:
        return target_stem_count / stem_count
    return None


def scale_strata(
    strata: PerSpecies[TreeStratum], scaling_factor: float
) -> PerSpecies[TreeStratum]:
    """New strata with every species' basal area and stem count multiplied by
    scaling_factor. Age, mean diameter and mean height are unchanged: the
    scaled stand has fewer trees of the same size."""

    def scale(stratum: TreeStratum) -> TreeStratum:
        return replace(
            stratum,
            basal_area=stratum.basal_area * scaling_factor,
            stem_count=stratum.stem_count * scaling_factor,
        )

    return PerSpecies(
        pine=scale(strata.pine),
        spruce=scale(strata.spruce),
        deciduous=scale(strata.deciduous),
    )


def strata_to_grow_from(
    strata: PerSpecies[TreeStratum], scaling_factor: float | None
) -> PerSpecies[TreeStratum]:
    """The strata a stand's growth table is built from: the scaled ones when
    the stand has a scaling factor, the recorded ones otherwise."""
    if scaling_factor is None:
        return strata
    return scale_strata(strata, scaling_factor)


def decide_scaling_factors(
    strata_per_stand: dict[StandID, PerSpecies[TreeStratum]],
    config: DenseYoungStandScalingConfig,
) -> dict[StandID, float | None]:
    """
    Every stand's scaling factor for one run of a generating tool, decided
    once: the rule's answer when the option is on (config.enabled), None for
    every stand when it is off. This is the one place that reads
    config.enabled.
    """
    return {
        stand_id: (
            dense_young_stand_scaling_factor(strata, config) if config.enabled else None
        )
        for stand_id, strata in strata_per_stand.items()
    }


# %% The warning


def stands_the_default_rule_would_scale(
    strata_per_stand: dict[StandID, PerSpecies[TreeStratum]],
    scaling_factors: dict[StandID, float | None],
) -> list[StandID]:
    """
    The stands that are dense young stands by the DEFAULT numbers, judged on
    the strata each stand is about to be grown from (strata_to_grow_from: its
    recorded strata, scaled by its factor when it has one). In
    strata_per_stand's order.

    This is what the tools warn about. With the option off, it lists every
    stand the option would scale with its default numbers. With the option on
    and the default numbers, it lists nothing: a scaled stand starts at its
    target, below its threshold. With the option on and other numbers, it
    lists what those numbers left denser than the defaults allow.
    """
    default_config = DenseYoungStandScalingConfig()
    return [
        stand_id
        for stand_id, strata in strata_per_stand.items()
        if dense_young_stand_scaling_factor(
            strata_to_grow_from(strata, scaling_factors[stand_id]), default_config
        )
        is not None
    ]
