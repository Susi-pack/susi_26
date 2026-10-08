"""Console progress-report formatting shared by xml_to_allometry.py and
metsakeskus_to_allometry.py: the sectioned Reading/Filtering/Writing
headings, the StandSkipped type + printer for reporting which stands
were dropped and why instead of silently vanishing, and the two dense young
stand scaling reports (which stands were scaled, and the warning).
"""

from dataclasses import dataclass

from susi.io.load_output_data import StandID
from tools.shared_allometry_tool_utils.dense_young_stand_scaling import (
    DENSE_YOUNG_STAND_SCALING_DOCS_URL,
    strata_to_grow_from,
    total_stem_count,
)
from tools.shared_allometry_tool_utils.tree_stratum import PerSpecies, TreeStratum


@dataclass(frozen=True)
class StandSkipped:
    """A stand that does not survive some filter or check, so no allometry
    file gets written for it."""

    stand_id: StandID
    reason: str


def print_section(title: str) -> None:
    """Marks one phase of main()'s reading -> filtering -> writing pipeline
    in the console output, so the phases are visually separated."""
    print()
    print(title)
    print("-" * len(title))


def print_skips(skips: list[StandSkipped], label: str) -> None:
    if not skips:
        return
    print(f"{label}: {len(skips)}")
    for skip in skips:
        print(f"  {skip.stand_id}: {skip.reason}")
    print()


def print_scaled_stands(
    strata_per_stand: dict[StandID, PerSpecies[TreeStratum]],
    scaling_factors: dict[StandID, float | None],
) -> None:
    """One line per stand dense young stand scaling scales down: its stand
    ID, its stem count as recorded and as the growth model will get it, and
    its scaling factor. strata_per_stand holds the recorded (unscaled)
    strata. Prints nothing when no stand has a factor."""
    scaled = {
        stand_id: scaling_factor
        for stand_id, scaling_factor in scaling_factors.items()
        if scaling_factor is not None
    }
    if not scaled:
        return
    print()
    print(
        f"Dense young stand scaling -- {len(scaled):,} stand(s) scaled down "
        "before the growth model runs:"
    )
    for stand_id, scaling_factor in scaled.items():
        recorded_strata = strata_per_stand[stand_id]
        # The "after" figure is read off the very strata the growth model
        # gets, not recomputed here as recorded * factor.
        scaled_strata = strata_to_grow_from(recorded_strata, scaling_factor)
        print(
            f"  {stand_id}: {total_stem_count(recorded_strata):.0f} -> "
            f"{total_stem_count(scaled_strata):.0f} stems/ha "
            f"(scaling factor {scaling_factor:.3f})"
        )


def print_dense_young_stand_warning(stand_ids: list[StandID]) -> None:
    """The warning about the stands the default numbers of dense young stand
    scaling would scale down, but that are about to be grown with more stems
    than that (see dense_young_stand_scaling.stands_the_default_rule_would_
    scale, which picks them). Only a warning: it never blocks the run. Prints
    nothing for an empty list."""
    if not stand_ids:
        return
    print()
    print(f"Warning: {len(stand_ids):,} dense young stand(s): {', '.join(stand_ids)}")
    print(
        "  These are young stands with more stems than the default limits of "
        "dense young stand scaling, and they will be grown that way. "
        "[dense_young_stand_scaling] in the config file scales such stands "
        f"down: {DENSE_YOUNG_STAND_SCALING_DOCS_URL}"
    )
