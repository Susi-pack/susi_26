"""
Stand-area lookup for the optimization.

The optimization combines whole stands into project-level totals, so every
stand needs an area in hectares: SUSI's outputs are per-hectare, and
`core.build_optimization_array` multiplies each stand's aggregated target
variables by its area before the stands are summed against each other.

Areas come from the project's own `stand_data.json` (`StandData.stand_area`),
which both generating tools populate, so any project can be optimized -- not
just the one site this module used to be hardcoded for (#216).

The inputs side (`<project>/inputs/stand_data.json`) and the outputs side
(`<project>/outputs/<run_id>/`) are both reached by composing downward from
`project_dir` via `susi.io.project_layout`. Nothing here walks up from a run
folder to guess which project it belongs to.

Note that areas are read at analysis time, not recorded at run time: editing a
project's `stand_data.json` after a run re-weights that run's already-computed
outputs. Accepted, and documented, rather than defended against here.
"""

from pathlib import Path

import susi.io.load_output_data as load_output
from susi.io import project_layout
from susi.io.load_output_data import StandID
from susi.io.stand_data import (
    StandDataDocument,
    load_stand_data_document_from_json,
)


def stand_areas_for_run(project_dir: Path, run_id: str) -> dict[StandID, float]:
    """
    Return the area in hectares of every stand in one run of a project.

    The stands are the subfolders of `<project_dir>/outputs/<run_id>/`, so the
    result covers exactly the stands the optimization will iterate over, and
    their areas come from `<project_dir>/inputs/stand_data.json`.

    Raises ValueError if any of those stands has no area there; see
    `areas_from_stand_data_document`.
    """
    stand_data_document = load_stand_data_document_from_json(
        path=project_layout.stand_data_path_for_project(project_dir=project_dir)
    )

    run_dirpath = project_layout.run_dir(project_dir=project_dir, run_id=run_id)
    stand_ids = [
        StandID(stand_dirpath.name)
        for stand_dirpath in load_output.list_subdirectories_sorted(path=run_dirpath)
    ]

    return areas_from_stand_data_document(
        stand_data_document=stand_data_document, stand_ids=stand_ids
    )


def areas_from_stand_data_document(
    stand_data_document: StandDataDocument, stand_ids: list[StandID]
) -> dict[StandID, float]:
    """
    Pick the area of each of stand_ids out of an already-loaded document.

    Split out from the file reading above so it can be unit-tested without a
    real project on disk: the lookup is a pure function over the document plus
    a list of stand IDs.

    Raises ValueError, naming *every* offending stand rather than just the
    first, if a stand is absent from the document or carries no
    `stand_area`. There is deliberately no equal-area fallback: `stand_area`
    is optional in `StandData` because a source may genuinely not carry it,
    and weighting stands by invented areas would yield a plausible-looking but
    wrong Pareto front. Partial coverage raises exactly like total absence --
    mixing real and invented areas is worse than either.
    """
    areas_ha: dict[StandID, float] = {}
    missing_stands: list[StandID] = []
    stands_without_area: list[StandID] = []

    # One pass over the stands, collecting both the areas and both kinds of
    # problem, so that every offending stand can be named at once. The
    # half-filled areas_ha is never returned: any problem at all raises.
    for stand_id in stand_ids:
        stand_data = stand_data_document.stands.get(stand_id)
        if stand_data is None:
            missing_stands.append(stand_id)
        elif stand_data.stand_area is None:
            stands_without_area.append(stand_id)
        else:
            areas_ha[stand_id] = float(stand_data.stand_area)

    if missing_stands or stands_without_area:
        raise ValueError(
            _no_area_message(
                missing_stands=missing_stands,
                stands_without_area=stands_without_area,
            )
        )

    return areas_ha


def _no_area_message(
    missing_stands: list[StandID], stands_without_area: list[StandID]
) -> str:
    """
    Spell out which stands have no area, and what the user can do about it.

    The two cases are reported separately because they have different causes:
    a stand missing from the document means the run and the document disagree
    about which stands exist, while a stand with no `stand_area` means the
    source data carried no area for it.

    Each list keeps the order it was collected in, which is the run folder's
    own natural sort (stand_2 before stand_10), rather than being re-sorted
    lexicographically here.
    """
    problems = []
    if missing_stands:
        problems.append(f"not present in the project's stand data: {missing_stands}")
    if stands_without_area:
        problems.append(f"have no stand_area recorded: {stands_without_area}")

    return (
        "Cannot determine stand areas for this run. The following stands are "
        + "; ".join(problems)
        + ". The optimization weights every stand by its area, so it will not "
        "run on a partial set -- there is no equal-area fallback, because a "
        "fabricated area produces a plausible-looking but wrong Pareto front. "
        "Rerun the tool that generated stand_data.json against a source that "
        "carries stand areas (stand_area is optional in StandData by design)."
    )
