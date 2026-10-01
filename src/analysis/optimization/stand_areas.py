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


def stand_areas_for_run(run_dir: Path) -> dict[StandID, float]:
    """
    Return the area in hectares of every stand in one run of a project.

    The stands are the stand folders of `<project_dir>/outputs/<run_id>/`, as
    listed by `load_output.list_stand_folders`: the same listing
    `core.read_data` uses, so the result covers exactly the stands the
    optimization will iterate over, in the same order. Their areas come from
    `<project_dir>/inputs/stand_data.json`, where each stand is looked up by
    its folder's name as it is. A stand folder is named by its stand ID, so
    nothing is mapped, stripped or translated here.

    Raises ValueError if any of those stands has no area there; see
    `areas_from_stand_data_document`. Both frontends call this before the
    user picks target variables, so a run whose folder names do not match the
    document fails here, before the slow netcdf read.
    """
    project_dir = project_layout.project_dir_from_run_dir(run_dir)

    stand_data_document = load_stand_data_document_from_json(
        path=project_layout.stand_data_path_for_project(project_dir=project_dir)
    )
    stand_ids = [
        StandID(stand_dirpath.name)
        for stand_dirpath in load_output.list_stand_folders(run_dirpath=run_dir)
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
                document_stand_ids=list(stand_data_document.stands.keys()),
            )
        )

    return areas_ha


def _no_area_message(
    missing_stands: list[StandID],
    stands_without_area: list[StandID],
    document_stand_ids: list[StandID],
) -> str:
    """
    Spell out which stands have no area, and what the user can do about it.

    The two cases get a paragraph each, printed only when its list is
    non-empty, because they have different causes and different fixes:

    - A stand folder with no entry in the document means the run and the
      document disagree about what the stands are called. The usual cause is
      a run script that decorates the stand ID (`stand_1` for a stand keyed
      `1`), so the fix is in the run script. document_stand_ids, the
      document's own keys, are printed beside the folder names so that such a
      mismatch is visible at a glance. They are passed in rather than
      re-derived here so that this stays a pure function of what it is told.
    - A stand that is in the document but has no `stand_area` means the
      source data carried no area for it, so the fix is on the
      data-generation side.

    Each list keeps the order it was collected in, which is the run folder's
    own natural sort (stand 2 before stand 10), rather than being re-sorted
    lexicographically here.
    """
    paragraphs = ["Cannot determine stand areas for this run."]

    if missing_stands:
        paragraphs.append(
            "These stand folders of the run have no entry in the project's "
            f"stand_data.json: {missing_stands}. A stand's folder must be named "
            "by its stand ID, the key it has in stand_data.json (which has: "
            f"{_capped_listing(stand_ids=document_stand_ids)}). Check the "
            "`stand_id` the run script passes to SimulationMetaData."
        )

    if stands_without_area:
        paragraphs.append(
            f"These stands have no stand_area recorded: {stands_without_area}. "
            "Rerun the tool that generated stand_data.json against a source "
            "that carries stand areas (stand_area is optional in StandData by "
            "design)."
        )

    paragraphs.append(
        "The optimization weights every stand by its area, so it will not run "
        "on a partial set -- there is no equal-area fallback, because a "
        "fabricated area produces a plausible-looking but wrong Pareto front."
    )

    return "\n\n".join(paragraphs)


# How many of the document's stand IDs the missing-stand message prints. Ten
# is enough to show how the document names its stands, which is all the
# message needs them for.
_MAX_DOCUMENT_STAND_IDS_SHOWN = 10


def _capped_listing(stand_ids: list[StandID]) -> str:
    """
    Format stand_ids for an error message: the first few, then a count of the
    rest, so a project with hundreds of stands does not flood the message.

    Only used for the document's keys, which are shown as an example of how
    the document names its stands. The offending stands are never capped.
    """
    shown = stand_ids[:_MAX_DOCUMENT_STAND_IDS_SHOWN]
    number_not_shown = len(stand_ids) - len(shown)
    if number_not_shown == 0:
        return f"{shown}"
    return f"{shown} and {number_not_shown} more"
