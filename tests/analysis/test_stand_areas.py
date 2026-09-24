"""
Tests for the optimization's stand-area lookup.

The lookup is a pure function over a loaded `StandDataDocument` plus a list of
stand IDs, so almost everything is testable against an in-memory document.
Exactly one test goes through the filesystem, to pin down that the document is
really read from the path `susi.io.project_layout` computes.
"""

from pathlib import Path

import pytest

from analysis.optimization.stand_areas import (
    areas_from_stand_data_document,
    stand_areas_for_run,
)
import susi.io.project_layout as project_layout
from susi.io.load_output_data import StandID
from susi.io.susi_parameter_model import AllometryFileAndSpecies, CanopyLayerName
from susi.io.stand_data import (
    SOURCE_CRS,
    StandData,
    StandDataDocument,
    dump_stand_data_document,
)


# Where the stands' allometry CSVs notionally live. Never written: reading
# stand areas doesn't touch them. Absolute, as a StandDataDocument's paths
# are in memory (docs/adr/0005).
_ALLOMETRY_DIR = Path("/project/inputs/allometry")


def _stand_data(
    stand_area: float | None, allometry_dir: Path = _ALLOMETRY_DIR
) -> StandData:
    """A StandData carrying stand_area, with plausible values for the rest."""
    return StandData(
        site_fertility_class=3,
        allometry_file_per_layer={
            CanopyLayerName.dominant: AllometryFileAndSpecies(
                file_path=allometry_dir / "pines.csv", species_id=1
            )
        },
        x_ykj=339,
        y_ykj=6675,
        stand_area=stand_area,
    )


def _document(
    areas_by_stand_id: dict[str, float | None], allometry_dir: Path = _ALLOMETRY_DIR
) -> StandDataDocument:
    return StandDataDocument(
        crs=SOURCE_CRS,
        altitude=100.0,
        ddy=1200.0,
        stands={
            StandID(stand_id): _stand_data(
                stand_area=stand_area, allometry_dir=allometry_dir
            )
            for stand_id, stand_area in areas_by_stand_id.items()
        },
    )


class TestAreasFromStandDataDocument:
    def test_picks_the_area_of_each_requested_stand(self):
        document = _document({"stand-1": 2.4, "stand-2": 1.9, "stand-10": 0.5})

        areas_ha = areas_from_stand_data_document(
            stand_data_document=document,
            stand_ids=[StandID("stand-1"), StandID("stand-10")],
        )

        assert areas_ha == {StandID("stand-1"): 2.4, StandID("stand-10"): 0.5}

    def test_stands_the_document_does_not_know_raise_naming_all_of_them(self):
        # Every offending stand at once: a user fixing their data one
        # error message at a time would otherwise rerun the optimization
        # once per missing stand.
        document = _document({"stand-1": 2.4})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document,
                stand_ids=[StandID("stand-1"), StandID("stand-7"), StandID("stand-9")],
            )

        assert "stand-7" in str(error.value)
        assert "stand-9" in str(error.value)

    def test_stand_with_no_recorded_area_raises_naming_all_of_them(self):
        document = _document({"stand-1": None, "stand-2": 1.9, "stand-3": None})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document,
                stand_ids=[StandID("stand-1"), StandID("stand-2"), StandID("stand-3")],
            )

        assert "stand-1" in str(error.value)
        assert "stand-3" in str(error.value)

    def test_error_message_says_to_rerun_the_generating_tool(self):
        # stand_area is optional in StandData by design, so the fix is on the
        # data-generation side, not here -- the message has to say so.
        document = _document({"stand-1": None})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document, stand_ids=[StandID("stand-1")]
            )

        assert "Rerun the tool that generated stand_data.json" in str(error.value)

    def test_partial_coverage_raises_rather_than_returning_what_it_found(self):
        # No equal-area fallback and no partial result: mixing real and
        # invented areas yields a plausible-looking but wrong Pareto front.
        document = _document({"stand-1": 2.4, "stand-2": None})

        with pytest.raises(ValueError):
            areas_from_stand_data_document(
                stand_data_document=document,
                stand_ids=[StandID("stand-1"), StandID("stand-2")],
            )


def test_stand_areas_for_run_reads_the_document_from_the_layouts_path(tmp_path):
    """
    The one file-backed test: everything else above is pure.

    It pins down only the wiring -- that the document is loaded from the path
    `project_layout` computes for the project, and that the stand IDs come
    from the run folder's subdirectories. The lookup's own behaviour is
    covered in-memory.
    """
    project_dir = tmp_path / "some_project"

    stand_data_path = project_layout.stand_data_path_for_project(
        project_dir=project_dir
    )
    stand_data_path.parent.mkdir(parents=True)
    # A third stand the document knows about but this run did not simulate:
    # the run folder, not the document, decides which keys come back.
    dump_stand_data_document(
        output_path=stand_data_path,
        document=_document(
            {"stand-1": 2.4, "stand-2": 1.9, "stand-3": 7.7},
            allometry_dir=stand_data_path.parent / "allometry",
        ),
    )

    run_dirpath = project_layout.run_dir(project_dir=project_dir, run_id="run_a")
    for stand_id in ["stand-1", "stand-2"]:
        (run_dirpath / stand_id).mkdir(parents=True)

    areas_ha = stand_areas_for_run(project_dir=project_dir, run_id="run_a")

    assert areas_ha == {StandID("stand-1"): 2.4, StandID("stand-2"): 1.9}
