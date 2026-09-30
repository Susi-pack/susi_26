"""
Tests for the optimization's stand-area lookup.

The lookup is a pure function over a loaded `StandDataDocument` plus a list of
stand IDs, so almost everything is testable against an in-memory document.
Two tests go through the filesystem: one to pin down that the document is
really read from the path `susi.io.project_layout` computes, and one to pin
down that the optimization's two readers of a run folder (the area lookup and
the netcdf read) list its stands in the same order.
"""

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import susi.io.load_output_data as load_output
from analysis.optimization import core as opti_core
from analysis.optimization.stand_areas import (
    areas_from_stand_data_document,
    stand_areas_for_run,
)
from susi.io import project_layout
from susi.io.load_output_data import NetcdfVariablePath, StandID
from susi.io.stand_data import (
    SOURCE_CRS,
    StandData,
    StandDataDocument,
    dump_stand_data_document,
)
from susi.io.susi_parameter_model import AllometryFileAndSpecies, CanopyLayerName

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
        initial_age_per_layer={CanopyLayerName.dominant: 40.0},
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

    def test_missing_stand_message_shows_the_documents_keys_and_blames_stand_id(self):
        # The Paroninkorpi failure this was written for: the run script named
        # its folders `stand_1` while stand_data.json is keyed `1`. Seeing the
        # folder names beside the document's keys is what makes the prefix
        # visible at a glance, and the fix is in the run script's `stand_id`.
        document = _document({"1": 2.4, "2": 1.9})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document,
                stand_ids=[StandID("stand_1"), StandID("stand_2")],
            )

        message = str(error.value)
        assert "['stand_1', 'stand_2']" in message
        assert "which has: ['1', '2']" in message
        assert "`stand_id` the run script passes to SimulationMetaData" in message

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

    def test_message_for_missing_stands_alone_gives_no_rerun_the_tool_advice(self):
        # The two causes have different fixes. Rerunning the generating tool
        # does nothing for a folder-name mismatch, so that advice must not be
        # printed when every stand the document does know has an area.
        document = _document({"1": 2.4})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document, stand_ids=[StandID("stand_1")]
            )

        message = str(error.value)
        assert "have no entry in the project's stand_data.json" in message
        assert "have no stand_area recorded" not in message
        assert "Rerun the tool" not in message

    def test_message_for_stands_without_area_alone_does_not_blame_stand_id(self):
        document = _document({"1": None})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document, stand_ids=[StandID("1")]
            )

        message = str(error.value)
        assert "have no stand_area recorded: ['1']" in message
        assert "have no entry in the project's stand_data.json" not in message
        assert "SimulationMetaData" not in message

    def test_message_with_both_problems_has_both_paragraphs(self):
        document = _document({"1": None, "2": 1.9})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document,
                stand_ids=[StandID("1"), StandID("2"), StandID("stand_3")],
            )

        message = str(error.value)
        assert "have no entry in the project's stand_data.json: ['stand_3']" in message
        assert "have no stand_area recorded: ['1']" in message

    def test_message_always_says_there_is_no_equal_area_fallback(self):
        # Whichever problem it is, the user must learn why the optimization
        # refuses instead of carrying on with the stands it could resolve.
        document = _document({"1": 2.4})

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document, stand_ids=[StandID("stand_1")]
            )

        assert "there is no equal-area fallback" in str(error.value)

    def test_message_caps_the_documents_keys_but_names_every_missing_folder(self):
        # A large project must not flood the message with its whole stand
        # list: the first ten keys are enough to show how the document names
        # its stands. The offending folders are a different matter: every one
        # of them is named, however many there are.
        document = _document({str(number): 1.0 for number in range(1, 26)})
        missing_stands = [StandID(f"stand_{number}") for number in range(1, 26)]

        with pytest.raises(ValueError) as error:
            areas_from_stand_data_document(
                stand_data_document=document, stand_ids=missing_stands
            )

        message = str(error.value)
        assert (
            "which has: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] "
            "and 15 more" in message
        )
        assert "'11'" not in message
        for missing_stand in missing_stands:
            assert f"'{missing_stand}'" in message

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
    The file-backed test of the area lookup: everything above is pure.

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


def _write_one_scenario_stand_folder(run_dirpath: Path, stand_id: str) -> None:
    """
    Write the smallest stand folder the output loader can read: one scenario
    folder holding `metadata.json`, `params.json` and a one-variable netcdf.
    """
    scenario_dirpath = run_dirpath / stand_id / "scenario_a"
    scenario_dirpath.mkdir(parents=True)

    netcdf_filepath = scenario_dirpath / "output.nc"
    with netCDF4.Dataset(netcdf_filepath, "w") as nc:
        nc.createDimension("time", 2)
        nc.createDimension("column", 3)
        variable = nc.createVariable("volume", "f8", ("time", "column"))
        variable.units = "m3/ha"
        variable[:] = np.ones((2, 3))

    (scenario_dirpath / "metadata.json").write_text(
        json.dumps(
            {
                "stand_id": stand_id,
                "scenario_id": "scenario_a",
                "netcdf_output_filepath": str(netcdf_filepath),
                "timestamp_start": "2020-01-01T00:00:00",
                "timestamp_end": "2020-12-31T00:00:00",
            }
        )
    )
    (scenario_dirpath / "params.json").write_text(json.dumps({}))


def test_the_netcdf_read_and_the_area_lookup_list_a_runs_stands_in_the_same_order(
    tmp_path,
):
    """
    The optimization reads a run folder twice: `stand_areas_for_run` for the
    areas, `core.read_data` for the netcdfs. Both must list the same stands in
    the same order, the natural sort of the stand IDs, whatever order the
    filesystem returns the folders in.
    """
    # Enough stands, with one- and two-digit IDs, that neither the
    # filesystem's own order nor a lexicographic sort ("10" before "2") is
    # likely to coincide with the natural one by accident.
    stand_ids = ["20", "10", "3", "11", "2", "1"]
    natural_order = [
        StandID(stand_id) for stand_id in ["1", "2", "3", "10", "11", "20"]
    ]

    project_dir = tmp_path / "some_project"
    stand_data_path = project_layout.stand_data_path_for_project(
        project_dir=project_dir
    )
    stand_data_path.parent.mkdir(parents=True)
    dump_stand_data_document(
        output_path=stand_data_path,
        document=_document(
            {stand_id: 1.5 for stand_id in stand_ids},
            allometry_dir=stand_data_path.parent / "allometry",
        ),
    )

    run_dirpath = project_layout.run_dir(project_dir=project_dir, run_id="run_a")
    for stand_id in stand_ids:
        _write_one_scenario_stand_folder(run_dirpath=run_dirpath, stand_id=stand_id)

    areas_ha = stand_areas_for_run(project_dir=project_dir, run_id="run_a")
    data_store = opti_core.read_data(
        run_dirpath=run_dirpath,
        variable_info={
            NetcdfVariablePath("/volume"): opti_core.TargetVariableProperties(
                aggregation_function=load_output.NetcdfVariableArray.mean_of_all_values,
                invert_optimization=False,
            )
        },
    )

    assert list(areas_ha.keys()) == natural_order
    assert data_store.stands == natural_order
    assert list(data_store.scenarios.keys()) == natural_order
