import argparse
import ast
import dataclasses
import inspect
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pydantic
import pytest
import shapely
from hypothesis import assume, given
from hypothesis import strategies as st
from shapely.geometry import MultiPolygon, Point, Polygon

from susi.io.load_output_data import StandID
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
)
from susi.io import susi_parameter_model
from tools.shared_allometry_tool_utils import (
    allometry_generation_defaults,
    cli_paths,
    growth_and_yield_table,
    input_validation,
    print_formatting,
    shared_utils,
    stand_data,
    tree_stratum,
)

# %% input_validation.make_existing_file_validator


@pytest.fixture
def dummy_xml_file(tmp_path):
    path = tmp_path / "stand.xml"
    path.write_text("<ForestPropertyData></ForestPropertyData>")
    return path


def test_make_existing_file_validator_accepts_matching_suffix(dummy_xml_file):
    validator = input_validation.make_existing_file_validator(".xml")
    assert validator(str(dummy_xml_file)) == dummy_xml_file


def test_make_existing_file_validator_rejects_missing_file(tmp_path):
    validator = input_validation.make_existing_file_validator(".xml")
    with pytest.raises(argparse.ArgumentTypeError):
        validator(str(tmp_path / "does_not_exist.xml"))


def test_make_existing_file_validator_rejects_a_directory(tmp_path):
    validator = input_validation.make_existing_file_validator(".xml")
    with pytest.raises(argparse.ArgumentTypeError):
        validator(str(tmp_path))


def test_make_existing_file_validator_rejects_wrong_suffix(tmp_path):
    validator = input_validation.make_existing_file_validator(".xml")
    wrong = tmp_path / "stand.gpkg"
    wrong.write_bytes(b"")
    with pytest.raises(argparse.ArgumentTypeError):
        validator(str(wrong))


def test_make_existing_file_validator_is_reusable_for_different_suffixes(tmp_path):
    # Same factory, two independent validators -- what replaces valid_xml_path/
    # valid_gpkg_path/valid_config_path each being hand-written separately.
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("")
    xml_validator = input_validation.make_existing_file_validator(".xml")
    toml_validator = input_validation.make_existing_file_validator(".toml")
    assert toml_validator(str(toml_path)) == toml_path
    with pytest.raises(argparse.ArgumentTypeError):
        xml_validator(str(toml_path))


# %% input_validation.valid_existing_directory


def test_valid_existing_directory_accepts_an_existing_directory(tmp_path):
    assert input_validation.valid_existing_directory(str(tmp_path)) == tmp_path


@pytest.mark.parametrize("bad_value", ["", "   "])
def test_valid_existing_directory_rejects_empty_value(bad_value):
    with pytest.raises(argparse.ArgumentTypeError):
        input_validation.valid_existing_directory(bad_value)


def test_valid_existing_directory_rejects_missing_directory(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        input_validation.valid_existing_directory(str(tmp_path / "nope"))


def test_valid_existing_directory_rejects_a_file(tmp_path):
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("x")
    with pytest.raises(argparse.ArgumentTypeError):
        input_validation.valid_existing_directory(str(file_path))


# %% input_validation.out_of_range_message


@pytest.mark.parametrize(
    "value,min_value,max_value",
    [
        (0, 0, 1000),  # lower bound, inclusive
        (1000, 0, 1000),  # upper bound, inclusive
        (500, 0, 1000),  # comfortably inside
    ],
)
def test_out_of_range_message_within_range_returns_none(value, min_value, max_value):
    assert (
        input_validation.out_of_range_message("altitude", value, min_value, max_value)
        is None
    )


def test_out_of_range_message_below_range_returns_message():
    message = input_validation.out_of_range_message("altitude", -1, 0, 1000)
    assert message is not None
    assert "altitude" in message
    assert "-1" in message


def test_out_of_range_message_above_range_returns_message():
    message = input_validation.out_of_range_message("ddy", 2001, 500, 2000)
    assert message is not None
    assert "ddy" in message
    assert "2001" in message


# %% input_validation.validate_altitude_ddy


def _parser():
    return argparse.ArgumentParser()


def test_validate_altitude_ddy_accepts_in_range_values_without_warning(capsys):
    input_validation.validate_altitude_ddy(_parser(), 150.0, 1200.0, False)
    assert capsys.readouterr().out == ""


def test_validate_altitude_ddy_blocks_out_of_range_by_default(capsys):
    with pytest.raises(SystemExit):
        input_validation.validate_altitude_ddy(_parser(), 1500.0, 1200.0, False)
    stderr = capsys.readouterr().err
    assert "altitude" in stderr
    assert "--allow-out-of-range-values" in stderr


def test_validate_altitude_ddy_reports_both_violations_together(capsys):
    with pytest.raises(SystemExit):
        input_validation.validate_altitude_ddy(_parser(), 1500.0, 2500.0, False)
    stderr = capsys.readouterr().err
    assert "1500" in stderr
    assert "2500" in stderr


def test_validate_altitude_ddy_allows_out_of_range_with_override(capsys):
    input_validation.validate_altitude_ddy(_parser(), 1500.0, 1200.0, True)
    assert "Warning" in capsys.readouterr().out


def test_validate_altitude_ddy_rejects_nan_even_with_override(capsys):
    with pytest.raises(SystemExit):
        input_validation.validate_altitude_ddy(_parser(), float("nan"), 1200.0, True)
    assert "altitude" in capsys.readouterr().err


# %% input_validation.validate_x_y_ykj
#
# Pure bounds logic over the YKJ grid units point_to_ykj produces, applied
# by new_growth_allometry.py to whichever pair of coordinates a run
# resolved -- converted from the config's ETRS-TM35FIN x/y in standalone
# mode, read straight off a StandData in sourced mode. The validator
# itself never learns which, which is exactly why one set of tests covers
# both.
#
# The bounds themselves belong to shared_utils, next to point_to_ykj, and
# StandData's x_ykj/y_ykj fields use the same ones -- see
# test_stand_data_bounds_come_from_the_shared_ykj_range below.

IN_RANGE_X_YKJ = 338  # a real Finnish stand's easting, 10 km units
IN_RANGE_Y_YKJ = 7042  # ... and its northing, 1 km units


def test_validate_x_y_ykj_accepts_in_range_values_without_warning(capsys):
    input_validation.validate_x_y_ykj(_parser(), IN_RANGE_X_YKJ, IN_RANGE_Y_YKJ, False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "x_ykj, y_ykj",
    [
        (shared_utils.X_YKJ_MIN, shared_utils.Y_YKJ_MIN),
        (shared_utils.X_YKJ_MAX, shared_utils.Y_YKJ_MAX),
    ],
)
def test_validate_x_y_ykj_accepts_its_own_bounds(x_ykj, y_ykj, capsys):
    # The bounds are inclusive, same as the altitude/ddy ones.
    input_validation.validate_x_y_ykj(_parser(), x_ykj, y_ykj, False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "x_ykj, y_ykj, expected_name",
    [
        (shared_utils.X_YKJ_MIN - 1, IN_RANGE_Y_YKJ, "x_ykj"),
        (shared_utils.X_YKJ_MAX + 1, IN_RANGE_Y_YKJ, "x_ykj"),
        (IN_RANGE_X_YKJ, shared_utils.Y_YKJ_MIN - 1, "y_ykj"),
        (IN_RANGE_X_YKJ, shared_utils.Y_YKJ_MAX + 1, "y_ykj"),
    ],
)
def test_validate_x_y_ykj_blocks_just_outside_each_bound(
    x_ykj, y_ykj, expected_name, capsys
):
    with pytest.raises(SystemExit):
        input_validation.validate_x_y_ykj(_parser(), x_ykj, y_ykj, False)
    stderr = capsys.readouterr().err
    assert expected_name in stderr
    assert "--allow-out-of-range-values" in stderr


def test_validate_x_y_ykj_reports_both_violations_together(capsys):
    with pytest.raises(SystemExit):
        input_validation.validate_x_y_ykj(_parser(), 0, 0, False)
    stderr = capsys.readouterr().err
    assert "x_ykj" in stderr
    assert "y_ykj" in stderr


def test_validate_x_y_ykj_allows_out_of_range_with_override(capsys):
    input_validation.validate_x_y_ykj(_parser(), 0, 0, True)
    assert "Warning" in capsys.readouterr().out


@given(
    x_ykj=st.integers(
        min_value=shared_utils.X_YKJ_MIN, max_value=shared_utils.X_YKJ_MAX
    ),
    y_ykj=st.integers(
        min_value=shared_utils.Y_YKJ_MIN, max_value=shared_utils.Y_YKJ_MAX
    ),
)
def test_validate_x_y_ykj_accepts_anything_inside_the_box(x_ykj, y_ykj):
    input_validation.validate_x_y_ykj(_parser(), x_ykj, y_ykj, False)


@given(
    x_ykj=st.integers(min_value=-100_000, max_value=100_000),
    y_ykj=st.integers(min_value=-100_000, max_value=100_000),
)
def test_validate_x_y_ykj_rejects_anything_outside_the_box(x_ykj, y_ykj):
    assume(
        not (shared_utils.X_YKJ_MIN <= x_ykj <= shared_utils.X_YKJ_MAX)
        or not (shared_utils.Y_YKJ_MIN <= y_ykj <= shared_utils.Y_YKJ_MAX)
    )
    with pytest.raises(SystemExit):
        input_validation.validate_x_y_ykj(_parser(), x_ykj, y_ykj, False)


# %% input_validation.load_toml_config
#
# check_config_fields (the manual presence/unknown-field checker
# load_toml_config's callers used to run before Pydantic did that job
# instead) is gone -- see xml_to_allometry.py's XmlConfig,
# metsakeskus_to_allometry.py's ExtractionConfig, and
# new_growth_allometry.py's NewGrowthConfig, all StrictFrozenModel now.
# _DummyConfig below stays a plain dataclass: it only needs to be some
# `parse`-constructible type, unrelated to what check_config_fields used to
# validate against.


@dataclass(frozen=True)
class _DummyConfig:
    required_field: int
    optional_field: int = 0


def test_load_toml_config_reads_the_file_and_hands_the_raw_dict_to_parse(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("required_field = 3\n")

    config = input_validation.load_toml_config(
        config_path, lambda raw: _DummyConfig(**raw)
    )

    assert config == _DummyConfig(required_field=3)


# %% allometry_generation_defaults.AllometryGenerationDefaults
#
# ExtractionConfig (metsakeskus_to_allometry.py), XmlConfig
# (xml_to_allometry.py), and NewGrowthSourcedConfig (new_growth_allometry.py)
# each subclass this now instead of redeclaring the same four fields; their
# own test files already cover that each config resolves these defaults
# unchanged (test_..._config_applies_defaults). This just covers the
# defaults model itself.


def test_allometry_generation_defaults_resolve_to_20_5_80_5():
    defaults = allometry_generation_defaults.AllometryGenerationDefaults()
    assert (
        defaults.n_trees,
        defaults.start_year,
        defaults.end_year,
        defaults.step_years,
    ) == (20, 5, 80, 5)


def test_allometry_generation_defaults_rejects_unknown_field():
    with pytest.raises(pydantic.ValidationError, match="typo_field"):
        allometry_generation_defaults.AllometryGenerationDefaults.model_validate(
            {"typo_field": 1}
        )


def test_allometry_generation_defaults_is_frozen():
    defaults = allometry_generation_defaults.AllometryGenerationDefaults()
    with pytest.raises(Exception):  # noqa: B017 -- pydantic's frozen-model error
        setattr(defaults, "n_trees", 5)  # noqa: B010 -- setattr, not `.` access, to keep this a runtime-only check


# %% cli_paths.check_output_dir_available


def test_check_output_dir_available_allows_a_free_folder(tmp_path):
    cli_paths.check_output_dir_available(tmp_path / "allometry", _parser())


def test_check_output_dir_available_refuses_an_existing_folder(tmp_path, capsys):
    output_dir = tmp_path / "allometry"
    output_dir.mkdir()
    with pytest.raises(SystemExit):
        cli_paths.check_output_dir_available(output_dir, _parser())
    assert "already exists" in capsys.readouterr().err


# %% cli_paths.resolve_config_path
#
# config_filename is passed in by each tool rather than shared: several
# tools' configs can live in one project's inputs/ folder, so they must not
# all default to the same name. Parametrized over both names in use to keep
# the helper honestly filename-agnostic.

CONFIG_FILENAMES = ["config.toml", "new_growth_config.toml"]


def _project_with_inputs(tmp_path) -> Path:
    """A project folder shaped the way the tools now expect: a root with an
    inputs/ folder under it, which is where the config file is looked up."""
    project_dir = tmp_path / "myproject"
    (project_dir / "inputs").mkdir(parents=True)
    return project_dir


@pytest.mark.parametrize("config_filename", CONFIG_FILENAMES)
def test_resolve_config_path_prefers_explicit_config(tmp_path, config_filename):
    explicit = tmp_path / "somewhere_else.toml"
    explicit.write_text("")
    project_dir = _project_with_inputs(tmp_path)
    (project_dir / "inputs" / config_filename).write_text("")

    assert (
        cli_paths.resolve_config_path(
            explicit, project_dir, _parser(), config_filename
        )
        == explicit
    )


@pytest.mark.parametrize("config_filename", CONFIG_FILENAMES)
def test_resolve_config_path_defaults_to_the_given_name_in_project_inputs(
    tmp_path, config_filename
):
    project_dir = _project_with_inputs(tmp_path)
    default_config = project_dir / "inputs" / config_filename
    default_config.write_text("")

    assert (
        cli_paths.resolve_config_path(
            None, project_dir, _parser(), config_filename
        )
        == default_config
    )


@pytest.mark.parametrize("config_filename", CONFIG_FILENAMES)
def test_resolve_config_path_ignores_a_config_loose_in_the_project_root(
    tmp_path, config_filename
):
    # --project-dir is the project root, not the folder the config sits in.
    # A file left directly in the root is not the default location, and
    # picking it up would quietly reinstate the old layout.
    project_dir = _project_with_inputs(tmp_path)
    (project_dir / config_filename).write_text("")

    with pytest.raises(SystemExit):
        cli_paths.resolve_config_path(
            None, project_dir, _parser(), config_filename
        )


@pytest.mark.parametrize("config_filename", CONFIG_FILENAMES)
def test_resolve_config_path_errors_when_default_is_missing(
    tmp_path, capsys, config_filename
):
    project_dir = _project_with_inputs(tmp_path)
    with pytest.raises(SystemExit):
        cli_paths.resolve_config_path(
            None, project_dir, _parser(), config_filename
        )
    stderr = capsys.readouterr().err
    assert str(project_dir / "inputs" / config_filename) in stderr
    assert "--config" in stderr


def test_resolve_config_path_ignores_another_tools_config_file(tmp_path, capsys):
    # The whole point of the parameter: new_growth_allometry.py must not
    # pick up the config.toml that xml_to_allometry.py left in the same
    # project's inputs/ folder.
    project_dir = _project_with_inputs(tmp_path)
    (project_dir / "inputs" / "config.toml").write_text("")

    with pytest.raises(SystemExit):
        cli_paths.resolve_config_path(
            None, project_dir, _parser(), "new_growth_config.toml"
        )
    assert "new_growth_config.toml" in capsys.readouterr().err


# %% cli_paths.finalize_cli_config
#
# The validate-then-resolve-output-dir tail every tool's parse_CLI_arguments
# repeats once its own config is loaded: metsakeskus_to_allometry.py's and
# xml_to_allometry.py's altitude/ddy come straight off their config;
# new_growth_allometry.py passes its *resolved* SiteInputs values instead
# (see that tool's own parse_CLI_arguments) -- this helper doesn't care
# which, it only ever sees two plain floats.


def test_finalize_cli_config_returns_the_allometry_dir_when_everything_is_fine(
    tmp_path,
):
    output_dir = cli_paths.finalize_cli_config(
        _parser(), 150.0, 1200.0, tmp_path, False
    )
    assert output_dir == tmp_path / "inputs" / "allometry"
    assert not output_dir.exists()  # creating it is main()'s job, not this one's


def test_finalize_cli_config_blocks_out_of_range_altitude_by_default(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli_paths.finalize_cli_config(_parser(), 1500.0, 1200.0, tmp_path, False)
    assert "altitude" in capsys.readouterr().err


def test_finalize_cli_config_allows_out_of_range_with_override(tmp_path, capsys):
    cli_paths.finalize_cli_config(_parser(), 1500.0, 1200.0, tmp_path, True)
    assert "Warning" in capsys.readouterr().out


def test_finalize_cli_config_refuses_an_existing_output_dir(tmp_path, capsys):
    (tmp_path / "inputs" / "allometry").mkdir(parents=True)
    with pytest.raises(SystemExit):
        cli_paths.finalize_cli_config(_parser(), 150.0, 1200.0, tmp_path, False)
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "altitude, ddy",
    [
        (input_validation.ALTITUDE_MIN, input_validation.DDY_MIN),
        (input_validation.ALTITUDE_MAX, input_validation.DDY_MAX),
        (150.0, 1200.0),
    ],
)
def test_finalize_cli_config_accepts_anything_in_range(tmp_path, altitude, ddy):
    # tmp_path fresh per parametrize case -- avoids the function-scoped-
    # fixture/@given interaction a Hypothesis property here would run into,
    # since finalize_cli_config's output_dir check depends on the
    # filesystem, not just its own arguments.
    output_dir = cli_paths.finalize_cli_config(
        _parser(), altitude, ddy, tmp_path, False
    )
    assert output_dir == tmp_path / "inputs" / "allometry"


# %% print_formatting


def test_print_section_prints_title_and_underline(capsys):
    print_formatting.print_section("Reading")
    printed = capsys.readouterr().out
    assert "Reading" in printed
    assert "-------" in printed  # len("Reading") dashes


def test_print_skips_does_nothing_for_an_empty_list(capsys):
    print_formatting.print_skips([], "Skipped")
    assert capsys.readouterr().out == ""


def test_print_skips_reports_stand_id_and_reason(capsys):
    skips = [
        print_formatting.StandSkipped(stand_id=StandID("1"), reason="no TreeStrata")
    ]
    print_formatting.print_skips(skips, "Skipped")
    printed = capsys.readouterr().out
    assert "Skipped: 1" in printed
    assert "1: no TreeStrata" in printed


# %% shared_utils.point_to_ykj


def test_point_to_ykj_returns_plausible_helsinki_area_coordinates():
    # A point roughly at Helsinki, in ETRS-TM35FIN (EPSG:3067).
    x, y = shared_utils.point_to_ykj(385000, 6685000)
    # Real YKJ eastings in southern Finland are ~3.3-3.4 million metres --
    # scaled by /10000 that's low-to-mid 300s, matching Helsinki's well-known
    # YKJ coordinates (~3387000, 6673000).
    assert 330 < x < 345
    assert 6650 < y < 6700


def test_point_to_ykj_reuses_one_transformer_across_calls():
    # Building a pyproj.Transformer is comparatively expensive; point_to_ykj
    # runs once per stand, so it must not rebuild one every call.
    shared_utils._ykj_transformer.cache_clear()
    shared_utils.point_to_ykj(385000, 6685000)
    shared_utils.point_to_ykj(385000, 6685000)
    assert shared_utils._ykj_transformer.cache_info().hits >= 1


# %% shared_utils.centroid_to_ykj
#
# Moved here from metsakeskus_to_allometry.py (ticket 22), now that
# xml_to_allometry.py takes its YKJ grid location from the centroid too.


def test_centroid_to_ykj_returns_plausible_helsinki_area_coordinates():
    # A small disc roughly at Helsinki, in ETRS-TM35FIN (EPSG:3067).
    polygon = Point(385000, 6685000).buffer(50)
    x, y = shared_utils.centroid_to_ykj(polygon)
    assert 330 < x < 345
    assert 6650 < y < 6700


def test_centroid_to_ykj_uses_the_centroid_not_the_first_vertex():
    # 20 km wide: the first vertex and the centroid sit 10 km apart in
    # easting, one whole YKJ easting unit, so the two choices can't agree.
    polygon = Polygon(
        [(380_000, 6_685_000), (400_000, 6_685_000), (400_000, 6_686_000), (380_000, 6_686_000)]
    )
    first_vertex_x, _ = shared_utils.point_to_ykj(*polygon.exterior.coords[0])
    centroid_x, _ = shared_utils.point_to_ykj(polygon.centroid.x, polygon.centroid.y)
    assert first_vertex_x != centroid_x

    assert shared_utils.centroid_to_ykj(polygon) == shared_utils.point_to_ykj(
        polygon.centroid.x, polygon.centroid.y
    )


def test_centroid_to_ykj_reuses_one_transformer_across_calls():
    # It runs once per stand, so it must not rebuild a Transformer every call.
    shared_utils._ykj_transformer.cache_clear()
    polygon = Point(385000, 6685000).buffer(50)
    shared_utils.centroid_to_ykj(polygon)
    shared_utils.centroid_to_ykj(polygon)
    assert shared_utils._ykj_transformer.cache_info().hits >= 1


# %% shared_utils.require_source_crs


def test_require_source_crs_accepts_the_source_crs():
    shared_utils.require_source_crs(shared_utils.SOURCE_CRS, where="test input")


@pytest.mark.parametrize("declared", ["EPSG:4326", "EPSG:2393", None])
def test_require_source_crs_rejects_anything_else_naming_the_source(declared):
    with pytest.raises(ValueError, match="test input.*EPSG:3067"):
        shared_utils.require_source_crs(declared, where="test input")


# %% tree_stratum.TreeStratum / ZERO_STRATUM


def test_tree_stratum_constructs_from_valid_values():
    stratum = tree_stratum.TreeStratum(
        age=30, basal_area=15.0, stem_count=400, mean_diameter=20.0, mean_height=18.0
    )
    assert stratum.age == 30
    assert stratum.basal_area == 15.0


def test_tree_stratum_rejects_a_malformed_value():
    with pytest.raises(pydantic.ValidationError):
        tree_stratum.TreeStratum(
            age="not-a-number",  # ty: ignore[invalid-argument-type]
            basal_area=15.0,
            stem_count=400,
            mean_diameter=20.0,
            mean_height=18.0,
        )


def test_tree_stratum_is_frozen():
    stratum = tree_stratum.TreeStratum(
        age=30, basal_area=15.0, stem_count=400, mean_diameter=20.0, mean_height=18.0
    )
    with pytest.raises(Exception):  # noqa: B017 -- pydantic's frozen-dataclass error
        stratum.age = 31  # noqa: B010  # ty: ignore[invalid-assignment]


def test_tree_stratum_is_hashable():
    stratum = tree_stratum.TreeStratum(
        age=30, basal_area=15.0, stem_count=400, mean_diameter=20.0, mean_height=18.0
    )
    assert len({stratum, tree_stratum.ZERO_STRATUM}) == 2


def test_zero_stratum_is_the_all_zero_value():
    assert tree_stratum.ZERO_STRATUM == tree_stratum.TreeStratum(
        age=0, basal_area=0.0, stem_count=0, mean_diameter=0.0, mean_height=0.0
    )


def test_tree_stratum_recurses_correctly_under_dataclasses_asdict():
    # This is exactly why TreeStratum is a pydantic *dataclass* rather than a
    # BaseModel: metsakeskus_to_allometry.py's informational JSON dump relies
    # on dataclasses.asdict() walking a plain dataclass that nests a
    # TreeStratum, and getting a plain dict back for it -- not a TreeStratum
    # instance untouched.
    @dataclass(frozen=True)
    class _Wrapper:
        stratum: tree_stratum.TreeStratum

    wrapper = _Wrapper(stratum=tree_stratum.ZERO_STRATUM)
    data = dataclasses.asdict(wrapper)
    assert data == {
        "stratum": {
            "age": 0,
            "basal_area": 0.0,
            "stem_count": 0,
            "mean_diameter": 0.0,
            "mean_height": 0.0,
        }
    }


# %% tree_stratum.PerSpecies
#
# Promoted here from metsakeskus_to_allometry.py (ticket 12) -- xml_to_
# allometry.py and new_growth_allometry.py now use it too, in place of their
# previous tuple[TreeStratum, TreeStratum, TreeStratum] convention.


def test_per_species_holds_one_value_per_species_slot():
    per_species = tree_stratum.PerSpecies(pine=1, spruce=2, deciduous=3)
    assert (per_species.pine, per_species.spruce, per_species.deciduous) == (1, 2, 3)


def test_per_species_is_frozen():
    per_species = tree_stratum.PerSpecies(
        pine=tree_stratum.ZERO_STRATUM,
        spruce=tree_stratum.ZERO_STRATUM,
        deciduous=tree_stratum.ZERO_STRATUM,
    )
    with pytest.raises(Exception):  # noqa: B017 -- stdlib frozen-dataclass error
        setattr(per_species, "pine", tree_stratum.ZERO_STRATUM)  # noqa: B010 -- setattr, not `.` access, to keep this a runtime-only check


# %% growth_and_yield_table.build_growth_and_yield_table
#
# Shared by all three tools now (ticket 12): metsakeskus_to_allometry.py's
# own build_growth_and_yield_table isolates a layer first (isolate_species_
# layer) and delegates here; xml_to_allometry.py and new_growth_allometry.py
# call this directly with their full, never-isolated PerSpecies. Species
# isolation itself is the caller's job -- this function never isolates
# anything, so both shapes are exercised here directly, against in-memory
# PerSpecies fixtures (no real CSV/growth-model output is asserted on
# beyond basic sanity -- that's read_allometry_info_from_csv round-trip
# territory, covered per-tool).


def _one_species_stratum(stem_count: float) -> tree_stratum.TreeStratum:
    return tree_stratum.TreeStratum(
        age=25, basal_area=8.0, stem_count=stem_count, mean_diameter=14.0, mean_height=11.0
    )


@pytest.mark.parametrize("active_field", ["pine", "spruce", "deciduous"])
def test_build_growth_and_yield_table_uses_whichever_species_slot_is_populated(
    active_field,
):
    # An already-isolated single-species PerSpecies -- the shape
    # metsakeskus_to_allometry.py's isolate_species_layer produces. Confirms
    # the deciduous slot feeds the table exactly like pine/spruce do:
    # nothing here relabels it as "birch" or otherwise treats it specially.
    stratum = _one_species_stratum(stem_count=500)
    zero = tree_stratum.ZERO_STRATUM
    strata = tree_stratum.PerSpecies(
        pine=stratum if active_field == "pine" else zero,
        spruce=stratum if active_field == "spruce" else zero,
        deciduous=stratum if active_field == "deciduous" else zero,
    )

    table = growth_and_yield_table.build_growth_and_yield_table(
        strata=strata,
        fertility_class=3,
        x_ykj=339,
        y_ykj=6675,
        altitude=100.0,
        ddy=1200.0,
        n_trees=5,
        start_year=5,
        end_year=10,
        step_years=5,
    )
    assert len(table) > 0
    assert table["N"].iloc[0] == pytest.approx(500, rel=0.1)
    assert table["Age"].iloc[0] == 25


def test_build_growth_and_yield_table_combines_all_three_populated_species():
    # A normal, never-isolated 3-species mix -- the shape xml_to_allometry.py
    # and new_growth_allometry.py always pass.
    strata = tree_stratum.PerSpecies(
        pine=_one_species_stratum(300),
        spruce=_one_species_stratum(250),
        deciduous=_one_species_stratum(200),
    )

    table = growth_and_yield_table.build_growth_and_yield_table(
        strata=strata,
        fertility_class=3,
        x_ykj=339,
        y_ykj=6675,
        altitude=100.0,
        ddy=1200.0,
        n_trees=5,
        start_year=5,
        end_year=10,
        step_years=5,
    )
    assert len(table) > 0
    assert table["N"].iloc[0] == pytest.approx(300 + 250 + 200, rel=0.1)


def test_build_growth_and_yield_table_peat_defaults_to_1():
    strata = tree_stratum.PerSpecies(
        pine=_one_species_stratum(300),
        spruce=tree_stratum.ZERO_STRATUM,
        deciduous=tree_stratum.ZERO_STRATUM,
    )
    default_peat = growth_and_yield_table.build_growth_and_yield_table(
        strata=strata,
        fertility_class=3,
        x_ykj=339,
        y_ykj=6675,
        altitude=100.0,
        ddy=1200.0,
        n_trees=5,
        start_year=5,
        end_year=10,
        step_years=5,
    )
    explicit_peat = growth_and_yield_table.build_growth_and_yield_table(
        strata=strata,
        fertility_class=3,
        x_ykj=339,
        y_ykj=6675,
        altitude=100.0,
        ddy=1200.0,
        n_trees=5,
        start_year=5,
        end_year=10,
        step_years=5,
        peat=1,
    )
    pd.testing.assert_frame_equal(default_peat, explicit_peat)


# %% The single YKJ range, shared by the data model and the CLI check
#
# shared_utils owns the bounds, next to the point_to_ykj call that produces
# coordinates in those units. Both enforcement points read them from there,
# so a coordinate is held to the same range however it reaches the tools.
# These tests fail if either consumer ever grows its own numbers again.


@pytest.mark.parametrize(
    "field_name, min_value, max_value",
    [
        ("x_ykj", shared_utils.X_YKJ_MIN, shared_utils.X_YKJ_MAX),
        ("y_ykj", shared_utils.Y_YKJ_MIN, shared_utils.Y_YKJ_MAX),
    ],
)
def test_stand_data_bounds_come_from_the_shared_ykj_range(
    field_name, min_value, max_value
):
    constraints = stand_data.StandData.model_fields[field_name].metadata
    assert {type(constraint).__name__: constraint for constraint in constraints}[
        "Ge"
    ].ge == min_value
    assert {type(constraint).__name__: constraint for constraint in constraints}[
        "Le"
    ].le == max_value


@pytest.mark.parametrize(
    "field_name, x_ykj, y_ykj",
    [
        ("x_ykj", shared_utils.X_YKJ_MAX + 1, 6675),
        ("x_ykj", shared_utils.X_YKJ_MIN - 1, 6675),
        ("y_ykj", 339, shared_utils.Y_YKJ_MAX + 1),
        ("y_ykj", 339, shared_utils.Y_YKJ_MIN - 1),
    ],
)
def test_stand_data_rejects_a_coordinate_the_cli_check_would_also_reject(
    field_name, x_ykj, y_ykj
):
    # The same value validate_x_y_ykj blocks in standalone mode is
    # unrepresentable in a StandData -- which is what "one source of truth"
    # has to mean in practice.
    with pytest.raises(pydantic.ValidationError, match=field_name):
        stand_data.StandData(
            site_fertility_class=3,
            canopy_layer_files={},
            x_ykj=x_ykj,
            y_ykj=y_ykj,
        )


# %% stand_data.StandData / StandDataDocument round-trip

# Fabricated AllometryFileAndSpecies values, mirroring
# tests/test_canopy_layer_allometry.py's strategy -- no real CSV is ever read here.
allometry_file_and_species_strategy = st.builds(
    AllometryFileAndSpecies,
    file_path=st.text(min_size=1, max_size=20).map(lambda name: Path(f"{name}.csv")),
    species_id=st.integers(min_value=1, max_value=100),
)

canopy_layer_files_strategy = st.dictionaries(
    keys=st.sampled_from(list(CanopyLayerName)),
    values=allometry_file_and_species_strategy,
    max_size=len(CanopyLayerName),
)



@st.composite
def stand_polygon_strategy(draw):
    """A real stand boundary in EPSG:3067 metres: an axis-aligned rectangle
    somewhere in Finland's ETRS-TM35FIN extent, optionally with one
    rectangular hole strictly inside it -- enough to exercise both the
    exterior ring and an interior ring through the WKT round-trip. Arbitrary
    (non-round) float coordinates on purpose: WKT must not lose precision."""
    coordinate = st.floats(allow_nan=False, allow_infinity=False)
    x0 = draw(coordinate.filter(lambda v: 50_000 <= v <= 750_000))
    y0 = draw(coordinate.filter(lambda v: 6_600_000 <= v <= 7_800_000))
    width = draw(st.floats(min_value=10, max_value=2_000))
    height = draw(st.floats(min_value=10, max_value=2_000))
    exterior = [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)]
    holes = []
    if draw(st.booleans()):
        # Middle third of the rectangle: always strictly inside it.
        hx0, hy0 = x0 + width / 3, y0 + height / 3
        hx1, hy1 = x0 + 2 * width / 3, y0 + 2 * height / 3
        holes.append([(hx0, hy0), (hx1, hy0), (hx1, hy1), (hx0, hy1)])
    return Polygon(exterior, holes)


_optional_positive_int = st.one_of(st.none(), st.integers(min_value=1, max_value=20))
_optional_nonneg_float = st.one_of(
    st.none(),
    st.floats(min_value=0, max_value=2000, allow_nan=False, allow_infinity=False),
)

stand_data_strategy = st.builds(
    stand_data.StandData,
    site_fertility_class=st.integers(min_value=1, max_value=10),
    canopy_layer_files=canopy_layer_files_strategy,
    x_ykj=st.integers(
        min_value=shared_utils.X_YKJ_MIN, max_value=shared_utils.X_YKJ_MAX
    ),
    y_ykj=st.integers(
        min_value=shared_utils.Y_YKJ_MIN, max_value=shared_utils.Y_YKJ_MAX
    ),
    polygon=st.one_of(st.none(), stand_polygon_strategy()),
    main_group=_optional_positive_int,
    sub_group=_optional_positive_int,
    stand_area=st.one_of(
        st.none(),
        st.floats(
            min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False
        ),
    ),
    basal_area=_optional_nonneg_float,
    mean_height=_optional_nonneg_float,
    mean_diameter=_optional_nonneg_float,
    total_volume=_optional_nonneg_float,
    stem_count=_optional_nonneg_float,
    developmentclass=_optional_positive_int,
    drainagestate=_optional_positive_int,
)

stand_data_document_strategy = st.builds(
    stand_data.StandDataDocument,
    crs=st.just(shared_utils.SOURCE_CRS),
    altitude=st.floats(
        min_value=-500, max_value=3000, allow_nan=False, allow_infinity=False
    ),
    ddy=st.floats(min_value=0, max_value=3000, allow_nan=False, allow_infinity=False),
    stands=st.dictionaries(
        keys=st.text(min_size=1, max_size=10).map(StandID),
        values=stand_data_strategy,
        max_size=5,
    ),
)


@given(document=stand_data_document_strategy)
def test_stand_data_document_roundtrips_through_json_in_memory(document):
    # Pure property, no file involved -- load_stand_data_document_from_json's
    # own file-reading behaviour is covered separately below.
    dumped = document.model_dump_json()
    assert stand_data.StandDataDocument.model_validate_json(dumped) == document


# %% stand_data.load_stand_data_document_from_json


def test_load_stand_data_document_from_json_reads_a_real_file(tmp_path):
    document = stand_data.StandDataDocument(
        crs=shared_utils.SOURCE_CRS,
        altitude=100.0,
        ddy=1200.0,
        stands={
            StandID("stand-1"): stand_data.StandData(
                site_fertility_class=3,
                canopy_layer_files={
                    CanopyLayerName.dominant: AllometryFileAndSpecies(
                        file_path=Path("pines.csv"), species_id=1
                    )
                },
                x_ykj=339,
                y_ykj=6675,
            )
        },
    )
    path = tmp_path / stand_data.STAND_DATA_FILENAME
    path.write_text(document.model_dump_json())

    loaded = stand_data.load_stand_data_document_from_json(path)

    assert loaded == document


# %% stand_data.dump_stand_data_document
#
# Moved here from being duplicated, byte-identically, in both
# metsakeskus_to_allometry.py and xml_to_allometry.py (ticket 12) -- each
# tool's own test file now only checks that it imports this function rather
# than defining a local copy.


def test_dump_stand_data_document_writes_stand_data_json(tmp_path):
    document = stand_data.StandDataDocument(
        crs=shared_utils.SOURCE_CRS,
        altitude=150.0,
        ddy=1200.0,
        stands={
            StandID("1"): stand_data.StandData(
                site_fertility_class=3,
                canopy_layer_files={
                    CanopyLayerName.dominant: AllometryFileAndSpecies(
                        file_path=Path("1_dominant.csv"), species_id=1
                    )
                },
                x_ykj=339,
                y_ykj=6675,
            )
        },
    )
    # The real layout: the document sits in <project-dir>/inputs/, beside
    # (not inside) the allometry/ folder holding the per-stand CSVs -- but
    # this function itself just writes to whatever path it's given; that
    # layout choice is stand_data_path_for_project's job, exercised by each
    # tool's own dump_stand_data_document-imports-the-shared-function test.
    output_path = tmp_path / "inputs" / "stand_data.json"
    output_path.parent.mkdir(parents=True)

    stand_data.dump_stand_data_document(output_path=output_path, document=document)

    assert output_path.exists()
    assert stand_data.load_stand_data_document_from_json(output_path) == document


# %% stand_data.StandPolygon (StandData.polygon) and StandDataDocument.crs

# A 10x10 m square with a 2x2 m hole, in EPSG:3067 metres near Paroninkorpi.
# shapely's area excludes the hole: 100 - 4 = 96.
_SQUARE_WITH_HOLE = Polygon(
    [(377_000.0, 6_767_000.0), (377_010.0, 6_767_000.0), (377_010.0, 6_767_010.0), (377_000.0, 6_767_010.0)],
    [[(377_004.0, 6_767_004.0), (377_006.0, 6_767_004.0), (377_006.0, 6_767_006.0), (377_004.0, 6_767_006.0)]],
)


def _stand_with_polygon(polygon) -> stand_data.StandData:
    return stand_data.StandData(
        site_fertility_class=3,
        canopy_layer_files={},
        x_ykj=338,
        y_ykj=6770,
        polygon=polygon,
    )


def test_stand_polygon_roundtrips_through_dump_and_load_with_its_holes(tmp_path):
    document = stand_data.StandDataDocument(
        crs=shared_utils.SOURCE_CRS,
        altitude=120.0,
        ddy=1250.0,
        stands={StandID("1"): _stand_with_polygon(_SQUARE_WITH_HOLE)},
    )
    output_path = tmp_path / stand_data.STAND_DATA_FILENAME

    stand_data.dump_stand_data_document(output_path=output_path, document=document)
    loaded = stand_data.load_stand_data_document_from_json(output_path)

    polygon = loaded.stands[StandID("1")].polygon
    assert isinstance(polygon, Polygon)
    assert polygon.equals(_SQUARE_WITH_HOLE)
    assert len(polygon.interiors) == 1
    assert polygon.area == 96.0


def test_stand_polygon_is_written_to_json_as_wkt():
    stand = _stand_with_polygon(_SQUARE_WITH_HOLE)

    dumped = stand.model_dump(mode="json")["polygon"]

    assert isinstance(dumped, str)
    assert shapely.from_wkt(dumped).equals(_SQUARE_WITH_HOLE)


def test_stand_polygon_accepts_a_wkt_string_in_python_mode_too():
    # The validator parses a str whichever way it arrives, not only from JSON.
    stand = _stand_with_polygon(_SQUARE_WITH_HOLE.wkt)

    assert isinstance(stand.polygon, Polygon)
    assert stand.polygon.equals(_SQUARE_WITH_HOLE)


@pytest.mark.parametrize(
    "bad_polygon",
    [
        pytest.param(
            MultiPolygon([_SQUARE_WITH_HOLE]), id="MultiPolygon-object"
        ),
        pytest.param(
            MultiPolygon([_SQUARE_WITH_HOLE]).wkt, id="MultiPolygon-wkt"
        ),
        pytest.param(Polygon(), id="empty-Polygon"),
        pytest.param("POLYGON EMPTY", id="empty-wkt"),
        pytest.param("POINT (377000 6767000)", id="Point-wkt"),
        # The pre-ticket-22 on-disk format: raw gml:coordinates pairs.
        pytest.param(
            "377000.0,6767000.0 377010.0,6767000.0 377010.0,6767010.0 377000.0,6767000.0",
            id="gml-coordinates-string",
        ),
        pytest.param(42, id="not-a-string"),
    ],
)
def test_stand_polygon_rejects_anything_but_a_non_empty_polygon(bad_polygon):
    with pytest.raises(pydantic.ValidationError):
        _stand_with_polygon(bad_polygon)


def test_stand_data_json_schema_still_builds():
    # The polygon type must not need arbitrary_types_allowed: the model has to
    # describe itself as JSON, with polygon as a plain string.
    schema = stand_data.StandDataDocument.model_json_schema()

    polygon_schema = schema["$defs"]["StandData"]["properties"]["polygon"]
    assert sorted(branch["type"] for branch in polygon_schema["anyOf"]) == [
        "null",
        "string",
    ]


def test_stand_data_document_requires_crs():
    with pytest.raises(pydantic.ValidationError, match="crs"):
        stand_data.StandDataDocument(altitude=100.0, ddy=1200.0, stands={})  # ty: ignore[missing-argument]


def test_stand_data_document_rejects_a_crs_other_than_the_source_crs():
    with pytest.raises(pydantic.ValidationError, match="EPSG:3067"):
        stand_data.StandDataDocument(
            crs="EPSG:4326", altitude=100.0, ddy=1200.0, stands={}
        )


def test_stand_data_document_accepts_the_source_crs():
    document = stand_data.StandDataDocument(
        crs=shared_utils.SOURCE_CRS, altitude=100.0, ddy=1200.0, stands={}
    )

    assert document.crs == "EPSG:3067"


# %% stand_data._build_stand_params_from_stand_data_document


def _single_stand_document(canopy_layer_files, site_fertility_class=3):
    return stand_data.StandDataDocument(
        crs=shared_utils.SOURCE_CRS,
        altitude=100.0,
        ddy=1200.0,
        stands={
            StandID("known"): stand_data.StandData(
                site_fertility_class=site_fertility_class,
                canopy_layer_files=canopy_layer_files,
                x_ykj=339,
                y_ykj=6675,
            )
        },
    )


def test_build_stand_params_raises_a_clear_error_for_an_unknown_stand_id():
    document = _single_stand_document(canopy_layer_files={})

    with pytest.raises(KeyError, match="unknown"):
        stand_data._build_stand_params_from_stand_data_document(
            document, StandID("unknown"), n=3
        )


def test_build_stand_params_matches_with_single_allometry_per_layer():
    canopy_layer_files = {
        CanopyLayerName.dominant: AllometryFileAndSpecies(
            file_path=Path("pines.csv"), species_id=1
        ),
        CanopyLayerName.under: AllometryFileAndSpecies(
            file_path=Path("spruces.csv"), species_id=2
        ),
    }
    document = _single_stand_document(
        canopy_layer_files=canopy_layer_files, site_fertility_class=4
    )

    params = stand_data._build_stand_params_from_stand_data_document(
        document, StandID("known"), n=5
    )

    assert params.site_fertility_class == 4
    assert params.canopy_layer_allometry == (
        CanopyLayerAllometry.with_single_allometry_per_layer(
            layers=canopy_layer_files, n=5
        )
    )


@given(
    canopy_layer_files=canopy_layer_files_strategy.filter(
        lambda layers: len(layers) > 0
    ),
    n1=st.integers(min_value=0, max_value=30),
    n2=st.integers(min_value=0, max_value=30),
)
def test_build_stand_params_pointer_lengths_track_n(canopy_layer_files, n1, n2):
    # Regression test for the bug this design was fixing: a CanopyLayerAllometry
    # built for one n must not be reused/cached for another -- each call must
    # produce pointer lists whose length matches the n passed to that call.
    assume(n1 != n2)
    document = _single_stand_document(canopy_layer_files=canopy_layer_files)

    params1 = stand_data._build_stand_params_from_stand_data_document(
        document, StandID("known"), n=n1
    )
    params2 = stand_data._build_stand_params_from_stand_data_document(
        document, StandID("known"), n=n2
    )

    for layer_name in canopy_layer_files:
        pointers1 = params1.canopy_layer_allometry.pointers[layer_name]
        pointers2 = params2.canopy_layer_allometry.pointers[layer_name]
        assert pointers1 is not None
        assert pointers2 is not None
        assert len(pointers1) == n1
        assert len(pointers2) == n2


# %% Dependency direction: susi_parameter_model.py must not import stand_data


def test_susi_parameter_model_does_not_import_from_tools():
    # tools/ depends on susi/, never the reverse -- stand_data.py lives in
    # tools/ specifically so susi_parameter_model.py can stay ignorant of it.
    source = inspect.getsource(susi_parameter_model)
    tree = ast.parse(source)

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(module.startswith("tools") for module in imported_modules)
