import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pydantic
import pytest
import shapely
import shapely.ops
from hypothesis import assume, given
from hypothesis import strategies as st
from pyproj import Transformer
from shapely.geometry import Polygon

from susi.io import stand_data
from susi.io.load_output_data import StandID
from tools.shared_allometry_tool_utils import (
    allometry_generation_defaults,
    cli_paths,
    growth_and_yield_table,
    input_validation,
    print_formatting,
    shared_utils,
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


def test_valid_existing_directory_returns_a_relative_directory_absolute(
    tmp_path, monkeypatch
):
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path)

    assert input_validation.valid_existing_directory("project") == (
        tmp_path / "project"
    )


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
        (stand_data.X_YKJ_MIN, stand_data.Y_YKJ_MIN),
        (stand_data.X_YKJ_MAX, stand_data.Y_YKJ_MAX),
    ],
)
def test_validate_x_y_ykj_accepts_its_own_bounds(x_ykj, y_ykj, capsys):
    # The bounds are inclusive, same as the altitude/ddy ones.
    input_validation.validate_x_y_ykj(_parser(), x_ykj, y_ykj, False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "x_ykj, y_ykj, expected_name",
    [
        (stand_data.X_YKJ_MIN - 1, IN_RANGE_Y_YKJ, "x_ykj"),
        (stand_data.X_YKJ_MAX + 1, IN_RANGE_Y_YKJ, "x_ykj"),
        (IN_RANGE_X_YKJ, stand_data.Y_YKJ_MIN - 1, "y_ykj"),
        (IN_RANGE_X_YKJ, stand_data.Y_YKJ_MAX + 1, "y_ykj"),
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
    x_ykj=st.integers(min_value=stand_data.X_YKJ_MIN, max_value=stand_data.X_YKJ_MAX),
    y_ykj=st.integers(min_value=stand_data.Y_YKJ_MIN, max_value=stand_data.Y_YKJ_MAX),
)
def test_validate_x_y_ykj_accepts_anything_inside_the_box(x_ykj, y_ykj):
    input_validation.validate_x_y_ykj(_parser(), x_ykj, y_ykj, False)


@given(
    x_ykj=st.integers(min_value=-100_000, max_value=100_000),
    y_ykj=st.integers(min_value=-100_000, max_value=100_000),
)
def test_validate_x_y_ykj_rejects_anything_outside_the_box(x_ykj, y_ykj):
    assume(
        not (stand_data.X_YKJ_MIN <= x_ykj <= stand_data.X_YKJ_MAX)
        or not (stand_data.Y_YKJ_MIN <= y_ykj <= stand_data.Y_YKJ_MAX)
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
    error = capsys.readouterr().err
    assert "already exists" in error
    # A project is one folder (ADR 0003): the way out is moving the old
    # allometry/ aside, not a second --project-dir.
    assert "--project-dir" not in error
    assert "rename or move" in error


def test_check_output_file_available_allows_a_free_file_in_an_existing_folder(
    tmp_path,
):
    cli_paths.check_output_file_available(tmp_path / "new_growth_pine.csv", _parser())


def test_check_output_file_available_refuses_an_existing_file(tmp_path, capsys):
    output_path = tmp_path / "new_growth_pine.csv"
    output_path.write_text("")
    with pytest.raises(SystemExit):
        cli_paths.check_output_file_available(output_path, _parser())
    error = capsys.readouterr().err
    assert "already exists" in error
    assert "delete" in error.lower()


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
        cli_paths.resolve_config_path(explicit, project_dir, _parser(), config_filename)
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
        cli_paths.resolve_config_path(None, project_dir, _parser(), config_filename)
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
        cli_paths.resolve_config_path(None, project_dir, _parser(), config_filename)


@pytest.mark.parametrize("config_filename", CONFIG_FILENAMES)
def test_resolve_config_path_errors_when_default_is_missing(
    tmp_path, capsys, config_filename
):
    project_dir = _project_with_inputs(tmp_path)
    with pytest.raises(SystemExit):
        cli_paths.resolve_config_path(None, project_dir, _parser(), config_filename)
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


# %% shared_utils.to_source_crs


# A 50x50 m square with a 10x10 m hole, in EPSG:3067 near Helsinki.
_POLYGON_IN_SOURCE_CRS = Polygon(
    [
        (385_000, 6_685_000),
        (385_050, 6_685_000),
        (385_050, 6_685_050),
        (385_000, 6_685_050),
    ],
    [
        [
            (385_010, 6_685_010),
            (385_020, 6_685_010),
            (385_020, 6_685_020),
            (385_010, 6_685_020),
        ]
    ],
)


def _reprojected(polygon: Polygon, from_crs: str, to_crs: str) -> Polygon:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return shapely.ops.transform(transformer.transform, polygon)


def test_to_source_crs_returns_a_polygon_already_in_the_source_crs_unchanged():
    result = shared_utils.to_source_crs(
        _POLYGON_IN_SOURCE_CRS, stand_data.SOURCE_CRS, where="test input"
    )
    assert result is _POLYGON_IN_SOURCE_CRS


@pytest.mark.parametrize(
    "declared_crs",
    [
        pytest.param("EPSG:4326", id="WGS84-lon-lat"),
        pytest.param("EPSG:2393", id="YKJ"),
        pytest.param("EPSG:3035", id="LAEA-Europe"),
    ],
)
def test_to_source_crs_reprojects_other_crss_holes_included(declared_crs):
    source = _reprojected(_POLYGON_IN_SOURCE_CRS, stand_data.SOURCE_CRS, declared_crs)

    result = shared_utils.to_source_crs(source, declared_crs, where="test input")

    # Back where it started, to well under a millimetre.
    assert result.equals_exact(_POLYGON_IN_SOURCE_CRS, tolerance=1e-3)
    assert len(result.interiors) == 1


def test_to_source_crs_fails_without_a_declared_crs():
    # Nothing to reproject from: guessing would give silently wrong geometry.
    with pytest.raises(ValueError, match="test input.*no CRS"):
        shared_utils.to_source_crs(_POLYGON_IN_SOURCE_CRS, None, where="test input")


def test_to_source_crs_fails_on_an_unrecognised_crs():
    with pytest.raises(ValueError, match="test input.*EPSG:999999"):
        shared_utils.to_source_crs(
            _POLYGON_IN_SOURCE_CRS, "EPSG:999999", where="test input"
        )


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
        stratum.age = 31  # ty: ignore[invalid-assignment]


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
        age=25,
        basal_area=8.0,
        stem_count=stem_count,
        mean_diameter=14.0,
        mean_height=11.0,
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

