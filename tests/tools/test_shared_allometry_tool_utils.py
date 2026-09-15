import argparse
import dataclasses
from dataclasses import dataclass

import pydantic
import pytest

from tools.shared_allometry_tool_utils import (
    input_validation,
    print_formatting,
    project_layout,
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


# %% project_layout.output_dir_for_project


def test_output_dir_for_project_appends_allometry(tmp_path):
    project_dir = tmp_path / "myproject"
    assert (
        project_layout.output_dir_for_project(project_dir) == project_dir / "allometry"
    )


# %% project_layout.check_output_dir_available


def test_check_output_dir_available_allows_a_free_folder(tmp_path):
    project_layout.check_output_dir_available(tmp_path / "allometry", _parser())


def test_check_output_dir_available_refuses_an_existing_folder(tmp_path, capsys):
    output_dir = tmp_path / "allometry"
    output_dir.mkdir()
    with pytest.raises(SystemExit):
        project_layout.check_output_dir_available(output_dir, _parser())
    assert "already exists" in capsys.readouterr().err


# %% project_layout.resolve_config_path


def test_resolve_config_path_prefers_explicit_config(tmp_path):
    explicit = tmp_path / "somewhere_else.toml"
    explicit.write_text("")
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    (project_dir / "config.toml").write_text("")

    assert (
        project_layout.resolve_config_path(explicit, project_dir, _parser()) == explicit
    )


def test_resolve_config_path_defaults_to_config_toml_in_project_dir(tmp_path):
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    default_config = project_dir / "config.toml"
    default_config.write_text("")

    assert (
        project_layout.resolve_config_path(None, project_dir, _parser())
        == default_config
    )


def test_resolve_config_path_errors_when_default_is_missing(tmp_path, capsys):
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    with pytest.raises(SystemExit):
        project_layout.resolve_config_path(None, project_dir, _parser())
    stderr = capsys.readouterr().err
    assert str(project_dir / "config.toml") in stderr
    assert "--config" in stderr


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
    skips = [print_formatting.StandSkipped(stand_id="1", reason="no TreeStrata")]
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
            age="not-a-number",
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
        stratum.age = 31  # noqa: B010


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
