import json
from pathlib import Path

import pytest
import xmltodict
from hypothesis import given
from hypothesis import strategies as st

from susi.io.load_output_data import StandID
from susi.io.susi_parameter_model import read_allometry_info_from_csv
from tools.shared_allometry_tool_utils import input_validation
from tools.xml_to_allometry import xml_to_allometry


# %% out_of_range_message reuse (from the shared package)


def test_out_of_range_message_reused_from_shared_package():
    assert input_validation.out_of_range_message("altitude", 500, 0, 1000) is None
    message = input_validation.out_of_range_message("altitude", -1, 0, 1000)
    assert message is not None
    assert "altitude" in message


# %% XmlConfig
#
# XmlConfig is a StrictFrozenModel (susi.io.extra_pydantic_types): presence/
# unknown-field checking, required-vs-defaulted fields, and extra="forbid"
# all come from Pydantic itself, the same as metsakeskus_to_allometry.py's
# ExtractionConfig and new_growth_allometry.py's NewGrowthConfig.


def test_xml_config_requires_altitude():
    with pytest.raises(ValueError, match="altitude"):
        xml_to_allometry.XmlConfig.model_validate({"ddy": 1200.0})


def test_xml_config_requires_ddy():
    with pytest.raises(ValueError, match="ddy"):
        xml_to_allometry.XmlConfig.model_validate({"altitude": 150.0})


def test_xml_config_reports_all_missing_fields_together():
    with pytest.raises(ValueError) as exc_info:
        xml_to_allometry.XmlConfig.model_validate({})
    message = str(exc_info.value)
    assert "altitude" in message
    assert "ddy" in message


def test_xml_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="typo_field"):
        xml_to_allometry.XmlConfig.model_validate(
            {"altitude": 150.0, "ddy": 1200.0, "typo_field": 1}
        )


def test_xml_config_applies_defaults():
    config = xml_to_allometry.XmlConfig.model_validate(
        {"altitude": 150.0, "ddy": 1200.0}
    )
    assert config.n_trees == 20
    assert config.start_year == 5
    assert config.end_year == 80
    assert config.step_years == 5


def test_xml_config_overrides_defaults():
    config = xml_to_allometry.XmlConfig.model_validate(
        {"altitude": 150.0, "ddy": 1200.0, "n_trees": 5, "step_years": 10}
    )
    assert config.n_trees == 5
    assert config.step_years == 10


def test_xml_config_is_frozen():
    config = xml_to_allometry.XmlConfig(altitude=150.0, ddy=1200.0)
    with pytest.raises(Exception):  # noqa: B017 -- pydantic's frozen-model error
        setattr(config, "altitude", 200.0)  # noqa: B010


def test_load_xml_config_reads_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 150.0\nddy = 1200.0\nn_trees = 15\n")
    config = xml_to_allometry.load_xml_config(config_path)
    assert config.altitude == 150.0
    assert config.n_trees == 15


# %% default_config.toml
#
# Guards against the shipped default/template config drifting from
# XmlConfig's own field defaults -- see that file's header comment.

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "tools"
    / "xml_to_allometry"
    / "default_config.toml"
)


def test_default_config_toml_optional_fields_match_model_defaults():
    config = xml_to_allometry.load_xml_config(DEFAULT_CONFIG_PATH)
    defaults = xml_to_allometry.XmlConfig(altitude=0.0, ddy=0.0)
    assert config.n_trees == defaults.n_trees
    assert config.start_year == defaults.start_year
    assert config.end_year == defaults.end_year
    assert config.step_years == defaults.step_years


def test_default_config_toml_required_fields_are_deliberately_out_of_range():
    """The shipped file's altitude/ddy placeholders must stay outside
    [ALTITUDE_MIN, ALTITUDE_MAX] / [DDY_MIN, DDY_MAX] -- that's what makes an
    untouched copy fail loudly instead of running silently."""
    config = xml_to_allometry.load_xml_config(DEFAULT_CONFIG_PATH)
    assert not (
        input_validation.ALTITUDE_MIN
        <= config.altitude
        <= input_validation.ALTITUDE_MAX
    )
    assert not (input_validation.DDY_MIN <= config.ddy <= input_validation.DDY_MAX)


# %% XML stand fixtures


def _polygon_coordinates() -> str:
    return "385000,6685000 385050,6685000 385050,6685050 385000,6685050"


def _stand_xml_block(stand_id: str, *, include_tree_strata: bool) -> str:
    tree_strata_block = (
        """
          <tst:TreeStrata>
            <tst:TreeStratum>
              <tst:TreeSpecies>1</tst:TreeSpecies>
              <tst:Age>40</tst:Age>
              <tst:BasalArea>20.0</tst:BasalArea>
              <tst:StemCount>500</tst:StemCount>
              <tst:MeanDiameter>20.0</tst:MeanDiameter>
              <tst:MeanHeight>18.0</tst:MeanHeight>
            </tst:TreeStratum>
          </tst:TreeStrata>
        """
        if include_tree_strata
        else ""
    )
    return f"""
    <st:Stand id="{stand_id}">
      <st:StandBasicData>
        <st:FertilityClass>3</st:FertilityClass>
        <st:MainGroup>1</st:MainGroup>
        <st:SubGroup>2</st:SubGroup>
        <st:Area>1.5</st:Area>
        <gdt:PolygonGeometry>
          <gml:polygonProperty>
            <gml:Polygon>
              <gml:exterior>
                <gml:LinearRing>
                  <gml:coordinates>{_polygon_coordinates()}</gml:coordinates>
                </gml:LinearRing>
              </gml:exterior>
            </gml:Polygon>
          </gml:polygonProperty>
        </gdt:PolygonGeometry>
      </st:StandBasicData>
      <ts:TreeStandData>
        <ts:TreeStandDataDate>
          <tss:TreeStandSummary>
            <tss:MeanDiameter>20.0</tss:MeanDiameter>
            <tss:MeanAge>40</tss:MeanAge>
            <tss:BasalArea>20.0</tss:BasalArea>
            <tss:MeanHeight>18.0</tss:MeanHeight>
            <tss:Volume>150.0</tss:Volume>
          </tss:TreeStandSummary>
          {tree_strata_block}
        </ts:TreeStandDataDate>
      </ts:TreeStandData>
    </st:Stand>
    """


def _forest_property_xml(stand_blocks: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<ForestPropertyData>\n"
        "  <st:Stands>\n"
        f"{stand_blocks}"
        "  </st:Stands>\n"
        "</ForestPropertyData>\n"
    )


def _parsed_stand(stand_id: str, *, include_tree_strata: bool) -> dict:
    """One <st:Stand> parsed back into the dict shape
    get_stand_data_from_xml/build_stand_datas expect (i.e. what
    read_stands_from_xml_file would hand them for a single stand)."""
    xml_text = _forest_property_xml(
        _stand_xml_block(stand_id, include_tree_strata=include_tree_strata)
    )
    parsed = xmltodict.parse(xml_text)
    return parsed["ForestPropertyData"]["st:Stands"]["st:Stand"]


# %% read_stands_from_xml_file


def test_read_stands_from_xml_file_wraps_a_single_stand_in_a_list(tmp_path):
    # xmltodict collapses a single repeated <st:Stand> element to a bare
    # dict instead of a one-item list. Exercised through the real parser on
    # actual XML text (not a hand-built list) so this catches the xmltodict
    # quirk itself, not just a mocked-out shape (#281).
    xml_path = tmp_path / "single_stand.xml"
    xml_path.write_text(
        _forest_property_xml(_stand_xml_block("1", include_tree_strata=True))
    )

    stands = xml_to_allometry.read_stands_from_xml_file(xml_path)

    assert isinstance(stands, list)
    assert len(stands) == 1


def test_read_stands_from_xml_file_keeps_multiple_stands_as_a_list(tmp_path):
    xml_path = tmp_path / "two_stands.xml"
    xml_path.write_text(
        _forest_property_xml(
            _stand_xml_block("1", include_tree_strata=True)
            + _stand_xml_block("2", include_tree_strata=True)
        )
    )

    stands = xml_to_allometry.read_stands_from_xml_file(xml_path)

    assert isinstance(stands, list)
    assert len(stands) == 2


def test_build_stand_datas_handles_a_single_stand_file_end_to_end(tmp_path):
    # Regression test for #281: a real single-stand XML file used to crash
    # build_stand_datas with "TypeError: string indices must be integers,
    # not 'str'" because xmltodict handed it a dict, not a list.
    xml_path = tmp_path / "single_stand.xml"
    xml_path.write_text(
        _forest_property_xml(_stand_xml_block("1", include_tree_strata=True))
    )

    stands = xml_to_allometry.read_stands_from_xml_file(xml_path)
    stand_datas, skipped = xml_to_allometry.build_stand_datas(stands)

    assert [sd.id for sd in stand_datas] == [StandID("1")]
    assert skipped == []


# %% get_stand_data_from_xml / NoTreeStrataError / build_stand_datas


def test_get_stand_data_from_xml_parses_a_full_stand():
    stand_data = xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand("1", include_tree_strata=True)
    )
    assert stand_data.id == StandID("1")
    assert stand_data.fertility_class == 3
    assert stand_data.main_species == 1
    assert stand_data.tree_strata[0].basal_area == pytest.approx(20.0)


def test_get_stand_data_from_xml_raises_no_tree_strata_error_when_missing():
    with pytest.raises(xml_to_allometry.NoTreeStrataError, match="2"):
        xml_to_allometry.get_stand_data_from_xml(
            _parsed_stand("2", include_tree_strata=False)
        )


def test_build_stand_datas_reports_skip_by_stand_id_and_reason():
    stands = [
        _parsed_stand("1", include_tree_strata=True),
        _parsed_stand("2", include_tree_strata=False),
    ]
    stand_datas, skipped = xml_to_allometry.build_stand_datas(stands)

    assert [sd.id for sd in stand_datas] == [StandID("1")]
    assert len(skipped) == 1
    assert skipped[0].stand_id == StandID("2")
    assert "TreeStrata" in skipped[0].reason


def test_build_stand_datas_keeps_every_stand_when_none_are_skipped():
    stands = [_parsed_stand("1", include_tree_strata=True)]
    stand_datas, skipped = xml_to_allometry.build_stand_datas(stands)
    assert len(stand_datas) == 1
    assert skipped == []


# %% parse_polygon_to_coords


@given(
    coords=st.lists(
        st.tuples(
            st.floats(allow_nan=False, allow_infinity=False),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
    )
)
def test_parse_coordinates_roundtrip(coords):
    # Build the input string in the same format the function expects
    input_str = " ".join(f"{x},{y}" for x, y in coords)

    result = xml_to_allometry.parse_polygon_to_coords(input_str)

    # Check structure
    assert len(result) == len(coords)
    for (rx, ry), (ox, oy) in zip(result, coords):
        assert rx == pytest.approx(ox)
        assert ry == pytest.approx(oy)


# %% plan_stand_output / process_stand / dump_json_info_to_file


def _stand_data(stand_id="1") -> xml_to_allometry.StandData:
    return xml_to_allometry.get_stand_data_from_xml(
        _parsed_stand(stand_id, include_tree_strata=True)
    )


def test_plan_stand_output_names_the_csv_by_stand_id(tmp_path):
    stand_data = _stand_data("42")
    assert xml_to_allometry.plan_stand_output(stand_data, tmp_path) == (
        tmp_path / "42.csv"
    )


def test_process_stand_writes_a_csv_round_tripping_through_the_real_reader(tmp_path):
    stand_data = _stand_data("1")
    config = xml_to_allometry.XmlConfig(altitude=150.0, ddy=1200.0, end_year=10)

    xml_to_allometry.process_stand(config, stand_data, PEAT=1, output_dir=tmp_path)

    output_path = xml_to_allometry.plan_stand_output(stand_data, tmp_path)
    assert output_path.exists()
    df = read_allometry_info_from_csv(output_path)
    assert len(df) > 0


def test_dump_json_info_to_file_writes_extra_xml_info_json(tmp_path):
    stand_data = _stand_data("1")
    many = xml_to_allometry._create_many_stand_datas([stand_data])

    xml_to_allometry.dump_json_info_to_file(output_dir=tmp_path, many_stand_datas=many)

    json_path = tmp_path / "extra_xml_info.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert "1" in payload["stand_datas"]


def test_print_dry_run_plan_reports_counts_and_writes_nothing(tmp_path, capsys):
    output_dir = tmp_path / "allometry"
    stand_datas = [_stand_data("1"), _stand_data("2")]

    xml_to_allometry.print_dry_run_plan(stand_datas, output_dir)

    printed = capsys.readouterr().out
    assert "2 stand(s) -- 2 CSV(s)" in printed
    assert "extra_xml_info.json" in printed
    assert not output_dir.exists()


# %% parse_CLI_arguments


@pytest.fixture
def dummy_xml_file(tmp_path):
    xml_path = tmp_path / "stand.xml"
    xml_path.write_text(
        _forest_property_xml(_stand_xml_block("1", include_tree_strata=True))
    )
    return xml_path


@pytest.fixture
def project_dir(tmp_path):
    # --project-dir must already exist (shared valid_existing_directory) --
    # the folder the docs have the user set up beforehand, with the XML file
    # and config.toml colocated inside it.
    path = tmp_path / "myproject"
    path.mkdir()
    return path


@pytest.fixture
def dummy_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 150.0\nddy = 1200.0\n")
    return config_path


def _run_parse_CLI_arguments(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["xml_to_allometry.py", *argv])
    return xml_to_allometry.parse_CLI_arguments()


def test_parse_CLI_arguments_requires_project_dir(
    monkeypatch, dummy_xml_file, dummy_config_file, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [str(dummy_xml_file), f"--config={dummy_config_file}"]
        )
    assert "--project-dir" in capsys.readouterr().err


def test_parse_CLI_arguments_reports_a_missing_default_config(
    monkeypatch, dummy_xml_file, project_dir, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [str(dummy_xml_file), f"--project-dir={project_dir}"]
        )
    stderr = capsys.readouterr().err
    assert str(project_dir / "config.toml") in stderr
    assert "--config" in stderr


def test_parse_CLI_arguments_finds_config_toml_inside_project_dir_by_default(
    monkeypatch, dummy_xml_file, project_dir
):
    (project_dir / "config.toml").write_text("altitude = 150.0\nddy = 1200.0\n")
    cli_args = _run_parse_CLI_arguments(
        monkeypatch, [str(dummy_xml_file), f"--project-dir={project_dir}"]
    )
    assert cli_args.config_path == project_dir / "config.toml"
    assert cli_args.config.altitude == 150.0


def test_parse_CLI_arguments_explicit_config_overrides_the_default_lookup(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    (project_dir / "config.toml").write_text("altitude = 10.0\nddy = 600.0\n")
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert cli_args.config_path == dummy_config_file
    assert (
        cli_args.config.altitude == 150.0
    )  # from dummy_config_file, not project_dir's own


def test_parse_CLI_arguments_blocks_out_of_range_altitude_by_default(
    monkeypatch, dummy_xml_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 1500.0\nddy = 1200.0\n")
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={config_path}",
                f"--project-dir={project_dir}",
            ],
        )
    stderr = capsys.readouterr().err
    assert "altitude" in stderr
    assert "--allow-out-of-range-values" in stderr


def test_parse_CLI_arguments_allows_out_of_range_with_override(
    monkeypatch, dummy_xml_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = 1500.0\nddy = 1200.0\n")
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={config_path}",
            f"--project-dir={project_dir}",
            "--allow-out-of-range-values",
        ],
    )
    assert cli_args.config.altitude == 1500.0
    assert "Warning" in capsys.readouterr().out


def test_parse_CLI_arguments_rejects_nan_even_with_override(
    monkeypatch, dummy_xml_file, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("altitude = nan\nddy = 1200.0\n")
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={config_path}",
                f"--project-dir={project_dir}",
                "--allow-out-of-range-values",
            ],
        )
    assert "altitude" in capsys.readouterr().err


def test_parse_CLI_arguments_refuses_existing_default_output_dir(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir, capsys
):
    (project_dir / "allometry").mkdir()
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
            ],
        )
    stderr = capsys.readouterr().err
    assert "already exists" in stderr
    assert "--project-dir" in stderr


def test_parse_CLI_arguments_creates_no_output_folder(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert not (project_dir / "allometry").exists()


def test_parse_CLI_arguments_dry_run_defaults_to_false(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
        ],
    )
    assert cli_args.dry_run is False


def test_parse_CLI_arguments_dry_run_creates_no_output_folder(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            str(dummy_xml_file),
            f"--config={dummy_config_file}",
            f"--project-dir={project_dir}",
            "--dry-run",
        ],
    )
    assert cli_args.dry_run is True
    assert not (project_dir / "allometry").exists()


def test_parse_CLI_arguments_dry_run_still_refuses_existing_output_dir(
    monkeypatch, dummy_xml_file, dummy_config_file, project_dir, capsys
):
    (project_dir / "allometry").mkdir()
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                str(dummy_xml_file),
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
                "--dry-run",
            ],
        )
    assert "already exists" in capsys.readouterr().err
