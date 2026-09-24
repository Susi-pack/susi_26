# %% Imports
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from susi.io.load_output_data import StandID
from susi.io.utils import SRC_DIR
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerName,
    read_allometry_info_from_csv,
)
from tools.new_growth_allometry import new_growth_allometry as nga
from susi.io import stand_data
from tools.shared_allometry_tool_utils import (
    input_validation,
    tree_stratum,
)

# %% NewGrowthConfig
#
# NewGrowthConfig is a StrictFrozenModel (susi.io.extra_pydantic_types):
# presence/unknown-field checking (missing/unknown field detection,
# reporting every violation together), required-vs-defaulted fields, and
# extra="forbid" all come from Pydantic itself, the same as
# xml_to_allometry.py's XmlConfig and metsakeskus_to_allometry.py's
# ExtractionConfig. These tests only cover what is specific to *this* tool:
# which fields are required, their defaulting, and the species/stems_count
# validation that has no equivalent in the other two.


def _valid_raw(**overrides) -> dict:
    raw = {
        "altitude": 100.0,
        "ddy": 1200.0,
        "fertility_class": 3,
        "x": 379930.3,
        "y": 7039150.8,
        "species": "pine",
        "stems_count": 2000,
    }
    raw.update(overrides)
    return raw


# Fields with no default -- what REQUIRED_CONFIG_FIELDS used to spell out by
# hand; now it's just "no default", read straight off the model itself.
REQUIRED_FIELDS = [
    name
    for name, field in nga.NewGrowthConfig.model_fields.items()
    if field.is_required()
]


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_new_growth_config_requires_each_field(missing_field):
    raw = _valid_raw()
    del raw[missing_field]
    with pytest.raises(ValueError, match=missing_field):
        nga.NewGrowthConfig.model_validate(raw)


def test_new_growth_config_reports_all_missing_fields_together():
    with pytest.raises(ValueError) as exc_info:
        nga.NewGrowthConfig.model_validate({})
    message = str(exc_info.value)
    for field in REQUIRED_FIELDS:
        assert field in message


def test_new_growth_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="typo_field"):
        nga.NewGrowthConfig.model_validate(_valid_raw(typo_field=1))


def test_new_growth_config_applies_defaults():
    config = nga.NewGrowthConfig.model_validate(_valid_raw())
    assert config.n_trees == 20
    assert config.start_year == 5
    assert config.end_year == 80
    assert config.step_years == 5


def test_new_growth_config_overrides_defaults():
    config = nga.NewGrowthConfig.model_validate(
        _valid_raw(n_trees=10, start_year=1, end_year=40, step_years=2)
    )
    assert config.n_trees == 10
    assert config.start_year == 1
    assert config.end_year == 40
    assert config.step_years == 2


def test_new_growth_config_is_frozen():
    config = nga.NewGrowthConfig.model_validate(_valid_raw())
    with pytest.raises(Exception):  # noqa: B017 -- pydantic's frozen-model error
        config.altitude = 200.0  # noqa: B010


def test_load_new_growth_config_reads_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "altitude = 100.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "pine"\nstems_count = 2000\n'
    )
    config = nga.load_new_growth_config(config_path)
    assert config == nga.NewGrowthConfig.model_validate(_valid_raw())


# %% species validation
#
# validate_species no longer exists as a standalone function -- it's now the
# NewGrowthConfig._normalize_species before-validator, so these go through
# the model, varying only the species field.


@pytest.mark.parametrize("species", [s.value for s in nga.Species])
def test_new_growth_config_accepts_every_valid_species(species):
    config = nga.NewGrowthConfig.model_validate(_valid_raw(species=species))
    assert config.species == nga.Species(species)


def test_new_growth_config_normalizes_species_case_and_whitespace():
    config = nga.NewGrowthConfig.model_validate(_valid_raw(species=" Pine "))
    assert config.species == nga.Species.PINE


def test_new_growth_config_rejects_an_unknown_species():
    with pytest.raises(ValueError, match="species"):
        nga.NewGrowthConfig.model_validate(_valid_raw(species="oak"))


def test_new_growth_config_rejects_a_non_string_species():
    with pytest.raises(ValueError, match="species"):
        nga.NewGrowthConfig.model_validate(_valid_raw(species=1))


# %% stems_count validation
#
# validate_stems_count no longer exists as a standalone function either --
# stems_count: PositiveInt (susi.io.extra_pydantic_types) does the positivity
# check now, so these also go through the model.


def test_new_growth_config_accepts_a_positive_stems_count():
    config = nga.NewGrowthConfig.model_validate(_valid_raw(stems_count=2000))
    assert config.stems_count == 2000


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_new_growth_config_rejects_non_positive_stems_count(bad_value):
    with pytest.raises(ValueError, match="stems_count"):
        nga.NewGrowthConfig.model_validate(_valid_raw(stems_count=bad_value))


def test_new_growth_config_coerces_a_numeric_string_stems_count():
    config = nga.NewGrowthConfig.model_validate(_valid_raw(stems_count="2000"))
    assert config.stems_count == 2000


# %% default_config.toml
#
# Guards against the shipped default/template config drifting from
# NewGrowthConfig's own field defaults -- see that file's header comment.
# The required fields' placeholders are deliberately invalid (species=""
# is not a valid Species, stems_count=-1 is non-positive) so an untouched
# copy fails loudly rather than running silently -- unlike
# xml_to_allometry.py's altitude/ddy placeholders, whose "deliberately out
# of range" nature is checked separately (also done below), because
# altitude/ddy support a real --allow-out-of-range-values override that
# species/stems_count do not.

DEFAULT_CONFIG_PATH = SRC_DIR / "tools" / "new_growth_allometry" / "default_config.toml"


def _read_raw_default_config() -> dict:
    with open(DEFAULT_CONFIG_PATH, "rb") as config_file:
        return tomllib.load(config_file)


def test_default_config_toml_optional_fields_match_dataclass_defaults():
    raw = _read_raw_default_config()
    defaults = nga.NewGrowthConfig(
        altitude=0.0,
        ddy=0.0,
        fertility_class=0,
        x=0.0,
        y=0.0,
        species=nga.Species.PINE,
        stems_count=1,
    )
    assert raw["n_trees"] == defaults.n_trees
    assert raw["start_year"] == defaults.start_year
    assert raw["end_year"] == defaults.end_year
    assert raw["step_years"] == defaults.step_years


def test_default_config_toml_altitude_ddy_placeholders_are_deliberately_out_of_range():
    raw = _read_raw_default_config()
    assert not (
        input_validation.ALTITUDE_MIN
        <= raw["altitude"]
        <= input_validation.ALTITUDE_MAX
    )
    assert not (input_validation.DDY_MIN <= raw["ddy"] <= input_validation.DDY_MAX)


def test_default_config_toml_species_placeholder_is_invalid():
    raw = _read_raw_default_config()
    assert raw["species"] not in [s.value for s in nga.Species]


def test_default_config_toml_stems_count_placeholder_is_invalid():
    raw = _read_raw_default_config()
    assert raw["stems_count"] <= 0


def test_default_config_toml_is_rejected_when_loaded_untouched():
    with pytest.raises(ValueError):
        nga.load_new_growth_config(DEFAULT_CONFIG_PATH)


# %% build_strata


@pytest.mark.parametrize("species", list(nga.Species))
def test_build_strata_places_the_species_in_its_own_slot(species):
    strata = nga.build_strata(species, stems_count=1500)
    field = {
        nga.Species.PINE: "pine",
        nga.Species.SPRUCE: "spruce",
        nga.Species.BIRCH: "deciduous",
    }[species]

    populated = getattr(strata, field)
    assert populated.age == nga.AGE
    assert populated.basal_area == nga.BASAL_AREA
    assert populated.stem_count == 1500
    assert populated.mean_diameter == nga.MEAN_DIAMETER
    assert populated.mean_height == nga.STARTING_HEIGHT[species]

    for other_field in ("pine", "spruce", "deciduous"):
        if other_field != field:
            assert getattr(strata, other_field) == tree_stratum.ZERO_STRATUM


def test_species_code_covers_every_valid_species_with_distinct_codes():
    assert set(nga.SPECIES_CODE) == set(nga.Species)
    assert len(set(nga.SPECIES_CODE.values())) == len(list(nga.Species))


def test_starting_height_covers_every_valid_species():
    assert set(nga.STARTING_HEIGHT) == set(nga.Species)


# %% plan_output


@pytest.mark.parametrize("species", [s.value for s in nga.Species])
def test_plan_output_names_the_csv_after_the_species(species, tmp_path):
    config = nga.NewGrowthConfig.model_validate(_valid_raw(species=species))
    assert nga.plan_output(config, tmp_path) == tmp_path / f"new_growth_{species}.csv"


# %% build_growth_and_yield_table


def test_build_growth_and_yield_table_writes_a_csv_round_tripping_through_the_real_reader(
    tmp_path,
):
    config = nga.NewGrowthConfig.model_validate(
        _valid_raw(species="pine", stems_count=2000)
    )
    site_inputs = nga.resolve_standalone_site_inputs(config)

    table = nga.build_growth_and_yield_table(config, site_inputs)
    output_path = tmp_path / "new_growth_pine.csv"
    table.to_csv(output_path, index=False)

    df = read_allometry_info_from_csv(output_path)
    # New growth must start at age 1 -- this is exactly what
    # ClearCut.new_allometry_includes_age_one enforces downstream.
    assert df["Age"].min() == 1
    assert df.loc[df["Age"] == 1, "N"].iloc[0] == 2000
    assert df.loc[df["Age"] == 1, "Hg"].iloc[0] == nga.STARTING_HEIGHT[nga.Species.PINE]


# %% parse_CLI_arguments


@pytest.fixture
def project_dir(tmp_path):
    # --project-dir is the project root, and must already exist (shared
    # valid_existing_directory). Everything this tool reads and writes lives
    # in its inputs/ folder: new_growth_config.toml going in, the allometry
    # CSV coming out.
    path = tmp_path / "myproject"
    (path / "inputs").mkdir(parents=True)
    return path


@pytest.fixture
def dummy_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "altitude = 150.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "pine"\nstems_count = 2000\n'
    )
    return config_path


def _run_parse_CLI_arguments(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["new_growth_allometry.py", *argv])
    return nga.parse_CLI_arguments()


def test_parse_CLI_arguments_requires_project_dir(
    monkeypatch, dummy_config_file, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(monkeypatch, [f"--config={dummy_config_file}"])


def test_parse_CLI_arguments_reports_a_missing_default_config(
    monkeypatch, project_dir, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(monkeypatch, [f"--project-dir={project_dir}"])


def test_parse_CLI_arguments_finds_its_own_config_inside_project_dir_by_default(
    monkeypatch, project_dir
):
    (project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME).write_text(
        "altitude = 150.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "spruce"\nstems_count = 1800\n'
    )
    cli_args = _run_parse_CLI_arguments(monkeypatch, [f"--project-dir={project_dir}"])
    assert cli_args.config_path == project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME
    assert cli_args.config.species == nga.Species.SPRUCE


def test_parse_CLI_arguments_ignores_another_tools_config_toml(
    monkeypatch, project_dir, capsys
):
    # xml_to_allometry.py/metsakeskus_to_allometry.py default to
    # config.toml, and all three tools can share a project folder. Picking
    # up a neighbour's config would be silently wrong, so this tool looks
    # only for its own name.
    (project_dir / "inputs" / "config.toml").write_text(
        "altitude = 150.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "pine"\nstems_count = 2000\n'
    )
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(monkeypatch, [f"--project-dir={project_dir}"])
    assert nga.DEFAULT_CONFIG_FILENAME in capsys.readouterr().err


def test_parse_CLI_arguments_explicit_config_overrides_the_default_lookup(
    monkeypatch, dummy_config_file, project_dir
):
    (project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME).write_text(
        ""
    )  # would fail to parse if used
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [f"--config={dummy_config_file}", f"--project-dir={project_dir}"],
    )
    assert cli_args.config_path == dummy_config_file


def test_parse_CLI_arguments_blocks_out_of_range_altitude_by_default(
    monkeypatch, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "bad_config.toml"
    config_path.write_text(
        "altitude = 1500.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "pine"\nstems_count = 2000\n'
    )
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [f"--config={config_path}", f"--project-dir={project_dir}"]
        )
    assert "altitude" in capsys.readouterr().err


def test_parse_CLI_arguments_allows_out_of_range_with_override(
    monkeypatch, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "out_of_range_config.toml"
    config_path.write_text(
        "altitude = 1500.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "pine"\nstems_count = 2000\n'
    )
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            f"--config={config_path}",
            f"--project-dir={project_dir}",
            "--allow-out-of-range-values",
        ],
    )
    assert cli_args.config.altitude == 1500.0
    assert "Warning" in capsys.readouterr().out


def test_parse_CLI_arguments_rejects_an_invalid_species(
    monkeypatch, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "bad_species_config.toml"
    config_path.write_text(
        "altitude = 150.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "oak"\nstems_count = 2000\n'
    )
    with pytest.raises(ValueError, match="species"):
        _run_parse_CLI_arguments(
            monkeypatch, [f"--config={config_path}", f"--project-dir={project_dir}"]
        )


def _existing_output_file(project_dir: Path, species: str) -> Path:
    """Leave a previous run's new_growth_<species>.csv where this tool writes."""
    output_dir = project_dir / "inputs" / "allometry" / "new_growth"
    output_dir.mkdir(parents=True)
    output_path = output_dir / f"new_growth_{species}.csv"
    output_path.write_text("left by a previous run\n")
    return output_path


def test_parse_CLI_arguments_refuses_an_existing_output_file(
    monkeypatch, dummy_config_file, project_dir, capsys
):
    # dummy_config_file's species is pine.
    output_path = _existing_output_file(project_dir, "pine")
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [f"--config={dummy_config_file}", f"--project-dir={project_dir}"],
        )
    error = capsys.readouterr().err
    assert "already exists" in error
    # argparse may wrap a long message, so match on the filename alone.
    assert output_path.name in error


def test_parse_CLI_arguments_accepts_an_existing_allometry_folder(
    monkeypatch, dummy_config_file, project_dir
):
    # The stand tools' allometry/ (and even a neighbouring species' file in
    # new_growth/) is no reason to refuse: only this run's own file is.
    _existing_output_file(project_dir, "spruce")
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [f"--config={dummy_config_file}", f"--project-dir={project_dir}"],
    )
    assert cli_args.config.species == nga.Species.PINE


def test_parse_CLI_arguments_creates_no_output_folder(
    monkeypatch, dummy_config_file, project_dir
):
    _run_parse_CLI_arguments(
        monkeypatch, [f"--config={dummy_config_file}", f"--project-dir={project_dir}"]
    )
    assert not (project_dir / "inputs" / "allometry").exists()


def test_parse_CLI_arguments_dry_run_defaults_to_false(
    monkeypatch, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch, [f"--config={dummy_config_file}", f"--project-dir={project_dir}"]
    )
    assert cli_args.dry_run is False


def test_parse_CLI_arguments_dry_run_still_refuses_an_existing_output_file(
    monkeypatch, dummy_config_file, project_dir, capsys
):
    _existing_output_file(project_dir, "pine")
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                f"--config={dummy_config_file}",
                f"--project-dir={project_dir}",
                "--dry-run",
            ],
        )
    assert "already exists" in capsys.readouterr().err


# %% main -- dry run vs. real run


def _write_valid_config(project_dir: Path, **overrides) -> None:
    raw = _valid_raw(**overrides)
    lines = [
        f"altitude = {raw['altitude']}",
        f"ddy = {raw['ddy']}",
        f"fertility_class = {raw['fertility_class']}",
        f"x = {raw['x']}",
        f"y = {raw['y']}",
        f'species = "{raw["species"]}"',
        f"stems_count = {raw['stems_count']}",
    ]
    (project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME).write_text(
        "\n".join(lines) + "\n"
    )


def test_main_dry_run_writes_nothing(monkeypatch, project_dir, capsys):
    _write_valid_config(project_dir)
    monkeypatch.setattr(
        "sys.argv",
        ["new_growth_allometry.py", f"--project-dir={project_dir}", "--dry-run"],
    )
    nga.main()
    assert not (project_dir / "inputs" / "allometry").exists()
    assert "Would write" in capsys.readouterr().out


def test_main_real_run_writes_the_expected_csv(monkeypatch, project_dir):
    _write_valid_config(project_dir, species="birch", stems_count=3000)
    monkeypatch.setattr(
        "sys.argv", ["new_growth_allometry.py", f"--project-dir={project_dir}"]
    )
    nga.main()

    output_path = (
        project_dir / "inputs" / "allometry" / "new_growth" / "new_growth_birch.csv"
    )
    assert output_path.exists()
    df = read_allometry_info_from_csv(output_path)
    assert df["Age"].min() == 1


def test_main_two_species_in_a_row_both_succeed(monkeypatch, project_dir):
    # One file per species into the same new_growth/ folder: the second run
    # must not be refused because the first one created the folder.
    new_growth_dir = project_dir / "inputs" / "allometry" / "new_growth"
    for species in ["pine", "spruce"]:
        _write_valid_config(project_dir, species=species)
        monkeypatch.setattr(
            "sys.argv", ["new_growth_allometry.py", f"--project-dir={project_dir}"]
        )
        nga.main()
    assert sorted(p.name for p in new_growth_dir.iterdir()) == [
        "new_growth_pine.csv",
        "new_growth_spruce.csv",
    ]


# %% Sourced mode: NewGrowthSourcedConfig
#
# Sourced mode (--stand-data/--stand-id) reads altitude, ddy,
# fertility_class and the YKJ coordinates off a StandDataDocument, so its
# config declares none of them. NewGrowthConfig subclasses
# NewGrowthSourcedConfig -- a standalone config is a sourced config plus
# those five fields -- which is what makes the extra="forbid" rejection
# below fall out for free rather than needing a cross-field validator.

SOURCED_ONLY_FIELDS = ["altitude", "ddy", "fertility_class", "x", "y"]


def _valid_sourced_raw(**overrides) -> dict:
    raw = {"species": "pine", "stems_count": 2000}
    raw.update(overrides)
    return raw


def test_new_growth_config_is_a_sourced_config_plus_the_sourced_fields():
    assert issubclass(nga.NewGrowthConfig, nga.NewGrowthSourcedConfig)
    assert set(nga.NewGrowthConfig.model_fields) - set(
        nga.NewGrowthSourcedConfig.model_fields
    ) == set(SOURCED_ONLY_FIELDS)


@pytest.mark.parametrize("sourced_field", SOURCED_ONLY_FIELDS)
def test_sourced_config_rejects_a_hand_typed_sourced_field(sourced_field):
    # Silently ignoring a stale `altitude = 100` while actually using the
    # document's value would be the worst of both worlds, so extra="forbid"
    # has to reject it outright.
    raw = _valid_sourced_raw(**{sourced_field: 100.0})
    with pytest.raises(ValueError, match=sourced_field):
        nga.NewGrowthSourcedConfig.model_validate(raw)


@pytest.mark.parametrize("missing_field", ["species", "stems_count"])
def test_sourced_config_still_requires_species_and_stems_count(missing_field):
    raw = _valid_sourced_raw()
    del raw[missing_field]
    with pytest.raises(ValueError, match=missing_field):
        nga.NewGrowthSourcedConfig.model_validate(raw)


def test_sourced_config_applies_the_same_projection_defaults():
    config = nga.NewGrowthSourcedConfig.model_validate(_valid_sourced_raw())
    assert (config.n_trees, config.start_year, config.end_year, config.step_years) == (
        20,
        5,
        80,
        5,
    )


def test_sourced_config_normalizes_species_the_same_way():
    config = nga.NewGrowthSourcedConfig.model_validate(
        _valid_sourced_raw(species=" Spruce ")
    )
    assert config.species == nga.Species.SPRUCE


def test_load_new_growth_sourced_config_reads_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('species = "pine"\nstems_count = 2000\n')
    config = nga.load_new_growth_sourced_config(config_path)
    assert config == nga.NewGrowthSourcedConfig.model_validate(_valid_sourced_raw())


# %% Sourced mode: resolve_sourced_site_inputs
#
# Driven by an in-memory StandDataDocument rather than a JSON file on disk:
# load_stand_data_document_from_json actually reading a real file is
# covered once, in test_shared_allometry_tool_utils.py, and repeating it
# here would only re-test Pydantic.

SOURCED_STAND_ID = StandID("stand-1")


def _stand_data_document(
    site_fertility_class: int = 3,
    x_ykj: int = 338,
    y_ykj: int = 7042,
    allometry_dir: Path = Path("/project/inputs/allometry"),
) -> stand_data.StandDataDocument:
    """A one-stand document, with only the fields sourced mode actually
    reads varied. allometry_file_per_layer is required by StandData but never
    looked at here -- this tool generates an allometry file, it does not
    consume the stand's existing ones -- so the CSV is never written.
    allometry_dir must be absolute (a document's paths are, in memory), and
    inside the document's folder if the document is dumped to disk."""
    return stand_data.StandDataDocument(
        crs=stand_data.SOURCE_CRS,
        altitude=100.0,
        ddy=1200.0,
        stands={
            SOURCED_STAND_ID: stand_data.StandData(
                site_fertility_class=site_fertility_class,
                allometry_file_per_layer={
                    CanopyLayerName.dominant: AllometryFileAndSpecies(
                        file_path=allometry_dir / "dominant.csv", species_id=1
                    )
                },
                x_ykj=x_ykj,
                y_ykj=y_ykj,
            )
        },
    )


def test_resolve_sourced_site_inputs_reads_each_value_from_its_own_place():
    document = _stand_data_document(site_fertility_class=4, x_ykj=340, y_ykj=7000)
    site_inputs = nga.resolve_sourced_site_inputs(
        stand_data_document=document, stand_id=SOURCED_STAND_ID
    )
    # altitude/ddy are project-global (document root); the rest belong to
    # the one named stand.
    assert site_inputs == nga.SiteInputs(
        altitude=100.0, ddy=1200.0, site_fertility_class=4, x_ykj=340, y_ykj=7000
    )


def test_resolve_sourced_site_inputs_does_not_convert_the_coordinates():
    # StandData.x_ykj/.y_ykj are already YKJ grid units -- running
    # point_to_ykj over them again would be a second, bogus conversion.
    document = _stand_data_document(x_ykj=338, y_ykj=7042)
    site_inputs = nga.resolve_sourced_site_inputs(
        stand_data_document=document, stand_id=SOURCED_STAND_ID
    )
    assert (site_inputs.x_ykj, site_inputs.y_ykj) == (338, 7042)


def test_resolve_sourced_site_inputs_raises_on_an_unknown_stand():
    document = _stand_data_document()
    with pytest.raises(KeyError):
        nga.resolve_sourced_site_inputs(
            stand_data_document=document, stand_id=StandID("no-such-stand")
        )


def test_sourced_mode_produces_the_same_table_as_standalone_mode():
    # The whole point of sourced mode: same site facts in, same allometry
    # out -- only the route they travelled differs.
    standalone_config = nga.NewGrowthConfig.model_validate(
        _valid_raw(species="pine", stems_count=2000, fertility_class=3)
    )
    standalone_inputs = nga.resolve_standalone_site_inputs(standalone_config)

    document = _stand_data_document(
        site_fertility_class=standalone_config.fertility_class,
        x_ykj=standalone_inputs.x_ykj,
        y_ykj=standalone_inputs.y_ykj,
    )
    sourced_config = nga.NewGrowthSourcedConfig.model_validate(
        _valid_sourced_raw(species="pine", stems_count=2000)
    )
    sourced_inputs = nga.resolve_sourced_site_inputs(
        stand_data_document=document, stand_id=SOURCED_STAND_ID
    )

    assert sourced_inputs == standalone_inputs
    pd.testing.assert_frame_equal(
        nga.build_growth_and_yield_table(sourced_config, sourced_inputs),
        nga.build_growth_and_yield_table(standalone_config, standalone_inputs),
    )


# %% Sourced mode: CLI flag pairing
#
# Thin parser logic: --stand-data and --stand-id are meaningless apart, so
# either alone is a parser.error. These only check the pairing and which
# mode it selects, not a full end-to-end run.


@pytest.fixture
def stand_data_file(tmp_path):
    path = tmp_path / stand_data.STAND_DATA_FILENAME
    stand_data.dump_stand_data_document(
        output_path=path,
        document=_stand_data_document(allometry_dir=tmp_path / "allometry"),
    )
    return path


@pytest.fixture
def sourced_config_file(project_dir):
    (project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME).write_text(
        'species = "pine"\nstems_count = 2000\n'
    )
    return project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME


def test_parse_CLI_arguments_rejects_stand_data_without_stand_id(
    monkeypatch, project_dir, sourced_config_file, stand_data_file, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [f"--project-dir={project_dir}", f"--stand-data={stand_data_file}"],
        )
    assert (
        "--stand-data and --stand-id must be given together" in capsys.readouterr().err
    )


def test_parse_CLI_arguments_rejects_stand_id_without_stand_data(
    monkeypatch, project_dir, sourced_config_file, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [f"--project-dir={project_dir}", f"--stand-id={SOURCED_STAND_ID}"],
        )
    assert (
        "--stand-data and --stand-id must be given together" in capsys.readouterr().err
    )


def test_parse_CLI_arguments_without_either_flag_stays_in_standalone_mode(
    monkeypatch, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch, [f"--config={dummy_config_file}", f"--project-dir={project_dir}"]
    )
    assert isinstance(cli_args.origin, nga.StandaloneOrigin)
    assert isinstance(cli_args.config, nga.NewGrowthConfig)
    # Standalone mode still converts the config's ETRS-TM35FIN pair itself.
    assert cli_args.site_inputs == nga.resolve_standalone_site_inputs(cli_args.config)


def test_parse_CLI_arguments_with_both_flags_runs_sourced_mode(
    monkeypatch, project_dir, sourced_config_file, stand_data_file
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            f"--project-dir={project_dir}",
            f"--stand-data={stand_data_file}",
            f"--stand-id={SOURCED_STAND_ID}",
        ],
    )
    assert cli_args.origin == nga.SourcedOrigin(
        stand_data_path=stand_data_file, stand_id=SOURCED_STAND_ID
    )
    assert not isinstance(cli_args.config, nga.NewGrowthConfig)
    assert cli_args.site_inputs == nga.SiteInputs(
        altitude=100.0, ddy=1200.0, site_fertility_class=3, x_ykj=338, y_ykj=7042
    )


def test_parse_CLI_arguments_reports_an_unknown_stand_id(
    monkeypatch, project_dir, sourced_config_file, stand_data_file, capsys
):
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                f"--project-dir={project_dir}",
                f"--stand-data={stand_data_file}",
                "--stand-id=no-such-stand",
            ],
        )
    error = capsys.readouterr().err
    assert "no-such-stand" in error
    # The document's own keys are listed, since a typo is likelier than a
    # genuinely absent stand.
    assert SOURCED_STAND_ID in error


def test_parse_CLI_arguments_rejects_a_sourced_config_holding_a_sourced_field(
    monkeypatch, project_dir, stand_data_file
):
    (project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME).write_text(
        'species = "pine"\nstems_count = 2000\naltitude = 100.0\n'
    )
    with pytest.raises(ValueError, match="altitude"):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                f"--project-dir={project_dir}",
                f"--stand-data={stand_data_file}",
                f"--stand-id={SOURCED_STAND_ID}",
            ],
        )


# %% The YKJ sanity range, in both modes
#
# validate_x_y_ykj itself is tested in test_shared_allometry_tool_utils.py;
# these check that this tool actually applies it, to the *resolved* values,
# whichever mode produced them.


def test_parse_CLI_arguments_blocks_an_out_of_range_ykj_coordinate(
    monkeypatch, project_dir, tmp_path, capsys
):
    # y = 4_000_000 m is a real EPSG:3067 northing, so it converts happily;
    # it just lands nowhere near Finland (y_ykj = 4002, well below
    # Y_YKJ_MIN). Exactly the mistyped-coordinate case the range is for.
    config_path = tmp_path / "far_south_config.toml"
    config_path.write_text(
        "altitude = 150.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 4000000.0\nspecies = "pine"\nstems_count = 2000\n'
    )
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch, [f"--config={config_path}", f"--project-dir={project_dir}"]
        )
    assert "y_ykj" in capsys.readouterr().err


def test_parse_CLI_arguments_allows_an_out_of_range_ykj_coordinate_with_override(
    monkeypatch, project_dir, tmp_path, capsys
):
    config_path = tmp_path / "far_south_config.toml"
    config_path.write_text(
        "altitude = 150.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 4000000.0\nspecies = "pine"\nstems_count = 2000\n'
    )
    cli_args = _run_parse_CLI_arguments(
        monkeypatch,
        [
            f"--config={config_path}",
            f"--project-dir={project_dir}",
            "--allow-out-of-range-values",
        ],
    )
    assert cli_args.site_inputs.y_ykj == 4002
    assert "Warning" in capsys.readouterr().out


def test_a_sourced_out_of_range_ykj_coordinate_is_a_clean_cli_error(
    monkeypatch, project_dir, sourced_config_file, tmp_path, capsys
):
    # StandData.x_ykj/.y_ykj carry the same shared bounds validate_x_y_ykj
    # checks (stand_data.X/Y_YKJ_MIN/MAX), so in sourced mode the
    # coordinate is stopped while the document is being read rather than by
    # the CLI check afterwards. Same range, earlier catch. What matters
    # here is that it still exits the way every other bad input to these
    # tools does -- parser.error naming the file, not a raw traceback.
    stand_data_path = tmp_path / stand_data.STAND_DATA_FILENAME
    stand_data.dump_stand_data_document(
        output_path=stand_data_path,
        document=_stand_data_document(allometry_dir=tmp_path / "allometry"),
    )
    stand_data_path.write_text(
        stand_data_path.read_text().replace('"y_ykj":7042', '"y_ykj":4002')
    )
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                f"--project-dir={project_dir}",
                f"--stand-data={stand_data_path}",
                f"--stand-id={SOURCED_STAND_ID}",
            ],
        )
    error = capsys.readouterr().err
    assert str(stand_data_path) in error
    assert "y_ykj" in error


def test_a_malformed_stand_data_document_is_a_clean_cli_error(
    monkeypatch, project_dir, sourced_config_file, tmp_path, capsys
):
    stand_data_path = tmp_path / stand_data.STAND_DATA_FILENAME
    stand_data_path.write_text('{"not": "a stand data document"}')
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [
                f"--project-dir={project_dir}",
                f"--stand-data={stand_data_path}",
                f"--stand-id={SOURCED_STAND_ID}",
            ],
        )
    assert "Could not read" in capsys.readouterr().err


# %% main -- a sourced run end to end


def test_main_sourced_run_writes_the_expected_csv_and_names_its_source(
    monkeypatch, project_dir, stand_data_file, capsys
):
    (project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME).write_text(
        'species = "spruce"\nstems_count = 1800\n'
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "new_growth_allometry.py",
            f"--project-dir={project_dir}",
            f"--stand-data={stand_data_file}",
            f"--stand-id={SOURCED_STAND_ID}",
        ],
    )
    nga.main()

    output_path = (
        project_dir / "inputs" / "allometry" / "new_growth" / "new_growth_spruce.csv"
    )
    assert output_path.exists()
    assert read_allometry_info_from_csv(output_path)["Age"].min() == 1

    # With no informational JSON for this tool, the Reading section is the
    # only place the user ever sees which document/stand was read.
    printed = capsys.readouterr().out
    assert str(stand_data_file.resolve()) in printed
    assert SOURCED_STAND_ID in printed
    assert "from document root" in printed


def test_main_sourced_run_in_a_project_whose_stand_allometry_already_exists(
    monkeypatch, project_dir
):
    # The bug ticket 26 fixed: sourced mode reads the stand_data.json that a
    # stand tool wrote, and that same stand tool created inputs/allometry/
    # with it. Refusing that folder meant sourced mode could never run in
    # its own project.
    allometry_dir = project_dir / "inputs" / "allometry"
    allometry_dir.mkdir()
    stand_csv = allometry_dir / "dominant.csv"
    stand_csv.write_text("the stand tool's own CSV\n")
    stand_data_path = project_dir / "inputs" / stand_data.STAND_DATA_FILENAME
    stand_data.dump_stand_data_document(
        output_path=stand_data_path,
        document=_stand_data_document(allometry_dir=allometry_dir),
    )
    (project_dir / "inputs" / nga.DEFAULT_CONFIG_FILENAME).write_text(
        'species = "pine"\nstems_count = 2000\n'
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "new_growth_allometry.py",
            f"--project-dir={project_dir}",
            f"--stand-data={stand_data_path}",
            f"--stand-id={SOURCED_STAND_ID}",
        ],
    )
    nga.main()

    assert (allometry_dir / "new_growth" / "new_growth_pine.csv").exists()
    # The stand set is left exactly as it was.
    assert sorted(p.name for p in allometry_dir.iterdir()) == [
        "dominant.csv",
        "new_growth",
    ]
    assert stand_csv.read_text() == "the stand tool's own CSV\n"
