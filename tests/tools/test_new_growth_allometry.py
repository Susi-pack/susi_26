# %% Imports
import tomllib
from pathlib import Path

import pytest

from susi.io.susi_parameter_model import read_allometry_info_from_csv
from tools.new_growth_allometry import new_growth_allometry as nga
from tools.shared_allometry_tool_utils import input_validation, tree_stratum

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

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "tools"
    / "new_growth_allometry"
    / "default_config.toml"
)


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
    slot = {nga.Species.PINE: 0, nga.Species.SPRUCE: 1, nga.Species.BIRCH: 2}[species]

    populated = strata[slot]
    assert populated.age == nga.AGE
    assert populated.basal_area == nga.BASAL_AREA
    assert populated.stem_count == 1500
    assert populated.mean_diameter == nga.MEAN_DIAMETER
    assert populated.mean_height == nga.STARTING_HEIGHT[species]

    for other_slot in range(3):
        if other_slot != slot:
            assert strata[other_slot] == tree_stratum.ZERO_STRATUM


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
    x_ykj, y_ykj = nga.point_to_ykj(config.x, config.y)

    table = nga.build_growth_and_yield_table(config, x_ykj, y_ykj)
    output_path = tmp_path / "new_growth_pine.csv"
    table.to_csv(output_path, index=False)

    df, species_id = read_allometry_info_from_csv(output_path)
    assert species_id == nga.SPECIES_CODE[nga.Species.PINE]
    # New growth must start at age 1 -- this is exactly what
    # ClearCut.new_allometry_includes_age_one enforces downstream.
    assert df["Age"].min() == 1
    assert df.loc[df["Age"] == 1, "N"].iloc[0] == 2000
    assert df.loc[df["Age"] == 1, "Hg"].iloc[0] == nga.STARTING_HEIGHT[nga.Species.PINE]


# %% parse_CLI_arguments


@pytest.fixture
def project_dir(tmp_path):
    # --project-dir must already exist (shared valid_existing_directory) --
    # the folder the docs have the user set up beforehand, with config.toml
    # inside it.
    path = tmp_path / "myproject"
    path.mkdir()
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


def test_parse_CLI_arguments_finds_config_toml_inside_project_dir_by_default(
    monkeypatch, project_dir
):
    (project_dir / "config.toml").write_text(
        "altitude = 150.0\nddy = 1200.0\nfertility_class = 3\n"
        'x = 379930.3\ny = 7039150.8\nspecies = "spruce"\nstems_count = 1800\n'
    )
    cli_args = _run_parse_CLI_arguments(monkeypatch, [f"--project-dir={project_dir}"])
    assert cli_args.config_path == project_dir / "config.toml"
    assert cli_args.config.species == nga.Species.SPRUCE


def test_parse_CLI_arguments_explicit_config_overrides_the_default_lookup(
    monkeypatch, dummy_config_file, project_dir
):
    (project_dir / "config.toml").write_text("")  # would fail to parse if used
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


def test_parse_CLI_arguments_refuses_existing_default_output_dir(
    monkeypatch, dummy_config_file, project_dir, capsys
):
    (project_dir / "allometry").mkdir()
    with pytest.raises(SystemExit):
        _run_parse_CLI_arguments(
            monkeypatch,
            [f"--config={dummy_config_file}", f"--project-dir={project_dir}"],
        )
    assert "already exists" in capsys.readouterr().err


def test_parse_CLI_arguments_creates_no_output_folder(
    monkeypatch, dummy_config_file, project_dir
):
    _run_parse_CLI_arguments(
        monkeypatch, [f"--config={dummy_config_file}", f"--project-dir={project_dir}"]
    )
    assert not (project_dir / "allometry").exists()


def test_parse_CLI_arguments_dry_run_defaults_to_false(
    monkeypatch, dummy_config_file, project_dir
):
    cli_args = _run_parse_CLI_arguments(
        monkeypatch, [f"--config={dummy_config_file}", f"--project-dir={project_dir}"]
    )
    assert cli_args.dry_run is False


def test_parse_CLI_arguments_dry_run_still_refuses_existing_output_dir(
    monkeypatch, dummy_config_file, project_dir, capsys
):
    (project_dir / "allometry").mkdir()
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
    (project_dir / "config.toml").write_text("\n".join(lines) + "\n")


def test_main_dry_run_writes_nothing(monkeypatch, project_dir, capsys):
    _write_valid_config(project_dir)
    monkeypatch.setattr(
        "sys.argv",
        ["new_growth_allometry.py", f"--project-dir={project_dir}", "--dry-run"],
    )
    nga.main()
    assert not (project_dir / "allometry").exists()
    assert "Would write" in capsys.readouterr().out


def test_main_real_run_writes_the_expected_csv(monkeypatch, project_dir):
    _write_valid_config(project_dir, species="birch", stems_count=3000)
    monkeypatch.setattr(
        "sys.argv", ["new_growth_allometry.py", f"--project-dir={project_dir}"]
    )
    nga.main()

    output_path = project_dir / "allometry" / "new_growth_birch.csv"
    assert output_path.exists()
    df, species_id = read_allometry_info_from_csv(output_path)
    assert species_id == nga.SPECIES_CODE[nga.Species.BIRCH]
    assert df["Age"].min() == 1
