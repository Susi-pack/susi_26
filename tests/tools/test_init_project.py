# %% Imports
import argparse
import tomllib
from pathlib import Path

import pytest

from tools.init_project import init_project
from tools.init_project.init_project import DataSource

# init_project.py's work is split so that everything that touches a
# filesystem takes the target path as an argument: these tests drive
# `create_project` straight against `tmp_path` and never go near the real
# projects root. Only the one smoke test at the bottom exercises the
# prompting layer and `main()`, and it redirects the projects root with
# SUSI_PROJECTS_ROOT first.


@pytest.fixture
def new_project(tmp_path) -> Path:
    """A project folder path that does not exist yet -- what main() hands
    create_project after refusing an existing one."""
    return tmp_path / "paroninkorpi"


# %% create_project -- the layout


def test_create_project_creates_the_layout(new_project):
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    assert (new_project / "inputs").is_dir()
    assert (new_project / "outputs").is_dir()
    assert (new_project / "susi_calls.py").is_file()
    assert (new_project / "README.md").is_file()
    assert (new_project / ".gitignore").is_file()


def test_create_project_leaves_the_folders_empty(new_project):
    # No .gitkeep in either folder. A project committed to a git repository
    # of its own before its first run loses both (git does not track empty
    # directories), but that repository is the user's to manage -- this tool
    # does not create it and does not work around it. README.md says so.
    init_project.create_project("paroninkorpi", DataSource.NONE, new_project)

    assert list((new_project / "outputs").iterdir()) == []
    assert list((new_project / "inputs").iterdir()) == []


def test_create_project_never_creates_the_allometry_folder(new_project):
    # check_output_dir_available refuses to run an allometry tool into an
    # existing folder, so pre-creating inputs/allometry/ here would break
    # every fresh project on its first tool run.
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    assert not (new_project / "inputs" / "allometry").exists()


def test_create_project_refuses_an_existing_project(new_project):
    init_project.create_project("paroninkorpi", DataSource.NONE, new_project)

    with pytest.raises(FileExistsError):
        init_project.create_project("paroninkorpi", DataSource.NONE, new_project)


def test_create_project_refuses_any_existing_folder_and_seeds_nothing(new_project):
    # Not just a folder that looks like a project: an existing folder with
    # someone's data loose in it must be refused too, and left untouched.
    new_project.mkdir()
    (new_project / "my_data.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(FileExistsError):
        init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    assert [path.name for path in new_project.iterdir()] == ["my_data.txt"]


def test_create_project_reports_every_path_it_created(new_project):
    created = init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    assert set(created) == {path for path in new_project.rglob("*")} | {new_project}


# %% create_project -- the config seed


@pytest.mark.parametrize("source", [DataSource.XML, DataSource.METSAKESKUS])
def test_create_project_copies_the_chosen_tools_config_template(source, new_project):
    init_project.create_project("paroninkorpi", source, new_project)

    # config.toml, not a per-tool name: that is the filename both generating
    # tools look up inside a project's inputs/ folder.
    config_path = new_project / "inputs" / "config.toml"
    template_path = init_project.SOURCE_SEEDS[source].config_template_path
    assert template_path is not None
    assert config_path.read_text(encoding="utf-8") == template_path.read_text(
        encoding="utf-8"
    )

    with open(config_path, "rb") as config_file:
        assert tomllib.load(config_file)  # a real, parseable config, not a stub


def test_create_project_writes_no_config_for_source_none(new_project):
    init_project.create_project("paroninkorpi", DataSource.NONE, new_project)

    assert not (new_project / "inputs" / "config.toml").exists()
    assert list((new_project / "inputs").iterdir()) == []


# %% create_project -- the seeded script


def test_seed_script_path_exists():
    # The one place the seed's location is written down. Ticket 17 repoints
    # it at src/example_scripts/susi_calls.py; this test is what catches the
    # repoint being forgotten.
    assert init_project.SEED_SCRIPT_PATH.is_file()


def test_create_project_copies_the_seed_script_verbatim_under_a_header(new_project):
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    seeded = (new_project / "susi_calls.py").read_text(encoding="utf-8")
    assert seeded.endswith(init_project.SEED_SCRIPT_PATH.read_text(encoding="utf-8"))
    # The header says what to edit first, and names the project it was made
    # for -- the copy's project_id is still the seed's placeholder.
    assert "paroninkorpi" in seeded
    assert "project_id" in seeded


# %% create_project -- .gitignore


def active_lines(text: str) -> list[str]:
    """The lines git would actually act on: no blanks, no comments."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_gitignore_is_written_for_every_source(new_project):
    # Written whether or not the user ever runs git init: harmless if
    # unused, correct if they version the project later.
    init_project.create_project("paroninkorpi", DataSource.NONE, new_project)

    assert (new_project / ".gitignore").is_file()


def test_gitignore_ignores_the_heavy_artifacts(new_project):
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    patterns = active_lines((new_project / ".gitignore").read_text(encoding="utf-8"))
    assert "*.nc" in patterns
    assert "*.tif" in patterns


def test_gitignore_keeps_the_provenance_files_tracked(new_project):
    # params.json and metadata.json are tiny and they are the entire
    # provenance record of a run: ignoring them is what would turn a shared
    # project from reproducible into merely readable.
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    patterns = active_lines((new_project / ".gitignore").read_text(encoding="utf-8"))
    assert not [
        pattern
        for pattern in patterns
        if "params.json" in pattern or "metadata.json" in pattern
    ]


def test_gitignore_leaves_weather_and_gpkg_for_the_user_to_decide(new_project):
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    gitignore = (new_project / ".gitignore").read_text(encoding="utf-8")
    assert "weather" in gitignore
    assert ".gpkg" in gitignore
    # ...but only as commented-out suggestions.
    assert not [
        pattern
        for pattern in active_lines(gitignore)
        if "weather" in pattern or ".gpkg" in pattern
    ]


# %% create_project -- README


def test_readme_points_at_the_standard_analysis_notebooks(new_project):
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    assert "paroninkorpi" in readme
    assert "src/analysis/notebooks/" in readme
    # index.ipynb is deliberately not copied into a project: its links are
    # relative to its siblings and would all be broken on arrival.
    assert not (new_project / "index.ipynb").exists()


def test_readme_warns_about_git_clean(new_project):
    # git clean -xdf run from the SUSI checkout deletes an ordinary
    # data-only project folder, but skips a nested git repository.
    init_project.create_project("paroninkorpi", DataSource.XML, new_project)

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    assert "git clean -xdf" in readme
    # ...and about the empty folders a first commit would drop, which is why
    # no .gitkeep is written for them.
    assert "does not track empty folders" in readme


# %% valid_project_id


def test_valid_project_id_strips_and_returns():
    assert init_project.valid_project_id("  paroninkorpi \n") == "paroninkorpi"


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "a/b", "../escape"])
def test_valid_project_id_rejects_non_names(name):
    # ArgumentTypeError, like the shared validators in input_validation.py:
    # it is what lets --name be validated by argparse itself.
    with pytest.raises(argparse.ArgumentTypeError):
        init_project.valid_project_id(name)


# %% The prompting layer


def test_main_prompts_for_what_the_flags_did_not_give(tmp_path, monkeypatch, capsys):
    """The one test that drives the interactive path end to end: no flags at
    all, both answers typed, everything landing under a redirected projects
    root. Every other test above calls create_project directly -- flags and
    pure functions are what tests are for, not a prompt transcript."""
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("SUSI_PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr("sys.argv", ["init_project.py"])

    answers = iter(["paroninkorpi", "1"])  # name, then "1" = the xml source
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    init_project.main()

    assert (projects_root / "paroninkorpi" / "inputs" / "config.toml").is_file()
    # The completion line: the folder is the user's, not this repo's.
    printed = capsys.readouterr().out
    assert "not tracked" in printed
    assert "repository of its own" in printed


def test_main_refuses_an_existing_project(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    (projects_root / "paroninkorpi").mkdir(parents=True)
    monkeypatch.setenv("SUSI_PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(
        "sys.argv",
        ["init_project.py", "--name", "paroninkorpi", "--source", "none"],
    )

    with pytest.raises(SystemExit):
        init_project.main()
