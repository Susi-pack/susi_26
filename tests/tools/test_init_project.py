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
    init_project.create_project("paroninkorpi", DataSource.XML, new_project, True, True)

    assert (new_project / "inputs").is_dir()
    assert (new_project / "outputs").is_dir()
    # Not required by anything -- a script names its own data paths -- but
    # the preferred home for a project's raw data, so it is there from the
    # start.
    assert (new_project / "data").is_dir()
    assert (new_project / "susi_calls.py").is_file()
    assert (new_project / "README.md").is_file()
    assert (new_project / ".gitignore").is_file()


def test_create_project_leaves_the_folders_empty(new_project):
    # No .gitkeep in either folder. A project committed to a git repository
    # of its own before its first run loses both (git does not track empty
    # directories), but that repository is the user's to manage -- this tool
    # does not create it and does not work around it. A seeded README says
    # so, for the projects that asked for one.
    init_project.create_project(
        "paroninkorpi", DataSource.NONE, new_project, False, False
    )

    assert list((new_project / "outputs").iterdir()) == []
    assert list((new_project / "inputs").iterdir()) == []
    assert list((new_project / "data").iterdir()) == []


def test_create_project_never_creates_the_allometry_folder(new_project):
    # check_output_dir_available refuses to run an allometry tool into an
    # existing folder, so pre-creating inputs/allometry/ here would break
    # every fresh project on its first tool run.
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, False, False
    )

    assert not (new_project / "inputs" / "allometry").exists()


def test_create_project_refuses_an_existing_project(new_project):
    init_project.create_project(
        "paroninkorpi", DataSource.NONE, new_project, False, False
    )

    with pytest.raises(FileExistsError):
        init_project.create_project(
            "paroninkorpi", DataSource.NONE, new_project, False, False
        )


def test_create_project_refuses_any_existing_folder_and_seeds_nothing(new_project):
    # Not just a folder that looks like a project: an existing folder with
    # someone's data loose in it must be refused too, and left untouched.
    new_project.mkdir()
    (new_project / "my_data.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(FileExistsError):
        init_project.create_project(
            "paroninkorpi", DataSource.XML, new_project, False, False
        )

    assert [path.name for path in new_project.iterdir()] == ["my_data.txt"]


def test_create_project_reports_every_path_it_created(new_project):
    created = init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, True, True
    )

    assert set(created) == {path for path in new_project.rglob("*")} | {new_project}


# %% create_project -- the two optional files


def test_create_project_writes_neither_optional_file_by_default(new_project):
    # The defaults the CLI applies when the user just presses Enter twice.
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, False, False
    )

    assert not (new_project / "README.md").exists()
    assert not (new_project / ".gitignore").exists()
    # The run script is not one of the optional extras: a project with
    # nothing to run is not worth creating.
    assert (new_project / "susi_calls.py").is_file()


@pytest.mark.parametrize(
    ("with_readme", "with_gitignore"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_create_project_writes_exactly_what_was_asked_for(
    with_readme, with_gitignore, new_project
):
    created = init_project.create_project(
        "paroninkorpi", DataSource.NONE, new_project, with_readme, with_gitignore
    )

    assert (new_project / "README.md").is_file() == with_readme
    assert (new_project / ".gitignore").is_file() == with_gitignore
    # The returned list is what main() prints, so it has to agree with the
    # folder rather than with the request.
    assert ((new_project / "README.md") in created) == with_readme
    assert ((new_project / ".gitignore") in created) == with_gitignore


# %% create_project -- the config seed


@pytest.mark.parametrize("source", [DataSource.XML, DataSource.METSAKESKUS])
def test_create_project_copies_the_chosen_tools_config_template(source, new_project):
    init_project.create_project("paroninkorpi", source, new_project, False, False)

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
    init_project.create_project(
        "paroninkorpi", DataSource.NONE, new_project, False, False
    )

    assert not (new_project / "inputs" / "config.toml").exists()
    assert list((new_project / "inputs").iterdir()) == []


# %% create_project -- the seeded script


def test_seed_script_path_exists():
    # The one place the seed's location is written down. Ticket 17 repoints
    # it at src/example_scripts/susi_calls.py; this test is what catches the
    # repoint being forgotten.
    assert init_project.SEED_SCRIPT_PATH.is_file()


def test_create_project_copies_the_seed_script_verbatim_under_a_header(new_project):
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, False, False
    )

    seeded = (new_project / "susi_calls.py").read_text(encoding="utf-8")
    assert seeded.endswith(init_project.SEED_SCRIPT_PATH.read_text(encoding="utf-8"))
    # The header says what to edit first, and names the project it was made
    # for -- the copy's project_dir still names the seed's placeholder.
    assert 'project_dir("paroninkorpi")' in seeded


def test_seed_script_header_points_at_the_readme_when_there_is_one(new_project):
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, True, False
    )

    header = (new_project / "susi_calls.py").read_text(encoding="utf-8")
    assert "See README.md next to this file." in header


def test_seed_script_header_does_not_point_at_a_missing_readme(new_project):
    # The header is the first thing the user reads in the file they are told
    # to start from, so it must not send them to a file that is not there.
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, False, False
    )

    header = (new_project / "susi_calls.py").read_text(encoding="utf-8")
    assert "README" not in header
    # ...and the rest of the header is intact.
    assert "python susi_calls.py" in header


# %% create_project -- .gitignore


def active_lines(text: str) -> list[str]:
    """The lines git would actually act on: no blanks, no comments."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_gitignore_is_written_when_asked_for(new_project):
    # Harmless if the project never becomes a git repository, correct if it
    # does -- but only when asked for.
    init_project.create_project(
        "paroninkorpi", DataSource.NONE, new_project, False, True
    )

    assert (new_project / ".gitignore").is_file()


def test_gitignore_ignores_the_heavy_artifacts(new_project):
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, False, True
    )

    patterns = active_lines((new_project / ".gitignore").read_text(encoding="utf-8"))
    assert "*.nc" in patterns
    assert "*.tif" in patterns


def test_gitignore_keeps_the_provenance_files_tracked(new_project):
    # params.json and metadata.json are tiny and they are the entire
    # provenance record of a run: ignoring them is what would turn a shared
    # project from reproducible into merely readable.
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, False, True
    )

    patterns = active_lines((new_project / ".gitignore").read_text(encoding="utf-8"))
    assert not [
        pattern
        for pattern in patterns
        if "params.json" in pattern or "metadata.json" in pattern
    ]


def test_gitignore_leaves_weather_and_gpkg_for_the_user_to_decide(new_project):
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, False, True
    )

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
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, True, False
    )

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    assert "paroninkorpi" in readme
    assert "src/analysis/notebooks/" in readme
    # index.ipynb is deliberately not copied into a project: its links are
    # relative to its siblings and would all be broken on arrival.
    assert not (new_project / "index.ipynb").exists()


def test_readme_warns_about_git_clean(new_project):
    # git clean -xdf run from the SUSI checkout deletes an ordinary
    # data-only project folder, but skips a nested git repository.
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, True, False
    )

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    assert "git clean -xdf" in readme
    # ...and about the empty folders a first commit would drop, which is why
    # no .gitkeep is written for them.
    assert "does not track empty folders" in readme


def test_readme_layout_lists_the_gitignore_when_one_was_written(new_project):
    init_project.create_project("paroninkorpi", DataSource.XML, new_project, True, True)

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    assert "├── README.md" in readme
    assert "└── .gitignore" in readme


def test_readme_layout_omits_the_gitignore_when_none_was_written(new_project):
    # A layout block that lists a file the project does not have is worse
    # than no layout block at all.
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, True, False
    )

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    layout_block = readme.split("## Start here")[0]
    assert "└── README.md" in layout_block
    assert ".gitignore" not in layout_block


def test_readme_says_to_write_a_gitignore_when_none_was_written(new_project):
    # The git section's advice is to make the folder a repository of its
    # own; following it without a .gitignore commits every netcdf.
    init_project.create_project(
        "paroninkorpi", DataSource.XML, new_project, True, False
    )

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    assert "no `.gitignore`" in readme
    assert "*.nc" in readme


def test_readme_does_not_mention_a_missing_gitignore_when_it_has_one(new_project):
    init_project.create_project("paroninkorpi", DataSource.XML, new_project, True, True)

    readme = (new_project / "README.md").read_text(encoding="utf-8")
    assert "no `.gitignore`" not in readme


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


@pytest.mark.parametrize("answer", ["", "   ", "n", "N", "no", "NO"])
def test_prompt_yes_no_is_no_for_no_and_for_a_bare_enter(answer, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)

    assert init_project.prompt_yes_no("Add it?") is False


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "  YES  "])
def test_prompt_yes_no_is_yes_only_when_asked_for(answer, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)

    assert init_project.prompt_yes_no("Add it?") is True


def test_prompt_yes_no_shows_no_as_the_default(monkeypatch):
    # [y/N]: the capital is the whole of how a user learns that Enter is no.
    prompts: list[str] = []
    monkeypatch.setattr(
        "builtins.input", lambda prompt="": prompts.append(prompt) or ""
    )

    init_project.prompt_yes_no("Add it?")

    assert prompts == ["Add it? [y/N]: "]


def test_prompt_yes_no_asks_again_after_an_unusable_answer(monkeypatch, capsys):
    answers = iter(["maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert init_project.prompt_yes_no("Add it?") is True
    assert "Answer y or n" in capsys.readouterr().out


def test_main_prompts_for_what_the_flags_did_not_give(tmp_path, monkeypatch, capsys):
    """The one test that drives the interactive path end to end: no flags at
    all, every answer typed, everything landing under a redirected projects
    root. Every other test above calls create_project directly -- flags and
    pure functions are what tests are for, not a prompt transcript."""
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("SUSI_PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr("sys.argv", ["init_project.py"])

    # name, "1" = the xml source, then Enter twice: no README, no .gitignore.
    answers = iter(["paroninkorpi", "1", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    init_project.main()

    assert (projects_root / "paroninkorpi" / "inputs" / "config.toml").is_file()
    assert not (projects_root / "paroninkorpi" / "README.md").exists()
    assert not (projects_root / "paroninkorpi" / ".gitignore").exists()
    # The completion line: the folder is the user's, not this repo's.
    printed = capsys.readouterr().out
    assert "not tracked" in printed
    assert "repository of its own" in printed


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--readme", "--gitignore"], True),
        (["--no-readme", "--no-gitignore"], False),
    ],
)
def test_main_takes_the_optional_files_from_the_flags(
    flags, expected, tmp_path, monkeypatch
):
    """Both halves of each BooleanOptionalAction pair, and the point of
    having them: a scripted run is never prompted at all, so input() here
    fails the test rather than hanging it."""
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("SUSI_PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(
        "sys.argv",
        ["init_project.py", "--name", "paroninkorpi", "--source", "none", *flags],
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail(f"prompted despite {flags}"),
    )

    init_project.main()

    assert (projects_root / "paroninkorpi" / "README.md").is_file() == expected
    assert (projects_root / "paroninkorpi" / ".gitignore").is_file() == expected


def test_main_refuses_an_existing_project(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    (projects_root / "paroninkorpi").mkdir(parents=True)
    monkeypatch.setenv("SUSI_PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(
        "sys.argv",
        ["init_project.py", "--name", "paroninkorpi", "--source", "none"],
    )
    # --readme and --gitignore were not passed, but the refusal comes first:
    # the name is all it takes to know there is nothing to create here, so
    # the user is not asked two questions and then turned away.
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail("prompted before refusing an existing project"),
    )

    with pytest.raises(SystemExit):
        init_project.main()
