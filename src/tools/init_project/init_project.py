# Creates a new, empty project folder.
#
# A project is `projects/<project_id>/`: everything one study is simulated
# from in its `inputs/`, everything its runs produce in its `outputs/`, its
# raw data in `data/`, and its own run script alongside them. `susi.io.project_layout` owns that
# layout; this tool is the one place that *creates* it.
#
# It creates the folders, not the run script: the user writes that, taking the
# projects in the checkout's example_projects/ as models. A copied example
# script would not run on its own anyway -- each example needs its own data.

# %% Imports
import argparse
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from susi.io.project_layout import (
    data_dir_for_project,
    inputs_dir_for_project,
    outputs_dir_for_project,
    project_dir,
)
from susi.io.utils import SRC_DIR

# %% Constants

# What xml_to_allometry.py and metsakeskus_to_allometry.py both look up when --config is not given.
CONFIG_FILENAME = "config.toml"


class DataSource(Enum):
    """
    Where a project's stand data comes from.
    Decides which tool's config template is seeded into `inputs/`.
    """

    XML = "xml"
    METSAKESKUS = "metsakeskus"
    NONE = "none"


@dataclass(frozen=True)
class SourceSeed:
    """
    Everything that differs between the `--source` choices.
    """

    # How the numbered prompt describes this choice.
    prompt_label: str
    # The generating tool's own template config, copied in as the project's
    # config.toml rather than restated here
    # None for DataSource.NONE: no config at all is written.
    config_template_path: Optional[Path]
    # What to tell the user to do next, printed on completion.
    next_step: str


SOURCE_SEEDS = {
    DataSource.XML: SourceSeed(
        prompt_label="an XML stand-data export  (xml_to_allometry.py)",
        config_template_path=SRC_DIR
        / "tools"
        / "xml_to_allometry"
        / "default_config.toml",
        next_step=(
            "Next: fill in the REQUIRED fields of {config_path}, then run "
            "xml_to_allometry.py against your XML export -- see "
            "docs/how_to_generate_allometry_from_xml.md. Then write a run script, "
            "modelled on the projects in example_projects/."
        ),
    ),
    DataSource.METSAKESKUS: SourceSeed(
        prompt_label="a Metsakeskus .gpkg      (metsakeskus_to_allometry.py)",
        config_template_path=SRC_DIR
        / "tools"
        / "metsakeskus_to_allometry"
        / "default_config.toml",
        next_step=(
            "Next: fill in the REQUIRED fields of {config_path}, then run "
            "metsakeskus_to_allometry.py against your .gpkg -- see "
            "docs/how_to_generate_allometry_from_metsakeskus.md. Then write a run "
            "script, modelled on the projects in example_projects/."
        ),
    ),
    DataSource.NONE: SourceSeed(
        prompt_label="nothing yet -- I will set the inputs up myself",
        config_template_path=None,
        next_step=(
            "No config.toml was written: set inputs/ up yourself, or re-run this "
            "tool for another project with --source xml/metsakeskus. Then write a "
            "run script, modelled on the projects in example_projects/."
        ),
    ),
}

# Printed last, whatever was created. The folder is not this repo's, and the
# most useful thing a team can do with it is make it a repository of its own
UNTRACKED_NOTE = (
    "This folder is not tracked by the SUSI repository -- it is yours, and may be "
    "used as a git repository of its own."
)

# Written only when asked for (--gitignore): harmless if the project never
# becomes a git repository, correct if it does.
GITIGNORE_CONTENTS = """# What to keep out of version control, if this project becomes a git
# repository of its own. It is not tracked by the SUSI repo either way.
#
# Only the heavy artifacts are ignored. Note what is NOT here: params.json
# and metadata.json, written beside every run's netcdf, stay tracked. They
# are tiny, and they are the entire provenance record of a run -- which is
# what makes a shared project reproducible rather than merely readable.

# Simulation results: one netcdf per stand and scenario, large and
# regenerable from the parameters kept beside them.
*.nc

# Rasters -- ditch depths, elevation models, ...
*.tif

# Your call, so left commented out: weather and vector inputs are usually
# worth tracking, unless yours are large or you may not redistribute them.
# weather.csv
# *.gpkg
"""

# No absolute paths in here, and no path back to the SUSI checkout: a
# project folder can be moved, and SUSI_PROJECTS_ROOT exists precisely so it
# can live on a different filesystem from the code. Anything of the form
# `../../src/...` would be broken the first time either happens -- the same
# reason index.ipynb is not copied into a project.
README_TEMPLATE = """# {project_id}

A SUSI project: everything this study is simulated from, everything its runs
produce, and the script that runs it, in one folder. Created by
`init_project.py`.

## Layout

```
{project_id}/
├── inputs/              <- everything the project is simulated from
│   ├── config.toml      <- config for the tool that generates your allometry
│   ├── stand_data.json  <- written by that tool: one entry per stand
│   └── allometry/       <- written by that tool, or placed by hand when no
│       │                   tool is used -- not both: the tools refuse to run
│       │                   into a folder that exists
│       └── new_growth/  <- post-clearcut allometry, one CSV per species,
│                           written by new_growth_allometry.py afterwards
├── data/                <- created for you: the preferred home for your raw
│                           data (weather.csv, XML export, rasters, ...), not
│                           an enforced one. Your script names these paths
├── outputs/             <- outputs/<run_id>/<stand_id>/<scenario_id>/
{layout_tail}
```

## Start here

1. Generate this project's allometry from your stand data. Run from the SUSI
   checkout, pointing `--project-dir` at this folder:

   ```
   python src/tools/xml_to_allometry/xml_to_allometry.py <export.xml> --project-dir <this folder>
   python src/tools/metsakeskus_to_allometry/metsakeskus_to_allometry.py <stands.gpkg> --project-dir <this folder>
   ```

   Fill in the REQUIRED fields of `inputs/config.toml` first; both tools take
   `--dry-run`, which reports what a run would produce without writing
   anything.

2. Write a run script in this folder. The projects in `example_projects/`
   of the SUSI checkout are the models to start from: complete project
   folders, laid out like this one. Pass
   `project_dir=project_dir("{project_id}")` to its `SimulationMetaData`, so
   the run writes into this project's `outputs/`.

3. Run it, with the SUSI environment active: `python <your script>.py`.

## Analysing the results

The standard analysis notebooks live in `src/analysis/notebooks/` in the SUSI
checkout: `project_summary`, `single_netcdf`, `compare_scenarios_for_stand`,
`single_scenario_dashboard` and `optimization`. They are project-agnostic --
each opens a folder selector, so point it at this project's `outputs/`. There
is a Streamlit frontend over the same data too: `susi-analyze`.

They are not copied into a project on purpose: a copy per project is five
stale forks waiting to happen. A project may of course keep bespoke notebooks
of its own alongside them.

## This folder and git

It is not tracked by the SUSI repository. A team that wants to share it makes
it a git repository of its own (`git init` here) -- the SUSI checkout's
`git status` stays clean either way.

**Worth knowing before you rely on it:** `git clean -xdf` run from the SUSI
checkout **deletes** an ordinary, data-only project folder, but **skips** a
nested git repository. Only `git clean -xdff` destroys a nested one. Making
this project a repository is, counter-intuitively, what protects it.
{gitignore_note}
One git quirk to know if you do: git does not track empty folders, so
committing this project before its first run leaves `inputs/` and `outputs/`
out of the commit, and a clone of it arrives without them. SUSI then refuses
to run, naming the folder it wants; `mkdir` it, or commit a placeholder file
inside each.

Nothing in the SUSI repo checks the code in here, either: type checking and
the pre-commit hooks only see the repo's own tracked files. Run
`uv run ty check <this folder>` yourself if you want it covered.
"""

# The Layout block's last lines. README.md is always among them -- the block
# only exists inside a README -- but .gitignore is listed only when one was
# actually written, so the tree never advertises a file the project does not
# have. Two whole lines rather than one conditional entry because the
# box-drawing character of the last entry differs.
LAYOUT_TAIL_WITH_GITIGNORE = "├── README.md\n└── .gitignore"
LAYOUT_TAIL_WITHOUT_GITIGNORE = "└── README.md"

# Spliced into the README's git section when no .gitignore was written. That
# section tells the user that making this folder a repository is what
# protects it from `git clean -xdf`; following that advice without a
# .gitignore commits every netcdf the project will ever produce. Carries its
# own blank lines, so the empty case leaves the section's paragraphs spaced
# exactly as they are written above.
NO_GITIGNORE_NOTE = """
**This project has no `.gitignore`.** Write one before you commit anything, or
`*.nc` and `*.tif` -- the simulation outputs and the rasters -- go into the
repository with everything else. `susi-init-project --gitignore` writes a
starting point you can copy.
"""

# %% Creating a project


def create_project(
    project_id: str,
    source: DataSource,
    project_dir: Path,
    with_readme: bool,
    with_gitignore: bool,
) -> list[Path]:
    """Create one project folder and return every path it created.

    Takes the target path rather than deriving it from `project_id`, so the
    whole of the real work is testable against a tmp_path and main() keeps
    the single responsibility of deciding *where* a project goes.
    `project_id` is passed alongside it, redundantly, because it is the name
    the README talks about -- not a path.

    `with_readme` and `with_gitignore` decide whether the two optional files
    are written at all. Both are asked for rather than assumed: the folder
    is the user's, and a README and a .gitignore are exactly the two files
    most likely to clash with conventions -- or a repository -- they already
    have. There is no default here; main() is where "no" is the default.

    Raises FileExistsError if the folder is already there: never seed into
    someone's existing data. main() catches that case earlier, with a
    message; this is the backstop for every other caller, which is why the
    project folder itself is created first and without `exist_ok`.
    """
    # First, and on its own: this is the call that refuses an existing
    # project. Creating inputs/ first would only refuse a folder that
    # already had an inputs/ in it, and would seed silently into any other.
    project_dir.mkdir(parents=True)

    inputs_dir = inputs_dir_for_project(project_dir)
    outputs_dir = outputs_dir_for_project(project_dir)
    data_dir = data_dir_for_project(project_dir)
    inputs_dir.mkdir()
    outputs_dir.mkdir()
    data_dir.mkdir()
    created = [project_dir, inputs_dir, outputs_dir, data_dir]

    # No .gitkeep in any of them, deliberately. Git does not track empty
    # directories, so a project committed to a repository of its own before
    # its first run arrives at a clone with no inputs/ or outputs/ -- but
    # that is the user's repository to manage, not this tool's, and the
    # failure it produces (require_outputs_dir / require_project_dir) names
    # the missing folder and says to create it. README.md mentions it.
    config_template_path = SOURCE_SEEDS[source].config_template_path
    if config_template_path is not None:
        config_path = inputs_dir / CONFIG_FILENAME
        shutil.copyfile(config_template_path, config_path)
        created.append(config_path)

    # The README describes the layout, so it has to know whether a
    # .gitignore is part of it.
    if with_readme:
        created.append(write_readme(project_id, project_dir, with_gitignore))
    if with_gitignore:
        created.append(write_gitignore(project_dir))
    return created


def write_readme(project_id: str, project_dir: Path, with_gitignore: bool) -> Path:
    """Write the project's own README: its layout, how to fill it, where the
    standard analysis notebooks are, and what git does and does not do to
    this folder.

    `with_gitignore` says whether one was written beside it, which the
    README's Layout block lists and its git section warns about the absence
    of. Passed in rather than probed off the filesystem: what this run
    created is the question, not what happens to be in the folder."""
    readme_path = project_dir / "README.md"
    readme_path.write_text(
        README_TEMPLATE.format(
            project_id=project_id,
            layout_tail=(
                LAYOUT_TAIL_WITH_GITIGNORE
                if with_gitignore
                else LAYOUT_TAIL_WITHOUT_GITIGNORE
            ),
            gitignore_note="" if with_gitignore else NO_GITIGNORE_NOTE,
        ),
        encoding="utf-8",
    )
    return readme_path


def write_gitignore(project_dir: Path) -> Path:
    """Write the project's .gitignore, for if it ever becomes a git
    repository -- see GITIGNORE_CONTENTS for what it keeps and drops."""
    gitignore_path = project_dir / ".gitignore"
    gitignore_path.write_text(GITIGNORE_CONTENTS, encoding="utf-8")
    return gitignore_path


# %% CLI args


@dataclass(frozen=True)
class CLIArguments:
    project_id: str
    source: DataSource
    project_dir: Path
    with_readme: bool
    with_gitignore: bool


def valid_project_id(value: str) -> str:
    """An argparse `type=` callable for a project's id, which is also its
    folder name under the projects root: `projects/<project_id>/` is the
    whole of the layout's nesting, so an id with a separator in it would
    silently write somewhere else entirely.

    Raises ArgumentTypeError like the shared validators in
    input_validation.py, so argparse reports a bad --name itself. The
    prompting path catches it and asks again."""
    project_id = value.strip()
    if not project_id:
        raise argparse.ArgumentTypeError("Project name must not be empty.")
    # Path(...).name drops any directory part, so a name that survives it
    # unchanged contains no path of any kind. ".." survives it -- pathlib
    # keeps it as an ordinary component -- so it is named here too, together
    # with ".", which Path(...).name already empties.
    if project_id in {".", ".."} or project_id != Path(project_id).name:
        raise argparse.ArgumentTypeError(
            f"A project name is a single folder name, not a path: {value!r}"
        )
    return project_id


# %% Prompting
#
# Flags first, prompts only for what is missing: the flags are what tests and
# scripts use, and the prompts are for a user who has just been told to run
# this once.


def ask(question: str) -> str:
    """input(), except that a closed stdin exits with a usable message
    instead of an EOFError traceback -- the likeliest way to reach it is a
    script that ran this tool without passing the flags."""
    try:
        return input(question)
    except EOFError:
        raise SystemExit(
            "\nNo answer given (stdin is closed). Pass the flags instead: "
            "--name, --source, --readme/--no-readme, --gitignore/--no-gitignore."
        )


def prompt_for_project_id() -> str:
    while True:
        try:
            return valid_project_id(ask("Project name: "))
        except argparse.ArgumentTypeError as invalid_project_id:
            print(invalid_project_id)


def prompt_for_source() -> DataSource:
    sources = list(DataSource)
    print("Where does this project's stand data come from?")
    for number, source in enumerate(sources, start=1):
        print(f"  {number}) {SOURCE_SEEDS[source].prompt_label}")
    while True:
        answer = ask(f"Choose 1-{len(sources)}, or the name: ").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= len(sources):
            return sources[int(answer) - 1]
        try:
            return DataSource(answer)
        except ValueError:
            print(f"Not one of the choices: {answer!r}")


def prompt_yes_no(question: str) -> bool:
    """Ask a yes/no question whose default -- a bare Enter -- is no.

    No for every question asked through here, which is why the default is
    baked in rather than passed: these are the optional extras in a folder
    that belongs to the user, so the answer that adds nothing is the one
    they get for free.
    """
    while True:
        answer = ask(f"{question} [y/N]: ").strip().lower()
        if answer in {"", "n", "no"}:
            return False
        if answer in {"y", "yes"}:
            return True
        print(f"Answer y or n -- Enter for no: {answer!r}")


# %% CLI


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new SUSI project folder -- its inputs/, outputs/ and data/ -- "
            "under the projects root. A README and a "
            ".gitignore are optional extras, off unless asked for."
        )
    )
    parser.add_argument(
        "--name",
        type=valid_project_id,
        default=None,
        help=(
            "Name of the new project. It is the project's id -- the folder "
            "name under the projects root, and what the run script's "
            "`project_dir(...)` must name. Prompted for if omitted."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=[source.value for source in DataSource],
        help=(
            "Where this project's stand data will come from. Decides which "
            "tool's template is seeded in as inputs/config.toml; 'none' "
            "writes no config at all. Prompted for if omitted."
        ),
    )
    # BooleanOptionalAction, so each of these is a pair: --readme and
    # --no-readme. Their default is None rather than False, which is what
    # separates "the user said no" from "the user said nothing" -- the
    # latter is what the prompt is for, the same way --name and --source
    # work. The prompt's own default is no, so an omitted flag and a bare
    # Enter land in the same place.
    parser.add_argument(
        "--readme",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write a README.md describing this project's layout, how to fill "
            "it, where the analysis notebooks are and what git does to the "
            "folder. Prompted for if omitted, and the prompt's default is no."
        ),
    )
    parser.add_argument(
        "--gitignore",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write a .gitignore keeping this project's netcdfs and rasters "
            "out of version control, for if the folder becomes a git "
            "repository of its own. Prompted for if omitted, and the "
            "prompt's default is no."
        ),
    )

    args = parser.parse_args()

    project_id = args.name if args.name is not None else prompt_for_project_id()

    new_project_dir = project_dir(project_id)
    # Checked here, before anything else is asked for: the name is all this
    # needs, and a user whose project already exists should hear so at once
    # rather than after answering three more questions.
    #
    # The same refusal the allometry tools make about their output folder,
    # for the same reason: seeding into an existing project would drop a
    # fresh config.toml -- and whatever else was asked for -- on top of
    # someone's work. Not check_output_dir_available, whose message tells
    # the user to "pass a different --project-dir", a flag this tool does
    # not have.
    if new_project_dir.exists():
        parser.error(
            f"Project folder already exists: {new_project_dir}. Pick a different "
            "--name, or work in that project directly."
        )

    source = DataSource(args.source) if args.source is not None else prompt_for_source()
    with_readme = (
        args.readme
        if args.readme is not None
        else prompt_yes_no("Add a README.md describing this project?")
    )
    with_gitignore = (
        args.gitignore
        if args.gitignore is not None
        else prompt_yes_no(
            "Add a .gitignore, in case this project becomes a git repository?"
        )
    )

    return CLIArguments(
        project_id=project_id,
        source=source,
        project_dir=new_project_dir,
        with_readme=with_readme,
        with_gitignore=with_gitignore,
    )


# %% main


def main() -> None:
    cli_args = parse_CLI_arguments()

    created = create_project(
        cli_args.project_id,
        cli_args.source,
        cli_args.project_dir,
        cli_args.with_readme,
        cli_args.with_gitignore,
    )

    print(f"\nCreated project {cli_args.project_id}:")
    for path in created:
        print(f"    {path}")

    print()
    print(
        SOURCE_SEEDS[cli_args.source].next_step.format(
            config_path=inputs_dir_for_project(cli_args.project_dir) / CONFIG_FILENAME
        )
    )
    print(UNTRACKED_NOTE)


if __name__ == "__main__":
    main()
