# Creates a new project folder, seeded so it can be run straight away.
#
# A project is `projects/<project_id>/`: everything one study is simulated
# from in its `inputs/`, everything its runs produce in its `outputs/`, and
# its own run script alongside them. `susi.io.project_layout` owns that
# layout; this tool is the one place that *creates* it.

# %% Imports
import argparse
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from susi.io.project_layout import (
    inputs_dir_for_project,
    outputs_dir_for_project,
    project_dir,
)
from susi.io.utils import SRC_DIR

# %% Constants

# Copy susi_calls.py, the easiest possible SUSI project there is. The path
# is composed from `susi.io.utils.SRC_DIR`, the one place that knows where
# the checkout is, and kept in a single named constant here so that moving
# src/scripts/ to src/example_scripts/ (ticket 17) is a one-line change.
SEED_SCRIPT_PATH = SRC_DIR / "scripts" / "susi_calls.py"

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
            "xml_to_allometry.py against your XML export -- see README.md."
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
            "metsakeskus_to_allometry.py against your .gpkg -- see README.md."
        ),
    ),
    DataSource.NONE: SourceSeed(
        prompt_label="nothing yet -- I will set the inputs up myself",
        config_template_path=None,
        next_step=(
            "No config.toml was written: set inputs/ up yourself, or re-run this "
            "tool for another project with --source xml/metsakeskus."
        ),
    ),
}

# Printed last, whatever was created. The folder is not this repo's, and the
# most useful thing a team can do with it is make it a repository of its own
UNTRACKED_NOTE = (
    "This folder is not tracked by the SUSI repository -- it is yours, and may be "
    "used as a git repository of its own."
)

# Prepended to the seeded copy of the run script. The copy itself is left
# byte-for-byte identical below this, so a diff against the example script
# shows exactly what the user has changed since.
SEED_SCRIPT_HEADER = """# Your project's run script, copied when this project was created from:
#     {seed_script_path}
# It is yours now -- the SUSI repo never updates this copy.
#
# Edit first:
#   - project_id -> "{project_id}", so the run writes into this project's
#     own outputs/ folder rather than some other project's
#   - run_id     -> a name for this run. A later run under a different id
#     lands beside this one instead of overwriting it.
#   - susi_params -> your own parameters, in place of the sample ones
#
# Then, with the SUSI environment active:
#     python susi_calls.py
#
# See README.md next to this file.

"""

# Written whether or not the user ever runs `git init`: harmless if unused,
# correct if they version the project later.
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
│   ├── allometry/       <- written by that tool. Do not create it by hand:
│   │                       the tools refuse to run into a folder that exists
│   └── ...              <- your weather.csv, XML export, rasters, ...
├── outputs/             <- outputs/<run_id>/<stand_id>/<scenario_id>/
├── susi_calls.py        <- your run script. Start here.
├── README.md
└── .gitignore
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

2. Edit `susi_calls.py` -- above all its `project_id`, which must be
   `"{project_id}"` for the run to write into this project.

3. Run it: `python susi_calls.py`.

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

One git quirk to know if you do: git does not track empty folders, so
committing this project before its first run leaves `inputs/` and `outputs/`
out of the commit, and a clone of it arrives without them. SUSI then refuses
to run, naming the folder it wants; `mkdir` it, or commit a placeholder file
inside each.

Nothing in the SUSI repo checks the code in here, either: type checking and
the pre-commit hooks only see the repo's own tracked files. Run
`uv run ty check <this folder>` yourself if you want it covered.
"""

# %% Creating a project


def create_project(
    project_id: str, source: DataSource, project_dir: Path
) -> list[Path]:
    """Create one project folder, seeded, and return every path it created.

    Takes the target path rather than deriving it from `project_id`, so the
    whole of the real work is testable against a tmp_path and main() keeps
    the single responsibility of deciding *where* a project goes.
    `project_id` is passed alongside it, redundantly, because it is the name
    the README and the seeded script talk about -- not a path.

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
    inputs_dir.mkdir()
    outputs_dir.mkdir()
    created = [project_dir, inputs_dir, outputs_dir]

    # No .gitkeep in either folder, deliberately. Git does not track empty
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

    created.append(write_seed_script(project_id, project_dir))
    created.append(write_readme(project_id, project_dir))
    created.append(write_gitignore(project_dir))
    return created


def write_seed_script(project_id: str, project_dir: Path) -> Path:
    """Copy SEED_SCRIPT_PATH in under a header saying what to edit first."""
    seed_script_path = project_dir / SEED_SCRIPT_PATH.name
    header = SEED_SCRIPT_HEADER.format(
        seed_script_path=SEED_SCRIPT_PATH, project_id=project_id
    )
    # Explicit utf-8 on both ends: the seed script is not pure ASCII, and
    # the default encoding is not utf-8 on Windows.
    seed_script_path.write_text(
        header + SEED_SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return seed_script_path


def write_readme(project_id: str, project_dir: Path) -> Path:
    """Write the project's own README: its layout, how to fill it, where the
    standard analysis notebooks are, and what git does and does not do to
    this folder."""
    readme_path = project_dir / "README.md"
    readme_path.write_text(
        README_TEMPLATE.format(project_id=project_id), encoding="utf-8"
    )
    return readme_path


def write_gitignore(project_dir: Path) -> Path:
    """Write the project's .gitignore, whether or not it is ever a git
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
            "\nNo answer given (stdin is closed). Pass --name and --source instead."
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


# %% CLI


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new SUSI project folder -- its inputs/ and outputs/, a "
            "run script to edit, a README and a .gitignore -- under the "
            "projects root."
        )
    )
    parser.add_argument(
        "--name",
        type=valid_project_id,
        default=None,
        help=(
            "Name of the new project. It is the project's id -- the folder "
            "name under the projects root, and what the run script's "
            "`project_id` must say. Prompted for if omitted."
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

    args = parser.parse_args()

    project_id = args.name if args.name is not None else prompt_for_project_id()
    source = DataSource(args.source) if args.source is not None else prompt_for_source()

    new_project_dir = project_dir(project_id)
    # The same refusal the allometry tools make about their output folder,
    # for the same reason: seeding into an existing project would drop a
    # fresh README and .gitignore on top of someone's work. Not
    # check_output_dir_available, whose message tells the user to "pass a
    # different --project-dir", a flag this tool does not have.
    if new_project_dir.exists():
        parser.error(
            f"Project folder already exists: {new_project_dir}. Pick a different "
            "--name, or work in that project directly."
        )

    return CLIArguments(
        project_id=project_id, source=source, project_dir=new_project_dir
    )


# %% main


def main() -> None:
    cli_args = parse_CLI_arguments()

    created = create_project(cli_args.project_id, cli_args.source, cli_args.project_dir)

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
