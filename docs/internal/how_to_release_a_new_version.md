---
icon: lucide/tag
---

# How to release a new SUSI version (maintainer)

For the repo maintainer only.
Users never need this page.

## How versions work

- SUSI's version is not written anywhere in the repo. `setuptools-scm` derives it from the nearest git
  tag every time SUSI is built or installed.
- Tags are plain `X.Y.Z` (e.g. `0.2.0`), **without** a `v` prefix.
- What a build gets:

| Built from | Version |
|---|---|
| The tagged commit `0.2.0` | `0.2.0` |
| One commit after it | `0.2.1.dev1+g936df7792` |
| The same, with uncommitted changes | `0.2.1.dev1+g936df7792.d20261014` |
| Any commit, while the repo has no tags at all | `0.0.1.dev523+g5146937b3` |
| A source tree without `.git` (e.g. GitHub "Download ZIP") | `0.0.0` |

- Until the first tag is pushed, every build is a `0.0.1.devN` version, which sorts *below* the old
  hard-coded `0.1.0`. The first tag should therefore be at least `0.1.0`.

- So releasing a version means creating one git tag. There is no file to edit, nothing to commit, and
  `uv.lock` does not change.
- There is no release automation: no CI job reacts to tags.

## Choosing the version number

- Format `X.Y.Z`, three numbers.
- While SUSI is `0.y.z`:
    - bump `Y` (`0.2.0` → `0.3.0`) for new features or anything that breaks existing projects (input
      files, parameters, output variables, commands);
    - bump `Z` (`0.2.0` → `0.2.1`) for bug fixes only.
- See the existing tags with `git tag --sort=-v:refname`.

## Steps

1. Update your local `main` and check that it is clean:
    - `git switch main && git pull`
    - `git status` shows nothing to commit.
    - Optional sanity checks on that commit:
      - `uv lock --locked`
      - `uv run pytest`
2. Tag the commit: `git tag X.Y.Z`
3. Push the tag: `git push origin X.Y.Z`
    - Optional: create a GitHub Release from the tag, with notes generated from the merged PRs:
      - `gh release create X.Y.Z --generate-notes`
      - Creating the release in the GitHub web UI also works. If you let it create the tag, name it `X.Y.Z`, not `vX.Y.Z`, and pick `main` as the target.
4. Deploy the release where it is needed, e.g. on Roihu: see [How to set up SUSI on Roihu](how_to_set_up_susi_on_roihu.md).

## If something went wrong

- Tagged the wrong commit, **not pushed yet**: `git tag -d X.Y.Z` and tag again.
- Tag **already pushed**: do not move or delete it, someone may already have built from it. Fix the
  problem and release the next patch version (e.g. `0.2.0` → `0.2.1`).

## Notes

- Your own editable install (`uv sync`) records the version it had when it was installed. The
  `cache-keys` setting in `pyproject.toml` makes uv rebuild it (about a second) on the first `uv run` or
  `uv sync` after a new commit or tag, so it stays current. A `pip install -e .` does not get this,
  but that is harmless: in a checkout, SUSI asks git for the commit hash directly and never reads the
  installed version.
- Building SUSI needs the `.git` folder. A shallow clone of the tag is enough:
  `git clone --depth 1 --branch X.Y.Z https://github.com/Susi-pack/susi_26.git`.
    - Without `.git`, `SETUPTOOLS_SCM_PRETEND_VERSION=X.Y.Z` fixes only the version. The wheel still
      leaves out every non-Python file (the tools' `default_config.toml`, the notebooks): `setuptools-scm`
      finds them by asking git which files are tracked. Do not ship such a build.
