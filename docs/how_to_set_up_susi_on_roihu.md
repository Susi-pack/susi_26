---
icon: lucide/server-cog
---

# How to set up SUSI on Roihu (maintainer, one-time + updates)

DRAFT: schematic bullet points, to be rewritten while doing the setup.
Items marked **(verify)** have not been tried on Roihu yet.

## Goal

- One shared SUSI installation for everyone in CSC project `project_2020769`.
- Users load it and run jobs; they never build anything themselves.
- SUSI is installed **non-editable** inside a Tykky container: the code is sealed in the image, no git
  checkout is kept, and nobody can change an installed release.

## Prerequisites in the repo

- `.scratch/non-editable-install/` ticket 01 (git hash fallback) has landed. Without it every
  simulation crashes in a non-editable install.
- Ticket 02 (version from git tags) for meaningful versions, ticket 03 if users run
  `susi-init-project --source xml|metsakeskus` on Roihu.

## Folder layout

```
/projappl/project_<NNN>/susi/
├── releases/
│   ├── 0.1.0_2026-10-14/
│   │   ├── env/                 # Tykky install: Python + packages + SUSI itself
│   │   └── req-roihu.txt        # exact requirements the env was built from
│   └── 0.2.0_2026-11-20/
├── current -> releases/0.2.0_2026-11-20
└── modulefiles/susi/            # optional: `module load susi`
```

- Release name: `<git tag>_<build date>`.
- A release is never modified after it is built.
- `current` is what users get by default.

## One-time prerequisites

- [ ] All SUSI users are members of `project_<NNN>` (MyCSC).
- [ ] Check quotas (MyCSC):
    - `/projappl`: 15 GiB / 150 000 files default. One release ≈ the env size. **(verify env size)**
    - `/scratch`: 250 GiB / 500 000 files default. Raise it in MyCSC before large runs (up to 10 TiB is approved automatically). Outputs stay on scratch, no Allas for now.
- [ ] Create `/projappl/project_<NNN>/susi/{releases,modulefiles}`.

## Building a release (first install and every update)

- When: on demand, when main has something users need.
- [ ] Release a version `X.Y.Z`: see [How to release a new SUSI version](how_to_release_a_new_version.md).
- [ ] On Roihu: `module purge && module load tykky`
    - Use `sinteractive --account project_<NNN> --time 1:00:00 --cores 8` if the login node runs out of memory.
- [ ] Shallow-clone the tag into a temporary folder (e.g. on scratch). Keep its `.git`: `setuptools-scm` reads the version from it.
    - `git clone --depth 1 --branch X.Y.Z https://github.com/Susi-pack/susi_26.git <tmp>/susi_26`
- [ ] Export the locked runtime requirements, including Jupyter for the web interface **(verify uv availability on Roihu; otherwise export on a laptop at the same tag)**:
    - From `/projappl/project_<NNN>/susi`: `uv export --project <tmp>/susi_26 --locked --no-default-groups --group notebook --no-hashes --no-emit-project -o releases/<release>/req-roihu.txt`
- [ ] Append the clone's path (non-editable) to `req-roihu.txt`: `<tmp>/susi_26`
- [ ] Build: `pip-containerize new --slim --uv --prefix releases/<release>/env releases/<release>/req-roihu.txt` **(verify)**
    - `--slim`: a minimal container with its own Python. **(verify its Python version; `.python-version` says 3.11)**
- [ ] Delete the temporary clone: the code now lives in the image.
- Smoke test:
    - [ ] `releases/<release>/env/bin/python -c "import importlib.metadata as m; print(m.version('susi-26'))"` prints the tag.
    - [ ] Run a copy of `example_projects/minimal` (adapted to `project_dir("minimal")`) with `SUSI_PROJECTS_ROOT` pointing to scratch.
    - [ ] `metadata.json` says `could not get git commit hash. The package version was: <tag>`.
- [ ] Switch: `ln -sfn releases/<release> current`
- [ ] Announce the new release to users (what changed, release name).

## Module file (optional, but it hides the details from users)

- Puts `<release>/env/bin` first on `PATH`.
- One module file per release, plus a default pointing to `current`.
- Users: `module use /projappl/project_<NNN>/susi/modulefiles && module load susi`.

## Jupyter in the Roihu web interface

- The env includes the `notebook` group (jupyterlab, ipykernel).
- In the Jupyter app form: Python → "Custom path" → `/projappl/project_<NNN>/susi/current/env/bin/python` **(verify the symlink works there)**.
- Users create their own notebooks in their scratch folder; the repo's notebooks are templates.

## Retiring old releases

- Jobs already running keep using their release as long as its folder exists.
- Jobs queued before a switch pick up the new `current` when they start.
    - Users who need a fixed version load the release by name.
- Delete a release when nobody needs it any more, or when quota requires it.
