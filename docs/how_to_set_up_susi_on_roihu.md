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
- Add the install and build folders to `~/.bashrc` on Roihu:

    ```bash
    cat >> ~/.bashrc <<'EOF'
    export SUSI_ROOT=/projappl/project_2020769/susi
    export SUSI_TMP=/scratch/project_2020769/$USER/susi-build
    EOF
    source ~/.bashrc
    ```

- Create the install folders:

    ```bash
    mkdir -p $SUSI_ROOT/releases $SUSI_ROOT/modulefiles
    ```

## Building a release (first install and every update)

Every command below can be pasted as-is: only the variables block needs editing.
`SUSI_ROOT` and `SUSI_TMP` come from `~/.bashrc` (see One-time prerequisites).
On the other hand, `SUSI_TAG` and `SUSI_RELEASE` live only in the current shell, but are useful variables to copy-paste later code snippets.

1. Release a version `X.Y.Z`: see [How to release a new SUSI version](how_to_release_a_new_version.md).

2. Create `SUSI_TAG` and `SUSI_RELEASE`:
        ```bash
        SUSI_TAG=X.Y.Z
        SUSI_RELEASE=${SUSI_TAG}_$(date +%F)
        ```

3. On Roihu:
    ```bash
    module purge && module load tykky
    ```

    NOTE: Use `sinteractive --account project_2020769 --time 1:00:00 --cores 8` if the login node runs out of memory.

4. Create the folders
    ```bash
    cd $SUSI_ROOT
    mkdir -p releases/$SUSI_RELEASE $SUSI_TMP
    ```

5. Shallow-clone the tag into the temporary folder. Keep its `.git`, because `setuptools-scm` reads the version from it.
    ```bash
    git clone --depth 1 --branch $SUSI_TAG https://github.com/Susi-pack/susi_26.git $SUSI_TMP/susi_26
    ```

6. Export the locked runtime requirements, including Jupyter for the web interface:
    ```bash
    uv export --project $SUSI_TMP/susi_26 --locked --no-default-groups --group notebook \
      --no-hashes --no-emit-project -o releases/$SUSI_RELEASE/req-roihu.txt
    ```

7. Add SUSI as a dependency to be installed by append the git clone's path to `req-roihu.txt`:
    ```bash
    echo $SUSI_TMP/susi_26 >> releases/$SUSI_RELEASE/req-roihu.txt
    ```

8. Build:
    ```bash
    pip-containerize new --slim --uv --prefix releases/$SUSI_RELEASE/env releases/$SUSI_RELEASE/req-roihu.txt
    ```

    - `--slim`: a minimal container with its own Python. **(verify its Python version; `.python-version` says 3.11)**
- [ ] Delete the temporary clone: the code now lives in the image.

    ```bash
    rm -rf $SUSI_TMP/susi_26
    ```

- Smoke test:
    - [ ] Prints the tag (`$SUSI_TAG`):

        ```bash
        releases/$SUSI_RELEASE/env/bin/python -c "import importlib.metadata as m; print(m.version('susi-26'))"
        ```

    - [ ] Run a copy of `example_projects/minimal` (adapted to `project_dir("minimal")`) with `SUSI_PROJECTS_ROOT` pointing to scratch.
    - [ ] `metadata.json` says `could not get git commit hash. The package version was: <tag>`.
- [ ] Switch:

    ```bash
    ln -sfn releases/$SUSI_RELEASE current
    ```

- [ ] Announce the new release to users (what changed, release name `$SUSI_RELEASE`).

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
