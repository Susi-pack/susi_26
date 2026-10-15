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
└── modulefiles/
    └── susi/
        ├── 0.1.0.lua            # `module load susi/0.1.0` → releases/0.1.0_2026-10-14
        └── 0.2.0.lua            # `module load susi` loads the highest version
```

- Release name: `<git tag>_<build date>`.
- A release is never modified after it is built.
- Users reach a release only through its module file, named after the git tag. A release without a
  module file is invisible to them.
- `module load susi` loads the highest version (Lmod's rule when no default is marked), so
  publishing a release means writing its module file. Nothing else changes.

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
    mkdir -p $SUSI_ROOT/releases $SUSI_ROOT/modulefiles/susi
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

8. Build (automatically reads the python version):
    ```bash
    pip-containerize new --uv --pyver $(cat $SUSI_TMP/susi_26/.python-version) --prefix releases/$SUSI_RELEASE/env releases/$SUSI_RELEASE/req-roihu.txt
    ```

9. Test:
    - Print the tag (`$SUSI_TAG`):

        ```bash
        releases/$SUSI_RELEASE/env/bin/python -c "import importlib.metadata as m; print(m.version('susi-26'))"
        ```

    - Copy `example_projects/minimal` from the clone into a projects root of its own for this release:

        ```bash
        mkdir -p $SUSI_TMP/test_$SUSI_RELEASE
        cp -r $SUSI_TMP/susi_26/example_projects/minimal $SUSI_TMP/test_$SUSI_RELEASE/
        ```

    - Run the test simulation (about a minute) with the release's Python, `SUSI_PROJECTS_ROOT`. It writes to `minimal/outputs/run_single/`, and fails if that folder
      already exists. **(verify that the Tykky wrapper passes `SUSI_PROJECTS_ROOT` through)**

        ```bash
        SUSI_PROJECTS_ROOT=$SUSI_TMP/test_$SUSI_RELEASE \
          releases/$SUSI_RELEASE/env/bin/python $SUSI_TMP/test_$SUSI_RELEASE/minimal/csc_single_simulation.py
        ```

    - Check that the run recorded the tag. It must print
      `"git_commit_hash": "could not get git commit hash. The package version was: <tag>"`:

        ```bash
        grep git_commit_hash $SUSI_TMP/test_$SUSI_RELEASE/minimal/outputs/run_single/metadata.json
        ```

    - Delete the test copy once you are happy: `rm -rf $SUSI_TMP/test_$SUSI_RELEASE`.

10. Delete the temporary clone: the code now lives in the image.
    ```bash
    rm -rf $SUSI_TMP/susi_26
    ```

11. Publish the release by writing its module file. `module load susi` always loads the highest version:

    ```bash
    cat > $SUSI_ROOT/modulefiles/susi/$SUSI_TAG.lua <<EOF
    help([[SUSI $SUSI_TAG, release $SUSI_RELEASE.]])
    whatis("Name: SUSI")
    whatis("Version: $SUSI_TAG")
    prepend_path("PATH", "$SUSI_ROOT/releases/$SUSI_RELEASE/env/bin")
    EOF
    ```

    - The heredoc is unquoted on purpose: the variables are expanded now, so the file holds the
      release's absolute path.
    - Check it: `module use $SUSI_ROOT/modulefiles && module avail susi`, then
      `module load susi/$SUSI_TAG && which python` must print `$SUSI_ROOT/releases/$SUSI_RELEASE/env/bin/python`.

12. Announce the new release to users: what changed, and the module name `susi/$SUSI_TAG`.

## Module files

- One Lua module file per release, `modulefiles/susi/<tag>.lua`. It only puts that release's
  `env/bin` first on `PATH`: the Tykky way of using an installation, as the
  [Tykky docs](https://docs.csc.fi/computing/containers/tykky/) describe it.
- Users: `module use /projappl/project_2020769/susi/modulefiles`, then
    - `module load susi` for the highest version,
    - `module load susi/X.Y.Z` for a fixed one,
    - `module avail susi` to list them.
- To keep users on an older release as the default (e.g. a new one turns out to be broken), mark it
  in `modulefiles/susi/.modulerc.lua`:

    ```lua
    module_version("susi/0.1.0", "default")
    ```

    Delete that file to go back to "highest version wins".
- Rebuilding the same tag (e.g. after a failed build) gives a new release folder with a new date.
  Point the existing `<tag>.lua` at it by running step 11 again.

## Jupyter in the Roihu web interface

- The env includes the `notebook` group (jupyterlab, ipykernel).
- In the Jupyter app form: Python → "Custom path" → the release's Python, e.g.
  `/projappl/project_2020769/susi/releases/<release>/env/bin/python`.
    - Users find that path with `module load susi && which python`.
    - It names one release, so moving to a new one means changing this path.
- Users create their own notebooks in their scratch folder; the repo's notebooks are templates.

## Retiring old releases

- `module load susi` puts one release's folder on `PATH`, so a job keeps that release until it ends,
  even if a newer one is published while it runs.
- Jobs queued before a new release pick it up when they start, if they `module load susi` without
  a version.
    - Users who need a fixed version load it by name: `module load susi/X.Y.Z`.
- Delete a release when nobody needs it any more, or when quota requires it:
    - first its module file, so nobody new loads it;
    - then its folder in `releases/`, once no running job uses it.
