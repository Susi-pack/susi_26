---
icon: lucide/notebook-pen
---

# How to run notebooks on Roihu

[Official CSC docs](https://docs.csc.fi/computing/webinterface/jupyter/).

The SUSI default installation already includes Jupyter, so you don't need to install anything.
The Roihu web interface's Jupyter app uses the SUSI module's Python directly.

!!! note "Before you start"
    Do the once-per-user setup in [How to run SUSI on Roihu](how_to_run_susi_on_roihu.md) first.

## 1. Find the SUSI Python path

In a Roihu shell:
```bash
module use /projappl/project_2020769/susi/modulefiles
module load susi
which python
```

It prints something like `/projappl/project_2020769/susi/releases/<release>/env/bin/python`.
Copy it: you will paste it into the Jupyter form.

!!! warning "The path names one release"
    The path belongs to a single SUSI release, not to "the latest SUSI".
    To use a newer release, run the commands above again and paste the new path.

## 2. Launch Jupyter

Go to [www.roihu.csc.fi](https://www.roihu.csc.fi) and log in.
Then: Apps → Jupyter and fill the form:

- **Project**: `project_2020769`.
- **Partition**: a CPU partition, e.g. `small` or `interactive`.
- **Python**: "Custom path", then paste the path from step 1.
- **Working directory**: the folder with your notebooks, usually `/scratch/project_2020769/`.
- Leave the advanced settings ("Additional packages") off.

You will also need to provide the RAM and the number of cores.
This depends on the amount of data you want to analyze.

Launch and wait.

## 3. Set `SUSI_PROJECTS_ROOT` in the notebook

SUSI needs `SUSI_PROJECTS_ROOT` to find your projects, as in batch jobs.
Without it, the notebook fails with `ValueError: SUSI_PROJECTS_ROOT is not set`.
The Jupyter form has no field for environment variables, so set it in the first cell of every notebook:

```python
import os

os.environ["SUSI_PROJECTS_ROOT"] = os.path.expandvars(
    "/scratch/project_2020769/$USER/susi_projects"
)
```

!!! info "Setting it once in `~/.bashrc`"
    An `export SUSI_PROJECTS_ROOT=...` line in `~/.bashrc` may also reach the Jupyter session.
**(verify)**

## 4. Get the notebooks

- **Your own notebooks**: copy them into your scratch folder with `rsync` or `scp` ([CSC docs](https://docs.csc.fi/data/moving/)).
- **SUSI's notebooks** (`src/analysis/notebooks/`): download them from [GitHub](https://github.com/Susi-pack/susi_26) at the **same tag** as the SUSI release you loaded.
  Treat them as templates and keep your copies in your scratch folder.
  Notebooks from a different version may not match the SUSI code in the module.

!!! warning "Scratch is temporary"
    Notebooks in scratch are deleted like any other scratch file.
    Keep the ones you care about in git or somewhere else.

## Known issues

- **`project_summary.ipynb` cannot save its CSV**: it writes to `SRC_DIR / "analysis" / "exports"`, a folder that doesn't exist in the installed SUSI.
  Change `export_dir` to a folder in your scratch.
- **Hard-coded paths**: some notebooks (e.g. `example_projects/paroninkorpi/analysis/*.ipynb`) set `PROJECT_DIRPATH` to a path on someone's own machine.
  Change it to your project folder in scratch.
- **Extra packages**: only the packages in the SUSI module are available.
  CSC's "Virtual environment" option may not work with a custom-path Python. **(verify)**
  Ask the maintainer to add the package to the next SUSI release.
