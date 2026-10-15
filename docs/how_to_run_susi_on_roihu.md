---
icon: lucide/cpu
---

# How to run SUSI jobs on Roihu (users)

DRAFT: schematic bullet points, to be rewritten.
Items marked **(verify)** have not been tried on Roihu yet.

## Before the first job (once per user)

- [ ] Be a member of CSC project `project_<NNN>` (ask the project manager).
- [ ] Log in: `ssh <user>@roihu-cpu.csc.fi`.
- [ ] Create your folder on scratch: `/scratch/project_<NNN>/$USER/susi_projects/`.

## Where things live

- SUSI code + Python: `/projappl/project_<NNN>/susi/` (read-only for you, managed by the maintainer).
- Your SUSI projects (inputs, data, outputs): `/scratch/project_<NNN>/$USER/susi_projects/<project>/`.
    - Point SUSI at it with `export SUSI_PROJECTS_ROOT=/scratch/project_<NNN>/$USER/susi_projects`.
- Scratch is temporary:
    - Files not used for 90–180 days are deleted.
    - Nothing is backed up.
    - The quota (space and number of files) is shared by the whole project.
    - Each simulation writes 3 files.
    - Long-term storage of results: **(TBD)**

## Using SUSI

- `module use /projappl/project_<NNN>/susi/modulefiles && module load susi`
    - or `export PATH=/projappl/project_<NNN>/susi/current/env/bin:$PATH`
- `module load susi` gives the latest release.
    - For reproducible series of runs, load a fixed release: `module load susi/<release>`.
- Don't run simulations on the login node; use batch jobs, or `sinteractive` for small tests.

## Batch job template

- `#SBATCH --account=project_<NNN>`
- `#SBATCH --partition=small` (shared node) or a full-node partition for big runs **(verify partition names)**
- `#SBATCH --time=...`, `--cpus-per-task=...`, `--mem-per-cpu=...`
- `set -euo pipefail`
- `module load susi`
- `export SUSI_PROJECTS_ROOT=...`
- `python <project>/<run script>.py`
- Outputs land in `<project>/outputs/<run>/` on scratch.

## Large runs

- Estimate first: ≈ (number of stands) × (number of scenarios) × (size of one `susi.nc`), plus 3 files per simulation.
- If it doesn't fit, ask the maintainer to raise the project's scratch quota.

## Analysing results

- On your own machine: copy the run folder from scratch (`rsync`/`scp`) and use [Read data from runs](reading_output_data.md).
- On Roihu: Jupyter app in the web interface, Python → "Custom path" → `/projappl/project_<NNN>/susi/current/env/bin/python`. Create the notebook in your scratch folder. **(verify)**
- The notebooks in the repo are templates: create your own notebook for your own runs.
