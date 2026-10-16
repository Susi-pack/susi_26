---
icon: lucide/cpu
---

# How to run SUSI jobs on Roihu

## Setup before the first job (once per user)
- Make sure you are a member of the CSC project `project_2020769` (ask the project manager if you aren't).
- Log in to Roihu. Follow their docs.
- (Recommended) Inside Roihu, create a dedicated folder for SUSI projects: `/scratch/project_2020769/$USER/susi_projects/`.

## Where things live

- SUSI code + Python: `/projappl/project_2020769/susi/` (managed by the maintainer).
- Your SUSI projects (inputs, data, outputs): `/scratch/project_2020769/$USER/susi_projects/<project>/`.

Note that you will later need point each SUSI simulation to its project folder it before each run with `export SUSI_PROJECTS_ROOT=/scratch/project_<NNN>/$USER/susi_projects`.
More details below.

!!! warning "Scratch is temporary"
    Files not used for 90–180 days are deleted, and nothing is backed up.
    For longer storage, you should move the files to `Allas`, the CSC storage solution.

## Running SUSI

### Load the pre-installed SUSI module
1. Pre-load all susi modules
    ```bash
    module use /projappl/project_2020769/susi/modulefiles
    ```
2. Load the latest module
    ```bash
    module load susi
    ```

    !!! note "Permanent setup"
        You may also set these commands once in `~/.bashrc` to forget about them. They will load at startup.

    !!! info "Other useful commands"
        - `module avail susi` lists the available modules.
        - `module load susi` loads the latest SUSI version. If you want to use another specific version use `module load susi/X.Y.Z`, where X.Y.Z is the version tag.

After loading, `python` and the `susi-*` commands are the ones coming from the SUSI installation.

!!! warning "Don't run bare simulations"
    Don't run simulations on the login node!
    Instead, use batch jobs or `sinteractive` for small tests.

## Create a batch job

[Official CSC docs](https://docs.csc.fi/computing/running/creating-job-scripts-roihu/).

We offer a ready-to-run example in `example_projects/minimal/parallel_simulations.sh`, next to the `parallel_simulations.py` it runs.
Copy the `minimal` folder into your `susi_projects/` folder, `cd` into it and submit with `sbatch parallel_simulations.sh`.

### Number of parallel processes

Slurm (Roihu) exposes the number of cpus per task set as `--cpus-per-task` via `$SLURM_CPUS_PER_TASK`, and this job passes it on to the run script, so the cores you reserve and the processes SUSI starts are always the same number.

- Keep `--ntasks=1` and `--nodes=1`: SUSI's parallel runs are processes of one program on one node, not separate tasks.
- `--cpus-per-task` cannot be greater than the number of simulations (stands × scenarios) in the run; SUSI refuses to start otherwise, so you don't pay for idle cores.
- A Roihu node has 384 cores, which is the most one job can get on `small`.

### Memory

Every parallel process holds its own simulation, so memory grows with the number of processes.
That is why the template uses `--mem-per-cpu` (memory per core) rather than `--mem` (memory per job).

!!! info "how much memory?"
    To choose the value, run a single simulation with `--cpus-per-task=1` first, then check its peak memory with `sacct -j <job id> -o JobID,MaxRSS,Elapsed`.
    Set `--mem-per-cpu` a bit above that `MaxRSS`.
    Note that if a job uses more memory than it reserved, Slurm kills it.

## Run batch job
[Official CSC docs](https://docs.csc.fi/computing/running/submitting-jobs/).

First `cd` to the location your Python script needs to be called from.
In our example, that would be (`minimal/` inside the SUSI projects folder).

Then, run the example batch job with
```bash
sbatch parallel_simulations.sh
```

## Analysing results
THIS PART OF THE DOCUMENTATION IS INCOMPLETE YET.

There will be two ways of running the jupyter notebooks to analyze the results:
- On your own machine: copy the run folder from scratch (`rsync`/`scp`) and use [Read data from runs](reading_output_data.md).
- On Roihu: Jupyter app in the web interface, Python → "Custom path" → the path that
  `module load susi && which python` prints. Create the notebook in your scratch folder. **(verify)**
    - That path names one release: to use a newer one, look up the path again.
