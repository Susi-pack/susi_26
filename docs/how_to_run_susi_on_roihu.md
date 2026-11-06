---
icon: lucide/cpu
---

# How to run your project on Roihu

This guide shows how to run your own SUSI project on Roihu, CSC's supercomputer, as a batch job.
It assumes you already have a project with a run script that works on your own machine, like `parallel_simulations.py` in the example project `minimal`.

If this is your first time on Roihu, do the [tutorial](tutorial_csc.md) first: it runs `minimal` from start to end.


!!! question "Web browser or terminal?"
    There are two ways of working on CSC: the [web interface](https://docs.csc.fi/computing/webinterface/) and the terminal (through SSH).
    You can follow this guide in both ways.
    Every step that differs has tabs for you to choose from.
    Everything else is a command you type in a shell on Roihu, and it is the same in both.

## Before you start

You need:

- Membership in the CSC project `project_2020769`. If you aren't a member, ask the project manager to [add you](https://docs.csc.fi/accounts/how-to-add-members-to-project/).
- A shell on Roihu, through the web interface or SSH ([CSC docs: Connecting](https://docs.csc.fi/computing/connecting/)).
- Your SUSI projects folder, `/scratch/project_2020769/<your_username>/susi_projects/`.

Steps 1 and 2 of the [tutorial](tutorial_csc.md#1-log-in-to-roihu) show how to log in and create the folder.

!!! tip "The `$USER` shortcut"
    In commands you run on Roihu, you can write `$USER` instead of `<your_username>`: every Linux shell sets it to the name you logged in with, so there is nothing to set up.
    It doesn't work in commands you run on your own machine, such as `rsync`: there, `$USER` is your local username.

!!! info "Where things live"
    - **SUSI itself**: `/projappl/project_2020769/susi/`.
        One installation shared by everyone in the project and maintained by the SUSI maintainer. you only load it (through the batch script); you never need to modify it.
    - **Your SUSI projects folder**: `/scratch/project_2020769/<your_username>/susi_projects/`, one folder per project, each with its `inputs/`, `data/`, `outputs/` and run scripts.

    Notice that they go in different disk areas: `/projappl` is meant for software, and `/scratch` for the rest of the non-permanent data (see [CSC docs: Supercomputer storage](https://docs.csc.fi/computing/disk/)).

!!! warning "Scratch is temporary"
    Files on `/scratch` that nobody has accessed for 90 to 180 days are deleted, and nothing there is backed up ([CSC docs: Disk cleaning](https://docs.csc.fi/computing/usage-policy/#disk-cleaning)).
    If you need permanent storage, [Allas](https://docs.csc.fi/data/Allas/) is CSC's storage service.

## 1. Put your project on Roihu

Your project goes in its own folder inside the SUSI projects folder.
The folder name must be the name your run script gives SUSI: a script with `project_dir("my_project")` looks for `/scratch/project_2020769/<your_username>/susi_projects/my_project/`.

Bring the project's `inputs/`, `data/`, run scripts and parameters.
A folder called `outputs/` must exist too.

!!! tip "Keep your project in its own git repository"
    A project can be a git repository of its own, separate from SUSI's, like [wq_rosa](https://github.com/Susi-pack/wq_rosa).
    Your inputs, parameters, run scripts and notebooks are then version-controlled, and putting the project on Roihu is one `git clone` into your SUSI projects folder:
    ```bash
    cd /scratch/project_2020769/<your_username>/susi_projects
    git clone https://github.com/Susi-pack/wq_rosa.git
    ```
    After that, bringing a change from your machine to Roihu is a `git pull` away.
    Keep `outputs/` out of the repository (with `.gitignore`), and create it after cloning if the repository doesn't have it.

=== "Web interface"

    Open **Files** in the top bar, go to your SUSI projects folder, create the project's folder and upload its files into it ([CSC docs: Files](https://docs.csc.fi/computing/webinterface/file-browser/)). Note:

    - Uploading a file with the same name as an existing one overwrites it without asking.
    - Keep the browser tab open until the upload finishes.

    For a project with many files, it might be easier to zip it into one archive on your machine, upload that, and unpack it in the shell.

=== "SSH"

    There are [several ways of moving data](https://docs.csc.fi/data/moving/).
    We here choose `rsync`, because running the same command again later only copies what changed, instead of creating a full copy again.
    From a terminal on your own machine (not on Roihu), with your CSC username:
    ```bash
    rsync -av my_project/ <your_username>@roihu-cpu.csc.fi:/scratch/project_2020769/<your_username>/susi_projects/my_project/
    ```

## 2. Write the batch job script

On Roihu, simulations run as batch jobs.
A batch job script states what the job needs (cores, memory, time) and which commands it runs.
You submit it to Slurm, Roihu's queue system, which runs it on a compute node as soon as those resources are free ([CSC docs: Creating a batch job script](https://docs.csc.fi/computing/running/creating-job-scripts-roihu/)).

!!! warning "Don't run simulations in the shell"
    The shell you logged into runs on a login node, which all users share and which is only meant for light work such as editing files and submitting jobs ([CSC docs: Login nodes](https://docs.csc.fi/computing/usage-policy/#login-nodes)).
    Always run simulations as batch jobs, never in the shell.

Start from the example project's script, `example_projects/minimal/parallel_simulations.sh`, and copy it into your project folder, next to your run script.
Click the (+) signs to see what each line does and whether you need to change it:

```bash
#!/bin/bash
#SBATCH --job-name=susi_minimal_parallel (1)
#SBATCH --account=project_2020769 (2)
#SBATCH --partition=small (3)
#SBATCH --time=00:10:00 (4)
#SBATCH --nodes=1 (5)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2 (6)
#SBATCH --mem-per-cpu=400M (7)
#SBATCH --output=slurm-%j.out (8)

set -euo pipefail  # (9)!

module use /projappl/project_2020769/susi/modulefiles  # (10)!
module load susi

export SUSI_PROJECTS_ROOT=/scratch/project_2020769/<your_username>/susi_projects  # (11)!

srun python parallel_simulations.py "$SLURM_CPUS_PER_TASK"  # (12)!
```

1. The name shown in the job lists. **Change it** to something that tells your jobs apart.
2. The CSC project code for SUSI. Do not modify it.
3. `small` gives one node for at most 72 hours, which fits SUSI. See [CSC docs: Roihu partitions](https://docs.csc.fi/computing/running/batch-job-partitions/#roihu-partitions) for more information.
4. The longest the job may run, as `hh:mm:ss`. Slurm stops the job when it is reached. **Change it**: see the  [Set time and memory](#set-time-and-memory) section below.
5. Keep both `--nodes=1` and `--ntasks=1`.
    SUSI's parallel simulations are processes of one Python program on one machine, not separate Slurm tasks, so they need one task on one node.
6. How many cores the job gets, and how many simulations SUSI runs at the same time. **Change it**: see [How the number of cores reaches SUSI](#how-the-number-of-cores-reaches-susi).
7. Memory per core. **Change it**: see [Set time and memory](#set-time-and-memory).
8. Everything the job prints, including errors, goes to `slurm-<job ID>.out`, in the folder you submit the job from.
9. Stops the job at the first command that fails, instead of carrying on.
10. These two lines make `python` the SUSI installation's Python.
    `module load susi` loads the newest installed SUSI version; see [Pin a SUSI version](#optional-pin-a-susi-version) to choose one yourself.
11. Tells SUSI where your SUSI projects folder is. An installed SUSI has no default, so it fails without it.
    The example's own file writes `$USER` here, which works as well.
12. Runs your run script. **Change** `parallel_simulations.py` to your run script's name.
    `srun` starts it inside the job's reservation, and `$SLURM_CPUS_PER_TASK` passes it the number of cores.

### How the number of cores reaches SUSI

You choose the number of cores in one place only, `--cpus-per-task`, and SUSI starts exactly that many processes.
This is how the number travels from the batch script to SUSI:

1. `#SBATCH --cpus-per-task=N` reserves `N` cores.
2. Slurm puts that `N` in the environment variable `$SLURM_CPUS_PER_TASK`.
3. `srun python parallel_simulations.py "$SLURM_CPUS_PER_TASK"` hands it to the run script as its first argument.
4. The run script reads it and gives it to `MultipleSusis` as `n_parallel_processes`.
5. SUSI starts `N` processes, and each runs one simulation at a time until all are done.

In `parallel_simulations.py`, steps 4 and 5 are these lines:

```python
parser = argparse.ArgumentParser()
parser.add_argument("n_parallel_processes", help="Number of parallel processes", type=int)
cli_args = parser.parse_args()

execution_config = MultipleSusis(
    simulation_parameter_list=all_parameters,
    n_parallel_processes=cli_args.n_parallel_processes,
)

with Pool(processes=execution_config.n_parallel_processes) as pool:
    pool.map(func=run_susi, iterable=execution_config.simulation_parameter_list)
```

!!! warning "Your run script must do the same"
    Your run script has to take the number of processes as its first argument and pass it to `MultipleSusis`.
    If it sets the number itself instead, the cores you reserved and the processes SUSI starts no longer match: cores you pay for sit idle, or processes fight over too few cores.

!!! info "How many cores?"
    - At most the number of simulations in the run (stands × scenarios).
    SUSI refuses to start with more, so you never pay for cores with nothing to do.
    - At most 384, the cores of one Roihu node, which is the most a `small` job can get.
    - More cores finish the run sooner, but a big reservation may wait longer in the queue.

### [optional] Pin a SUSI version

`module load susi` always loads the newest SUSI installed on Roihu, and new versions come out regularly.
To keep getting the same results from the same project, load one version explicitly:

```bash
module load susi/X.Y.Z
```

`module avail susi` lists the installed versions.
Each run records the SUSI version it ran with in its `metadata.json`, in the `git_commit_hash` field.

### Set time and memory

Every SUSI process holds a whole simulation in memory, so a job's memory grows with its number of processes.

If a job uses more memory than it reserved, or runs longer than its `--time`, Slurm kills it.
Asking for far more than needed is not free either: the job may wait longer in the queue, and the reservation is [billed](https://docs.csc.fi/computing/hpc-billing/) to the project.

!!! info "Estimating `--time` and `--mem-per-cpu`"
    These empirical formulas have worked for SUSI so far, with a lot of margin:

    ```
    --time (seconds)    = 6 * ceil(n_simulations / n_cpus) * years_per_simulation + 5
    --mem-per-cpu (MB)  = 5 * years_per_simulation + 260 / n_cpus + 275
    ```

    Where:

    - `n_simulations` is the number of simulations in the run: stands × scenarios, or the number of `susi.nc` files the run will produce.
    - `n_cpus` is `--cpus-per-task`.
    - `years_per_simulation` is the number of years each simulation runs for.
    - `ceil()` rounds up to the next whole number.

    For example, 100 simulations of 50 years on 20 cores: `--time` = 6 × 5 × 50 + 5 = 1505 seconds, so `00:26:00`, and `--mem-per-cpu` = 250 + 13 + 275 = 538 MB, so `600M`.

To measure instead of estimate, run a single simulation with `--cpus-per-task=1`, check its peak memory once it has finished, and set `--mem-per-cpu` a bit above it:

=== "Web interface"

    Open **Job monitoring** in the top bar: it shows how much of its reserved memory and time each finished job used.
    Jobs shorter than one minute are not recorded, and a job shows up a few minutes after it ends.

=== "SSH"

    ```bash
    sacct -j <job ID> -o JobID,MaxRSS,Elapsed
    ```
    `MaxRSS` is the peak memory, and `Elapsed` the time the job ran.

## 3. Submit the job

`cd` into your project folder, where the batch script and the run script are, and submit:

```bash
cd /scratch/project_2020769/<your_username>/susi_projects/my_project
sbatch parallel_simulations.sh
```

Slurm answers with the job's ID, e.g. `Submitted batch job 1234567` ([CSC docs: Submitting a batch job](https://docs.csc.fi/computing/running/submitting-jobs/)).

## 4. Follow the job

=== "Web interface"

    Open **Active jobs** in the top bar.
    It lists your jobs with their status and the resources they asked for.
    Deleting a running job there cancels it.

=== "SSH"

    ```bash
    squeue --me          # your queued and running jobs
    scancel <job ID>     # cancel one
    ```

To see what the job printed so far, including any error, read its log in the `slurm-<job ID>.out` file in the project folder.

## 5. Get the results

Each simulation writes to its own folder, `outputs/<run_id>/<stand_id>/<scenario_id>/`, inside your project folder.

You can analyse them on Roihu, where they already are, with Jupyter: see [How to run notebooks on Roihu](how_to_run_notebooks_on_roihu.md).

Or copy them to your own machine and use [Read data from runs](reading_output_data.md) there:

=== "Web interface"

    Pack the run into one file in the shell:
    ```bash
    tar -czf my_run.tar.gz outputs/my_run
    ```
    Then download `my_run.tar.gz` from **Files**.

=== "SSH"

    From a terminal on your own machine (not on Roihu):
    ```bash
    rsync -av <your_username>@roihu-cpu.csc.fi:/scratch/project_2020769/<your_username>/susi_projects/my_project/outputs/my_run ./
    ```

Remember that scratch is [temporary](#before-you-start): copy what you want to keep.
