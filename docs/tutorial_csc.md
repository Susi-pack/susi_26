---
icon: lucide/graduation-cap
---

# Tutorial: your first SUSI run on Roihu

In this tutorial you run SUSI's smallest example project, `minimal`, on Roihu.
`minimal` simulates one stand under two scenarios, and Roihu runs both simulations in parallel.

The tutorial takes about 10 minutes, and by the end you will have logged in to Roihu, submitted a batch job, watched it run and found its results.

!!! question "Web browser or terminal?"
    There are two ways of working on CSC: the [web interface](https://docs.csc.fi/computing/webinterface/) and the terminal (through SSH).
    You can follow this tutorial in both ways.
    Every step that differs has tabs for you to choose from.
    Everything else is a command you type in a shell on Roihu, and it is the same in both.

!!! note "What you need"
    A CSC account that is a member of the CSC project `project_2020769` (the codename assigned to SUSI in CSC).
    If you are not a member yet, ask the project manager to add you.

## 1. Log in to Roihu

=== "Web interface"

    1. Go to [www.roihu.csc.fi](https://www.roihu.csc.fi) and log in with your CSC account.
    2. Open **Login node shell**, under *Pinned Apps* or under *Tools* in the top bar.

    You now have a terminal that opens in your browser tab, which for our purposes is equivalent to a real terminal.
    You will type every command of this tutorial in it.

=== "SSH"

    1. [Sign your public SSH key in MyCSC](https://docs.csc.fi/computing/connecting/ssh-keys/#signing-public-key).
        The signature is valid for 24 hours, so you will have to sign again on another day.
    2. Connect, with your CSC username:
        ```bash
        ssh <your_username>@roihu-cpu.csc.fi
        ```

    You will type every command of this tutorial in that terminal.

## 2. Create your SUSI projects folder

Your SUSI projects will live in a folder of your own, inside the project's scratch disk:

```bash
mkdir -p /scratch/project_2020769/<your_username>/susi_projects
```

Replace `<your_username>` with your CSC username, here and in every command below.

!!! tip "The `$USER` shortcut"
    In commands you run on Roihu, you can write `$USER` instead of your username, e.g. `mkdir -p /scratch/project_2020769/$USER/susi_projects`.
    `$USER` is a standard variable that every Linux shell sets to the name you logged in with, so there is nothing to set up.
    It doesn't work in commands you run on your own machine, such as `ssh` or `scp`: there, `$USER` is your local username, which may differ from the CSC one.

## 3. Upload the example project

The example project `minimal` doesn't come with the SUSI installation on Roihu: it lives in the SUSI repository, in `example_projects/minimal/`.
Upload it from your own copy of SUSI into your SUSI projects folder.
If you don't have SUSI on your machine, download it from [GitHub](https://github.com/Susi-pack/susi_26) (*Code* → *Download ZIP*) and unpack it.

!!! warning "Leave old outputs behind"
    If you have already run `minimal` on your machine, delete its `outputs/run_parallel_01/` folder before uploading, but keep `outputs/` itself.
    SUSI never overwrites a run, so the job would fail if that folder is already there.

=== "Web interface"

    1. Open **Files** in the top bar and go to `/scratch/project_2020769/<your_username>/susi_projects/`.
    2. Click **Upload** and drag the `minimal` folder from your machine into the upload window.
    3. Keep the browser tab open until the upload finishes.

    See [CSC docs: Files](https://docs.csc.fi/computing/webinterface/file-browser/) for more on the Files app.

=== "SSH"

    In a terminal on your own machine (not on Roihu), from the folder of your SUSI copy:
    ```bash
    scp -r example_projects/minimal <your_username>@roihu-cpu.csc.fi:/scratch/project_2020769/<your_username>/susi_projects/
    ```

    See [CSC docs: scp](https://docs.csc.fi/data/moving/scp/) for more on `scp`.

Then, in the Roihu shell, go into the uploaded project and look at its contents:

```bash
cd /scratch/project_2020769/<your_username>/susi_projects/minimal
ls
```

You should see, among others, `parallel_simulations.py` and `parallel_simulations.sh`.

## 4. Submit the job

```bash
sbatch parallel_simulations.sh
```

Roihu answers with the job's ID:

```
Submitted batch job 1234567
```

`parallel_simulations.sh` is a batch script.
It asks Roihu for 2 cores for at most 10 minutes to run the two SUSI scenarios on (feel free to inspect the file.
More detailed information on [How to run your project on Roihu](how_to_run_susi_on_roihu.md#2-write-the-batch-job-script)).

After submitting it, your job now waits in the queue of all the rest of jobs submitted to CSC until there are two free cores to run SUSI on, and then executes `parallel_simulations.py`.

## 5. Watch it run

=== "Web interface"

    Open **Active jobs** in the top bar.
    Your job is listed with its ID and its status: *Queued* while it waits, *Running* while it runs.
    When it finishes, its status changes to *Completed*.

=== "SSH"

    ```bash
    squeue --me
    ```

    Your job is listed with its ID and its state: `PD` (pending) while it waits, `R` while it runs.
    When it finishes, it disappears from the list.

Expect it to start within a few minutes and to runs for less than one minute.

## 6. Find the results

When the job has finished, look at the run's outputs:

=== "Web interface"

    Open **Files** and go to `/scratch/project_2020769/<your_username>/susi_projects/minimal/outputs/run_parallel_01/stand_01/`.
    You should see one folder per scenario, `deep_ditch` and `shallow_ditch`, each holding `metadata.json`, `params.json` and `susi.nc`.

    Everything the job printed is in `slurm-<job ID>.out`, in the `minimal` folder: click it to read it.

=== "SSH"

    ```bash
    ls outputs/run_parallel_01/stand_01/*
    ```

    You should see one folder per scenario, each with its own results:

    ```
    outputs/run_parallel_01/stand_01/deep_ditch:
    metadata.json  params.json  susi.nc

    outputs/run_parallel_01/stand_01/shallow_ditch:
    metadata.json  params.json  susi.nc
    ```

    Everything the job printed is in `slurm-<job ID>.out`, next to the batch script:

    ```bash
    cat slurm-1234567.out
    ```

`susi.nc` holds the simulation's results, and `params.json` and `metadata.json` record what was simulated.

Congratulations: you have run SUSI on Roihu!

!!! tip "Running it again"
    SUSI never overwrites a run.
    To submit the job a second time, either delete/rename the previous outputs or change the `run_id` parameter.

!!! abstract "WORK IN PROGRESS..."
    The next part of this tutorial, analysing these results in a notebook, will come when the example project gets its own notebook.

    Until then:

    - To analyse results on Roihu, see [How to run notebooks on Roihu](how_to_run_notebooks_on_roihu.md).
    - To run your own project, see [How to run your project on Roihu](how_to_run_susi_on_roihu.md).
