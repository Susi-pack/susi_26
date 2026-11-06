---
icon: lucide/notebook-pen
---

# How to run notebooks on Roihu

Through Roihu's web interface we can start Jupyter sessions that run directly in CSC.
This is a good idea, because you can analyze a run's outputs where they already are, instead of copying them to your own machine first ([CSC docs: Jupyter](https://docs.csc.fi/computing/webinterface/jupyter/)).
This is most worth it when a run is big, and copying its outputs would be slow.

Roihu's Jupyter sessions  can run SUSI through a simple `import susi`.
In this document we show you how.

A Jupyter session is a Slurm job like any batch job: it runs on a compute node, with the cores and memory you ask for, and it is billed to the project for as long as it runs.

!!! note "Before you start"
    You need a shell on Roihu and your SUSI projects folder: see [How to run your project on Roihu](how_to_run_susi_on_roihu.md#before-you-start).

## 1. Find the SUSI Python path

The Jupyter app's own Python environments don't include SUSI, so you have to tell it where SUSI's Python is.

In a login node shell (the web interface's **Login node shell**, or SSH):

```bash
module use /projappl/project_2020769/susi/modulefiles
module load susi
which python
```

It prints something like `/projappl/project_2020769/susi/releases/<release>/env/bin/python`.
Copy it: you will paste it into the Jupyter form.

!!! warning "Choose a different release"
    The path points to the latest SUSI release.
    To use a different release, load it as shown in [Pin a SUSI version](how_to_run_susi_on_roihu.md#optional-pin-a-susi-version), run `which python` again and paste the new path.

## 2. Launch Jupyter

Go to [www.roihu.csc.fi](https://www.roihu.csc.fi) and log in.
Then open *Apps* → *Jupyter* and fill in the form:

- **Project**: `project_2020769`.
- **Partition**: `interactive`, which usually starts right away and allows up to 32 cores and 64 GiB.
    Use `small` if you need more ([CSC docs: Roihu partitions](https://docs.csc.fi/computing/running/batch-job-partitions/#roihu-partitions)).
- **Cores and memory**: start small, e.g. 1 core and 4 GiB.
    What you need depends on how much of a run you load at once.
    If the kernel dies while loading data, relaunch with more memory.
- **Time**: how long the session may run, at most 16 hours.
    When it runs out the session stops, and anything unsaved is lost.
- **Python**: "Custom path", then paste the path from step 1.
- **Working directory**: your SUSI projects folder, `/scratch/project_2020769/<your_username>/susi_projects/`.
    This is only where Jupyter's file browser opens; each notebook still runs in its own folder (see step 3).
- Leave "Enable advanced settings" off.

Launch and wait: the session is queued like any job.
It appears on *My Interactive Sessions*, and once it is running you can connect to it from there.

!!! tip "If the session fails to start"
    On *My Interactive Sessions*, open the session's `output.log` and look for errors.
    Check first that the Python path in the form is the one `which python` printed.

## 3. Point your notebook at its run

Keep each notebook inside its project folder and write the paths to runs relative to the notebook, e.g.:

```python
RUN_PATH = Path("outputs/run_01").resolve()
```

Jupyter runs a notebook in the notebook's own folder, so the same path works on Roihu and on your own machine.
See [Read data from runs](reading_output_data.md) for how to load the outputs.

## 4. Stop the session when you are done

Closing the browser tab does **not** stop the session: it keeps running, holding its cores and memory, until its time runs out.
When you are done, cancel it on *My Interactive Sessions* ([CSC docs: Interactive apps](https://docs.csc.fi/computing/webinterface/apps/)).

