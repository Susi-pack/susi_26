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
