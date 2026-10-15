#!/bin/bash
# Roihu batch job for parallel_simulations.py.
# Submit from inside the copied project folder: `sbatch parallel_simulations.sh`.
# See docs/how_to_run_susi_on_roihu.md.
#SBATCH --job-name=susi_minimal_parallel
#SBATCH --account=project_2020769
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The only place the number of parallel processes is set.
# It is then passed to the script as $SLURM_CPUS_PER_TASK. 
#SBATCH --cpus-per-task=2
# Per core, so the total memory grows linearly with --cpus-per-task
#SBATCH --mem-per-cpu=1000M
#SBATCH --output=slurm-%j.out       # Standard output of the job script

set -euo pipefail # Extra security for early error raising

module use /projappl/project_2020769/susi/modulefiles
module load susi

# Set the env var that tells SUSI where the projects are stored
export SUSI_PROJECTS_ROOT=/scratch/project_2020769/$USER/susi_projects

srun python parallel_simulations.py "$SLURM_CPUS_PER_TASK"
