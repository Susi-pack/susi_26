---
icon: lucide/layers
---

# Single vs Multiple SUSI Runs
SUSI simulations can be run in two ways: as a single run or as multiple runs (batch). 

If you want to run more than one simulation, we recommend using the `MultipleSusis` class.

## Why use `MultipleSusis`?

When running multiple simulations (batch mode), several issues can arise if not handled properly:

- **Duplicate parameters**: Running the same simulation twice wastes computational resources
- **Duplicate output paths**: Accidentally overwriting previous results with new simulations
- **Inconsistent run IDs**: Mixing different runs in one batch makes results hard to organize
- **Missing identifiers**: Forgetting to specify which stand or scenario a run belongs to

The `MultipleSusis` class addresses these problems through built-in validation checks:

1. **Consistent run ID**: All runs must share the same `run_id`
2. **Complete identifiers**: All runs must have both `stand_id` and `scenario_id` (or neither)
3. **Unique output paths**: No two runs can write to the same folder
4. **Unique parameter sets**: Duplicate SUSI parameter configurations are rejected
5. **Folder-safe names**: `stand_id` and `scenario_id` cannot contain path separators
6. **Same tag keys**: All simulations must use the same set of [tag](#tags) keys; the values may differ

---

## Single Run

When running a single simulation using `SimulationParams`, outputs are stored in a flat structure:

```
<project_dir>/outputs/
└── <run_id>/
    ├── metadata.json
    ├── params.json
    └── susi.nc
```

### Example: `susi_calls.py`

```python
# parameters.py beside this script: the project's SusiParams.
import parameters

from susi.io.execution_config import SimulationParams
from susi.io.metadata_model import SimulationMetaData
from susi.io.project_layout import project_dir

simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(project_dir=project_dir("testing"), run_id="testing2"),
    susi_params=parameters.PARAMETERS,
)
```

In this case, only `project_dir` and `run_id` are required in `SimulationMetaData`.
`project_dir` is the project's folder -- here `project_dir("testing")`, the user
project `testing` under the projects root. The run always writes into that
folder's own `outputs/`, which must already exist, and the output folder will be:
```
projects/testing/outputs/testing2/
├── metadata.json
├── params.json
└── susi.nc
```

---

## Multiple Susis (Batch)

When running multiple simulations using `MultipleSusis`, outputs are organized in a hierarchical structure:

```
<project_dir>/outputs/
└── <run_id>/
    ├── stand_A/
    │   ├── scenario_1/
    │   │   ├── metadata.json
    │   │   ├── params.json
    │   │   └── susi.nc
    │   └── scenario_2/
    │       └── ...
    └── stand_B/
        └── scenario_1/
            └── ...
```

### Example: `susi_parallel.py`

```python
from susi.io.execution_config import MultipleSusis, SimulationParams
from susi.io.metadata_model import SimulationMetaData
from susi.io.project_layout import project_dir

all_parameters = [
    SimulationParams(
        metadata=SimulationMetaData(
            project_dir=project_dir("ditch_depth"),
            run_id="run_01",
            stand_id="stand_01",
            scenario_id="deep_ditch",
        ),
        susi_params=deep,
    ),
    SimulationParams(
        metadata=SimulationMetaData(
            project_dir=project_dir("ditch_depth"),
            run_id="run_01",
            stand_id="stand_01",
            scenario_id="shallow_ditch",
        ),
        susi_params=shallow,
    ),
]

execution_config = MultipleSusis(
    simulation_parameter_list=all_parameters,
    n_parallel_processes=2,
)
```

### Tags

A run's simulations sometimes fall into groups you want to compare in the
analysis: stands from two regions, say, simulated in one run. Record the group
as a **tag** on each simulation's metadata, a free-text key-value pair:

```python
SimulationMetaData(
    project_dir=project_dir("phase4"),
    run_id="baseline",
    stand_id="171161",
    scenario_id="control",
    tags={"region": "pirkanmaa"},
)
```

Tags group simulations *within* a run. To tell one run from another (before
and after a parameter change, say), use a different `run_id` instead.

SUSI gives tags no meaning: it never reads them, it only writes them to each
simulation's `metadata.json`, and you choose the keys. A tag belongs to one
simulation, not to a stand, so a stand's scenarios may carry different values
for the same key. In a `MultipleSusis` batch, every simulation must use the same
keys: a simulation without the `region` key would drop out of any grouping by
region. See [Reading Simulation Output Data](reading_output_data.md#grouping-a-runs-simulations-by-tag)
for how to read tags back.


---

## Comparison

| Aspect | Single Run | Multiple Runs |
|--------|-----------|---------------|
| `stand_id` | Not required | Required |
| `scenario_id` | Not required | Required |
| `project_dir` | Required | Required |
| `run_id` | Required | Required, must be same for all runs |
| `tags` | Optional | Optional, but every simulation must use the same keys |
| Folder structure | `<run_id>/` | `<run_id>/stand_id/scenario_id/` |
| Class used | `SimulationParams` | `MultipleSusis` |

---

## Validation Rules

When using `MultipleSusis`, the following rules are enforced:

1. **Paired fields**: `stand_id` and `scenario_id` must either both be set or both be unset
2. **Consistent run ID**: All runs must share the same `run_id`
3. **Unique paths**: No two runs can have the same output folder path
4. **Folder-safe names**: `stand_id` and `scenario_id` cannot contain path separators (`/` or `\`)
5. **Same tag keys**: All simulations must use the same set of [tag](#tags) keys; the values may differ
