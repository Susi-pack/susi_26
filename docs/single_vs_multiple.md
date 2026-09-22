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

---

## Single Run

When running a single simulation using `SimulationParams`, outputs are stored in a flat structure:

```
projects/<project_id>/outputs/
└── <run_id>/
    ├── metadata.json
    ├── params.json
    └── susi.nc
```

### Example: `susi_calls.py`

```python
from susi.io.execution_config import SimulationParams
from susi.io.metadata_model import SimulationMetaData
from inputs.parameters import golden_test

simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(project_id="testing", run_id="testing2"),
    susi_params=golden_test.PARAMETERS,
)
```

In this case, only `project_id` and `run_id` are required in `SimulationMetaData`.
`parent_output_folder` is left out, so it is derived from `project_id` as that
project's own `outputs/` folder, and the output folder will be:
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
projects/<project_id>/outputs/
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

all_parameters = [
    SimulationParams(
        metadata=SimulationMetaData(
            project_id="ditch_depth_experiment",
            run_id="run_01",
            stand_id="stand_01",
            scenario_id="deep_ditch",
        ),
        susi_params=deep,
    ),
    SimulationParams(
        metadata=SimulationMetaData(
            project_id="ditch_depth_experiment",
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


---

## Comparison

| Aspect | Single Run | Multiple Runs |
|--------|-----------|---------------|
| `stand_id` | Not required | Required |
| `scenario_id` | Not required | Required |
| `project_id` | Required | Required |
| `run_id` | Required | Required, must be same for all runs |
| Folder structure | `<run_id>/` | `<run_id>/stand_id/scenario_id/` |
| Class used | `SimulationParams` | `MultipleSusis` |

---

## Validation Rules

When using `MultipleSusis`, the following rules are enforced:

1. **Paired fields**: `stand_id` and `scenario_id` must either both be set or both be unset
2. **Consistent run ID**: All runs must share the same `run_id`
3. **Unique paths**: No two runs can have the same output folder path
4. **Folder-safe names**: `stand_id` and `scenario_id` cannot contain path separators (`/` or `\`)
