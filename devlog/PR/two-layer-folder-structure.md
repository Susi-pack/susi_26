# PR: Two-Layer Folder Structure for SUSI Simulations

## Summary

Introduced a consistent hierarchical folder layout for batch SUSI simulations while keeping individual single runs unchanged.

## Changes

### 1. Metadata Model Updates (`SimulationMetaData`)

- Added optional fields `stand_id` and `scenario_id` for batch run identification
- Updated `experiment_folder_path` computed field:
  - **Single run**: `parent_output_folder/experiment_id/`
  - **Batch run**: `parent_output_folder/experiment_id/stand_id/scenario_id/`
- Added validator ensuring `stand_id` and `scenario_id` are either both set or both None
- Added folder-safe character validation (rejects `/` and `\` in folder names)

### 2. MultipleSusis Model Updates (`execution_config.py`)

- Added validator `_check_stand_and_scenario_ids_are_set()` - all runs in a batch must have `stand_id` and `scenario_id`
- Added validator `_check_single_experiment_id()` - all runs must share the same `experiment_id`
- Added validator `_check_for_duplicated_experiment_folder_paths()` - prevents filesystem conflicts

### 3. Backward Compatibility

- **Individual SUSI runs are unaffected**: No `stand_id`/`scenario_id` required, path remains `experiment_id/`
- Existing `SimulationMetaData` usage continues to work without modification

## Example Folder Structure

```
output/
└── my_experiment/                    # experiment_id
    ├── stand_A/                      # stand_id
    │   ├── scenario_1/               # scenario_id
    │   │   ├── metadata.json
    │   │   ├── params.json
    │   │   └── susi.nc
    │   └── scenario_2/
    │       └── ...
    └── stand_B/
        └── scenario_1/
            └── ...
```

## UI Impact

When using `MultipleSusis` to create batch simulations:
- Users must provide both `stand_id` and `scenario_id` for each run
- All runs must share the same `experiment_id`
- Output folders are automatically structured as `experiment_id/stand_id/scenario_id/`

Individual `SusiParams` runs continue to use the flat `experiment_id/` structure.
