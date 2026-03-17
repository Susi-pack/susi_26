# PR: SoA Output Storage and Hardened Folder Structure

## Summary

Refactored the way SUSI stores and reads simulation output data:
- **Input API**: Added `stand_id` and `scenario_id` fields to `SimulationMetaData` for hierarchical batch runs
- **Output API**: Replaced deprecated `netcdf_utils.py` with new `load_output_data.py` using SoA (Structure of Arrays) pattern
- **Folder Structure**: Hardened to strict `project/stand_id/scenario_id` hierarchy

## Changes

### 1. Input API: `SimulationMetaData` (`src/susi/io/metadata_model.py`)

**New Fields:**
- `stand_id: str | None` - First-level folder hierarchy for batch runs
- `scenario_id: str | None` - Second-level folder hierarchy for batch runs

**Updated `experiment_folder_path`:**
```
Single run:  parent_output_folder/experiment_id/
Batch run:   parent_output_folder/experiment_id/stand_id/scenario_id/
```

### 2. Output Reading: New SoA Data Structure (`src/susi/io/load_output_data.py`)

**Replaces:** `netcdf_utils.py` (deprecated)

**New Classes:**
- `NetcdfVariableArray` - Wraps raw NetCDF arrays with spatial/temporal interfaces (`processed`, `last_timestep()`, `spatial_mean_at_last_timestep()`, etc.)
- `OutputDataStore` - Main SoA structure keyed by variable first:
  ```python
  stands: list[StandID]
  scenarios: dict[StandID, list[ScenarioID]]
  variables: Sequence[NetcdfVariablePath]
  data: dict[NetcdfVariablePath, dict[(StandID, ScenarioID), NetcdfVariableArray]]
  ```

### 3. Folder Structure (Hardened)

```
output/
└── my_experiment/        # experiment_id
    ├── stand_A/          # stand_id
    │   ├── scenario_1/   # scenario_id
    │   │   ├── metadata.json
    │   │   ├── params.json
    │   │   └── susi.nc
    │   └── scenario_2/
    └── stand_B/
        └── ...
```

## Breaking Changes

1. Output reading API: Migrate from `netcdf_utils.py` to `load_output_data.py`
2. Data structure: AoS → SoA (variable-first keys)
3. Variable access: Now accessed by variable path first, then (stand_id, scenario_id)

## Migration

**Old:** `data[stand_idx][scenario_name][variable_path]`  
**New:** `store.get_variable_value_for_scenario_and_stand(variable_path, stand_id, scenario_id)`
