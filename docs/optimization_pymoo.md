# Optimization with PyMOO

This document explains the data structures and computational representations used in `analysis/optimization/core.py`, which performs multi-objective optimization (NSGA-II) across 21 forest stands.

## Two computational representations
There are two representations for the same data. One is analogous to the underlying Netcdf file structure, and it is useful understand it in a humna-readable way; the other is useful for the optimization algorighm 

### 1. OutputDataStore (SoA - Structure of Arrays)

The primary data representation in this codebase is a **SoA structure keyed by variable first**, defined in `src/susi/io/load_output_data.py`:

```python
OutputDataStore(
    stands: list[StandID],                                    # ["stand_A", "stand_B", ...]
    scenarios: dict[StandID, list[ScenarioID]],              # {"stand_A": ["scen_1", "scen_2"], ...}
    variables: Sequence[NetcdfVariablePath],                 # ["/balance/K", "/volume", ...]
    data: dict[NetcdfVariablePath, dict[(StandID, ScenarioID), NetcdfVariableArray]]
)
```

#### Structure Hierarchy

| Level | Type | Description |
|-------|------|-------------|
| `stands` | `list[StandID]` | List of stand identifiers |
| `scenarios` | `dict[StandID, list[ScenarioID]]` | Scenarios per stand |
| `variables` | `Sequence[NetcdfVariablePath]` | Variable paths in netcdf |
| `data` | `dict` | SoA: variable → (stand, scenario) → values |

#### Classes from `src/susi/io/load_output_data.py`

- **`NetcdfVariableArray`** - Wraps raw NetCDF arrays with spatial/temporal interfaces
- **`NetcdfVariableInfo`** - Metadata about a variable (name, dimensions, shape, units)
- **`NetcdfVariablePath`** - Type alias: `str`
- **`StandID`** - Type alias: `str`
- **`ScenarioID`** - Type alias: `str`


Example:
```python
store = OutputDataStore(
    stands=["stand_01", "stand_02"],
    scenarios={"stand_01": ["default", "fertilized"], "stand_02": ["default"]},
    variables=["/stand/volume", "/balance/C/soileq"],
    data={
        "/stand_c_balance_co2/volume": {
            ("stand_01", "default"): NetcdfVariableArray(...),
            ("stand_01", "fertilized"): NetcdfVariableArray(...),
            ("stand_02", "default"): NetcdfVariableArray(...),
        },
        "/balance/C/soil_c_balance_co2eq": {...}
    }
)

# Access a specific value
value = store.get_variable_value_for_scenario_and_stand(
    "/stand/volume", "stand_01", "default"
)

# Or get all scenarios for a variable
all_volumes = store.get_variable_values_all_scenarios("/stand/volume")
value = all_volumes[("stand_01", "default")]
```

### 2. NetcdfVariableArray (Optimization)

Each variable value in the SoA is wrapped in a `NetcdfVariableArray` which provides clean interfaces for optimization:

```python
NetcdfVariableArray(
    raw: np.ndarray  # Raw data from NetCDF
)
```

**Processed view:** Removes implementation quirks (ditch columns, single-scenario axis reduction)

**Available methods:**
- `last_timestep()` - Get final timestep values
- `spatial_mean_at_last_timestep()` - Mean over locations at final timestep
- `spatial_sum_at_last_timestep()` - Sum over locations at final timestep
- `mean_over_space()` - Time-series of spatial means (1-D)
- `mean_over_time()` - Spatial profile of temporal means (1-D)
- `mean_of_all_values()` - Mean of every value

## Reading Data

The main function to read netcdf files into the SoA structure:

```python
from susi.io.load_output_data import (
    read_netcdf_files_for_selected_variables,
    list_stand_folders,
    load_all_metadatas_from_stands,
)

# 1. Load metadata from the stand folders of one run
run_dirpath = Path("projects/my_project/outputs/my_run")
stand_folders = list_stand_folders(run_dirpath=run_dirpath)
metadata_by_stand = load_all_metadatas_from_stands(stand_folders)

# 2. Read selected variables
selected_variables = ["/stand/volume", "/balance/C/soil_c_balance_co2eq"]
store = read_netcdf_files_for_selected_variables(
    selected_variables=selected_variables,
    metadata_by_stand=metadata_by_stand,
)

# 3. Access data
value = store.get_variable_value_for_scenario_and_stand(
    variable_path="/stand/volume",
    stand_id="stand_01",
    scenario_id="default",
)
```

## Usage in Optimization

In `optimization_pymoo.py`:

1. **Read netcdf data** → produces `OutputDataStore`
2. **Extract variable values** → use `get_variable_value_for_scenario_and_stand()`
3. **Scale by stand area** → multiply by stand's area
4. **Run optimization** → direct array access for fast lookups
