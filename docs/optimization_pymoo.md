# Optimization with PyMOO

This document explains the data structures and computational representations used in `analysis/optimization_pymoo.py`, which performs multi-objective optimization (NSGA-II) across 21 forest stands.

## Two computational representations
There are two representations for the same data. One is analogous to the underlying Netcdf file structure, and it is useful understand it in a humna-readable way; the other is useful for the optimization algorighm 

### 1. List of Dicts of NetCDF Variables

The primary data representation in this codebase is a **nested list-dict structure** that mirrors the structure of the Susi netcdf files:


```python
list[dict[ScenarioName, TargetVariableDict]]
```

#### Structure Hierarchy

| Level | Type | Description |
|-------|------|-------------|
| Outer list | `list` | 21 stands (one per forest stand) |
| Dict keys | `ScenarioName` | Scenario names: "default", "DNM", "fertilized", etc. |
| Dict values | `list[NetcdfVariableValue]` | List of variables with their netcdf paths and values |

#### Dataclasses from `src/susi/io/netcdf_utils.py`

- **`NetcdfVariableInfo`** - Metadata about a variable (path, name, dimensions, shape, units)
- **`NetcdfVariableValue`** - Contains the variable path and its `np.ndarray` values
- **`NetcdfVariablePath`** - Type alias: `str`
- **`TargetVariableDict`** - Type alias: `dict[NetcdfVariablePath, float]`
- **`ScenarioName`** - Type alias: `str`



Example (without full typing):
```python
vars_of_interest_by_stand = [
    {  # Stand 0
        "default": {"/stand/volume": 150.5, "/balance/C/soil_c_balance_co2eq": 12.3},
        "DNM": {"/stand/volume": 145.2, "/balance/C/soil_c_balance_co2eq": 14.1},
        "fertilized": {"/stand/volume": 160.8, "/balance/C/soil_c_balance_co2eq": 11.0},
    },
    {  # Stand 1
        # ... more scenarios
    },
    # ... 21 stands total
]
```

### 2. List of np.ndarrays (Optimization)
`ScenarioArrayData` is an representation capturing the arrays that go to the optimization, and the `scenario_names` of each scenario. Keeping the scenario names is necessary in order to be able to know which array corresponds to which scenario.

```python
ScenarioArrayData(
    arrays: list[np.ndarray],           # One array per stand
    scenario_names: list[list[ScenarioName]]  # Names for each scenario
)
```

Each array has shape `(n_scenarios, n_variables)`:

```python
arrays[0]  # Stand 0
# array shape: (3, 2) for 3 scenarios and 2 target variables
# So different columns correspond to different target variables
# array content:
# [[150.5, 12.3],   # default
#  [145.2, 14.1],   # DNM
#  [160.8, 11.0]]   # fertilized
```

## Conversion Functions

The following functions in `src/susi/io/netcdf_utils.py` allow switching between representations:

### Netcdf variables dict -> optimization arrays

```python
transform_list_of_scenarios_to_optimization_array_structure(
    vars_of_interest_by_stand: list[dict[ScenarioName, TargetVariableDict]],
    n_stands: int,
    target_variable_paths: list[NetcdfVariablePath],
) -> ScenarioArrayData
```

### optimization arrays -> Netcdf variables dict

```python
transform_array_data_to_list_of_scenarios(
    array_data: ScenarioArrayData,
    n_stands: int,
    target_variable_paths: list[NetcdfVariablePath],
) -> list[dict[ScenarioName, TargetVariableDict]]
```

## Usage in Optimization

In `optimization_pymoo.py`:

1. **Read netcdf data** → produces list of dicts
2. **Transform to arrays** → `transform_list_of_scenarios_to_optimization_array_structure()`
3. **Scale by stand area** → multiply each array by its stand's area
4. **Run optimization** → target function uses array indexing for fast lookups
