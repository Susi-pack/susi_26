---
icon: lucide/database
---

# Reading Simulation Output Data

## Output data layout

Each simulation run produces a **folder** containing:

* `metadata.json` — run-level metadata (start/end timestamps, scenario name, …)
* `params.json` — all SUSI input parameters (flattened into a single JSON)
* A `.nc` NetCDF4 file with the output variables (path stored in `metadata.json` under `netcdf_output_filepath`)

Folders are organised by **project → stand → scenario**.  A typical layout:

```
data/paroninkorpi/
├── stand_1/
│   ├── baseline/
│   │   ├── metadata.json
│   │   ├── params.json
│   │   └── output.nc
│   ├── fertilized/
│   │   └── ...
│   └── dnm/
│       └── ...
├── stand_2/
└── …
```

Reading `params.json` / `metadata.json` is faster than the `.nc` (a few KB vs a few MB). 
That is why the workflows below read the `.json` files first, and only open the
large NetCDF files once it is know which variables are needed.

---

## How to read a single NetCDF file

??? question "`NetcdfVariablePath`, `StandID` and `ScenarioID`"

    In the examples below you will note that we make heavy use of the following types.

    | Alias | What it holds | Example |
    |-------|---------------|---------|
    | `NetcdfVariablePath` | Path to a variable inside the NetCDF, e.g. `"/stand/volume"` | `NetcdfVariablePath("/stand/volume")` |
    | `StandID` | The name of a stand (same as its folder name) | `StandID("stand_A")` |
    | `ScenarioID` | The name of a scenario (same as its folder name) | `ScenarioID("baseline")` |

    Under the hood they are all really Python `str`!

    This may be strange if you are not used to type annotations.
    If they are strings, why not pass just a string?

    There are two main benefits:
    1. To make the function signatures self-documenting.
    2. To be warned by the type checker if you accidentally pass a scenario ID where a stand ID is expected.


    Compare these two:

    ```python
    # Without type aliases — easy to swap arguments:
    get_variable("stand_A", "/stand/volume", "baseline")

    # With type aliases — the signature tells you what each argument is:
    get_variable(
        stand_id=StandID("stand_A"),
        variable_path=NetcdfVariablePath("/stand/volume"),
        scenario_id=ScenarioID("baseline"),
    )
    ```


Use this when exploring one scenario in isolation.

This is used in the following places in the codebase:

* `src/analysis/streamlit/pages/1_single_netcdf.py` — folder picker, variable explorer, plots
* `src/analysis/streamlit/pages/single_scenario_dashboard.py` — ~90 variables read at once, multiple themed plot pages
* `src/analysis/streamlit/pages/optimization.py` — golden test file for variable structure, then reads from chosen scenario

```python
from pathlib import Path
import susi.io.load_output_data as load_output
from susi.io.load_output_data import NetcdfVariablePath

# 1. Read params from JSON (fast!)
params = load_output.read_params_from_jsons(
    simulation_folderpath=Path("data/paroninkorpi/stand_1/baseline")
)

# 2. Locate the NetCDF file and list all available variables
netcdf_file = Path(params.metadata["netcdf_output_filepath"])
all_variables = load_output.list_all_netcdf_variables(netcdf_file)

# 3. Pick a few variables and read their values
variable_paths = [
    NetcdfVariablePath("/stand/volume"),
    NetcdfVariablePath("/strip/dwtyr"),
    NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
]
data = load_output.read_value_several_variables_from_single_file(
    netcdf_filepath=netcdf_file,
    variable_paths=variable_paths,
)

# 4. Select a variable
arr = data[NetcdfVariablePath("/stand/volume")]

# 5. (Optional) choose aggregation method
print(arr.spatial_mean_at_last_timestep())   # float
print(arr.mean_over_space())                 # time-series (1-D array)
print(arr.last_timestep())                   # raw slice
```
??? question "Why return `NetcdfVariableArray` instead of a plain numpy array?"

    The raw arrays inside the NetCDF files come in three shapes:

    * **1-D** — a simple time series (no spatial dimension)
    * **2-D** — time × location
    * **3-D** — scenario × time × location

    The 3-D shape carries implementation details that most analyses should ignore:
    the scenario axis is always length 1 for single-scenario files, and the first
    and last location columns are *ditch columns* that should normally be excluded.

    `NetcdfVariableArray` hides this complexity so you don't have to think about it:

    ```python
    # Without the wrapper — you need to know the internals:
    raw = nc.variables["/stand/volume"][:]       # shape (1, 120, 52)
    clean = raw[0, :, 1:-1]                      # drop scenario + ditches

    # With the wrapper — it just works:
    arr = NetcdfVariableArray(raw)
    ```

    The aggregation methods, some of them shown above at the last, optional step, are shortcuts for the most common reductions.  They save writing `arr.processed[-1, :].mean()` every time, and they can be passed around as callbacks — see `src/analysis/streamlit/pages/optimization.py` where users pick one from a dropdown at runtime.


---

## How to read multiple NetCDF files (Stand × Scenario)

Use this when you need data from every stand & scenario in a project.

This is used in the following places in the codebase:

* `src/analysis/streamlit/pages/project_summary.py` — batch-loads all stands/scenarios and builds an aggregated DataFrame; uses the manual approach for caching
* `src/analysis/streamlit/pages/compare_scenarios_for_stand.py` — metadata-only: compares JSON parameters across scenarios without loading any NetCDFs

```python
import susi.io.load_output_data as load_output
from susi.io.load_output_data import (
    StandID, ScenarioID, NetcdfVariablePath,
)

# 1. Discover stand folders under the project
stand_folders = load_output.list_subdirectories(
    Path("data/paroninkorpi")
)

# 2. Load all metadata (fast — only JSONs)
metadata_by_stand = load_output.load_all_metadatas_from_stands(
    folders=stand_folders
)

# 3. Define variables and batch-read
selected = [
    NetcdfVariablePath("/stand/volume"),
    NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
    NetcdfVariablePath("/strip/dwtyr"),
]
data_store = load_output.read_netcdf_files_for_selected_variables_from_metadatas(
    selected_variables=selected,
    metadata_by_stand=metadata_by_stand,
)

# 4. Query individual (stand, scenario) values
arr = data_store.get_variable_value_for_scenario_and_stand(
    variable_path=NetcdfVariablePath("/stand/volume"),
    stand_id=StandID("stand_1"),
    scenario_id=ScenarioID("baseline"),
)

# 5. (Optional) aggregate
print(arr.spatial_mean_at_last_timestep())
```


**Used in:**

* `src/analysis/streamlit/pages/project_summary.py` — batch-loads all stands/scenarios and builds an aggregated DataFrame; uses the manual approach for caching
* `src/analysis/streamlit/pages/compare_scenarios_for_stand.py` — metadata-only: compares JSON parameters across scenarios without loading any NetCDFs

---

## Core Data Model

All the loading code lives in the module `susi.io.load_output_data`.
The sections below explain the key types and functions so the auto-generated
API reference at the end makes sense.

### Main data types

| Type | Role |
|------|------|
| `OutputDataStore` | Holds all loaded data: `data[var_path][(stand, scenario)] → NetcdfVariableArray` |
| `NetcdfVariableArray` | Wraps a raw numpy array; provides aggregation helpers (spatial mean, time series, etc.) |
| `NetcdfVariableInfo` | Shape, dimension names, units of a variable — metadata only, no values |


### Module contents

All functions, classes, and types live under `susi.io.load_output_data`.
The auto-generated reference below lists everything; here is a quick map:

| Group | Functions |
|-------|-----------|
| **Metadata (JSON) loading** | `read_params_from_jsons`, `load_all_metadatas_from_stands`, `modify_after_load`, `coerce_datetime_format` |
| **NetCDF introspection** | `list_all_netcdf_variables` |
| **Single-file value reading** | `read_value_several_variables_from_single_file` |
| **Multi-file batch reading** | `read_netcdf_files_for_selected_variables`, `read_netcdf_files_for_selected_variables_from_metadatas` |
| **Filesystem helpers** | `list_subdirectories`, `get_scenarios_for_stand`, `get_netcdf_filepaths_for_stand` |

The aggregation methods on `NetcdfVariableArray` include `.last_timestep()`,
`.initial_timestep()`, `.spatial_mean_at_last_timestep()`, `.mean_over_space()`,
`.mean_over_time()`, `.mean_of_all_values()`, and more — see the class
documentation below.


::: susi.io.load_output_data
    handler: python
