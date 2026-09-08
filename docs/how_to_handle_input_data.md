---
icon: lucide/folder-input
---

# How to handle input data

SUSI needs three pieces of input data to run its simulations:

- an allometry path to guide the stand development,
- weather data for the simulated period, and
- the parameters for the simulation.

This guide covers where you should store that data.
For more information about each dataset, see [SUSI's three input data types](input_data_types.md).

!!! info "system data vs user data"

    Apart from the data you choose to input (the *user data*), SUSI also comes with some default datasets that are used for its development and testing (the *system data*).
    Your data should not be tracked in the repository, but the system data should.
    To keep those concers separated, these datasets are stored in different folders.

    - System data lives in `src/inputs/system`. That is the only folder with tracked contents within `src/inputs`.
    - User data lives in 2 places, both untracked.
        - Parameters live in `src/inputs/user_parameters`
        - Weather and allometry data should stay in the root-level `inputs/` folder.

##  Your weather and allometry data: `inputs/<project>/`

The repo root has an `inputs/` folder, tracked the same way `outputs/` is:
the folder itself is tracked (via `.gitkeep`) but everything you put inside it is gitignored.
This is the conventional place to drop your own datasets (weather CSVs, allometry files, XML stand exports).

We recommend that you name the top-level folder after your **project**, mirroring the way
in which simulation outputs are organized.
The way you organize your data inside that folder is up to you.
If you have few datasets for a given project, a flat structure works well:

```
inputs/
└── my_project/
    ├── weather.csv
    └── my_pines_allometry.csv
```
If you have several datasets in the same project (e.g., because you want to simulate many stands, weather scenarios and/or allometry paths) giving some more structure might work better.
For instance, grouping by type of data:

```
inputs/
└── my_project/
    ├── weather/
    │   └── weather_1.csv
    │   └── weather_2.csv
    │   └── ...
    └── allometry/
        └── my_pines.csv
```

Grouping by scenario instead (mirroring how outputs are organized) is just as valid.
Nothing under the project folder is enforced.


## Your model parameters: `src/inputs/user_parameters/`

`src/inputs/user_parameters/` is a reserved, empty spot to define your own model parameters.
These are `.py` files that build a `SusiParams` instance, the `Pydantic` class that declares the set of parameters required by SUSI.
The folder is tracked, but its contents are not.
The reason to put these inside `src/inputs/` instead of inside the root level `inputs/` with the rest of the data is that this makes the parameters importable by Python:

```python
from inputs.user_parameters import my_site_parameters
```

!!! info
    You are not required to put the input parameters into the `src/inputs/user_parameters/` folder.
    As long as you pass an instance of `SusiParams` to the simulator, anything works.
    The current suggestion is recommended because it allows for a clean separation between model parameters and the simulation instructions.
    This setup works well for simple projects, but there are legitimate reasons for not following this recommendation!

## Building a `SusiParams` instance from your own data

Once you have your data in place, you must point SUSI to it.
`WeatherParams.FMI_weather_filepath` and `CanopyLayerAllometry.allometry_dir_path`
both accept any path.
Build them off `user_input_folder`
instead of `input_folder` to point at your own data:

```python
from susi.io.app_settings import AppSettings
from susi.io.susi_parameter_model import WeatherParams, CanopyLayerAllometry, CanopyLayerName

app_settings = AppSettings()

weather_parameters = WeatherParams(
    FMI_weather_filepath=app_settings.user_input_folder.joinpath("my_project/weather/weather.csv"),
)

allometry_parameters = CanopyLayerAllometry(
    allometry_dir_path=app_settings.user_input_folder.joinpath("my_project/allometry"),
    allometry_file_registry={1: "my_pines.csv"},
    pointers={
        CanopyLayerName.dominant: [1],
        CanopyLayerName.subdominant: None,
        CanopyLayerName.under: None,
    },
)
```

See [`src/inputs/system/parameters/sample_parameters.py`](https://github.com/Susi-pack/susi_26/blob/main/src/inputs/system/parameters/sample_parameters.py)
for a full, working `SusiParams` built the same way off `input_folder` — swap in
`user_input_folder` and it becomes a template for your own site.


## Change default input folders via `AppSettings`

All this works because `AppSettings` holds the default paths for the two input folders.
```python
from susi.io.app_settings import AppSettings

app_settings = AppSettings()

app_settings.input_folder       # <repo_root>/src/inputs
app_settings.user_input_folder  # <repo_root>/inputs
```

This way, we have resolved, validated paths, and you never have to hardcode an absolute path or guess the repo layout.

If you want to change the defaults because you have your data elsewhere, simply instantiate `AppSettings` with different values:

```python
from susi.io.app_settings import AppSettings

app_settings = AppSettings(input_folder=..., user_input_folder=...)

...
```

Here all the fields available in `AppSettings`:
::: susi.io.app_settings.AppSettings
    handler: python
    options:
      show_source: false

## Summary

| Folder | Tracked? | Importable? | Use for |
|---|---|---|---|
| `inputs/<project>/` | Folder only | No | Your weather/allometry/stand data |
| `src/inputs/user_parameters/` | Folder only | Yes | Your own parameter-model scripts |
