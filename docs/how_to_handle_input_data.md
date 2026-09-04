---
icon: lucide/folder-input
---

# How to handle input data

SUSI needs two very different kinds of input data:

- **Dev data** — small, reference weather/allometry files and parameter models that tests
  and example scripts depend on. This has to stay tracked in the repo so a fresh checkout
  keeps working.
- **Your own data** — site-specific weather files, allometry tables, XML stand exports,
  parameter models, and so on. This is personal to your machine and must never end up in a
  commit.

The repo keeps these apart with two folders and one settings object.

## `src/inputs/` — tracked dev data

`src/inputs/` lives inside the Python package tree, so anything under it is importable.
Only `src/inputs/system/` is actually tracked in git; everything else under `src/inputs/`
is gitignored.

```
src/inputs/
├── system/              # tracked — data used by tests and example scripts
│   ├── allometry/
│   │   └── CF_41.csv
│   ├── parameters/
│   │   ├── sample_parameters.py
│   │   ├── golden_test.py
│   │   └── para_2021.py
│   └── weather/
│       └── CFw.csv
└── user_parameters/     # tracked as a folder, contents gitignored
    └── .gitkeep
```

`src/inputs/system/parameters/` holds the Pydantic parameter-model scripts that tests and
example scripts import directly, e.g.:

```python
from inputs.system.parameters import sample_parameters
```

`src/inputs/user_parameters/` is a reserved, empty spot for your own parameter-model
scripts. It's tracked as a folder (so it always exists after checkout) but its contents are
gitignored — put a parameter-model `.py` file there if you want it importable the same way
`system/parameters/` scripts are, without it ever being committed.

## `inputs/` — untracked personal data

The repo root also has an `inputs/` folder, tracked the same way `outputs/` is: the folder
itself is tracked (via `.gitkeep`) but everything you put inside it is gitignored. This is
the conventional place to drop your own datasets — weather CSVs, allometry files, XML stand
exports — that aren't Python and don't need to be importable.

```
inputs/
└── .gitkeep   # only this is tracked; drop your own files anywhere under inputs/
```

## Pointing `AppSettings` at these folders

::: susi.io.app_settings.AppSettings
    handler: python
    options:
      show_source: false

`AppSettings.input_folder` and `AppSettings.user_input_folder` give you both roots as
resolved, validated paths, so you never have to hardcode an absolute path or guess the repo
layout:

```python
from susi.io.app_settings import AppSettings

app_settings = AppSettings()

app_settings.input_folder       # <repo_root>/src/inputs
app_settings.user_input_folder  # <repo_root>/inputs
```

## Building `SusiParams` from your own data

`WeatherParams.FMI_weather_filepath` and `CanopyLayerAllometry.allometry_dir_path` both
accept any path, tracked or not — build them off `user_input_folder` instead of
`input_folder` to point at your own data:

```python
from susi.io.app_settings import AppSettings
from susi.io.susi_parameter_model import WeatherParams, CanopyLayerAllometry, CanopyLayerName

app_settings = AppSettings()

weather_parameters = WeatherParams(
    FMI_weather_filepath=app_settings.user_input_folder.joinpath("my_site/weather.csv"),
)

allometry_parameters = CanopyLayerAllometry(
    allometry_dir_path=app_settings.user_input_folder.joinpath("my_site/allometry"),
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

## Summary

| Folder | Tracked? | Importable? | Use for |
|---|---|---|---|
| `src/inputs/system/` | Yes | Yes | Dev data used by tests/example scripts |
| `src/inputs/user_parameters/` | Folder only | Yes | Your own parameter-model scripts |
| `inputs/` (repo root) | Folder only | No | Your own weather/allometry/stand data |
