# susi_2021 -- water-table validation

A **testing project** (see `CONTEXT.md`): it checks SUSI against field measurements and does not
study a site. SUSI simulates 11 drained peatland sites and `figures_2021.py` compares the results
with measured water tables and measured stand growth.

The code and parameters are tracked. The data is **not**: it is 27 MB, and the measurements are
Luke / University of Helsinki data whose redistribution terms haven't been checked. The script runs
only once you have put the data in place by hand.

## Files

| File | What it is |
| --- | --- |
| `parameters_2021.py` | Per-site parameters, plus `PROJECT_DIR`, `RUN_ID` and the data paths. Both scripts use this one definition. |
| `susi_2021.py` | Runs every site in parallel and writes to `outputs/run_01/<stand>/<scenario>/`. |
| `figures_2021.py` | Reads those outputs and the measurements, then draws the validation figures. |

## Data to put in place (untracked)

```
susi_2021/
├── data/
│   ├── weather/                    FMI weather, one CSV per weather station:
│   │                               jaakkoinsuo, koirasuo, muhos, nevajarvi, parkano _weather.csv
│   └── measurements/
│       ├── gr_bio.xlsx             measured stand growth (biomass) per site
│       └── Pohjavesiaineistot/     measured water tables: *_pohjavesi_koottu.xlsx
│                                   per site (sheet "CSV"), with the raw logger
│                                   folders beside them
└── inputs/
    └── allometry/                  Motti growth tables, one .xls per site and
                                    scenario (e.g. ansa21_A.xls)
```

`parameters_2021.py` says which file each site reads (`wfile`, `mottifile`), and
`figures_2021.py`'s `WT_MEASUREMENT_INFO` says which measurement workbook and tubes each site uses.

Where the data came from: the SUSI 2021 validation material (A. Laurén, University of Helsinki,
with measurements from Luke). If you have an old `projects/susi_2021/inputs/`, move it here by
hand:

| Old location | New location |
| --- | --- |
| `inputs/vesitase_wfiles/*` | `data/weather/` |
| `inputs/Pohjavesiaineistot/` | `data/measurements/Pohjavesiaineistot/` |
| `inputs/gr_bio.xlsx` | `data/measurements/gr_bio.xlsx` |
| `inputs/motti_files/*` | `inputs/allometry/` |

## Running

From the repo root:

```sh
uv run python test_projects/susi_2021/susi_2021.py
uv run python test_projects/susi_2021/figures_2021.py
```

Run the scripts directly, because this folder is not a package. The scripts import
`parameters_2021` as a sibling module. A run fails if `outputs/run_01/` already exists, so delete
it before you run again.
