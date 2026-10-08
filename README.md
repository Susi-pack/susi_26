Peatland simulator SUSI version used in Saari et al. (nimi) 2025 and Niemi et al. (2025) (nimi)

📖 **Documentation:** <https://susi-pack.github.io/susi_26/>

# Installation
Clone this repo.
Then, navigate to the project directory and install it using some Python package manager.

## Using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended)
Create a virtual environment and install the project + core dependencies:
```
uv sync
```

If you want to install developer dependencies (required for running tests), run
```
`uv sync --group dev`
```

Then, activate the environment:
```
`source .venv/bin/activate`
```

Or skip activation and run commands directly:
```
`uv run python ...`
```

## Using `pip`
NOTE: at the moment we are using Python 3.11.
Installing with `uv` picks the right Python version, but installing with `pip` doesn't.
There are in principle no reasons why higher Python versions shouldn't work.
But, if installing via `pip`, you are responsible for managing the Python version yourself.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

# Run
## SUSI model
The script `example_projects/minimal/single_simulation.py` runs a single SUSI simulation. It is the simplest entry point.
`python example_projects/minimal/single_simulation.py`

More complete examples live in `example_projects/`: each is a whole project folder (inputs, data, run scripts and outputs), laid out like a project of your own.

## Tools
Installing SUSI also installs these commands (run them with the environment
active, or prefixed with `uv run`):

| Command | What it does | Docs |
|---|---|---|
| `susi-init-project` | Creates a new, empty project folder | — |
| `susi-xml-to-allometry` | XML stand export → allometry CSVs + `stand_data.json` | `docs/xml_to_allometry.md` |
| `susi-metsakeskus-to-allometry` | Metsäkeskus `.gpkg` → allometry CSVs + `stand_data.json` | `docs/metsakeskus_to_allometry.md` |
| `susi-new-growth-allometry` | Post-clearcut allometry for one stand | `docs/new_growth_allometry.md` |

Each takes `--help`.

## GUI for output analysis (under development)
`susi-analyze`


# Project structure
= To be done =
