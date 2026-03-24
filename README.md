Peatland simulator SUSI version used in Saari et al. (nimi) 2025 and Niemi et al. (2025) (nimi)

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
The script located in `src/scripts/susi_calls.py` runs a single SUSI simulation. It is the simplest entry point.
`python src/scripts/susi_calls.py`

More complex scripts might be found in the same folder.

## GUI for output analysis (under development)
`susi-analyze`

# Docs
The repo is not public yet, so the docs only exist if you compile them locally.
We use `zensical`.

# Project structure
= To be done =
