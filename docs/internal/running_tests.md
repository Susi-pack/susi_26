---
icon: lucide/flask-conical
---

# Running the tests

The test suite is split into three groups, so the bare `pytest` command we run while developing stays fast.

| Group | What it holds | Command | When to run it |
|---|---|---|---|
| fast | Every test without a group marker. | `uv run pytest` | On every change while developing. |
| slow | Tests marked `@pytest.mark.slow`. | `uv run pytest -m slow` | Once at the end of a development cycle. CI runs it on every PR. |
| golden | Tests marked `@pytest.mark.golden`. | `uv run pytest -m golden` | Once at the end of a development cycle. CI runs it on every PR. |
| all | All three groups. | `uv run pytest --all` | Before a release. |

Plain `pytest` deselects the slow and golden groups, and its summary line says how many tests it left out (`N deselected`).

## Which group a new test goes in

- **golden**: the test compares a run's output against a stored reference output. Today that is `tests/golden_file_test/test_golden_file.py`, which compares a full SUSI run with `golden_susi.nc`.
- **slow**: the test runs the growth model (the allometry tools' `Growth_and_Yield_Table`) or a full SUSI simulation, or it takes more than 0.5 s when run on its own.
- **fast**: everything else. Leave it unmarked.

A test belongs to one group only.
A golden test runs a full simulation too, but it gets only the `golden` mark, so `pytest -m slow` doesn't run it again.

Mark a test with a decorator.
If the test has other decorators, put the mark on top:

```python
import pytest


@pytest.mark.slow
def test_process_stand_writes_two_csvs(tmp_path):
    ...
```

### Measuring "more than 0.5 s"

`uv run pytest --durations=20` lists the slowest tests.
A test can look slow in the full run only because it is the first one to import or start up something heavy (geopandas' file drivers, Hypothesis).
Run it on its own before marking it:

```bash
uv run pytest --durations=1 "tests/path/to/test_file.py::test_name"
```

Mark it slow only if it is still over 0.5 s alone.

## Running a single slow or golden test

Naming a slow or golden test's file or ID is not enough.
The default `-m` still deselects it, pytest prints `1 deselected` and runs nothing.
Add the test's group, or `--all`:

```bash
uv run pytest -m golden tests/golden_file_test/test_golden_file.py
uv run pytest -m slow "tests/tools/test_xml_to_allometry.py::test_process_stand_writes_a_csv_round_tripping_through_the_real_reader"
```

## How it is wired

- The markers and the default `-m "not slow and not golden"` are in `[tool.pytest]` in `pyproject.toml`. A `-m` on the command line replaces the default one.
- `--all` is defined in `tests/conftest.py`. It clears the marker filter.
- CI (`.github/workflows/ci.yml`) runs each group as its own job: `test` (fast), `slow` and `golden`.
