"""System data: the small weather/allometry/parameter fixtures tracked in the repo.

These files ship *with* the package, so where they live is a fact, not a setting:
they are addressed relative to this file rather than through `AppSettings`.
User-authored datasets live under `projects/`, not here.
"""

from pathlib import Path

# Root of this package, i.e. the folder holding weather/, allometry/ and parameters/.
SYSTEM_INPUTS_DIR = Path(__file__).parent
