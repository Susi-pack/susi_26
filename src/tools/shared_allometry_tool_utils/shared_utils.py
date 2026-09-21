"""The YKJ coordinate-transform arithmetic shared by xml_to_allometry.py and
metsakeskus_to_allometry.py: EPSG:3067 (ETRS-TM35FIN, what both tools' input
data ships in) -> EPSG:2393 (YKJ, what Growth_and_Yield_Table's x/y expect),
rounded/scaled to the grid units it wants (10 km easting units, 1 km
northing units).

Each tool still supplies its own input point -- xml_to_allometry.py uses a
stand's first polygon vertex, metsakeskus_to_allometry.py uses the stand
polygon's centroid -- that choice is unchanged and stays with each tool.
"""

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pyproj import Transformer

SOURCE_CRS = "EPSG:3067"  # ETRS-TM35FIN, the CRS both tools' input geometries ship in
YKJ_CRS = "EPSG:2393"  # Finnish YKJ grid, what Growth_and_Yield_Table's x/y expect

# The plausible range of a YKJ grid coordinate, in exactly the units
# point_to_ykj returns below -- which is why they live here, next to the
# function that produces them, rather than in whichever module happens to
# check them. This is the ONE definition: StandData's x_ykj/y_ykj fields
# (stand_data.py) and input_validation.validate_x_y_ykj both import it, so a
# coordinate is held to the same bounds however it reaches the tools -- read
# out of a stand-data document, or converted from a config's ETRS-TM35FIN
# x/y.
#
# Deliberately wider than Finland itself (real mainland stands land around
# x 305-376, y 6600-7780), for the same reason ALTITUDE_MIN/MAX and
# DDY_MIN/MAX are: this is a typo-catcher, not a border. It should catch a
# coordinate that never went through the EPSG:3067 -> YKJ conversion, or
# went through it with x and y swapped, while never rejecting a real stand.
X_YKJ_MIN = 200  # YKJ easting, 10 km units
X_YKJ_MAX = 550
Y_YKJ_MIN = 6500  # YKJ northing, 1 km units
Y_YKJ_MAX = 7900

# Named types so a model field says which coordinate it holds, and picks up
# the bounds by doing so.
YkjEasting = Annotated[int, Field(ge=X_YKJ_MIN, le=X_YKJ_MAX)]
YkjNorthing = Annotated[int, Field(ge=Y_YKJ_MIN, le=Y_YKJ_MAX)]


@lru_cache(maxsize=1)
def _ykj_transformer() -> Transformer:
    """Built once and reused -- constructing a Transformer is comparatively
    expensive, and point_to_ykj runs once per stand (thousands per real run)."""
    return Transformer.from_crs(SOURCE_CRS, YKJ_CRS, always_xy=True)


def point_to_ykj(x: float, y: float) -> tuple[int, int]:
    """A single (x, y) point in EPSG:3067 -> YKJ grid coordinates, scaled to
    the units Growth_and_Yield_Table expects."""
    transformer = _ykj_transformer()
    easting, northing = transformer.transform(x, y)
    return round(easting / 10000), round(northing / 1000)
