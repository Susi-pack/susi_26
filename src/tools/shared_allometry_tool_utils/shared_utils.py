"""The coordinate arithmetic shared by xml_to_allometry.py and
metsakeskus_to_allometry.py.

Stand geometry is brought into SOURCE_CRS (EPSG:3067, ETRS-TM35FIN) as soon
as it's read -- to_source_crs below for one polygon; the Metsäkeskus tool
reprojects its whole stand layer at once -- so everything downstream,
stand_data.json included, only ever sees EPSG:3067.

From there, EPSG:3067 -> EPSG:2393 (YKJ, what Growth_and_Yield_Table's x/y expect),
rounded/scaled to the grid units it wants (10 km easting units, 1 km
northing units).

Both tools take a stand's YKJ grid location at its polygon centroid, through
the one centroid_to_ykj below (xml_to_allometry.py used the first polygon
vertex until ticket 22). That location feeds only the sawlog-reduction
equation (StemCurve.sawlogReduction), so it's a growth-model input, not a
second set of coordinates for the stand.
"""

from functools import lru_cache
from typing import Annotated

from pydantic import Field
import shapely.ops
from pyproj import Transformer
from pyproj.exceptions import CRSError
from shapely.geometry import Polygon

SOURCE_CRS = "EPSG:3067"  # ETRS-TM35FIN, the CRS all stand geometry is brought into
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


def centroid_to_ykj(polygon: Polygon) -> tuple[int, int]:
    """A stand polygon's centroid -> YKJ grid coordinates, via point_to_ykj.
    The polygon must be in SOURCE_CRS (EPSG:3067)."""
    return point_to_ykj(polygon.centroid.x, polygon.centroid.y)


def require_source_crs(declared_crs: str | None, where: str) -> None:
    """The "is this SOURCE_CRS?" check behind StandDataDocument.crs.
    point_to_ykj hard-codes SOURCE_CRS, so geometry in any other CRS would
    give wrong YKJ grid cells (and wrong raster pixels downstream) without
    any error. The generating tools never hit this: they reproject into
    SOURCE_CRS on the way in (to_source_crs). `where` names what declared
    the CRS, for the error message."""
    if declared_crs != SOURCE_CRS:
        raise ValueError(
            f"{where} is in CRS {declared_crs!r}, but only {SOURCE_CRS} is supported"
        )


@lru_cache(maxsize=None)
def _to_source_crs_transformer(declared_crs: str) -> Transformer:
    """One Transformer per source CRS, reused: an XML export repeats the
    same srsName on every stand."""
    return Transformer.from_crs(declared_crs, SOURCE_CRS, always_xy=True)


def to_source_crs(polygon: Polygon, declared_crs: str | None, where: str) -> Polygon:
    """
    A polygon in declared_crs -> the same polygon in SOURCE_CRS, holes
    included. Returned as-is when it's already there.

    Coordinates are read x-first (easting/longitude before northing/
    latitude), whatever axis order the CRS officially declares: the same
    always_xy convention point_to_ykj and geopandas use.

    Raises ValueError when there is no declared CRS (there is nothing to
    reproject from, and guessing would give silently wrong geometry) or when
    pyproj doesn't recognise it. `where` names what declared the CRS, for
    the error message.
    """
    if declared_crs is None:
        raise ValueError(f"{where} declares no CRS, so it can't be reprojected")
    if declared_crs == SOURCE_CRS:
        return polygon
    try:
        transformer = _to_source_crs_transformer(declared_crs)
    except CRSError as error:
        raise ValueError(
            f"{where} declares CRS {declared_crs!r}, which pyproj doesn't "
            f"recognise: {error}"
        ) from error
    return shapely.ops.transform(transformer.transform, polygon)
