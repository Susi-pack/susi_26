"""Reprojection of stand geometry into SOURCE_CRS, for xml_to_allometry.py.

Stand geometry is brought into SOURCE_CRS (EPSG:3067, ETRS-TM35FIN) as soon
as it's read -- to_source_crs below for one polygon; the Metsäkeskus tool
reprojects its whole stand layer at once -- so everything downstream,
stand_data.json included, only ever sees EPSG:3067.

SOURCE_CRS itself, and the EPSG:3067 -> YKJ arithmetic that follows it,
live in susi.io.stand_data, next to the StandData fields that hold the
result.
"""

from functools import lru_cache

import shapely.ops
from pyproj import Transformer
from pyproj.exceptions import CRSError
from shapely.geometry import Polygon

from susi.io.stand_data import SOURCE_CRS


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
