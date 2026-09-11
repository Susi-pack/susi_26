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

from pyproj import Transformer

SOURCE_CRS = "EPSG:3067"  # ETRS-TM35FIN, the CRS both tools' input geometries ship in
YKJ_CRS = "EPSG:2393"  # Finnish YKJ grid, what Growth_and_Yield_Table's x/y expect


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
