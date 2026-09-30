"""Suggests a DDY (effective temperature sum) value for a region's
config.toml, estimated from the mean YKJ northing (y-coordinate) of that
region's stands -- run this once per gpkg, before writing config.toml.

Why this is legitimate for DDY but not for altitude
-----------------------------------------------------
Neither DDY nor altitude is a per-stand attribute in the Metsakeskus gpkg
schema (checked every column the pipeline touches -- there is no elevation
or temperature-sum field anywhere), so both were historically filled in by
hand per region, via MK_to_susi_allometry_v2.ipynb's REGION_META_MAP:

    REGION_META_MAP = {
        "Uusimaa":           dict(DDY=1250, y=6700, x=338, altitude=50),
        "Kanta-Hame":        dict(DDY=1200, y=6785, x=336, altitude=100),
        "Pirkanmaa":         dict(DDY=1150, y=6835, x=332, altitude=130),
        "Pohjanmaa":         dict(DDY=1050, y=7020, x=320, altitude=30),
        "Pohjois_Pohjanmaa": dict(DDY=950,  y=7180, x=347, altitude=40),
        "Kainuu":            dict(DDY=900,  y=7200, x=353, altitude=200),
        "Lappi":             dict(DDY=800,  y=7400, x=346, altitude=120),
    }

Plotting those 7 regions' y against DDY is close to a straight line (fits
with R^2 = 0.994) -- Finland's effective temperature sum tracks latitude
very tightly. Altitude does NOT follow any such pattern (it's local
topography: e.g. Kainuu y=7200/altitude=200 sits right next to
Pohjois-Pohjanmaa y=7180/altitude=40, a 5x difference at nearly the same
latitude), so altitude still needs either a DEM sampled at stand centroids
or a manual per-region value -- this script does not touch it.

Usage
-----
    python estimate_ddy_from_latitude.py path/to/region.gpkg

Prints the mean YKJ y-coordinate of every stand centroid in the gpkg's
`stand` layer, plus the DDY that linear fit predicts for it. Copy the
suggested value into that region's config.toml `ddy` field (altitude still
needs to be set by hand, e.g. by carrying over the nearest REGION_META_MAP
region's value, or a DEM sample if one becomes available).
"""

import argparse
from pathlib import Path

import geopandas as gpd

# Linear fit of DDY ~ y (YKJ northing, in THOUSANDS of meters -- i.e. the
# REGION_META_MAP convention, where Lappi's y=7400 means a real northing of
# 7,400,000 m) from REGION_META_MAP's 7 points (least squares, R^2 = 0.994):
# DDY = DDY_SLOPE * y_km + DDY_INTERCEPT
DDY_SLOPE = -0.6491565042491878
DDY_INTERCEPT = 5598.081069817156


def estimate_ddy_from_mean_y_km(mean_y_km: float) -> float:
    """DDY predicted by the region-metadata linear fit for a given mean YKJ
    y-coordinate (northing), IN THOUSANDS OF METERS (REGION_META_MAP's
    convention -- e.g. 7400 for Lappi, not 7400000). See module docstring
    for how this was derived."""
    return DDY_SLOPE * mean_y_km + DDY_INTERCEPT


def mean_stand_y_meters(input_gpkg: Path) -> float:
    """Mean YKJ y-coordinate (northing) across every stand centroid in the
    gpkg's `stand` layer, IN METERS (the gpkg's own units -- e.g.
    ~7,373,000 for a Lappi stand). Callers must divide by 1000 before
    passing this to estimate_ddy_from_mean_y_km, which expects
    REGION_META_MAP's thousands-of-meters convention instead."""
    stand = gpd.read_file(input_gpkg, layer="stand")
    centroids = stand.geometry.centroid
    return float(centroids.y.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_gpkg", type=Path)
    args = parser.parse_args()

    mean_y_m = mean_stand_y_meters(args.input_gpkg)
    mean_y_km = mean_y_m / 1000.0
    suggested_ddy = estimate_ddy_from_mean_y_km(mean_y_km)

    print(f"gpkg                  : {args.input_gpkg}")
    print(f"mean stand y (meters) : {mean_y_m:.1f}")
    print(f"mean stand y (REGION_META_MAP units, /1000): {mean_y_km:.1f}")
    print(f"suggested DDY         : {suggested_ddy:.0f}")
    print()
    print("Copy this into the region's config.toml, e.g.:")
    print(f"    ddy = {suggested_ddy:.0f}")
    print()
    print(
        "altitude has no such latitude relationship (it's local topography) "
        "and still needs a manual value or a DEM sample."
    )


if __name__ == "__main__":
    main()