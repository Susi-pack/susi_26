---
icon: lucide/layers
---

# SUSI's three input data types

Every SUSI run is built from three kinds of input: **weather**, **allometry**,
and a **`SusiParams`** instance.
This page explains what each one is and how
you get it.
For where to put your own copies of these and how to reference
them, see [How to handle input data](how_to_handle_input_data.md).

## Where input data lives

Your input data is *user data*: site-specific weather files, allometry
tables, XML stand exports, parameter models, and so on.
It is personal to your machine and must never end up in a commit.
It lives in a
project's own `inputs/` and `data/` folders under the projects root, found through the project folder a run script names.

The small weather/allometry files and parameter models in `tests/` are test
fixtures: they back the test suite, not your runs.
The example projects in
`example_projects/` carry their own copies.

## Weather

Weather is the meteorological forcing for the simulation: an FMI
interpolated daily weather record (mean/max/min temperature, rainfall,
radiation, humidity) for a given site, read from a semicolon-delimited CSV.

How to get it: To do.

Reference: [Weather data](weather_data.md), [`WeatherParams`](simulation_config.md#susi.io.susi_parameter_model.WeatherParams).

## Allometry

Allometry is a canopy layer's growth-and-yield table — age, basal area, stem
count, mean diameter, and mean height for each tree stratum — that drives
stand growth over the course of a run.

How to get it: generate it with one of two converters, depending on the
inventory data you start from.
Both write the same allometry CSV format.

- From a Finnish forest XML stand export (metsätietostandardit):
  [`xml_to_allometry.py`](xml_to_allometry.md).
- From a Metsäkeskus forest inventory GeoPackage:
  [`metsakeskus_to_allometry.py`](how_to_generate_allometry_from_metsakeskus.md),
  which converts every drained-peatland stand in the export, one file per
  canopy layer.

Reference: [XML → allometry file](xml_to_allometry.md),
[Metsäkeskus data → allometry files](metsakeskus_to_allometry.md),
[`CanopyLayerAllometry`](simulation_config.md#susi.io.susi_parameter_model.CanopyLayerAllometry).

## `SusiParams`

`SusiParams` is the Pydantic object that fully defines a single SUSI run —
weather, allometry, canopy, cutting-management, and every other run
parameter, bundled into one instance.

How to get it: unlike weather and allometry, it isn't derived from an
external dataset — you write it yourself, as a parameter-model script that
instantiates `SusiParams` (see [How to handle input data](how_to_handle_input_data.md)
for where to put your own).

Reference: [Simulation Parameters](simulation_config.md).

## Summary

| Type | What it is | How you get it | Reference |
|---|---|---|---|
| Weather | FMI interpolated daily weather CSV | To do | [Weather data](weather_data.md) |
| Allometry | Growth-and-yield table per canopy layer | Generate with `xml_to_allometry.py` from an XML stand export, or `metsakeskus_to_allometry.py` from a Metsäkeskus `.gpkg` | [XML → allometry file](xml_to_allometry.md), [Metsäkeskus data → allometry files](metsakeskus_to_allometry.md) |
| `SusiParams` | Full run configuration | Write it yourself as a parameter-model script | [Simulation Parameters](simulation_config.md) |
