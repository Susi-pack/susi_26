---
icon: lucide/file-input
---

# XML to Allometry Converter

## Overview

The `xml_to_allometry.py` script transforms XML stand data following Metsätietostandardit into the allometry CSV format required by SUSI's growth and yield model.

To generate the same kind of file from a Metsäkeskus GeoPackage instead, see
[How to generate allometry files from Metsäkeskus data](how_to_generate_allometry_from_metsakeskus.md).

This tool allows SUSI to work with this standardized forestry data format.


## Finnish Forestry XML Standard

The input XML files follow the **Finnish national forest information standards** (metsätietostandardit), managed by the Finnish Forest Centre. These standards define XML schema specifications for exchanging forest resource data between different operators in the forestry sector.

Key characteristics of the format:

- Uses **GML (Geography Markup Language)** for geospatial data (polygon geometries)
- Structured around forest **stands** (kuvio) as the basic unit
- Contains tree stratum data, stand statistics, and metadata

## Input XML Structure

The XML file must follow this hierarchical structure:

```xml
<ForestPropertyData>
  <Stands>
    <Stand id="...">
      <StandBasicData>
        <FertilityClass>...</FertilityClass>
        <MainGroup>...</MainGroup>
        <SubGroup>...</SubGroup>
        <SoilType>...</SoilType>
        <PolygonGeometry>
          <polygonProperty>
            <Polygon>
              <exterior>
                <LinearRing>
                  <coordinates> x1,y1 x2,y2 ... </coordinates>
                </LinearRing>
              </exterior>
            </Polygon>
          </polygonProperty>
        </PolygonGeometry>
      </StandBasicData>
      <TreeStandData>
        <TreeStandDataDate>
          <TreeStandSummary>
            <MeanDiameter>...</MeanDiameter>
            <MeanAge>...</MeanAge>
            <BasalArea>...</BasalArea>
            <MeanHeight>...</MeanHeight>
            <Volume>...</Volume>
          </TreeStandSummary>
          <TreeStrata>
            <TreeStratum>
              <Age>...</Age>
              <BasalArea>...</BasalArea>
              <StemCount>...</StemCount>
              <MeanDiameter>...</MeanDiameter>
              <MeanHeight>...</MeanHeight>
            </TreeStratum>
            <!-- Two more TreeStratum elements (3 total) -->
          </TreeStrata>
        </TreeStandDataDate>
      </TreeStandData>
    </Stand>
  </Stands>
</ForestPropertyData>
```

### Required Elements

| Element | Description |
|---------|-------------|
| `Stand/@id` | Unique stand identifier |
| `StandBasicData/FertilityClass` | Site fertility class (integer) |
| `StandBasicData/PolygonGeometry` | Stand boundary as GML polygon |
| `TreeStandSummary/MeanDiameter` | Mean diameter at breast height (cm) |
| `TreeStandSummary/MeanAge` | Mean stand age (years) |
| `TreeStandSummary/BasalArea` | Basal area (m²/ha) |
| `TreeStandSummary/MeanHeight` | Mean height (m) |
| `TreeStandSummary/Volume` | Total volume (m³/ha) |
| `TreeStrata/TreeStratum` | Exactly 3 tree strata with age, basal area, stem count, diameter, and height |

### Optional Elements

These are parsed for informational purposes and included in the JSON output:

- `StandBasicData/MainGroup` - Main species group
- `StandBasicData/SubGroup` - Sub group
- `StandBasicData/SoilType` - Soil type

## Output Files

### Allometry CSV files

For each stand in the XML file, the script generates a CSV file named
`susi_input_{stand_id}.csv`: the allometric road map produced by
`Growth_and_Yield_Table`, with a leading `Species_ID` column naming the stand's
main species. It follows the canonical allometry schema — the columns declared
in `susi.core.allometry_columns.ALLOMETRY_COLUMNS` — and is read directly by
`read_allometry_info_from_csv`.

### Informational JSON File

An additional file `extra_XML_info.json` is created in the output directory, containing the complete parsed stand data in JSON format. This includes:

- Stand ID and fertility class
- Polygon coordinates
- Tree stratum data (all 3 strata)
- Main species (determined by largest basal area)
- Optional metadata: main group, sub group, soil type, mean age, basal area, mean height, total volume

## Usage

```bash
python xml_to_allometry.py <input.xml> <output_directory> --altitude=<value> --ddy=<value>
```

**Arguments:**

- `input.xml` - Path to the input XML file
- `output_directory` - Path to the output directory for generated files
- `--altitude` - **Required.** Altitude above sea level, in metres, applied to every stand in this run. Not present in the XML standard, so it must be supplied explicitly.
- `--ddy` - **Required.** Temperature sum (degree days per year). Also not present in the XML standard, so it must be supplied explicitly.
- `--allow-out-of-range-values` - Optional flag. See [Altitude / DDY Validation](#altitude--ddy-validation) below.
- `--do-thinning` - Optional flag. Apply sapling stand thinning where needed (see [Thinning Rate Calculation](#thinning-rate-calculation)).

**Example:**

```bash
python src/tools/xml_to_allometry/xml_to_allometry.py data/forest_stands.xml output/allometry_files/ --altitude=150 --ddy=1200
```

### Altitude / DDY Validation

`--altitude` and `--ddy` apply the same value to every stand processed in this run — there is currently no way to vary them per stand or per XML file in a single run.

Values are checked against an enforced range, deliberately set a bit wider than what's typical for Finnish forest land, to allow some margin without silently accepting nonsense input:

| Parameter | Enforced range | Typical Finnish value | Unit |
|-----------|-----------------|------------------------|------|
| `--altitude` | 0 – 1000 | sea level – 700 | metres above sea level |
| `--ddy` | 500 – 2000 | ~600 (Lapland) – ~1500 (southern Finland) | degree days per year |

By default, a value outside the **enforced range** blocks the run with an error naming every violation found (not just the first). Pass `--allow-out-of-range-values` to proceed anyway — the tool will still print a warning for each out-of-range value used, so it stays visible in the run's output. `NaN` is always rejected outright, regardless of `--allow-out-of-range-values`.

## Technical Details

### Coordinate Transformation

The script transforms stand coordinates from EPSG:3067 (ETRS-TM35FIN) to EPSG:2393 (YKJ) coordinate reference system.

### Thinning Rate Calculation

For young stands with high stem counts and small diameters, the script makes a decision about whether thinning is needed, and calculates a thinning rate:

- **Threshold:** 2500 stems/ha (species ≠ 2) or 2200 stems/ha (species = 2)
- **Condition:** mean diameter < 8 cm AND stem count > threshold
- **Target:** Reduce to 2000 stems/ha (species = 2) or 1800 stems/ha (species ≠ 2)

If thinning is needed, the basal area and stem count are adjusted accordingly before generating the allometry table.

### Main Species Determination

The main species is automatically determined as the tree stratum with the largest basal area.
