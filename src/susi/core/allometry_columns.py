from dataclasses import dataclass


@dataclass(frozen=True)
class AllometryColumn:
    """
    Documentation-only metadata about one column of the canonical allometry CSV
    schema (see ALLOMETRY_COLUMNS below). Nothing here is validated or enforced
    at runtime beyond `name`, which doubles as the CSV header and the working
    DataFrame column name after read_allometry_info_from_csv().
    """

    name: str
    unit: str
    former_finnish_name: str
    description: str


ALLOMETRY_COLUMNS: tuple[AllometryColumn, ...] = (
    AllometryColumn(
        name="Schedule",
        unit="",
        former_finnish_name="Kasvatus",
        description="Constant within a file. Not read by allometry_development(). See issue #204.",
    ),
    AllometryColumn(
        name="Year",
        unit="years",
        former_finnish_name="Vuosi",
        description="Years elapsed from the start of this growth-and-yield table. Not read by allometry_development(), which uses Age instead. See issue #204.",
    ),
    AllometryColumn(
        name="Age",
        unit="years",
        former_finnish_name="Ikä",
        description="Stand age at this row.",
    ),
    AllometryColumn(
        name="N", unit="stems/ha", former_finnish_name="N", description="Stem count."
    ),
    AllometryColumn(
        name="BA", unit="m2/ha", former_finnish_name="PPA", description="Basal area."
    ),
    AllometryColumn(
        name="Hg",
        unit="m",
        former_finnish_name="Hg",
        description="Basal-area-weighted mean height. Not read by allometry_development(), which uses Hdom (dominant height) instead. See issue #204.",
    ),
    AllometryColumn(
        name="Dg",
        unit="cm",
        former_finnish_name="Dg",
        description="Basal-area-weighted mean diameter.",
    ),
    AllometryColumn(
        name="Hdom",
        unit="m",
        former_finnish_name="Hdom",
        description="Dominant height (mean height of the 100 thickest stems/ha).",
    ),
    AllometryColumn(
        name="Volume",
        unit="m3/ha",
        former_finnish_name="Tilavuus",
        description="Total standing volume.",
    ),
    AllometryColumn(
        name="Logs",
        unit="m3/ha",
        former_finnish_name="Tukki",
        description="Saw-log assortment volume.",
    ),
    AllometryColumn(
        name="Pulp",
        unit="m3/ha",
        former_finnish_name="Kuitu",
        description="Pulpwood assortment volume.",
    ),
    AllometryColumn(
        name="Loss",
        unit="m3/ha",
        former_finnish_name="Hukka",
        description="Harvest residue / non-merchantable assortment volume. Not read by allometry_development(). See issue #204.",
    ),
    AllometryColumn(
        name="Yield",
        unit="m3/ha",
        former_finnish_name="Tuotos",
        description="Cumulative yield.",
    ),
    AllometryColumn(
        name="Mortality",
        unit="",
        former_finnish_name="Kuolleisuus",
        description="Cumulative mortality. Always 0 in xml_to_allometry.py-generated files. Not read by allometry_development(). See issue #204.",
    ),
    AllometryColumn(
        name="Stem_wood",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="runko(aines)",
        description="Stem wood biomass (merchantable). allometry.py converts this to kg/ha on read. See issue #205 for a possible unit mismatch with xml_to_allometry.py-generated files.",
    ),
    AllometryColumn(
        name="Stem_loss",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="runko(hukka)",
        description="Stem wood biomass lost to harvest residue. Not read by allometry_development(). See issues #204, #205.",
    ),
    AllometryColumn(
        name="Living_branches",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="elävät oksat",
        description="Living branch biomass. See issue #205.",
    ),
    AllometryColumn(
        name="Dead_branches",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="kuolleet oksat",
        description="Dead branch biomass. See issue #205.",
    ),
    AllometryColumn(
        name="Foliage",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="lehdet",
        description="Foliage (leaf/needle) biomass. See issue #205.",
    ),
    AllometryColumn(
        name="Stump",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="Kannot",
        description="Stump biomass. See issue #205.",
    ),
    AllometryColumn(
        name="Coarse_roots",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="Juuret >2mm",
        description="Coarse root biomass (>2mm diameter). See issue #205.",
    ),
    AllometryColumn(
        name="Fine_roots",
        unit="tonnes/ha (Mg/ha)",
        former_finnish_name="Hienojuuret",
        description="Fine root biomass (<=2mm diameter). See issue #205.",
    ),
)
