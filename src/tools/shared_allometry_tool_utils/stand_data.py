# Declares data structures for the JSON document storing external stand data information
from pathlib import Path

from pydantic import Field

from susi.io.load_output_data import StandID
from susi.io.susi_parameter_model import (
    CanopyLayerAllometry,
    CanopyLayerName,
    AllometryFileAndSpecies,
    StandParams,
)

# STAND_DATA_FILENAME is owned by susi.io.project_layout, so that `susi/` can
# name this file without importing `tools/`. Imported back here (the allowed
# direction) so `stand_data.STAND_DATA_FILENAME` keeps working. The tools
# themselves no longer need it -- they ask susi.io.project_layout for the
# whole path -- but the name stays reachable beside the document model it
# names, which is where the tool tests look for it.
from susi.io.project_layout import STAND_DATA_FILENAME as STAND_DATA_FILENAME
from susi.io.extra_pydantic_types import (
    StrictFrozenModel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)
from tools.shared_allometry_tool_utils.shared_utils import YkjEasting, YkjNorthing


class StandData(StrictFrozenModel):
    """
    A single stand as described by different output data sources (Metsäkeskus and XML standard)
    This is used in the three allometry generating tools.
    """

    site_fertility_class: PositiveInt = Field(description="Site fertility class")
    canopy_layer_files: dict[CanopyLayerName, AllometryFileAndSpecies] = Field(
        description="Allometry file and species per canopy layer (dominant/subdominant/under)."
    )
    x_ykj: YkjEasting = Field(
        description="Stand location, YKJ grid easting (10 km units). Computed from the source geometry via shared_utils.point_to_ykj, and bounded by the X_YKJ_MIN/MAX defined alongside it.",
    )
    y_ykj: YkjNorthing = Field(
        description="Stand location, YKJ grid northing (1 km units). Computed from the source geometry via shared_utils.point_to_ykj, and bounded by the Y_YKJ_MIN/MAX defined alongside it.",
    )
    polygon: str | None = Field(
        default=None,
        description="Stand boundary polygon, as whitespace-separated 'x,y' coordinate pairs in the source CRS. Plain metadata, kept for information/traceability only.",
    )

    stand_area: PositiveFloat | None = Field(
        default=None,
        description="Stand area, ha. Important for later analysis if the numbers have to be scaled to region level. Nothing in the SUSI simulation depends on the total stand area, but it is necessary later to run, e.g., the optimization algorithm.",
    )
    main_group: PositiveInt | None = Field(
        default=None,
        description=(
            "Finnish land-use main-group code (e.g. Metsakeskus 'maingroup' and XML st:MainGroup; 1 = forest land). Plain metadata here: the "
            "source tools use it to filter stands before StandData is built, "
            "not after."
        ),
    )
    sub_group: PositiveInt | None = Field(
        default=None,
        description=(
            "Finnish land-use sub-group code (e.g. Metsakeskus 'subgroup' and XML st:SubGroup; 2-3 = peatland). Plain metadata here, same "
            "filter-before-building caveat as main_group."
        ),
    )
    mean_height: NonNegativeFloat | None = Field(
        default=None,
        description="Mean tree height, m. Plain metadata, no current reader.",
    )
    mean_diameter: NonNegativeFloat | None = Field(
        default=None,
        description="Mean tree diameter, cm. Plain metadata, no current reader.",
    )
    total_volume: NonNegativeFloat | None = Field(
        default=None,
        description="Total stem volume, m3/ha. Plain metadata, no current reader.",
    )
    developmentclass: PositiveInt | None = Field(
        default=None,
        description=(
            "Stand development-class code (e.g. 1 = open/seedling, "
            "2 = young growing, 3 = grown-up). Plain metadata here: the "
            "source tools use it to filter stands before StandData is built, "
            "not after."
        ),
    )
    drainagestate: PositiveInt | None = Field(
        default=None,
        description=(
            "Drainage-state code (e.g. 7-9 = drained). Plain metadata "
            "here, the "
            "source tools use it to filter stands before StandData is built, "
            "not after."
        ),
    )
    soil_type: PositiveInt | None = Field(
        default=None,
        description=(
            "Finnish land-use soil-type code (e.g. XML st:SoilType / "
            "Metsakeskus soiltype). No SUSI-simulation consumer yet, but "
            "expected to eventually map onto SiteParams.peat_type/"
            "peat_type_bottom (susi_parameter_model.py) -- that mapping is "
            "still open work (#280)."
        ),
    )
    mean_age: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Mean stand age, years. No SUSI-simulation consumer yet, but "
            "expected to eventually feed an initial-stand-age field on "
            "SiteParams (susi_parameter_model.py), analogous to how "
            "initial_canopylayer_age_years is set today."
        ),
    )
    basal_area: NonNegativeFloat | None = Field(
        default=None,
        description="Stand basal area, m2/ha. Use to store basal area that is not species specific.",
    )
    basal_area_pine: NonNegativeFloat | None = Field(
        default=None, description="Basal area of pines (m^2/ha)."
    )
    basal_area_spruce: NonNegativeFloat | None = Field(
        default=None, description="Basal area of spruces (m^2/ha)."
    )
    basal_area_deciduous: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Basal area of deciduous trees (m^2/ha): the bucket for every "
            "species code >= 3, which at this inventory layer is a real mix "
            "of species rather than SUSI's birch simplification -- the "
            "engine's TreeSpecies.birch is the same bucket under the name "
            "src/susi/core uses. See CONTEXT.md, Species."
        ),
    )
    stem_count: NonNegativeFloat | None = Field(
        default=None,
        description="Stem count, trees/ha. Use to store number of stems that is not species specific.",
    )
    stem_count_pine: NonNegativeFloat | None = Field(
        default=None, description="Number of pines per hectare, trees/ha"
    )
    stem_count_spruce: NonNegativeFloat | None = Field(
        default=None, description="Number of spruces per hectare, trees/ha"
    )
    stem_count_deciduous: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Number of deciduous trees per hectare, trees/ha. Same "
            "species-code >= 3 bucketing as basal_area_deciduous."
        ),
    )


class StandDataDocument(StrictFrozenModel):
    """
    The serialized data structure that describes the whole stand data JSON document.
    It contains all stands.
    """

    altitude: float = Field(description="Project altitude, m.")
    ddy: float = Field(description="Project effective temperature sum (degree days).")
    stands: dict[StandID, StandData] = Field(
        description="All stands in the project, keyed by stand ID."
    )


def load_stand_data_document_from_json(path: Path) -> StandDataDocument:
    return StandDataDocument.model_validate_json(path.read_text())


def dump_stand_data_document(output_path: Path, document: StandDataDocument) -> None:
    """Writes a StandDataDocument as JSON to output_path -- the project's
    stand_data.json, normally susi.io.project_layout.stand_data_path_for_project.
    Takes the already-resolved path rather than a project_dir: both
    xml_to_allometry.py's and metsakeskus_to_allometry.py's main() already
    need that same path for their own status printing, so they derive it
    once and pass it in here, instead of each deriving it a second time."""
    output_path.write_text(document.model_dump_json())


def _build_stand_params_from_stand_data_document(
    stand_data_document: StandDataDocument,
    stand_id: StandID,
    n: int,
) -> StandParams:
    """
    Builds a simulation-ready StandParams for one stand out of a whole project's StandDataDocument.
    `n`, the number of soil columns, must be supplied by the caller:
    It is a property of the run being built, never of the stand data itself
    """
    stand_data = stand_data_document.stands[stand_id]

    canopy_layer_allometry = CanopyLayerAllometry.with_single_allometry_per_layer(
        layers=stand_data.canopy_layer_files, n=n
    )

    return StandParams(
        site_fertility_class=stand_data.site_fertility_class,
        canopy_layer_allometry=canopy_layer_allometry,
    )


def build_stand_params_from_stand_data_json(
    stand_data_json_path: Path,
    stand_id: StandID,
    n: int,
) -> StandParams:
    return _build_stand_params_from_stand_data_document(
        stand_data_document=load_stand_data_document_from_json(stand_data_json_path),
        stand_id=stand_id,
        n=n,
    )
