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
from susi.io.extra_pydantic_types import (
    StrictFrozenModel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)


STAND_DATA_FILENAME = "stand_data.json"


class StandData(StrictFrozenModel):
    """
    A single stand as described by different output data sources (Metsäkeskus and XML standard)
    This is used in the three allometry generating tools.
    """

    site_fertility_class: PositiveInt = Field(description="Site fertility class")
    canopy_layer_files: dict[CanopyLayerName, AllometryFileAndSpecies] = Field(
        description="Allometry file and species per canopy layer (dominant/subdominant/under)."
    )
    x_ykj: int = Field(
        ge=250,
        le=400,
        description="Stand location, YKJ grid easting (10 km units). Computed from the source geometry via shared_utils.point_to_ykj.",
    )
    y_ykj: int = Field(
        ge=6500,
        le=7800,
        description="Stand location, YKJ grid northing (1 km units). Computed from the source geometry via shared_utils.point_to_ykj.",
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
    basal_area: NonNegativeFloat | None = Field(
        default=None,
        description="Stand basal area, m2/ha. Plain metadata, no current reader.",
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
    stem_count: NonNegativeFloat | None = Field(
        default=None,
        description="Stem count, trees/ha. Plain metadata, no current reader.",
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
