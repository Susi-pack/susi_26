# Declares data structures for the JSON document storing external stand data information
from pathlib import Path
from typing import Annotated, Any

import shapely
from pydantic import (
    AfterValidator,
    Field,
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    WithJsonSchema,
    model_serializer,
    model_validator,
)
from shapely.errors import ShapelyError
from shapely.geometry import Polygon

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
from tools.shared_allometry_tool_utils.shared_utils import (
    YkjEasting,
    YkjNorthing,
    require_source_crs,
)


def _validate_stand_polygon(value: Any) -> Polygon:
    """
    StandPolygon's validator: a shapely Polygon passes through as-is, a str is
    parsed as WKT (what stand_data.json holds), and anything that doesn't end
    up as a non-empty Polygon is rejected -- a MultiPolygon included, since no
    source produces one (see ticket 22) and consumers call Polygon-only API.
    """
    if isinstance(value, str):
        try:
            value = shapely.from_wkt(value)
        except ShapelyError as error:
            # Also catches the pre-ticket-22 on-disk format (raw
            # gml:coordinates pairs), which isn't WKT.
            raise ValueError(f"polygon is not valid WKT: {error}") from error
    if not isinstance(value, Polygon):
        raise ValueError(
            f"polygon must be a shapely Polygon or its WKT, got {type(value).__name__}"
        )
    if value.is_empty:
        raise ValueError("polygon must not be empty")
    return value


# The stand boundary: a shapely Polygon in memory, a WKT string on disk. See
# docs/adr/0004 for why WKT, and why the CRS lives on the document rather
# than inside the string (shapely can't read EWKT's "SRID=...;" prefix).
# PlainValidator replaces pydantic's own validation for the type, and
# WithJsonSchema describes it as the string it serializes to -- together they
# let the model build without arbitrary_types_allowed, which on its own would
# still leave the model unable to write or read a Polygon as JSON.
StandPolygon = Annotated[
    Polygon,
    PlainValidator(_validate_stand_polygon),
    PlainSerializer(lambda polygon: polygon.wkt, return_type=str),
    WithJsonSchema({"type": "string", "description": "WKT POLYGON"}),
]


def _validate_document_crs(crs: str) -> str:
    """Only SOURCE_CRS is supported -- see shared_utils.require_source_crs."""
    require_source_crs(crs, where="stand_data.json")
    return crs


DocumentCrs = Annotated[str, AfterValidator(_validate_document_crs)]


class StandData(StrictFrozenModel):
    """
    A single stand as described by different output data sources (Metsäkeskus and XML standard)
    This is used in the three allometry generating tools.
    """

    site_fertility_class: PositiveInt = Field(description="Site fertility class")
    allometry_file_per_layer: dict[CanopyLayerName, AllometryFileAndSpecies] = Field(
        description=(
            "Allometry file and species per canopy layer (dominant/subdominant/"
            "under). Same element type StandParams uses; build_stand_params "
            "expands it into a CanopyLayerAllometry. Inside a "
            "StandDataDocument, each file_path is absolute in memory and "
            "relative to the folder holding stand_data.json on disk -- see "
            "docs/adr/0005."
        )
    )
    x_ykj: YkjEasting = Field(
        description=(
            "The stand's YKJ grid location, easting (10 km units): an input to "
            "the sawlog-reduction equation (StemCurve.sawlogReduction), taken at "
            "the polygon centroid via shared_utils.centroid_to_ykj, and read back "
            "by new_growth_allometry.py in its sourced mode. Not a second set of "
            "coordinates for the stand. Bounded by X_YKJ_MIN/MAX."
        ),
    )
    y_ykj: YkjNorthing = Field(
        description=(
            "The stand's YKJ grid location, northing (1 km units). Same role "
            "and source as x_ykj. Bounded by Y_YKJ_MIN/MAX."
        ),
    )
    polygon: StandPolygon | None = Field(
        default=None,
        description=(
            "Stand boundary (exterior ring plus holes), in the document's crs. "
            "A shapely Polygon in memory, WKT in stand_data.json. Not "
            "audit-trail-only: paroninkorpi.py reads it for the ditch-depth "
            "raster lookup."
        ),
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


# Pydantic context key through which load_stand_data_document_from_json and
# dump_stand_data_document tell StandDataDocument which folder the document
# sits in: the folder every allometry file_path is relative to on disk.
DOCUMENT_DIR_CONTEXT_KEY = "stand_data_document_dir"


def _document_dir_from_context(context: Any) -> Path | None:
    if not isinstance(context, dict):
        return None
    return context.get(DOCUMENT_DIR_CONTEXT_KEY)


def _absolute_allometry_file_path(
    file_path: str | Path, stand_id: StandID, document_dir: Path | None
) -> Path:
    """
    The in-memory (absolute) form of one allometry file_path.

    With document_dir (loading stand_data.json), file_path is the on-disk
    form: it must be relative, stay inside document_dir (no `..`), and is
    joined onto document_dir. Without it (a document built in memory, or
    validated with no context), file_path must already be absolute: a
    relative path is never quietly resolved against the current folder,
    since that's the bug docs/adr/0005 removes.
    """
    path = Path(file_path)
    if document_dir is None:
        if not path.is_absolute():
            raise ValueError(
                f"Stand {stand_id!r}: allometry file_path {str(path)!r} is relative, "
                "but no document folder to resolve it against was given. Read "
                "stand_data.json with load_stand_data_document_from_json, or build "
                "the document in memory with absolute paths."
            )
        return path
    if path.is_absolute():
        raise ValueError(
            f"Stand {stand_id!r}: allometry file_path {str(path)!r} in "
            "stand_data.json is absolute; it must be relative to the folder "
            "holding stand_data.json."
        )
    if ".." in path.parts:
        raise ValueError(
            f"Stand {stand_id!r}: allometry file_path {str(path)!r} in "
            "stand_data.json contains '..'; allometry files must be inside the "
            "folder holding stand_data.json."
        )
    return document_dir / path


def _with_absolute_allometry_paths(
    stand: Any, stand_id: StandID, document_dir: Path | None
) -> Any:
    """
    Returns `stand` with every allometry file_path made absolute (see
    _absolute_allometry_file_path). A stand arrives either as raw data (a
    dict, from JSON or keyword arguments) or as an already-built StandData
    (when the document is built in memory); anything else is left for
    pydantic's own validation to reject.
    """

    def absolute(entry: Any) -> Any:
        if isinstance(entry, AllometryFileAndSpecies):
            return entry.model_copy(
                update={
                    "file_path": _absolute_allometry_file_path(
                        entry.file_path, stand_id, document_dir
                    )
                }
            )
        if isinstance(entry, dict) and isinstance(entry.get("file_path"), (str, Path)):
            # str, not Path: when validating JSON, pydantic still validates
            # this raw dict in JSON mode afterwards, where a Path field only
            # accepts a string.
            return {
                **entry,
                "file_path": str(
                    _absolute_allometry_file_path(
                        entry["file_path"], stand_id, document_dir
                    )
                ),
            }
        return entry

    if isinstance(stand, StandData):
        # StandData is frozen: model_copy is the only way to swap a field.
        # It skips validation, which is fine -- absolute() only replaces a
        # Path with another Path.
        return stand.model_copy(
            update={
                "allometry_file_per_layer": {
                    layer: absolute(entry)
                    for layer, entry in stand.allometry_file_per_layer.items()
                }
            }
        )
    if isinstance(stand, dict) and isinstance(
        stand.get("allometry_file_per_layer"), dict
    ):
        return {
            **stand,
            "allometry_file_per_layer": {
                layer: absolute(entry)
                for layer, entry in stand["allometry_file_per_layer"].items()
            },
        }
    return stand


def _relative_allometry_file_path(
    file_path: Path, stand_id: StandID, document_dir: Path
) -> str:
    """
    The on-disk form of one (absolute, in-memory) allometry file_path:
    relative to document_dir, as a POSIX string so the document reads the
    same on any OS. Both sides are resolved first, so a `..` inside
    file_path, or a symlinked document_dir, can't make a file outside
    document_dir look inside it (or the reverse).
    """
    try:
        relative = file_path.resolve().relative_to(document_dir.resolve())
    except ValueError:
        raise ValueError(
            f"Stand {stand_id!r}: allometry file {str(file_path)!r} is not inside "
            f"{str(document_dir)!r}, the folder stand_data.json is written to. "
            "Every allometry file must live inside the project: copy it into "
            "the project's inputs/ folder first."
        ) from None
    return relative.as_posix()


class StandDataDocument(StrictFrozenModel):
    """
    The serialized data structure that describes the whole stand data JSON document.
    It contains all stands.

    Allometry file paths are absolute in memory and relative to the
    document's own folder on disk (docs/adr/0005). This class does the
    conversion, driven by the DOCUMENT_DIR_CONTEXT_KEY context that
    load_stand_data_document_from_json and dump_stand_data_document pass --
    the only two intended ways in and out. AllometryFileAndSpecies itself is
    left alone, since in SusiParams a relative path still means relative to
    the current folder.
    """

    crs: DocumentCrs = Field(
        description=(
            "The CRS every stand polygon in this document is in -- one per "
            "document, never per stand. Always shared_utils.SOURCE_CRS "
            "(EPSG:3067): the generating tools reproject a source in any "
            "other CRS into it on the way in. See docs/adr/0004."
        ),
    )
    altitude: float = Field(description="Project altitude, m.")
    ddy: float = Field(description="Project effective temperature sum (degree days).")
    stands: dict[StandID, StandData] = Field(
        description="All stands in the project, keyed by stand ID."
    )

    @model_validator(mode="before")
    @classmethod
    def allometry_paths_are_absolute_in_memory(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
        # mode="before" because StandData is frozen: the paths have to be
        # absolute before the StandDatas are built, not patched afterwards.
        if not isinstance(data, dict) or not isinstance(data.get("stands"), dict):
            return data
        document_dir = _document_dir_from_context(info.context)
        return {
            **data,
            "stands": {
                stand_id: _with_absolute_allometry_paths(stand, stand_id, document_dir)
                for stand_id, stand in data["stands"].items()
            },
        }

    @model_serializer(mode="wrap")
    def allometry_paths_are_relative_on_disk(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        data = handler(self)
        document_dir = _document_dir_from_context(info.context)
        # No context (e.g. a bare model_dump for inspection): paths stay
        # absolute, which a context-free model_validate also accepts.
        if document_dir is None:
            return data
        for stand_id, stand in self.stands.items():
            dumped_layers = data["stands"][stand_id]["allometry_file_per_layer"]
            for layer, entry in stand.allometry_file_per_layer.items():
                dumped_layers[layer]["file_path"] = _relative_allometry_file_path(
                    entry.file_path, stand_id, document_dir
                )
        return data


def load_stand_data_document_from_json(path: Path) -> StandDataDocument:
    """Reads stand_data.json. Its allometry file paths, relative to the
    file's own folder on disk, come back absolute. The CSVs themselves need
    not exist: whether they do is SusiParams' concern (docs/adr/0005)."""
    return StandDataDocument.model_validate_json(
        path.read_text(),
        context={DOCUMENT_DIR_CONTEXT_KEY: path.resolve().parent},
    )


def dump_stand_data_document(output_path: Path, document: StandDataDocument) -> None:
    """Writes a StandDataDocument as JSON to output_path -- the project's
    stand_data.json, normally susi.io.project_layout.stand_data_path_for_project.
    Takes the already-resolved path rather than a project_dir: both
    xml_to_allometry.py's and metsakeskus_to_allometry.py's main() already
    need that same path for their own status printing, so they derive it
    once and pass it in here, instead of each deriving it a second time.

    Allometry file paths are written relative to output_path's folder;
    raises if any of them is outside it (see docs/adr/0005)."""
    output_path.write_text(
        document.model_dump_json(context={DOCUMENT_DIR_CONTEXT_KEY: output_path.parent})
    )


def build_stand_params(
    stand_data_document: StandDataDocument,
    stand_id: StandID,
    n: int,
) -> StandParams:
    """
    Builds a simulation-ready StandParams for one stand out of a whole project's StandDataDocument.
    `n`, the number of soil columns, must be supplied by the caller:
    It is a property of the run being built, never of the stand data itself.
    """
    stand_data = stand_data_document.stands[stand_id]

    canopy_layer_allometry = CanopyLayerAllometry.with_single_allometry_per_layer(
        layers=stand_data.allometry_file_per_layer, n=n
    )

    return StandParams(
        site_fertility_class=stand_data.site_fertility_class,
        canopy_layer_allometry=canopy_layer_allometry,
    )
