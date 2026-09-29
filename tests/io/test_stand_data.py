import ast
import inspect
import json
import shutil
import string
from pathlib import Path

import pydantic
import pytest
import shapely
import shapely.ops
from hypothesis import assume, given
from hypothesis import strategies as st
from shapely.geometry import MultiPolygon, Point, Polygon

from susi.io import stand_data, susi_parameter_model
from susi.io.load_output_data import StandID
from susi.io.project_layout import STAND_DATA_FILENAME
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
)

# %% stand_data.point_to_ykj


def test_point_to_ykj_returns_plausible_helsinki_area_coordinates():
    # A point roughly at Helsinki, in ETRS-TM35FIN (EPSG:3067).
    x, y = stand_data.point_to_ykj(385000, 6685000)
    # Real YKJ eastings in southern Finland are ~3.3-3.4 million metres --
    # scaled by /10000 that's low-to-mid 300s, matching Helsinki's well-known
    # YKJ coordinates (~3387000, 6673000).
    assert 330 < x < 345
    assert 6650 < y < 6700


def test_point_to_ykj_reuses_one_transformer_across_calls():
    # Building a pyproj.Transformer is comparatively expensive; point_to_ykj
    # runs once per stand, so it must not rebuild one every call.
    stand_data._ykj_transformer.cache_clear()
    stand_data.point_to_ykj(385000, 6685000)
    stand_data.point_to_ykj(385000, 6685000)
    assert stand_data._ykj_transformer.cache_info().hits >= 1


# %% stand_data.centroid_to_ykj
#
# Moved here from metsakeskus_to_allometry.py (ticket 22), now that
# xml_to_allometry.py takes its YKJ grid location from the centroid too.


def test_centroid_to_ykj_returns_plausible_helsinki_area_coordinates():
    # A small disc roughly at Helsinki, in ETRS-TM35FIN (EPSG:3067).
    polygon = Point(385000, 6685000).buffer(50)
    x, y = stand_data.centroid_to_ykj(polygon)
    assert 330 < x < 345
    assert 6650 < y < 6700


def test_centroid_to_ykj_uses_the_centroid_not_the_first_vertex():
    # 20 km wide: the first vertex and the centroid sit 10 km apart in
    # easting, one whole YKJ easting unit, so the two choices can't agree.
    polygon = Polygon(
        [
            (380_000, 6_685_000),
            (400_000, 6_685_000),
            (400_000, 6_686_000),
            (380_000, 6_686_000),
        ]
    )
    first_vertex_x, _ = stand_data.point_to_ykj(*polygon.exterior.coords[0])
    centroid_x, _ = stand_data.point_to_ykj(polygon.centroid.x, polygon.centroid.y)
    assert first_vertex_x != centroid_x

    assert stand_data.centroid_to_ykj(polygon) == stand_data.point_to_ykj(
        polygon.centroid.x, polygon.centroid.y
    )


def test_centroid_to_ykj_reuses_one_transformer_across_calls():
    # It runs once per stand, so it must not rebuild a Transformer every call.
    stand_data._ykj_transformer.cache_clear()
    polygon = Point(385000, 6685000).buffer(50)
    stand_data.centroid_to_ykj(polygon)
    stand_data.centroid_to_ykj(polygon)
    assert stand_data._ykj_transformer.cache_info().hits >= 1


# %% stand_data.require_source_crs


def test_require_source_crs_accepts_the_source_crs():
    stand_data.require_source_crs(stand_data.SOURCE_CRS, where="test input")


@pytest.mark.parametrize("declared", ["EPSG:4326", "EPSG:2393", None])
def test_require_source_crs_rejects_anything_else_naming_the_source(declared):
    with pytest.raises(ValueError, match="test input.*EPSG:3067"):
        stand_data.require_source_crs(declared, where="test input")


# %% The single YKJ range, shared by the data model and the CLI check
#
# stand_data owns the bounds, next to the point_to_ykj call that produces
# coordinates in those units. Both enforcement points read them from there,
# so a coordinate is held to the same range however it reaches the tools.
# These tests fail if either consumer ever grows its own numbers again.


@pytest.mark.parametrize(
    "field_name, min_value, max_value",
    [
        ("x_ykj", stand_data.X_YKJ_MIN, stand_data.X_YKJ_MAX),
        ("y_ykj", stand_data.Y_YKJ_MIN, stand_data.Y_YKJ_MAX),
    ],
)
def test_stand_data_bounds_come_from_the_shared_ykj_range(
    field_name, min_value, max_value
):
    constraints = stand_data.StandData.model_fields[field_name].metadata
    assert {type(constraint).__name__: constraint for constraint in constraints}[
        "Ge"
    ].ge == min_value
    assert {type(constraint).__name__: constraint for constraint in constraints}[
        "Le"
    ].le == max_value


@pytest.mark.parametrize(
    "field_name, x_ykj, y_ykj",
    [
        ("x_ykj", stand_data.X_YKJ_MAX + 1, 6675),
        ("x_ykj", stand_data.X_YKJ_MIN - 1, 6675),
        ("y_ykj", 339, stand_data.Y_YKJ_MAX + 1),
        ("y_ykj", 339, stand_data.Y_YKJ_MIN - 1),
    ],
)
def test_stand_data_rejects_a_coordinate_the_cli_check_would_also_reject(
    field_name, x_ykj, y_ykj
):
    # The same value validate_x_y_ykj blocks in standalone mode is
    # unrepresentable in a StandData -- which is what "one source of truth"
    # has to mean in practice.
    with pytest.raises(pydantic.ValidationError, match=field_name):
        stand_data.StandData(
            site_fertility_class=3,
            allometry_file_per_layer={},
            initial_age_per_layer={},
            x_ykj=x_ykj,
            y_ykj=y_ykj,
        )


# %% StandData.initial_age_per_layer must cover exactly the layers with a file
#
# An age for a layer with no allometry file, or a file with no age, is a bug
# in whichever tool wrote the document, so StandData refuses both.


def _stand_with_files_and_ages(allometry_layers, age_layers) -> stand_data.StandData:
    return stand_data.StandData(
        site_fertility_class=3,
        allometry_file_per_layer={
            layer: AllometryFileAndSpecies(
                file_path=Path(f"/project/inputs/allometry/{layer.value}.csv"),
                species_id=1,
            )
            for layer in allometry_layers
        },
        initial_age_per_layer={layer: 30.0 for layer in age_layers},
        x_ykj=339,
        y_ykj=6675,
    )


def test_stand_data_rejects_an_age_for_a_layer_with_no_allometry_file():
    with pytest.raises(pydantic.ValidationError, match="subdominant"):
        _stand_with_files_and_ages(
            allometry_layers=[CanopyLayerName.dominant],
            age_layers=[CanopyLayerName.dominant, CanopyLayerName.subdominant],
        )


def test_stand_data_rejects_an_allometry_file_with_no_age():
    with pytest.raises(pydantic.ValidationError, match="subdominant"):
        _stand_with_files_and_ages(
            allometry_layers=[CanopyLayerName.dominant, CanopyLayerName.subdominant],
            age_layers=[CanopyLayerName.dominant],
        )


def test_stand_data_accepts_ages_for_exactly_the_allometry_layers():
    stand = _stand_with_files_and_ages(
        allometry_layers=[CanopyLayerName.dominant, CanopyLayerName.subdominant],
        age_layers=[CanopyLayerName.subdominant, CanopyLayerName.dominant],
    )

    assert stand.initial_age_per_layer == {
        CanopyLayerName.dominant: 30.0,
        CanopyLayerName.subdominant: 30.0,
    }


# %% stand_data.StandData / StandDataDocument round-trip

# Fabricated AllometryFileAndSpecies values, mirroring
# tests/test_canopy_layer_allometry.py's strategy -- no real CSV is ever read here.
# Absolute, because that's what a StandDataDocument holds in memory (a
# relative one is rejected without a document folder -- docs/adr/0005).
_FAKE_ALLOMETRY_DIR = Path("/project/inputs/allometry")
allometry_file_and_species_strategy = st.builds(
    AllometryFileAndSpecies,
    file_path=st.text(
        alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=20
    ).map(lambda name: _FAKE_ALLOMETRY_DIR / f"{name}.csv"),
    species_id=st.integers(min_value=1, max_value=100),
)

allometry_file_per_layer_strategy = st.dictionaries(
    keys=st.sampled_from(list(CanopyLayerName)),
    values=allometry_file_and_species_strategy,
    max_size=len(CanopyLayerName),
)


@st.composite
def stand_polygon_strategy(draw):
    """A real stand boundary in EPSG:3067 metres: an axis-aligned rectangle
    somewhere in Finland's ETRS-TM35FIN extent, optionally with one
    rectangular hole strictly inside it -- enough to exercise both the
    exterior ring and an interior ring through the WKT round-trip. Arbitrary
    (non-round) float coordinates on purpose: WKT must not lose precision."""
    coordinate = st.floats(allow_nan=False, allow_infinity=False)
    x0 = draw(coordinate.filter(lambda v: 50_000 <= v <= 750_000))
    y0 = draw(coordinate.filter(lambda v: 6_600_000 <= v <= 7_800_000))
    width = draw(st.floats(min_value=10, max_value=2_000))
    height = draw(st.floats(min_value=10, max_value=2_000))
    exterior = [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
    ]
    holes = []
    if draw(st.booleans()):
        # Middle third of the rectangle: always strictly inside it.
        hx0, hy0 = x0 + width / 3, y0 + height / 3
        hx1, hy1 = x0 + 2 * width / 3, y0 + 2 * height / 3
        holes.append([(hx0, hy0), (hx1, hy0), (hx1, hy1), (hx0, hy1)])
    return Polygon(exterior, holes)


_optional_positive_int = st.one_of(st.none(), st.integers(min_value=1, max_value=20))
_optional_nonneg_float = st.one_of(
    st.none(),
    st.floats(min_value=0, max_value=2000, allow_nan=False, allow_infinity=False),
)

_initial_age = st.floats(
    min_value=0, max_value=300, allow_nan=False, allow_infinity=False
)


@st.composite
def stand_data_strategy(draw):
    # initial_age_per_layer must have the same keys as allometry_file_per_layer,
    # so the ages are drawn per file layer rather than as an independent dict.
    allometry_file_per_layer = draw(allometry_file_per_layer_strategy)
    initial_age_per_layer = {
        layer: draw(_initial_age) for layer in allometry_file_per_layer
    }
    return draw(
        st.builds(
            stand_data.StandData,
            site_fertility_class=st.integers(min_value=1, max_value=10),
            allometry_file_per_layer=st.just(allometry_file_per_layer),
            initial_age_per_layer=st.just(initial_age_per_layer),
            x_ykj=st.integers(
                min_value=stand_data.X_YKJ_MIN, max_value=stand_data.X_YKJ_MAX
            ),
            y_ykj=st.integers(
                min_value=stand_data.Y_YKJ_MIN, max_value=stand_data.Y_YKJ_MAX
            ),
            polygon=st.one_of(st.none(), stand_polygon_strategy()),
            main_group=_optional_positive_int,
            sub_group=_optional_positive_int,
            stand_area=st.one_of(
                st.none(),
                st.floats(
                    min_value=0.01,
                    max_value=10_000,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            basal_area=_optional_nonneg_float,
            mean_height=_optional_nonneg_float,
            mean_diameter=_optional_nonneg_float,
            total_volume=_optional_nonneg_float,
            stem_count=_optional_nonneg_float,
            developmentclass=_optional_positive_int,
            drainagestate=_optional_positive_int,
        )
    )


stand_data_document_strategy = st.builds(
    stand_data.StandDataDocument,
    crs=st.just(stand_data.SOURCE_CRS),
    altitude=st.floats(
        min_value=-500, max_value=3000, allow_nan=False, allow_infinity=False
    ),
    ddy=st.floats(min_value=0, max_value=3000, allow_nan=False, allow_infinity=False),
    stands=st.dictionaries(
        keys=st.text(min_size=1, max_size=10).map(StandID),
        values=stand_data_strategy(),
        max_size=5,
    ),
)


@given(document=stand_data_document_strategy)
def test_stand_data_document_roundtrips_through_json_in_memory(document):
    # Pure property, no file involved, so no document folder either: the
    # (absolute) allometry paths pass through unchanged. The on-disk,
    # relative-to-the-document form is covered through dump/load below.
    dumped = document.model_dump_json()
    assert stand_data.StandDataDocument.model_validate_json(dumped) == document


# %% stand_data.load_stand_data_document_from_json / dump_stand_data_document
#
# dump_stand_data_document moved here from being duplicated, byte-identically,
# in both metsakeskus_to_allometry.py and xml_to_allometry.py (ticket 12) --
# each tool's own test file now only checks that it imports this function
# rather than defining a local copy.
#
# The pair is the seam for allometry paths (ticket 23, docs/adr/0005):
# relative to the document's folder on disk, absolute in memory.


def _document_with_allometry(files_per_stand: dict[str, Path]):
    """A StandDataDocument with one dominant-layer allometry file per stand."""
    return stand_data.StandDataDocument(
        crs=stand_data.SOURCE_CRS,
        altitude=150.0,
        ddy=1200.0,
        stands={
            StandID(stand_id): stand_data.StandData(
                site_fertility_class=3,
                allometry_file_per_layer={
                    CanopyLayerName.dominant: AllometryFileAndSpecies(
                        file_path=file_path, species_id=1
                    )
                },
                initial_age_per_layer={CanopyLayerName.dominant: 40.0},
                x_ykj=339,
                y_ykj=6675,
            )
            for stand_id, file_path in files_per_stand.items()
        },
    )


def _project_inputs_with_allometry(root: Path, stand_ids: list[str]):
    """The real layout: <root>/inputs/stand_data.json beside (not inside)
    <root>/inputs/allometry/<id>.csv. Returns (document, stand_data.json path);
    the CSVs are written, the document isn't."""
    inputs_dir = root / "inputs"
    allometry_dir = inputs_dir / "allometry"
    allometry_dir.mkdir(parents=True)
    files = {stand_id: allometry_dir / f"{stand_id}.csv" for stand_id in stand_ids}
    for file_path in files.values():
        file_path.write_text("Age\n1\n")
    return _document_with_allometry(files), inputs_dir / STAND_DATA_FILENAME


def _raw_file_paths(json_path: Path) -> dict[str, str]:
    """stand ID -> the dominant layer's file_path, exactly as written on disk."""
    raw = json.loads(json_path.read_text())
    return {
        stand_id: stand["allometry_file_per_layer"]["dominant"]["file_path"]
        for stand_id, stand in raw["stands"].items()
    }


def _write_raw_document_with_file_path(json_path: Path, file_path: str) -> None:
    """A stand_data.json whose one stand holds `file_path` verbatim."""
    raw = json.loads(
        _document_with_allometry({"1": Path("/placeholder.csv")}).model_dump_json()
    )
    raw["stands"]["1"]["allometry_file_per_layer"]["dominant"]["file_path"] = file_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(raw))


def test_dump_writes_allometry_paths_relative_to_the_document(tmp_path):
    document, json_path = _project_inputs_with_allometry(tmp_path, ["1", "2"])

    stand_data.dump_stand_data_document(output_path=json_path, document=document)

    assert _raw_file_paths(json_path) == {
        "1": "allometry/1.csv",
        "2": "allometry/2.csv",
    }
    # And nothing in the file names the folder it happens to be in.
    assert str(tmp_path) not in json_path.read_text()


def test_load_gives_back_the_original_absolute_paths(tmp_path):
    document, json_path = _project_inputs_with_allometry(tmp_path, ["1", "2"])
    stand_data.dump_stand_data_document(output_path=json_path, document=document)

    loaded = stand_data.load_stand_data_document_from_json(json_path)

    assert loaded == document
    for stand in loaded.stands.values():
        for entry in stand.allometry_file_per_layer.values():
            assert entry.file_path.is_absolute()


def test_a_moved_document_points_into_its_new_location(tmp_path):
    document, json_path = _project_inputs_with_allometry(tmp_path / "old", ["1"])
    stand_data.dump_stand_data_document(output_path=json_path, document=document)

    new_inputs_dir = tmp_path / "somewhere" / "else" / "inputs"
    new_inputs_dir.parent.mkdir(parents=True)
    shutil.move(json_path.parent, new_inputs_dir)
    loaded = stand_data.load_stand_data_document_from_json(
        new_inputs_dir / STAND_DATA_FILENAME
    )

    dominant = loaded.stands[StandID("1")].allometry_file_per_layer[
        CanopyLayerName.dominant
    ]
    assert dominant.file_path == new_inputs_dir / "allometry" / "1.csv"
    assert dominant.file_path.exists()


def test_load_does_not_need_the_allometry_files_to_exist(tmp_path):
    json_path = tmp_path / "inputs" / STAND_DATA_FILENAME
    _write_raw_document_with_file_path(json_path, "allometry/never-written.csv")

    loaded = stand_data.load_stand_data_document_from_json(json_path)

    dominant = loaded.stands[StandID("1")].allometry_file_per_layer[
        CanopyLayerName.dominant
    ]
    assert dominant.file_path == json_path.parent / "allometry" / "never-written.csv"
    assert not dominant.file_path.exists()


@pytest.mark.parametrize(
    "outside",
    [
        pytest.param(lambda root: root / "elsewhere" / "1.csv", id="sibling-folder"),
        pytest.param(
            lambda root: root / "inputs" / ".." / "elsewhere" / "1.csv",
            id="dotdot-escape",
        ),
    ],
)
def test_dump_rejects_an_allometry_file_outside_the_document_folder(tmp_path, outside):
    json_path = tmp_path / "inputs" / STAND_DATA_FILENAME
    json_path.parent.mkdir()
    file_path = outside(tmp_path)
    document = _document_with_allometry({"stand-7": file_path})

    with pytest.raises(ValueError) as error:
        stand_data.dump_stand_data_document(output_path=json_path, document=document)

    assert "stand-7" in str(error.value)
    assert str(file_path) in str(error.value)
    assert not json_path.exists()


@pytest.mark.parametrize(
    "bad_file_path",
    [
        pytest.param("/abs/allometry/1.csv", id="absolute"),
        pytest.param("../allometry/1.csv", id="leading-dotdot"),
        pytest.param("allometry/../../1.csv", id="inner-dotdot"),
    ],
)
def test_load_rejects_absolute_and_dotdot_paths(tmp_path, bad_file_path):
    json_path = tmp_path / "inputs" / STAND_DATA_FILENAME
    _write_raw_document_with_file_path(json_path, bad_file_path)

    with pytest.raises(pydantic.ValidationError) as error:
        stand_data.load_stand_data_document_from_json(json_path)

    assert "'1'" in str(error.value)
    assert bad_file_path in str(error.value)


def test_a_relative_path_without_a_document_folder_is_rejected():
    # No silent fallback to the current folder: outside
    # load_stand_data_document_from_json there's nothing to resolve against.
    raw = json.loads(
        _document_with_allometry(
            {"stand-3": Path("/placeholder.csv")}
        ).model_dump_json()
    )
    raw["stands"]["stand-3"]["allometry_file_per_layer"]["dominant"]["file_path"] = (
        "allometry/3.csv"
    )

    with pytest.raises(pydantic.ValidationError) as error:
        stand_data.StandDataDocument.model_validate_json(json.dumps(raw))

    assert "stand-3" in str(error.value)
    assert "allometry/3.csv" in str(error.value)


def test_building_a_document_in_memory_with_a_relative_path_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="pines.csv"):
        _document_with_allometry({"1": Path("pines.csv")})


# %% stand_data.StandPolygon (StandData.polygon) and StandDataDocument.crs

# A 10x10 m square with a 2x2 m hole, in EPSG:3067 metres near Paroninkorpi.
# shapely's area excludes the hole: 100 - 4 = 96.
_SQUARE_WITH_HOLE = Polygon(
    [
        (377_000.0, 6_767_000.0),
        (377_010.0, 6_767_000.0),
        (377_010.0, 6_767_010.0),
        (377_000.0, 6_767_010.0),
    ],
    [
        [
            (377_004.0, 6_767_004.0),
            (377_006.0, 6_767_004.0),
            (377_006.0, 6_767_006.0),
            (377_004.0, 6_767_006.0),
        ]
    ],
)


def _stand_with_polygon(polygon) -> stand_data.StandData:
    return stand_data.StandData(
        site_fertility_class=3,
        allometry_file_per_layer={},
        initial_age_per_layer={},
        x_ykj=338,
        y_ykj=6770,
        polygon=polygon,
    )


def test_stand_polygon_roundtrips_through_dump_and_load_with_its_holes(tmp_path):
    document = stand_data.StandDataDocument(
        crs=stand_data.SOURCE_CRS,
        altitude=120.0,
        ddy=1250.0,
        stands={StandID("1"): _stand_with_polygon(_SQUARE_WITH_HOLE)},
    )
    output_path = tmp_path / STAND_DATA_FILENAME

    stand_data.dump_stand_data_document(output_path=output_path, document=document)
    loaded = stand_data.load_stand_data_document_from_json(output_path)

    polygon = loaded.stands[StandID("1")].polygon
    assert isinstance(polygon, Polygon)
    assert polygon.equals(_SQUARE_WITH_HOLE)
    assert len(polygon.interiors) == 1
    assert polygon.area == 96.0


def test_stand_polygon_is_written_to_json_as_wkt():
    stand = _stand_with_polygon(_SQUARE_WITH_HOLE)

    dumped = stand.model_dump(mode="json")["polygon"]

    assert isinstance(dumped, str)
    assert shapely.from_wkt(dumped).equals(_SQUARE_WITH_HOLE)


def test_stand_polygon_accepts_a_wkt_string_in_python_mode_too():
    # The validator parses a str whichever way it arrives, not only from JSON.
    stand = _stand_with_polygon(_SQUARE_WITH_HOLE.wkt)

    assert isinstance(stand.polygon, Polygon)
    assert stand.polygon.equals(_SQUARE_WITH_HOLE)


@pytest.mark.parametrize(
    "bad_polygon",
    [
        pytest.param(MultiPolygon([_SQUARE_WITH_HOLE]), id="MultiPolygon-object"),
        pytest.param(MultiPolygon([_SQUARE_WITH_HOLE]).wkt, id="MultiPolygon-wkt"),
        pytest.param(Polygon(), id="empty-Polygon"),
        pytest.param("POLYGON EMPTY", id="empty-wkt"),
        pytest.param("POINT (377000 6767000)", id="Point-wkt"),
        # The pre-ticket-22 on-disk format: raw gml:coordinates pairs.
        pytest.param(
            "377000.0,6767000.0 377010.0,6767000.0 377010.0,6767010.0 377000.0,6767000.0",
            id="gml-coordinates-string",
        ),
        pytest.param(42, id="not-a-string"),
    ],
)
def test_stand_polygon_rejects_anything_but_a_non_empty_polygon(bad_polygon):
    with pytest.raises(pydantic.ValidationError):
        _stand_with_polygon(bad_polygon)


def test_stand_data_json_schema_still_builds():
    # The polygon type must not need arbitrary_types_allowed: the model has to
    # describe itself as JSON, with polygon as a plain string.
    schema = stand_data.StandDataDocument.model_json_schema()

    polygon_schema = schema["$defs"]["StandData"]["properties"]["polygon"]
    assert sorted(branch["type"] for branch in polygon_schema["anyOf"]) == [
        "null",
        "string",
    ]


def test_stand_data_document_requires_crs():
    with pytest.raises(pydantic.ValidationError, match="crs"):
        stand_data.StandDataDocument(altitude=100.0, ddy=1200.0, stands={})  # ty: ignore[missing-argument]


def test_stand_data_document_rejects_a_crs_other_than_the_source_crs():
    with pytest.raises(pydantic.ValidationError, match="EPSG:3067"):
        stand_data.StandDataDocument(
            crs="EPSG:4326", altitude=100.0, ddy=1200.0, stands={}
        )


def test_stand_data_document_accepts_the_source_crs():
    document = stand_data.StandDataDocument(
        crs=stand_data.SOURCE_CRS, altitude=100.0, ddy=1200.0, stands={}
    )

    assert document.crs == "EPSG:3067"


# %% stand_data.build_stand_params


def _single_stand_document(
    allometry_file_per_layer, site_fertility_class=3, initial_age_per_layer=None
):
    if initial_age_per_layer is None:
        initial_age_per_layer = {layer: 25.0 for layer in allometry_file_per_layer}
    return stand_data.StandDataDocument(
        crs=stand_data.SOURCE_CRS,
        altitude=100.0,
        ddy=1200.0,
        stands={
            StandID("known"): stand_data.StandData(
                site_fertility_class=site_fertility_class,
                allometry_file_per_layer=allometry_file_per_layer,
                initial_age_per_layer=initial_age_per_layer,
                x_ykj=339,
                y_ykj=6675,
            )
        },
    )


def test_build_stand_params_raises_a_clear_error_for_an_unknown_stand_id():
    document = _single_stand_document(allometry_file_per_layer={})

    with pytest.raises(KeyError, match="unknown"):
        stand_data.build_stand_params(document, StandID("unknown"), n=3)


def test_build_stand_params_matches_with_single_allometry_per_layer():
    allometry_file_per_layer = {
        CanopyLayerName.dominant: AllometryFileAndSpecies(
            file_path=_FAKE_ALLOMETRY_DIR / "pines.csv", species_id=1
        ),
        CanopyLayerName.under: AllometryFileAndSpecies(
            file_path=_FAKE_ALLOMETRY_DIR / "spruces.csv", species_id=2
        ),
    }
    document = _single_stand_document(
        allometry_file_per_layer=allometry_file_per_layer,
        site_fertility_class=4,
        initial_age_per_layer={
            CanopyLayerName.dominant: 45.0,
            CanopyLayerName.under: 12.0,
        },
    )

    params = stand_data.build_stand_params(document, StandID("known"), n=5)

    assert params.site_fertility_class == 4
    assert params.canopy_layer_allometry == (
        CanopyLayerAllometry.with_single_allometry_per_layer(
            layers=allometry_file_per_layer, n=5
        )
    )
    # Layers the stand has carry their recorded age; the one it doesn't
    # (subdominant) starts at 0.0, since the engine needs every layer.
    assert params.initial_canopylayer_age_years == {
        CanopyLayerName.dominant: 45.0,
        CanopyLayerName.subdominant: 0.0,
        CanopyLayerName.under: 12.0,
    }


@given(
    allometry_file_per_layer=allometry_file_per_layer_strategy.filter(
        lambda layers: len(layers) > 0
    ),
    n1=st.integers(min_value=0, max_value=30),
    n2=st.integers(min_value=0, max_value=30),
)
def test_build_stand_params_pointer_lengths_track_n(allometry_file_per_layer, n1, n2):
    # Regression test for the bug this design was fixing: a CanopyLayerAllometry
    # built for one n must not be reused/cached for another -- each call must
    # produce pointer lists whose length matches the n passed to that call.
    assume(n1 != n2)
    document = _single_stand_document(allometry_file_per_layer=allometry_file_per_layer)

    params1 = stand_data.build_stand_params(document, StandID("known"), n=n1)
    params2 = stand_data.build_stand_params(document, StandID("known"), n=n2)

    for layer_name in allometry_file_per_layer:
        pointers1 = params1.canopy_layer_allometry.pointers[layer_name]
        pointers2 = params2.canopy_layer_allometry.pointers[layer_name]
        assert pointers1 is not None
        assert pointers2 is not None
        assert len(pointers1) == n1
        assert len(pointers2) == n2


# %% Dependency direction: susi/ must not import tools/


@pytest.mark.parametrize("susi_module", [susi_parameter_model, stand_data])
def test_susi_modules_do_not_import_from_tools(susi_module):
    # tools/ depends on susi/, never the reverse. stand_data.py moved into
    # susi/io/ precisely so analysis/ can read a StandDataDocument without
    # importing tools/.
    source = inspect.getsource(susi_module)
    tree = ast.parse(source)

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(module.startswith("tools") for module in imported_modules)
