import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    SkipValidation,
    computed_field,
    field_validator,
)

from susi.io.extra_pydantic_types import (
    PositiveFloat,
    NonNegativeFloat,
    NonPositiveFloat,
    PositiveInt,
)


class StrictFrozenModel(BaseModel):
    """
    Defines a stricter Pydantic class
    """

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on assignment
        frozen=True,  # Force immutability
        extra="forbid",  # Forbid extra fields
        validate_default=True,  # Validate default values
        json_encoders={np.ndarray: lambda v: v.tolist()},
    )


class SimulationConfig(StrictFrozenModel):
    # Time
    start_date: datetime.datetime = Field(description="Simulation start date.")
    end_date: datetime.datetime = Field(description="Simulation end date.")


class WeatherParams(StrictFrozenModel):
    """
    Weather parameters
    """

    # FMI weather
    FMI_weather_filepath: FilePath = Field(description="Path to weather files.")

    # Rest of weather. Unused?
    # infolder: DirectoryPath = Field(description="Directory where weather files are")
    # infile_d: FilePath = Path("Tammela_weather_1.csv")
    # start_yr: int = 1980
    # end_yr: int = 1984
    # description: str = "Undefined, Finland"
    # lat: float = 65.00
    # lon: float = 25.00


class MottiFileParams(StrictFrozenModel):
    """
    Motti files to read
    """

    path: DirectoryPath = Field(description="Motti files input file folder")
    dominant: dict[int, str] = Field(
        description="int: 0 if not in use. str: Motti file for the dominant layer."
    )
    subdominant: dict[int, str] = Field(
        description="int: 0 if not in use. str: Motti file for the subdominant layer."
    )
    under: dict[int, str] = Field(
        description="int: 0 if not in use. str: Motti file for the understorey layer."
    )


class CanopyStateParams(StrictFrozenModel):
    """
    Canopy state parameters
    """

    lai_conif: float = Field(default=3.0, description="conifer 1-sided LAI (m2 m-2)")
    lai_decid_max: float = Field(
        default=0.01, description="maximum annual deciduous 1-sided LAI (m2 m-2):"
    )
    hc: float = Field(default=16.0, description="canopy height (m)")
    cf: float = Field(default=0.7, description="canopy closure fraction (-)")

    w: float = Field(default=0.0, description="Initial state of canopy storage (mm)")
    swe: float = Field(
        default=0.0, description="Initial state of snow water equivalent (mm)"
    )


class CanopyStateParamsArray:
    """
    Canopy and moss for each soil column (0, and n-1 are ditches)
    Same as CanopyStateParameters, but with all array elements.
    """

    lai_conif: np.ndarray
    lai_decid_max: np.ndarray
    hc: np.ndarray
    cf: np.ndarray
    # initial state of canopy storage [mm] and snow water equivalent [mm]
    w: np.ndarray
    swe: np.ndarray

    def __init__(self, canopy_state_parameters: CanopyStateParams, array_length: int):
        for name, value in canopy_state_parameters.model_dump().items():
            setattr(self, name, value * np.ones(array_length))


class CanopyParams(BaseModel):
    """
    Canopy parameters
    """

    dt: PositiveFloat = Field(default=86400.0, description="Canopy model timestep (s).")

    class Flow(StrictFrozenModel):
        """
        Flow field parameters
        """

        # Flow field
        zmeas: float = 2.0
        zground: float = Field(
            default=0.5, description="Reference height above ground (m)."
        )
        zo_ground: float = Field(
            default=0.01, description="ground roughness length (m)."
        )

    class Interception(StrictFrozenModel):
        """
        Interception parameters
        """

        # interception
        wmax: float = 0.5
        wmaxsnow: float = 4.0

    class Snow(StrictFrozenModel):
        """
        Snow parameters
        """

        # degree-day snow model
        kmelt: float = Field(
            default=2.8934e-05, description="melt coefficient in open (mm/s)"
        )
        kfreeze: float = Field(
            default=5.79e-6, description="freezing coefficient (mm/s)"
        )
        r: float = Field(
            default=0.05, description="maximum fraction of liquid water in snow (-)"
        )

    class Physpara(StrictFrozenModel):
        """
        Physpara parameters: canopy conductance and soil evaporation.
        """

        # canopy conductance
        amax_init: float = Field(
            frozen=False,
            default=10.0,
            description="Initial maximum photosynthetic rate (umolm-2(leaf)s-1)",
        )
        g1_conif: float = Field(default=2.1, description="stomatal parameter, conifers")
        g1_decid: float = Field(
            default=3.5, description="stomatal parameter, deciduous"
        )
        q50: float = Field(default=50.0, description="light response parameter (Wm-2)")
        kp: float = Field(default=0.6, description="light attenuation parameter (-)")
        rw: float = Field(default=0.20, description="critical value for REW (-),")
        rwmin: float = Field(
            default=0.02, description="minimum relative conductance (-)"
        )
        # soil evaporation
        gsoil: float = Field(
            default=1e-2,
            description="Soil surface conductance if soil is fully wet (m/s)",
        )

    class Phenology(StrictFrozenModel):
        """
        Phenology parameters. Seasonal cycle of physiology.
        """

        # seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
        smax: float = Field(default=18.5, description="degC")
        tau: float = Field(default=13.0, description="days")
        xo: float = Field(default=-4.0, description="degC")
        fmin: float = Field(
            default=0.05, description="minimum photosynthetic capacity in winter (-)"
        )

    flow: Flow = Flow()
    interception: Interception = Interception()
    snow: Snow = Snow()
    physpara: Physpara = Physpara()
    phenology: Phenology = Phenology()
    state: CanopyStateParams = CanopyStateParams()


class OrganicLayerParams(StrictFrozenModel):
    """
    Parameters for the organic layer
    """

    org_depth: PositiveFloat = Field(
        default=0.04, description="depth of organic top layer (m)"
    )
    org_poros: PositiveFloat = Field(default=0.9, description="porosity (-)")
    org_fc: PositiveFloat = Field(default=0.3, description="field capacity (-)")
    org_rw: PositiveFloat = Field(
        default=0.24,
        description="critical vol. moisture content (-) for decreasing phase in Ef.",
    )
    pond_storage_max: PositiveFloat = Field(
        default=0.01, description="max ponding allowed (m)"
    )

    # initial values
    org_sat: PositiveFloat = Field(
        default=1.0, description="organic top layer saturation ratio (-)"
    )
    pond_storage: NonNegativeFloat = Field(default=0.0, description="pond storage")


class OrganicLayerParamsArray:
    """
    Same class as OrganicLayerParams, but with all fields a numpy array for each soil column.
    """

    org_depth: np.ndarray
    org_poros: np.ndarray
    org_fc: np.ndarray
    org_rw: np.ndarray
    pond_storage_max: np.ndarray
    org_sat: np.ndarray
    pond_storage: np.ndarray

    def __init__(self, organic_layer_parameters: OrganicLayerParams, array_length: int):
        for name, value in organic_layer_parameters.model_dump().items():
            setattr(self, name, value * np.ones(array_length))


class OutputParams(StrictFrozenModel):
    """
    Output file IO parameters
    """

    outfolder: DirectoryPath
    netcdf: Path
    startday: int = 1
    startmonth: int = 7  # Päivä josta keskiarvojen laskenta alkaa
    endday: int = 31
    endmonth: int = 8  # Päivä johon keskiarvojen laskenta loppuu


class PhotoParameters(StrictFrozenModel):
    """
    Photosynthesis parameters for assimilation model (Mäkelä et al. 2008)
    """

    beta: float
    gamma: float
    kappa: float
    tau: float
    X0: float
    Smax: float
    alfa: float
    nu: float


class LocationsForPhotoParams(str, Enum):
    """
    Gives all options for the location of the photosynthesis parameters
    """

    all_data = "All_data"
    sodankyla = "Sodankyla"
    hyytiala = "Hyytiala"
    norunda = "Norunda"
    tharandt = "Tharandt"
    bray = "Bray"


PRESET_PHOTO_PARAMETERS: dict[str, PhotoParameters] = {
    "All_data": PhotoParameters(
        beta=0.513,
        gamma=0.0196,
        kappa=-0.389,
        tau=7.2,
        X0=-4.0,
        Smax=17.3,
        alfa=1.0,
        nu=5.0,
    ),
    "Sodankyla": PhotoParameters(
        beta=0.831,
        gamma=0.065,
        kappa=-0.150,
        tau=10.2,
        X0=-0.9,
        Smax=16.4,
        alfa=1.0,
        nu=5.0,
    ),
    "Hyytiala": PhotoParameters(
        beta=0.504,
        gamma=0.0303,
        kappa=-0.235,
        tau=11.1,
        X0=-3.1,
        Smax=17.3,
        alfa=1.0,
        nu=5.0,
    ),
    "Norunda": PhotoParameters(
        beta=0.500,
        gamma=0.0220,
        kappa=-0.391,
        tau=5.7,
        X0=-4.0,
        Smax=17.6,
        alfa=1.062,
        nu=11.27,
    ),
    "Tharandt": PhotoParameters(
        beta=0.742,
        gamma=0.0267,
        kappa=-0.512,
        tau=1.8,
        X0=-5.2,
        Smax=18.5,
        alfa=1.002,
        nu=442.0,
    ),
    "Bray": PhotoParameters(
        beta=0.459,
        gamma=-0.000669,
        kappa=-0.560,
        tau=2.6,
        X0=-17.6,
        Smax=45.0,
        alfa=0.843,
        nu=2.756,
    ),
}


def get_photo_parameters_by_location(
    location: LocationsForPhotoParams,
) -> PhotoParameters:
    return PRESET_PHOTO_PARAMETERS[location.value]


class TreeSpecies(str, Enum):
    """
    Possible tree species options
    """

    pine = "Pine"
    spruce = "Spruce"
    birch = "Birch"


class PeatTypes(str, Enum):
    """
    Possible choices for peat types
    """

    generic = "A"
    sphagnum = "S"


class NutrientFertilizationParameters(StrictFrozenModel):
    """
    Nutrient fertilization parameters
    """

    dose: NonNegativeFloat = Field(
        description="Dose of compound in fertilizer, kg ha-1"
    )
    decay_k: NonNegativeFloat = Field(description="Decay rate, yr-1")
    eff: NonNegativeFloat = Field(description="Nutrient use efficiency")


class FertilizationParameters(StrictFrozenModel):
    """
    Canopy parameters
    """

    application_year: int = 2201
    N: NutrientFertilizationParameters
    P: NutrientFertilizationParameters
    K: NutrientFertilizationParameters
    pH_increment: NonNegativeFloat = 1.0


class SiteParams(StrictFrozenModel):
    """
    Soil and stand parameters
    """

    # Forest
    # Age of different forest layers at the beginning of the simulation
    initial_dominant_stand_age_years: NonNegativeFloat = Field(
        description="Age of the dominant stand at the beginning of the simulation. This is set to all nodes in the strip."
    )
    initial_subdominant_stand_age_years: NonNegativeFloat = Field(
        description="Age of the subdominant layer at the beginning of the simulation. This is set to all nodes in the strip."
    )
    initial_understorey_age_years: NonNegativeFloat = Field(
        description="Age of the understorey at the beginning of the simulation. This is set to all nodes in the strip."
    )

    L: float = Field(description="Strip width, i.e., distance between ditches, m")

    site_fertility_class: PositiveInt = Field(
        description="Site fertility class. This is set to all nodes in the strip."
    )

    sitename: str
    species: TreeSpecies
    sfc_specification: float
    hdom: float | None
    vol: float | None
    smc: str
    nLyrs: int
    dzLyr: float
    L: float = Field(description="Strip width, i.e., distance between ditches, m")
    ditch_depth_west: list[NonPositiveFloat] = Field(
        description="ditch depth at the beginning of simulation (m). If given several values SUSI calculates scenarios for each ditch depth."
    )
    ditch_depth_east: list[NonPositiveFloat] = Field(
        description="ditch depth at the beginning of simulation (m). If given several values SUSI calculates scenarios for each ditch depth."
    )
    ditch_depth_20y_west: list[NonPositiveFloat] = Field(
        description="Ditch depth after 20 yrs, m, negative down"
    )
    ditch_depth_20y_east: list[NonPositiveFloat] = Field(
        description="Ditch depth after 20 yrs, m, negative down"
    )
    scenario_name: list[str] = Field(
        description="Scenario names, equal nmber of names than ditch depth scenarios."
    )

    drain_age: PositiveFloat = Field(description="Time since drainage (yrs).")
    initial_h: float
    slope: float
    peat_type: list[PeatTypes] = Field(
        description="'S' if Sphagnum, 'A' if woody or carex peat"
    )
    peat_type_bottom: list[PeatTypes]
    anisotropy: float = Field(description="Anisotropy of peat hydraulic conductivity")
    vonP: bool = Field(description="degree of decomposition, vonPost scale, int")
    vonP_top: list[int]
    vonP_bottom: int
    bd_top: float | None = Field(description="Bulk density (g/cm3).")
    bd_bottom: float
    peatN: float | None
    peatP: float | None
    peatK: float | None
    enable_peattop: bool
    enable_peatmiddle: bool
    enable_peatbottom: bool
    rho_mor: float = Field(description="bulk density of mor layer, kg m-3")
    h_mor: NonNegativeFloat | Callable[..., float] = Field(
        description="depth of mor layer, m"
    )
    cutting_yr: int = Field(
        description="Year for cutting. Not used if year is outside the simulation period."
    )
    cutting_to_ba: float = Field(description="basal area after cutting, m2/ha")
    depoN: float
    depoP: float
    depoK: float
    fertilization: FertilizationParameters

    @computed_field
    @property
    def n(self) -> int:
        """Number of computation nodes in the strip, 2-m width of node"""
        return int(self.L / 2)

    @property
    def age(self) -> SkipValidation[dict[str, np.ndarray]]:
        """Age of stand for all nodes along the strip"""
        return {
            "dominant": self.initial_dominant_stand_age_years * np.ones(self.n),
            "subdominant": self.initial_subdominant_stand_age_years * np.ones(self.n),
            "under": self.initial_understorey_age_years * np.ones(self.n),
        }

    @property
    def sfc(self) -> SkipValidation[np.ndarray]:
        """site fertility class for all nodes in the strip"""
        return np.ones(self.n, dtype=int) * self.site_fertility_class

    @property
    def canopylayers(self) -> dict[str, np.ndarray]:
        return {
            "dominant": np.ones(self.n, dtype=int),
            "subdominant": np.zeros(self.n, dtype=int),
            "under": np.zeros(self.n, dtype=int),
        }

    @field_validator("h_mor", mode="before")
    @classmethod
    def compute_if_callable(cls, hmor, info):
        if callable(hmor):
            drain_age = info.data.get("drain_age")
            rho_mor = info.data.get("rho_mor")
            if drain_age is None:
                raise ValueError("`drain_age` must be provided to compute hmor")
            if rho_mor is None:
                raise ValueError("`rho_mor` must be provided to compute hmor")
            try:
                return hmor(drain_age, rho_mor)
            except Exception as e:
                raise ValueError(f"Failed to compute h_mor: {e}")
        return hmor


class SusiParams(StrictFrozenModel):
    """
    Parameter class to be stantiated.
    """

    params_schema_version: int = 1
    weather_parameters: WeatherParams
    motti_file_parameters: MottiFileParams
    simulation_config: SimulationConfig
    canopy_parameters: CanopyParams
    organic_layer_parameters: OrganicLayerParams
    output_parameters: OutputParams
    photo_parameters: PhotoParameters
    site_parameters: SiteParams
