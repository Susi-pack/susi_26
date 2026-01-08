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
    StrictBool,
    computed_field,
    field_validator,
)

from susi.io.extra_pydantic_types import (
    PositiveFloat,
    NonNegativeFloat,
    NonPositiveFloat,
    PositiveInt,
)
from susi.io.utils import get_project_root


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

    outfolder: DirectoryPath = get_project_root() / Path("outputs/")
    netcdf: Path = Path("susi.nc")
    startday: int = 1
    startmonth: int = 7  # Päivä josta keskiarvojen laskenta alkaa
    endday: int = 31
    endmonth: int = 8  # Päivä johon keskiarvojen laskenta loppuu


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
