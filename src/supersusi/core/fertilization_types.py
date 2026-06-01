import numpy as np
from typing import Literal
from dataclasses import dataclass

Nutrient = Literal["N", "P", "K"]


@dataclass(frozen=True)
class State:
    pH_increment: float
    nutrient_release: dict[Nutrient, np.ndarray]


@dataclass(frozen=True)
class DynamicInputs:
    years_since_fertilization: int
