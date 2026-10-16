from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
NonPositiveFloat = Annotated[float, Field(le=0)]
PositiveInt = Annotated[int, Field(gt=0)]


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
