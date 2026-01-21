from typing import Annotated
from pydantic import Field

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
NonPositiveFloat = Annotated[float, Field(le=0)]
PositiveInt = Annotated[int, Field(gt=0)]
