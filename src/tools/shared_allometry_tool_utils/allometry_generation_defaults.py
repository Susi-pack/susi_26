"""The n_trees/start_year/end_year/step_years defaults shared by
ExtractionConfig (metsakeskus_to_allometry.py), XmlConfig
(xml_to_allometry.py), and NewGrowthSourcedConfig (new_growth_allometry.py):
previously defined identically in all three, each with a comment pointing at
one of the other two as the "same values" source.
"""

from susi.io.extra_pydantic_types import StrictFrozenModel


class AllometryGenerationDefaults(StrictFrozenModel):
    n_trees: int = 20
    start_year: int = 5
    end_year: int = 80
    step_years: int = 5
