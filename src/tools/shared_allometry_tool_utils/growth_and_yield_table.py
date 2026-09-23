"""The Growth_and_Yield_Table-construction shape all three allometry-
generating tools share: three species' worth of age_N/G_N/N_N/Dg_N/Hg_N
keyword arguments, read off a PerSpecies[TreeStratum], plus the site/
projection scalars.

Species isolation -- zeroing every slot but one, to model a single canopy
layer growing alone -- is the caller's job (see
metsakeskus_to_allometry.py's isolate_species_layer): pass an
already-isolated PerSpecies in for that. This function itself never isolates
anything, so it works unmodified for xml_to_allometry.py's and
new_growth_allometry.py's always-combined three-species tables too.
"""

import pandas as pd

from susi.core.allometric_road_map import Growth_and_Yield_Table
from tools.shared_allometry_tool_utils.tree_stratum import PerSpecies, TreeStratum


def build_growth_and_yield_table(
    strata: PerSpecies[TreeStratum],
    fertility_class: int,
    x_ykj: int,
    y_ykj: int,
    altitude: float,
    ddy: float,
    n_trees: int,
    start_year: int,
    end_year: int,
    step_years: int,
    peat: int = 1,
) -> pd.DataFrame:
    growth_and_yield_table = Growth_and_Yield_Table(
        age_1=strata.pine.age,
        G_1=strata.pine.basal_area,
        N_1=strata.pine.stem_count,
        Dg_1=strata.pine.mean_diameter,
        Hg_1=strata.pine.mean_height,
        age_2=strata.spruce.age,
        G_2=strata.spruce.basal_area,
        N_2=strata.spruce.stem_count,
        Dg_2=strata.spruce.mean_diameter,
        Hg_2=strata.spruce.mean_height,
        age_3=strata.deciduous.age,
        G_3=strata.deciduous.basal_area,
        N_3=strata.deciduous.stem_count,
        Dg_3=strata.deciduous.mean_diameter,
        Hg_3=strata.deciduous.mean_height,
        DDY=ddy,
        fertility_class=fertility_class,
        peat=peat,
        y=y_ykj,
        x=x_ykj,
        altitude=altitude,
        n_trees=n_trees,
    )
    return growth_and_yield_table.get_table(
        start_year=start_year, end_year=end_year, step_years=step_years
    )
