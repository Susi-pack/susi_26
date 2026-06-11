"""
Source: https://github.com/lukefi/metsi/blob/main/lukefi/metsi/domain/data_collection/biomass_repola.py
"""

import math

# Sources
# Repola J. (2013). Modelling tree biomasses in Finland
# https://dissertationesforestales.fi/pdf/article1941.pdf
# Repola J. (2009) Silva Fennica 43(4) Biomass equations for Scots pine and Norway spruce in Finland
# https://www.silvafennica.fi/pdf/article184.pdf
# Repola J. (2008) Silva Fennica 42(4) Biomass equations for birch in Finland
# https://www.silvafennica.fi/pdf/article236.pdf

# Model input explanations:
# d = tree diameter at breast height, cm
# h = tree height, m


def stump_diameter(diameter_at_breast_height) -> float:
    """
    Laasasenaho 1975, stump diameter f(d):
    Needed in biomass calculations
    """
    return 2.0 + 1.25 * diameter_at_breast_height


def stem_wood_biomass_1(species, DBH, height) -> float:
    """
    Repola J. (2013). Modelling tree biomasses in Finland, p. 25
    """
    if species == 1:  # Scots Pine
        lnbm = (
            -3.721
            + 8.103 * (stump_diameter(DBH) / (stump_diameter(DBH) + 14))
            + 5.066 * (height / (height + 12))
            + (0.002 + 0.009) / 2
        )
        bm = math.exp(lnbm)
    elif species == 2:  # Norway Spruce
        lnbm = (
            -3.555
            + 8.042 * (stump_diameter(DBH) / (stump_diameter(DBH) + 14))
            + 0.869 * math.log(height)
            + 0.015 * height
            + (0.009 + 0.009) / 2
        )
        bm = math.exp(lnbm)
    else:  # Others, model for Birch
        lnbm = (
            -4.879
            + 9.651 * (stump_diameter(DBH) / (stump_diameter(DBH) + 12))
            + 1.012 * math.log(height)
            + (0.00263 + 0.00544) / 2
        )
        bm = math.exp(lnbm)
    # bm = Biomass of component, kg
    # to tons:
    bm = bm / 1000
    return bm


# Stem bark biomass 1 #f(d,h)
def stem_bark_biomass_1(species, DBH, height) -> float:
    """
    Repola J. (2009) Silva Fennica 43(4) Biomass equations for Scots pine and Norway spruce in Finland p. 631-633
    Repola J. (2008) Silva Fennica 42(4) Biomass equations for birch in Finland p. 611-613
    """
    if species == 1:  # Scots Pine
        lnbm = (
            -4.548
            + 7.997 * (stump_diameter(DBH) / (stump_diameter(DBH) + 12))
            + 0.357 * (math.log(height))
            + (0.015 + 0.061) / 2
        )
        bm = math.exp(lnbm)
    elif species == 2:  # Norway Spruce
        lnbm = (
            -4.548
            + 9.448 * (stump_diameter(DBH) / (stump_diameter(DBH) + 18))
            + 0.436 * (math.log(height))
            + (0.023 + 0.041) / 2
        )
        bm = math.exp(lnbm)
    else:  # Others, model for Birch
        lnbm = (
            -5.401
            + 10.061 * (stump_diameter(DBH) / (stump_diameter(DBH) + 12))
            + 2.657 * (height / (height + 20))
            + (0.01043 + 0.04443) / 2
        )
        bm = math.exp(lnbm)
    # bm = Biomass of component, kg
    # to tons:
    bm = bm / 1000
    return bm


# Living branches biomass 1 f(d,h)
def living_branches_biomass_1(species, DBH, height) -> float:
    """
    Repola J. (2009) Silva Fennica 43(4) Biomass equations for Scots pine and Norway spruce in Finland p. 631-633
    Repola J. (2008) Silva Fennica 42(4) Biomass equations for birch in Finland p. 611-613
    """
    if species == 1:  # Scots Pine
        lnbm = (
            -6.162
            + 15.075 * (stump_diameter(DBH) / (stump_diameter(DBH) + 12))
            - 2.618 * (height / (height + 12))
            + (0.041 + 0.089) / 2
        )
        bm = math.exp(lnbm)
    elif species == 2:  # Norway Spruce
        lnbm = (
            -4.214
            + 14.508 * (stump_diameter(DBH) / (stump_diameter(DBH) + 13))
            - 3.277 * (height / (height + 5))
            + (0.039 + 0.081) / 2
        )
        bm = math.exp(lnbm)
    else:  # Others, model for Birch
        lnbm = (
            -4.152
            + 15.874 * (stump_diameter(DBH) / (stump_diameter(DBH) + 16))
            - 4.407 * (height / (height + 10))
            + (0.02733 + 0.07662) / 2
        )
        bm = math.exp(lnbm)
    # bm = Biomass of component, kg
    # to tons:
    bm = bm / 1000
    return bm


# Dead branches biomass 1 f(d,h)
def dead_branches_biomass_1(species, DBH, height) -> float:
    """
    Repola J. (2009) Silva Fennica 43(4) Biomass equations for Scots pine and Norway spruce in Finland p. 631-633
    Repola J. (2008) Silva Fennica 42(4) Biomass equations for birch in Finland p. 611-613
    """
    if species == 1:  # Scots Pine
        lnbm = -5.201 + 10.574 * (stump_diameter(DBH) / (stump_diameter(DBH) + 16))
        bm = math.exp(lnbm)
        bm = bm * 0.911
    elif species == 2:  # Norway Spruce
        lnbm = (
            -4.850
            + 7.702 * (stump_diameter(DBH) / (stump_diameter(DBH) + 18))
            + 0.513 * (math.log(height))
        )
        bm = math.exp(lnbm)
        bm = bm * 1.343
    else:  # Others, model for Birch
        lnbm = -8.335 + 12.402 * (stump_diameter(DBH) / (stump_diameter(DBH) + 16))
        bm = math.exp(lnbm)
        bm = bm * 2.0737
    # bm = Biomass of component, kg
    # to tons:
    bm = bm / 1000
    return bm


# Foliage/needles biomass 1 f(d,h)
def foliage_biomass_1(species, DBH, height) -> float:
    """
    Repola J. (2009) Silva Fennica 43(4) Biomass equations for Scots pine and Norway spruce in Finland p. 631-633
    Repola J. (2008) Silva Fennica 42(4) Biomass equations for birch in Finland p. 611-613
    """
    if species == 1:  # Scots Pine
        lnbm = (
            -6.303
            + 14.472 * (stump_diameter(DBH) / (stump_diameter(DBH) + 6))
            - 3.976 * (height / (height + 1))
            + (0.109 + 0.118) / 2
        )
        bm = math.exp(lnbm)
    elif species == 2:  # Norway Spruce
        lnbm = (
            -2.994
            + 12.251 * (stump_diameter(DBH) / (stump_diameter(DBH) + 10))
            - 3.415 * (height / (height + 1))
            + (0.107 + 0.089) / 2
        )
        bm = math.exp(lnbm)
    else:  # Others, model for Birch
        lnbm = (
            -29.566
            + 33.372 * (stump_diameter(DBH) / (stump_diameter(DBH) + 2))
            + (0.00 + 0.077) / 2
        )
        bm = math.exp(lnbm)
    # bm = Biomass of component, kg
    # to tons:
    bm = bm / 1000
    return bm


# Stump biomass f(d)
def stump_biomass_1(species, DBH) -> float:
    """
    Repola J. (2009) Silva Fennica 43(4) Biomass equations for Scots pine and Norway spruce in Finland p. 631-633
    Repola J. (2008) Silva Fennica 42(4) Biomass equations for birch in Finland p. 611-613
    """
    if species == 1:  # Scots Pine
        lnbm = (
            -6.753
            + 12.681 * (stump_diameter(DBH) / (stump_diameter(DBH) + 12))
            + (0.010 + 0.044) / 2
        )
        bm = math.exp(lnbm)
    elif species == 2:  # Norway Spruce
        lnbm = (
            -3.964
            + 11.730 * (stump_diameter(DBH) / (stump_diameter(DBH) + 26))
            + (0.065 + 0.058) / 2
        )
        bm = math.exp(lnbm)
    else:  # Others, model for Birch
        lnbm = (
            -3.574
            + 11.304 * (stump_diameter(DBH) / (stump_diameter(DBH) + 26))
            + (0.02154 + 0.04542) / 2
        )
        bm = math.exp(lnbm)
    # bm = Biomass of component, kg
    # to tons:
    bm = bm / 1000
    return bm


# Coarse roots (>1cm) biomass 1 f(d,h)
def roots_biomass_1(species, DBH, height) -> float:
    """
    Repola J. (2009) Silva Fennica 43(4) Biomass equations for Scots pine and Norway spruce in Finland p. 631-633
    Repola J. (2008) Silva Fennica 42(4) Biomass equations for birch in Finland p. 611-613
    """
    if species == 1:  # Scots Pine
        lnbm = (
            -5.550
            + 13.408 * (stump_diameter(DBH) / (stump_diameter(DBH) + 15))
            + (0.000 + 0.079) / 2
        )
        bm = math.exp(lnbm)
    elif species == 2:  # Norway Spruce
        lnbm = (
            -2.294
            + 10.646 * (stump_diameter(DBH) / (stump_diameter(DBH) + 24))
            + (0.105 + 0.114) / 2
        )
        bm = math.exp(lnbm)
    else:  # Others, model for Birch
        lnbm = (
            -3.223
            + 6.497 * (stump_diameter(DBH) / (stump_diameter(DBH) + 22))
            + 1.033 * math.log(height)
            + (0.048 + 0.02677) / 2
        )
        bm = math.exp(lnbm)
    # bm = Biomass of component, kg
    # to tons:
    bm = bm / 1000
    return bm
