"""
Source: https://github.com/lukefi/metsi/blob/main/lukefi/metsi/forestry/preprocessing/naslund.py
Module contains forestry domain spesific model functions
"""



def naslund_height(diameter: float | None, species: float | None) -> float | None:
    """
    Näslund height model, with parameters from one Siipilehto. As extracted from LueVMI12.py.
    :param diameter: diameter of the tree at 1.3m height
    :param species: species code of the tree in internal TreeSpecies terms
    :return estimated height of the tree in meters or None
    """
    if diameter is not None and diameter > 0:
        # scots pine or other coniferous (and deciduous trees also here)
        if species != 2:
            height = ((diameter**2) / (0.894 + 0.185 * diameter) ** 2) + 1.3
            return round(height, 2)

        # norway spruce
        else:
            height = ((diameter**3) / (1.811 + 0.308 * diameter) ** 3) + 1.3
            return round(height, 2)

        height = ((diameter**2) / (0.898 + 0.242 * diameter) ** 2) + 1.3
        return round(height, 2)
    return None


def naslund_correction(species: float, diameter: float, height: float) -> float:
    """Height correction coefficient by Naslund height model

    :spe: tree stratum species
    :diameter: tree stratum diameter
    :height: tree stratum height
    :return: height correction coefficient
    """
    h_computed = naslund_height(diameter, species)
    return height / h_computed
