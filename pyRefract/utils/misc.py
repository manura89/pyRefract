import json
from pathlib import Path

import numpy as np

FITTING_DATABASE_PATH = Path(__file__).parent.parent / "databases/fitting_database.json"

with open(FITTING_DATABASE_PATH, "r") as f:
    FITTING_DATABASE = json.load(f)

ELEMENT_LATTICE_PARAMETERS = FITTING_DATABASE.get("LATTICE_PARAMETERS", None)


def rule_of_mixtures_linear(
    elements: list[str],
    concentrations: list[float] | np.ndarray,
    element_values: dict[str, float],
) -> float:
    """
    Helper function to calculate the rule of mixtures properties for a given set
    of elements, concentrations, and element values.
    """
    return sum(
        concentration * element_values[element]
        for element, concentration in zip(elements, concentrations)
    )


def rule_of_mixtures_lattice(
    elements: list[str],
    concentrations: list[float] | np.ndarray,
    element_values: dict[str, float] | None = None,
) -> float:
    """
    Helper function to calculate the rule of mixtures lattice parameter for a
    given set of elements, concentrations, and element values.
    """
    if element_values is None:
        element_values = ELEMENT_LATTICE_PARAMETERS

    if element_values is None:
        raise ValueError("Lattice parameters not found in the fitting database.")

    # Volume of cubic unit cells (V) calculated from the lattice parameters

    element_volumes = {element: element_values[element] ** 3 for element in elements}

    # Weighted average of the unit cell volumes based on concentrations

    volume = sum(
        concentration * element_volumes[element]
        for element, concentration in zip(elements, concentrations)
    )

    return volume ** (1 / 3)
