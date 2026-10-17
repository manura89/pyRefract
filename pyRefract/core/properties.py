import json
from pathlib import Path

import numpy as np

FITTING_DATABASE_PATH = Path(__file__).parent.parent / "databases/fitting_database.json"

with open(FITTING_DATABASE_PATH, "r") as f:
    FITTING_DATABASE = json.load(f)

TRANSFORMATION_MATRICES = {
    "bcc_MELTING": [[0, 10, 10], [10, 0, 10], [20, 20, 0]],
    "bcc_TDE": [[0, 14, 14], [15, 0, 15], [16, 16, 0]],
    "bcc_SBE": [[0, 10, 10], [10, 0, 10], [10, 10, 0]],
    "bcc_100": [[0, 5, 5], [5, 0, 5], [5, 5, 0]],
    "bcc_110": [[3, -3, 0], [4, 4, 0], [4, 4, 8]],
    "bcc_112": [[0, 0, 5], [3, -3, 0], [6, 6, 4]],
}

OFFSETS = {
    "bcc_100": [0.25, 0.25, 0.25],
    "bcc_110": [0, 0.5, 0.25],
    "bcc_112": [-1 / 12, 5 / 12, 1 / 12],
}

SLIP_VECTORS = {
    "bcc_110": [1 / 6, 1 / 8, 0],
    "bcc_112": [1 / 5, 0, 0],
}

BURGERS_VECTORS = {  # Unit vectors of the slip directions
    "bcc_110": [np.sqrt(2 / 3), np.sqrt(1 / 3), 0],
    "bcc_112": [1, 0, 0],
}

ELASTIC_CONSTANTS = {
    "bcc": ["C11", "C12", "C44"],
    "fcc": ["C11", "C12", "C44"],
    "hcp": ["C11", "C12", "C13", "C33", "C44"],
}

BCC_CRACK_GEOMETRIES_PATH = (
    Path(__file__).parent.parent / "databases/bcc_crack_geometries.json"
)
with open(BCC_CRACK_GEOMETRIES_PATH, "r") as f:
    CRACK_GEOMETRY = json.load(f).get("110_112")


def get_elastic_parameters(
    elements: list[str],
    method: str,
    concentrations: list[float] | np.ndarray | None = None,
    structure_type: str = "bcc",
    fitting_database: dict | None = None,
    transformation_matrix: list[list[int]] | np.ndarray = TRANSFORMATION_MATRICES[
        "bcc_100"
    ],
    offset: list[float] | np.ndarray = OFFSETS["bcc_100"],
    elastic_constants: list[str] | None = None,
    calc=None,
    potential_file: str | None = None,
    pair_style: str | None = None,
    potential_cmd: list[str] | None = None,
    nimages: int = 3,
    custom_elastic_parameters: dict[str, float] | None = None,
    custom_lattice_parameters: dict[str, float] | None = None,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Calculate and return the elastic parameters and lattice parameters for the
    given alloy composition and structure type with the specified method.

    Args:
        elements (list[str]): List of elements in the alloy.
        method (str): Method used for calculating elastic parameters ('fitting',
            'ase', 'lammps', or 'custom').
        concentrations (list[float] | np.ndarray | None, optional):
            Concentrations (atomic) of the elements. Defaults is equiatomic.
        structure_type (str, optional): Type of crystal structure. Defaults to
            "bcc".
        fitting_database (dict | None, optional): Fitting database to use.
            Defaults in "databases/fitting_database.json".
        transformation_matrix (list[list[int]] | np.ndarray, optional):
            Transformation matrix for the structure. Defaults to
            TRANSFORMATION_MATRICES["bcc_100"].
        offset (list[float] | np.ndarray, optional): Offset of the atomic
            positions (in fractions of the lattice parameter) to avoid atoms
            moving across the simulation box. Defaults to OFFSETS["bcc_100"].
        calc_directions (list[str] | None, optional): List of directions for the
            calculation. Defaults to ["C11", "C12", "C44"].
        calc (optional): Calculator object. Defaults to None. Required if method
            is 'ase'.
        potential_file (str | None, optional): Path to the potential file.
            Defaults to None. Required if method is 'lammps'.
        pair_style (str | None, optional): Pair style for the potential.
            Defaults to None. Required if method is 'lammps'.
        potential_cmd (list[str] | None, optional): LAMMPS potential commands.
            This can be given instead of potential_file, pair_style, and will
            override them if provided. Defaults to None. Required if method is
            'lammps'.
        nimages (int, optional): Number of images for the calculation. Defaults
            to 3.
        custom_elastic_parameters (dict[str, float] | None, optional): Custom
            elastic parameters. Defaults to None. Required if method is
            'custom'.
        custom_lattice_parameters (dict[str, float] | None, optional): Custom
            lattice parameters. Defaults to None. Required if method is
            'custom'.
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed
            structure. Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure.
            Defaults to "relaxed_structures".

    Returns:
        tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]: A
            tuple containing the elastic parameters and lattice parameters and
            the strandard deviation of each parameter.
    """
    if concentrations is None:
        concentrations = [1.0 / len(elements)] * len(elements)

    # Normalize the concentrations to ensure they sum to 1.
    concentrations = np.array(concentrations) / np.sum(concentrations)

    if elastic_constants is None:
        elastic_constants = ELASTIC_CONSTANTS.get(structure_type, ["C11", "C12", "C44"])

    if method == "fitting":
        from ..utils.misc import rule_of_mixtures_lattice, rule_of_mixtures_linear

        if fitting_database is None:
            fitting_database = FITTING_DATABASE

        lattice_parameter = {
            "a": (
                rule_of_mixtures_lattice(
                    elements=elements,
                    concentrations=concentrations,
                    element_values=fitting_database.get("LATTICE_PARAMETERS", None),
                ),
                0.0,
            )
        }
        elastic_parameters = {
            C: (
                rule_of_mixtures_linear(
                    elements=elements,
                    concentrations=concentrations,
                    element_values=fitting_database.get(
                        "ELASTIC", fitting_database.get("ELASTIC", None)
                    ).get(C, None),
                ),
                0.0,
            )
            for C in elastic_constants
        }

    elif method == "ase":
        from ..calculators.ase_calculators import ase_elastic_properties

        lattice_parameter, elastic_parameters = ase_elastic_properties(
            elements=elements,
            concentrations=concentrations,
            calc=calc,
            transformation_matrix=transformation_matrix,
            offset=offset,
            nimages=nimages,
            check_cna=check_cna,
            save_relaxed_structure=save_relaxed_structure,
            save_path=save_path,
        )

    elif method == "lammps":
        from ..calculators.lammps_calculators import (
            lammps_elastic_properties,
            lammps_potential,
        )

        if potential_cmd is None:
            if potential_file is None or pair_style is None:
                raise ValueError(
                    "Both potential_file and pair_style must be provided for LAMMPS."
                )
            potential_cmd = lammps_potential(
                filename=potential_file,
                pair_style=pair_style,
                elements=elements,
            )

        lattice_parameter, elastic_parameters = lammps_elastic_properties(
            elements=elements,
            concentrations=concentrations,
            transformation_matrix=transformation_matrix,
            offset=offset,
            potential_cmd=potential_cmd,
            structure_type=structure_type,
            nimages=nimages,
            check_cna=check_cna,
            save_relaxed_structure=save_relaxed_structure,
            save_path=save_path,
        )

    elif method == "custom":
        lattice_parameter = {
            param: (custom_lattice_parameters[param], 0.0)
            for param in custom_lattice_parameters
        }
        elastic_parameters = {
            param: (custom_elastic_parameters[param], 0.0)
            for param in custom_elastic_parameters
        }

    return lattice_parameter, elastic_parameters


def get_surface_energies(
    elements: list[str],
    method: str,
    concentrations: list[float] | np.ndarray | None = None,
    structure_type: str = "bcc",
    transformation_matrices: dict[list[list[int]]]
    | dict[np.ndarray] = TRANSFORMATION_MATRICES,
    offsets: dict[str, list[float]] | dict[str, np.ndarray] = OFFSETS,
    calc=None,
    potential_file: str | None = None,
    pair_style: str | None = None,
    potential_cmd: list[str] | None = None,
    nimages: int = 3,
    custom_surface_energies: dict[str, float] | None = None,
    surface_planes: list[str] | None = None,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str = "relaxed_structures",
) -> dict[str, float]:
    """
    Calculate and return the surface energies for the given alloy composition
    and structure type with the specified method.

    Args:
        elements (list[str]): List of elements in the alloy.
        method (str): Method used for calculating surface energies ('ase',
            'lammps', or 'custom').
        concentrations (list[float] | np.ndarray | None, optional):
            Concentrations (atomic) of the elements. Defaults is equiatomic.
        structure_type (str, optional): Type of crystal structure. Defaults to
            "bcc".
        transformation_matrices (dict[list[list[int]]] | dict[np.ndarray],
            optional): Transformation matrices for the structure. The keys
            should be surface planes and the values should be the corresponding
            transformation matrices. Defaults to TRANSFORMATION_MATRICES.
        offsets (dict[str, list[float]] | dict[str, np.ndarray], optional):
            Offset of the atomic positions (in fractions of the lattice
            parameter) to avoid atoms moving across the simulation box. Defaults
            to OFFSETS.
        calc (optional): Calculator object. Defaults to None. Required if method
            is 'ase'.
        potential_file (str | None, optional): Path to the potential file.
            Defaults to None. Required if method is 'lammps'.
        pair_style (str | None, optional): Pair style for the potential.
            Defaults to None. Required if method is 'lammps'.
        potential_cmd (list[str] | None, optional): LAMMPS potential commands.
            This can be given instead of potential_file, pair_style, and will
            override them if provided. Defaults to None. Required if method is
            'lammps'.
        nimages (int, optional): Number of images for the calculation. Defaults
            to 3.
        custom_surface_energies (dict[str, float] | None, optional): Custom
            surface energies. Defaults to None. Required if method is 'custom'.
        surface_planes (list[str] | None, optional): List of surface planes to
            consider. Defaults to ["100", "110", "112"].
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed
            structure. Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure.
            Defaults to "relaxed_structures".

    Returns:
        dict[str, tuple[float, float]]: A dictionary where each key is a surface
            plane and the value is a tuple containing the surface energy and its
            standard deviation for that plane.
    """
    if concentrations is None:
        concentrations = [1.0 / len(elements)] * len(elements)

    concentrations = np.array(concentrations) / np.sum(concentrations)

    if surface_planes is None:
        surface_planes = ["100", "110", "112"]

    if method == "ase":
        from ..calculators.ase_calculators import ase_surface_energy

        surface_energies = {
            plane: ase_surface_energy(
                elements=elements,
                concentrations=concentrations,
                transformation_matrix=transformation_matrices[
                    f"{structure_type}_{plane}"
                ],
                plane=plane,
                offset=offsets[f"{structure_type}_{plane}"],
                calc=calc,
                structure_type=structure_type,
                nimages=nimages,
                check_cna=check_cna,
                save_relaxed_structure=save_relaxed_structure,
                save_path=save_path,
            )
            for plane in surface_planes
        }

    elif method == "lammps":
        from ..calculators.lammps_calculators import (
            lammps_potential,
            lammps_surface_energy,
        )

        if potential_cmd is None:
            if potential_file is None or pair_style is None:
                raise ValueError(
                    "Both potential_file and pair_style must be provided for LAMMPS."
                )
            potential_cmd = lammps_potential(
                filename=potential_file,
                pair_style=pair_style,
                elements=elements,
            )

        surface_energies = {
            plane: lammps_surface_energy(
                elements=elements,
                concentrations=concentrations,
                transformation_matrix=transformation_matrices[
                    f"{structure_type}_{plane}"
                ],
                plane=plane,
                offset=offsets[f"{structure_type}_{plane}"],
                potential_cmd=potential_cmd,
                structure_type=structure_type,
                nimages=nimages,
                check_cna=check_cna,
                save_relaxed_structure=save_relaxed_structure,
                save_path=save_path,
            )
            for plane in surface_planes
        }

    elif method == "custom":
        surface_energies = {
            plane: custom_surface_energies.get(plane, 0.0) for plane in surface_planes
        }

    return surface_energies
