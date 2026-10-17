import json
from pathlib import Path

import numpy as np
from mendeleev import element

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

BURGERS_VECTORS = {  # Unit vectors in the slip directions
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
    Calculate and return the elastic parameters and lattice parameters for the given
    alloy composition and structure type with the specified method.

    Args:
        elements (list[str]): List of elements in the alloy.
        method (str): Method used for calculating elastic parameters ('fitting', 'ase',
            'lammps', or 'custom').
        concentrations (list[float] | np.ndarray | None, optional): Concentrations
            (atomic) of the elements. Defaults is equiatomic.
        structure_type (str, optional): Type of crystal structure. Defaults to "bcc".
        fitting_database (dict | None, optional): Fitting database to use. Defaults in
            "databases/fitting_database.json".
        transformation_matrix (list[list[int]] | np.ndarray, optional): Transformation
            matrix for the structure. Defaults to TRANSFORMATION_MATRICES["bcc_100"].
        offset (list[float] | np.ndarray, optional): Offset of the atomic positions (in
            fractions of the lattice parameter) to avoid atoms moving across the
            simulation box. Defaults to OFFSETS["bcc_100"].
        calc_directions (list[str] | None, optional): List of directions for the
            calculation. Defaults to ["C11", "C12", "C44"].
        calc (optional): Calculator object. Defaults to None. Required if method is
            'ase'.
        potential_file (str | None, optional): Path to the potential file. Defaults to
            None. Required if method is 'lammps'.
        pair_style (str | None, optional): Pair style for the potential. Defaults to
            None. Required if method is 'lammps'.
        potential_cmd (list[str] | None, optional): LAMMPS potential commands. This can
            be given instead of potential_file, pair_style, and will override them if
            provided. Defaults to None. Required if method is 'lammps'.
        nimages (int, optional): Number of images for the calculation. Defaults to 3.
        custom_elastic_parameters (dict[str, float] | None, optional): Custom elastic
            parameters. Defaults to None. Required if method is 'custom'.
        custom_lattice_parameters (dict[str, float] | None, optional): Custom lattice
            parameters. Defaults to None. Required if method is 'custom'.
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed structure.
            Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure. Defaults
            to "relaxed_structures".

    Returns:
        tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]: A tuple
            containing the elastic parameters and lattice parameters and the strandard
            deviation of each parameter.
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
    Calculate and return the surface energies for the given alloy composition and
    structure type with the specified method.

    Args:
        elements (list[str]): List of elements in the alloy.
        method (str): Method used for calculating surface energies ('ase', 'lammps', or
            'custom').
        concentrations (list[float] | np.ndarray | None, optional): Concentrations
            (atomic) of the elements. Defaults is equiatomic.
        structure_type (str, optional): Type of crystal structure. Defaults to "bcc".
        transformation_matrices (dict[list[list[int]]] | dict[np.ndarray], optional):
        Transformation matrices for the structure. The keys should be surface planes and
            the values should be the corresponding transformation matrices. Defaults to
            TRANSFORMATION_MATRICES.
        offsets (dict[str, list[float]] | dict[str, np.ndarray], optional): Offset of
            the atomic positions (in fractions of the lattice parameter) to avoid atoms
            moving across the simulation box. Defaults to OFFSETS.
        calc (optional): Calculator object. Defaults to None. Required if method is
            'ase'.
        potential_file (str | None, optional): Path to the potential file. Defaults to
            None. Required if method is 'lammps'.
        pair_style (str | None, optional): Pair style for the potential. Defaults to
            None. Required if method is 'lammps'.
        potential_cmd (list[str] | None, optional): LAMMPS potential commands. This can
            be given instead of potential_file, pair_style, and will override them if
            provided. Defaults to None. Required if method is 'lammps'.
        nimages (int, optional): Number of images for the calculation. Defaults to 3.
        custom_surface_energies (dict[str, float] | None, optional): Custom surface
            energies. Defaults to None. Required if method is 'custom'.
        surface_planes (list[str] | None, optional): List of surface planes to consider.
            Defaults to ["100", "110", "112"].
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed structure.
            Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure. Defaults
            to "relaxed_structures".

    Returns:
        dict[str, tuple[float, float]]: A dictionary where each key is a surface plane
            and the value is a tuple containing the surface energy and its standard
            deviation for that plane.
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


def get_usf_energies(
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
    custom_usf_energies: dict[str, float] | None = None,
    usf_planes: list[str] | None = None,
    slip_vectors: list[float] | np.ndarray = SLIP_VECTORS,
    usf_steps: int = 25,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str = "relaxed_structures",
) -> dict[str, float]:
    """
    Calculate and return the unstable stacking fault (USF) energies for the given alloy
    composition and structure type with the specified method.

    Args:
        elements (list[str]): List of elements in the alloy.
        method (str): Method used for calculating USF energies ('ase', 'lammps', or
        'custom').
        concentrations (list[float] | np.ndarray | None, optional): Concentrations
            (atomic) of the elements. Defaults is equiatomic.
        structure_type (str, optional): Type of crystal structure. Defaults to "bcc".
        transformation_matrices (dict[list[list[int]]] | dict[np.ndarray], optional):
        Transformation matrices for the structure. The keys should be surface planes and
            the values should be the corresponding transformation matrices. Defaults to
            TRANSFORMATION_MATRICES.
        offsets (dict[str, list[float]] | dict[str, np.ndarray], optional): Offset of
            the atomic positions (in fractions of the lattice parameter) to avoid atoms
            moving across the simulation box. Defaults to OFFSETS.
        calc (optional): Calculator object. Defaults to None. Required if method is
            'ase'.
        potential_file (str | None, optional): Path to the potential file. Defaults to
            None. Required if method is 'lammps'.
        pair_style (str | None, optional): Pair style for the potential. Defaults to
            None. Required if method is 'lammps'.
        potential_cmd (list[str] | None, optional): LAMMPS potential commands. This can
            be given instead of potential_file, pair_style, and will override them if
            provided. Defaults to None. Required if method is 'lammps'.
        nimages (int, optional): Number of images for the calculation. Defaults to 3.
        custom_usf_energies (dict[str, float] | None, optional): Custom USF energies.
            Defaults to None. Required if method is 'custom'.
        usf_planes (list[str] | None, optional): List of USF planes to consider.
            Defaults to ["100", "110", "112"].
        slip_vectors (list[float] | np.ndarray, optional): Slip vectors for the USF
            calculation. Defaults to SLIP_VECTORS.
        usf_steps (int, optional): Number of steps for the USF calculation. Defaults to
            25.
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed structure.
            Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure. Defaults
            to "relaxed_structures".

    Returns:
        dict[str, tuple[float, float]]: A dictionary where each key is a USF plane
            and the value is a tuple containing the USF energy and its standard
            deviation for that plane.
    """
    if concentrations is None:
        concentrations = [1.0 / len(elements)] * len(elements)

    concentrations = np.array(concentrations) / np.sum(concentrations)

    if usf_planes is None:
        usf_planes = ["110", "112"]

    if method == "ase":
        from ..calculators.ase_calculators import ase_usf_energy

        usf_energies = {
            plane: ase_usf_energy(
                elements=elements,
                concentrations=concentrations,
                transformation_matrix=transformation_matrices[
                    f"{structure_type}_{plane}"
                ],
                plane=plane,
                slip_vector=slip_vectors[f"{structure_type}_{plane}"],
                usf_steps=usf_steps,
                offset=offsets[f"{structure_type}_{plane}"],
                calc=calc,
                structure_type=structure_type,
                nimages=nimages,
                check_cna=check_cna,
                save_relaxed_structure=save_relaxed_structure,
                save_path=save_path,
            )
            for plane in usf_planes
        }

    elif method == "lammps":
        from ..calculators.lammps_calculators import (
            lammps_potential,
            lammps_usf_energy,
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

        usf_energies = {
            plane: lammps_usf_energy(
                elements=elements,
                concentrations=concentrations,
                transformation_matrix=transformation_matrices[
                    f"{structure_type}_{plane}"
                ],
                plane=plane,
                slip_vector=slip_vectors[f"{structure_type}_{plane}"],
                usf_steps=usf_steps,
                offset=offsets[f"{structure_type}_{plane}"],
                potential_cmd=potential_cmd,
                structure_type=structure_type,
                nimages=nimages,
                check_cna=check_cna,
                save_relaxed_structure=save_relaxed_structure,
                save_path=save_path,
            )
            for plane in usf_planes
        }

    elif method == "custom":
        usf_energies = {
            plane: custom_usf_energies.get(plane, 0.0) for plane in usf_planes
        }

    return usf_energies


def get_slip_plane_stresses(
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
    custom_slip_plane_stresses: dict[str, float] | None = None,
    surface_planes: list[str] | None = None,
    slip_vectors: list[float] | np.ndarray = BURGERS_VECTORS,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str = "relaxed_structures",
) -> dict[str, float]:
    """
    Calculate and return the standard deviation of slip plane stresses for the given
    alloy composition and structure type with the specified method.

    Args:
        elements (list[str]): List of elements in the alloy.
        method (str): Method used for calculating slip plane stresses ('ase', 'lammps',
        or 'custom').
        concentrations (list[float] | np.ndarray | None, optional): Concentrations
            (atomic) of the elements. Defaults is equiatomic.
        structure_type (str, optional): Type of crystal structure. Defaults to "bcc".
        transformation_matrices (dict[list[list[int]]] | dict[np.ndarray], optional):
        Transformation matrices for the structure. The keys should be surface planes and
            the values should be the corresponding transformation matrices. Defaults to
            TRANSFORMATION_MATRICES.
        offsets (dict[str, list[float]] | dict[str, np.ndarray], optional): Offset of
            the atomic positions (in fractions of the lattice parameter) to avoid atoms
            moving across the simulation box. Defaults to OFFSETS.
        calc (optional): Calculator object. Defaults to None. Required if method is
            'ase'.
        potential_file (str | None, optional): Path to the potential file. Defaults to
            None. Required if method is 'lammps'.
        pair_style (str | None, optional): Pair style for the potential. Defaults to
            None. Required if method is 'lammps'.
        potential_cmd (list[str] | None, optional): LAMMPS potential commands. This can
            be given instead of potential_file, pair_style, and will override them if
            provided. Defaults to None. Required if method is 'lammps'.
        nimages (int, optional): Number of images for the calculation. Defaults to 3.
        custom_slip_plane_stresses (dict[str, float] | None, optional): Custom slip
            plane stresses. Defaults to None. Required if method is 'custom'.
        surface_planes (list[str] | None, optional): List of surface planes to consider.
            Defaults to ["100", "110", "112"].
        slip_vectors (list[float] | np.ndarray, optional): Slip vectors for the USF
            calculation. Defaults to SLIP_VECTORS.
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed structure.
            Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure. Defaults
            to "relaxed_structures".

    Returns:
        dict[str, tuple[float, float]]: A dictionary where each key is a surface plane
        and the value is the  standard deviation of the slip plane stresses for that
        plane.
    """
    if concentrations is None:
        concentrations = [1.0 / len(elements)] * len(elements)

    concentrations = np.array(concentrations) / np.sum(concentrations)

    if surface_planes is None:
        surface_planes = ["112"]  # Default surface planes

    if method == "ase":
        from ..calculators.ase_calculators import ase_slip_plane_stress

        slip_plane_stresses = {
            plane: ase_slip_plane_stress(
                elements=elements,
                concentrations=concentrations,
                transformation_matrix=transformation_matrices[
                    f"{structure_type}_{plane}"
                ],
                plane=plane,
                slip_vector=slip_vectors[f"{structure_type}_{plane}"],
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
    elif method == "lammps":
        from ..calculators.lammps_calculators import (
            lammps_potential,
            lammps_slip_plane_stress,
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

        slip_plane_stresses = {
            plane: lammps_slip_plane_stress(
                elements=elements,
                concentrations=concentrations,
                transformation_matrix=transformation_matrices[
                    f"{structure_type}_{plane}"
                ],
                plane=plane,
                slip_vector=slip_vectors[f"{structure_type}_{plane}"],
                offset=offsets[f"{structure_type}_{plane}"],
                potential_cmd=potential_cmd,
                calc=calc,
                structure_type=structure_type,
                nimages=nimages,
                check_cna=check_cna,
                save_relaxed_structure=save_relaxed_structure,
                save_path=save_path,
            )
            for plane in surface_planes
        }
    elif method == "custom":
        slip_plane_stresses = {
            plane: custom_slip_plane_stresses.get(plane, 0.0)
            for plane in surface_planes
        }

    return slip_plane_stresses


def density_function(
    elements: list[str],
    concentrations: list[float] | np.ndarray | None = None,
    element_lattice_parameters: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculate and return the density for the given alloy composition.

    Args:
        elements (list[str]): List of elements in the alloy.
        concentrations (list[float] | np.ndarray | None, optional): Concentrations
            (atomic) of the elements. Defaults to equiatomic.
        element_lattice_parameters (dict[str, float] | None, optional): Dictionary of
        element lattice parameters. Defaults is for bcc structure.

    Returns:
        np.ndarray: Density of the alloy in g/cm^3.
    """

    AVOGADRO_NUMBER = 6.02214076e23

    if element_lattice_parameters is None:
        element_lattice_parameters = FITTING_DATABASE.get("LATTICE_PARAMETERS")

    if concentrations is None:
        concentrations = [1.0 / len(elements)] * len(elements)

    concentrations = np.array(concentrations) / np.sum(concentrations)

    molar_mass_of_elements = {el: element(el).atomic_weight for el in elements}  # g/mol
    volume_of_elements = {
        el: element_lattice_parameters[el] ** 3 * 1e-24 / 2 for el in elements
    }  # For primitive BCC unit cell in cm^3

    total_volume = sum(
        conc * volume_of_elements.get(el, 0.0)
        for conc, el in zip(concentrations, elements)
    )
    total_molar_mass = sum(
        conc * molar_mass_of_elements.get(el, 0.0)
        for conc, el in zip(concentrations, elements)
    )

    return (total_molar_mass / total_volume) / AVOGADRO_NUMBER  # g/cm^3


def get_ductility_indices(
    crack_geometry: dict,
    alloy_properties: dict,
) -> dict:
    """
    Calculate ductility indices based on crack geometry and alloy properties.

    Args:
        crack_geometry (dict): Dictionary containing crack geometry data.
        alloy_properties (dict): Dictionary containing the relevant alloy properties.

    Returns:
        dict: Dictionary containing calculated ductility indices.
    """
    from .ductility import Fracture_potential

    fracture_potential = Fracture_potential(**crack_geometry)
    fracture_potential.assign_properties(**alloy_properties)
    fracture_potential.calculate_ductility_indices()

    crack_plane = fracture_potential.crack_plane
    emission_plane = fracture_potential.emission_plane

    crack_geometry_key = f"({crack_plane}, {emission_plane})"

    ductility_indices = {
        crack_geometry_key: {
            "D": fracture_potential.D,
            "D_local": getattr(fracture_potential, "D_local", None),
        }
    }

    return ductility_indices


def get_yield_strengths(
    elements: list[str],
    concentrations: list[float] | np.ndarray | None = None,
    alloy_properties: dict | None = None,
    strain_rate: float = 1e-3,
    element_lattice_parameters: dict[str, float] | None = None,
    temperatures: list[float] | None = None,
) -> dict[float, float]:
    """
    Calculate yield strengths for a given set of elements, concentrations, alloy
    properties, and temperatures.

    Args:
        elements (list[str]): List of chemical elements.
        concentrations (list[float] | np.ndarray): List or array of concentrations
            corresponding to the elements.
        element_lattice_parameters (dict[str, float]): Dictionary containing the
            lattice parameters for each element.
        alloy_properties (dict): Dictionary containing the relevant alloy properties.
        temperatures (list[float] | None): List of temperatures at which to calculate
            yield strengths.
        strain_rate (float): Strain rate at which to calculate yield strengths.

    Returns:
        dict[float, float]: Dictionary containing calculated yield strengths for each
        temperature.
    """

    alpha = 1 / 12
    kB = 8.6173303e-5
    reference_strain_rate = 1e4
    eVtoGPa = 160.21766208

    if concentrations is None:
        concentrations = [1.0 / len(elements)] * len(elements)

    concentrations = np.array(concentrations) / np.sum(concentrations)

    if element_lattice_parameters is None:
        element_lattice_parameters = FITTING_DATABASE.get("LATTICE_PARAMETERS", None)

    element_volumes = {
        element: 0.5 * element_lattice_parameters[element] ** 3 for element in elements
    }

    if alloy_properties is not None and "lattice_parameters" in alloy_properties:
        alloy_lattice_parameter = alloy_properties["lattice_parameters"]["a"]
        alloy_volume = 0.5 * alloy_lattice_parameter**3
    else:
        alloy_volume = sum(
            concentration * element_volumes[element]
            for element, concentration in zip(elements, concentrations)
        )
        alloy_lattice_parameter = (2 * alloy_volume) ** (1 / 3)

    b = alloy_lattice_parameter * np.sqrt(3) / 2
    misfit_volumes = {
        element: element_volumes[element] - alloy_volume for element in elements
    }

    misfit_sum = sum(
        concentration * misfit_volumes[element] ** 2
        for element, concentration in zip(elements, concentrations)
    )

    if alloy_properties is not None and "elastic_parameters" in alloy_properties:
        C11 = alloy_properties["elastic_parameters"]["C11"]
        C12 = alloy_properties["elastic_parameters"]["C12"]
        C44 = alloy_properties["elastic_parameters"]["C44"]
    else:
        _, elastic_parameters = get_elastic_parameters(
            elements=elements, concentrations=concentrations, method="fitting"
        )
        C11 = elastic_parameters["C11"][0]
        C12 = elastic_parameters["C12"][0]
        C44 = elastic_parameters["C44"][0]

    B = (C11 + 2 * C12) / 3  # Bulk modulus
    mu = np.sqrt(C44 * (C11 - C12) / 2)  # Shear modulus
    nu = (3 * B - 2 * mu) / (2 * (3 * B + mu))  # Poisson's ratio

    # 0 K yield strength calculation

    tau_y0 = (
        3.067
        * 0.04
        * (alpha ** (-1 / 3))
        * mu
        * (((1 + nu) / (1 - nu)) ** (4 / 3))
        * ((misfit_sum / (b**6)) ** (2 / 3))
    )
    d_Eb0 = (
        2.00
        * (alpha ** (1 / 3))
        * mu
        * (b**3)
        * (((1 + nu) / (1 - nu)) ** (2 / 3))
        * ((misfit_sum / (b**6)) ** (1 / 3))
        / eVtoGPa
    )

    yield_strengths = {}

    for temperature in temperatures:
        tau_y1 = tau_y0 * (
            1
            - (
                (kB * temperature * np.log(reference_strain_rate / strain_rate) / d_Eb0)
                ** (2 / 3)
            )
        )
        tau_y2 = tau_y0 * np.exp(
            -(
                (kB * temperature * np.log(reference_strain_rate / strain_rate) / d_Eb0)
                ** (0.91)
            )
            / 0.55
        )

        if tau_y1 / tau_y0 >= 0.5:
            yield_strengths[temperature] = tau_y1 * 1000
        else:
            yield_strengths[temperature] = tau_y2 * 1000

    return yield_strengths
