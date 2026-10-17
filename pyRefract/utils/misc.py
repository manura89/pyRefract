import json
from pathlib import Path

import numpy as np
from ase import Atoms

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
    Helper function to calculate the rule of mixtures lattice parameter for a given set
    of elements, concentrations, and element values.
    """
    if element_values is None:
        element_values = ELEMENT_LATTICE_PARAMETERS

    if element_values is None:
        raise ValueError("Lattice parameters not found in the fitting database.")

    element_volumes = {element: element_values[element] ** 3 for element in elements}

    volume = sum(
        concentration * element_volumes[element]
        for element, concentration in zip(elements, concentrations)
    )

    return volume ** (1 / 3)


def check_methods_completeness(
    methods: dict[str, callable],
) -> None | tuple[bool, list[str]]:
    """
    Check if all required methods are implemented.

    Args:
        methods (dict[str, callable]): Dictionary of method names to check.

    Returns:
        (None | tuple[bool, list[str]]): None if all methods are implemented,
        tuple[False, list[str]] otherwise. If False, returns a list of missing methods.
    """
    allowed_methods = {
        "elastic": ["fitting", "ase", "lammps", "custom"],
        "surface_energies": ["ase", "lammps", "custom"],
        "usf_energies": ["ase", "lammps", "custom"],
        "slip_plane_stresses": ["ase", "lammps", "custom"],
        "crack_geometry": [
            "crack_front",
            "crack_plane",
            "emission_plane",
            "burgers_direction_leading",
            "crack_extension_direction",
            "slip_direction",
            "rotation_matrix",
            "transformation_matrix",
        ],
    }
    missing_methods = []

    if "elastic" not in methods:
        missing_methods.append('Missing methods for "elastic" calculations')
    elif methods["elastic"]["method"] not in allowed_methods["elastic"]:
        missing_methods.append(
            f"Method {methods['elastic']['method']} is not allowed for "
            + "'elastic' calculations"
        )
    elif methods["elastic"]["method"] == "ase":
        calc = methods["elastic"].get("calc", None)
        if calc is None:
            missing_methods.append(
                "Missing 'calc' for 'ase' method in 'elastic' calculations"
            )
    elif methods["elastic"]["method"] == "lammps":
        pair_style = methods["elastic"].get("pair_style", None)
        potential_file = methods["elastic"].get("potential_file", None)

        if pair_style is None:
            missing_methods.append(
                "Missing 'pair_style' for 'lammps' method in"
                + " 'elastic' calculations"
            )
        if potential_file is None:
            missing_methods.append(
                "Missing 'potential_file' for 'lammps' method"
                + " in 'elastic' calculations"
            )
    elif methods["elastic"]["method"] == "custom":
        lattice_parameter = methods["elastic"].get("custom_lattice_parameters", None)
        if lattice_parameter is None:
            missing_methods.append(
                "Missing 'custom_lattice_parameters' for 'custom'"
                + " method in 'elastic' calculations"
            )
        elastic_parameters = methods["elastic"].get("custom_elastic_parameters", None)
        if elastic_parameters is None:
            missing_methods.append(
                "Missing 'custom_elastic_parameters' for 'custom'"
                + " method in 'elastic' calculations"
            )

    if "surface_energies" not in methods:
        missing_methods.append('Missing methods for "surface_energies" calculations')
    elif (
        methods["surface_energies"]["method"] not in allowed_methods["surface_energies"]
    ):
        missing_methods.append(
            f"Method {methods['surface_energies']['method']} is not allowed for "
            + "'surface_energies' calculations"
        )
    elif methods["surface_energies"]["method"] == "ase":
        calc = methods["surface_energies"].get("calc", None)
        if calc is None:
            missing_methods.append(
                "Missing 'calc' for 'ase' method in"
                + " 'surface_energies' calculations"
            )
    elif methods["surface_energies"]["method"] == "lammps":
        pair_style = methods["surface_energies"].get("pair_style", None)
        potential_file = methods["surface_energies"].get("potential_file", None)
        if pair_style is None:
            missing_methods.append(
                "Missing 'pair_style' for 'lammps' method in"
                + " 'surface_energies' calculations"
            )
        if potential_file is None:
            missing_methods.append(
                "Missing 'potential_file' for 'lammps' method in"
                + " 'surface_energies' calculations"
            )
    elif methods["surface_energies"]["method"] == "custom":
        lattice_parameter = methods["surface_energies"].get(
            "custom_surface_energies", None
        )
        if lattice_parameter is None:
            missing_methods.append(
                "Missing 'custom_surface_energies' for 'custom'"
                + " method in 'surface_energies' calculations"
            )
    elif methods["surface_energies"]["method"] == "custom":
        surface_energy = methods["surface_energies"].get("custom_surface_energy", None)
        if surface_energy is None:
            missing_methods.append(
                "Missing 'custom_surface_energy' for 'custom'"
                + " method in 'surface_energies' calculations\n"
            )

    if "usf_energies" not in methods:
        missing_methods.append('Missing methods for "usf_energies" calculations')
    elif methods["usf_energies"]["method"] not in allowed_methods["usf_energies"]:
        missing_methods.append(
            f"Method {methods['usf_energies']['method']} is not allowed for "
            + "'usf_energies' calculations"
        )
    elif methods["usf_energies"]["method"] == "ase":
        calc = methods["usf_energies"].get("calc", None)
        if calc is None:
            missing_methods.append(
                "Missing 'calc' for 'ase' method in 'usf_energies'" + " calculations\n"
            )
    elif methods["usf_energies"]["method"] == "lammps":
        pair_style = methods["usf_energies"].get("pair_style", None)
        potential_file = methods["usf_energies"].get("potential_file", None)
        if pair_style is None:
            missing_methods.append(
                "Missing 'pair_style' for 'lammps' method in"
                + " 'usf_energies' calculations\n"
            )
        if potential_file is None:
            missing_methods.append(
                "Missing 'potential_file' for 'lammps' method in"
                + " 'usf_energies' calculations\n"
            )
    elif methods["usf_energies"]["method"] == "custom":
        lattice_parameter = methods["usf_energies"].get("custom_usf_energies", None)
        if lattice_parameter is None:
            missing_methods.append(
                "Missing 'custom_usf_energies' for 'custom'"
                + " method in 'usf_energies' calculations\n"
            )
    elif methods["usf_energies"]["method"] == "custom":
        usf_energy = methods["usf_energies"].get("custom_usf_energy", None)
        if usf_energy is None:
            missing_methods.append(
                "Missing 'custom_usf_energy' for 'custom' method"
                + " in 'usf_energies' calculations\n"
            )

    if "slip_plane_stresses" not in methods:
        missing_methods.append('Missing methods for "slip_plane_stresses" calculations')
    elif (
        methods["slip_plane_stresses"]["method"]
        not in allowed_methods["slip_plane_stresses"]
    ):
        missing_methods.append(
            f"Method {methods['slip_plane_stresses']['method']} is not allowed for "
            + "'slip_plane_stresses' calculations"
        )
    elif methods["slip_plane_stresses"]["method"] == "ase":
        calc = methods["slip_plane_stresses"].get("calc", None)
        if calc is None:
            missing_methods.append(
                "Missing 'calc' for 'ase' method in"
                + " 'slip_plane_stresses' calculations\n"
            )
    elif methods["slip_plane_stresses"]["method"] == "lammps":
        pair_style = methods["slip_plane_stresses"].get("pair_style", None)
        potential_file = methods["slip_plane_stresses"].get("potential_file", None)
        calc = methods["slip_plane_stresses"].get("calc", None)
        if pair_style is None:
            missing_methods.append(
                "Missing 'pair_style' for 'lammps' method in"
                + " 'slip_plane_stresses' calculations\n"
            )
        if potential_file is None:
            missing_methods.append(
                "Missing 'potential_file' for 'lammps' method in"
                + " 'slip_plane_stresses' calculations\n"
            )
        if calc is None:
            missing_methods.append(
                "Missing 'calc' for 'lammps' method in"
                + " 'slip_plane_stresses' calculations\n"
            )
    elif methods["slip_plane_stresses"]["method"] == "custom":
        stress = methods["slip_plane_stresses"].get("custom_stress", None)
        if stress is None:
            missing_methods.append(
                "Missing 'custom_stress' for 'custom' method"
                + " in 'slip_plane_stresses' calculations\n"
            )

    crack_geometry = methods.get("crack_geometry", None)
    if crack_geometry is not None:
        for attribute in allowed_methods["crack_geometry"]:
            if crack_geometry.get(attribute, None) is None:
                missing_methods.append(
                    f"Missing '{attribute}' for 'crack_geometry' calculations\n"
                )

    if missing_methods:
        return missing_methods
    return None


def count_unique_within_tolerance(array: np.ndarray, tol: float) -> int:
    """
    Helper function to count unique values in an array within a specified tolerance.
    """
    if len(array) == 0:
        return 0

    sorted_arr = np.sort(array)

    unique_count = 1
    current_base = sorted_arr[0]

    for val in sorted_arr[1:]:
        if (val - current_base) > tol:
            # Value is outside the threshold; start a new group
            unique_count += 1
            current_base = val

    return unique_count


def get_net_forces(
    atoms: Atoms, calc, slip_vector: list[float] | np.ndarray
) -> np.ndarray:
    """
    Helper function to calculate net forces on the surface atoms by comparing relaxed
    bulk and unrelaxed surface.
    """

    atoms.calc = calc
    bulk_forces = atoms.get_forces()

    surface_atoms = atoms.copy()
    surface_cell = surface_atoms.get_cell()
    surface_cell[2, 2] += 8.0
    surface_atoms.set_cell(surface_cell, scale_atoms=False)
    surface_atoms.calc = calc
    surface_forces = surface_atoms.get_forces()

    return np.dot(surface_forces - bulk_forces, slip_vector)


def circle_intersection_area(d: float, R: float, r: float) -> float:
    """
    Helper function to calculate the intersection area of two circles of radii R and r
    with a distance d between their centers.
    """
    if d >= R + r:
        return 0.0
    if d <= abs(R - r):
        return np.pi * min(R, r) ** 2

    d1 = (d**2 - r**2 + R**2) / (2 * d)
    d2 = d - d1

    area = (
        R**2 * np.arccos(d1 / R)
        - d1 * np.sqrt(R**2 - d1**2)
        + r**2 * np.arccos(d2 / r)
        - d2 * np.sqrt(r**2 - d2**2)
    )
    return area


def get_subsurface_force_contributions(
    relaxed_bulk: Atoms,
    net_forces: np.ndarray,
    surface_idx: list[int],
    subsurface_idx: list[int],
    custom_radii: list[float] | None = None,
) -> np.ndarray:
    """
    Helper function to calculate the contributions of subsurface atoms to the net forces
    on surface atoms.
    """
    from ase.data import covalent_radii
    from ase.geometry import get_distances

    if custom_radii is None:
        radii = [covalent_radii[z] for z in relaxed_bulk.numbers]
    else:
        radii = custom_radii

    atoms_2d = relaxed_bulk.copy()
    pos_2d = atoms_2d.positions
    pos_2d[:, 2] = 0.0
    atoms_2d.set_positions(pos_2d)

    atoms_2d.pbc = [True, True, False]

    distances, _ = get_distances(
        atoms_2d.positions[subsurface_idx],
        p2=atoms_2d.positions[surface_idx],
        cell=atoms_2d.cell,
        pbc=atoms_2d.pbc,
    )

    contributions = np.zeros(len(surface_idx))

    for i, sub_idx in enumerate(subsurface_idx):
        sub_radius = radii[sub_idx]
        sub_force = net_forces[sub_idx]

        overlap_areas = np.zeros(len(surface_idx))

        for j, top_idx in enumerate(surface_idx):
            top_radius = radii[top_idx]
            d = np.linalg.norm(distances[i, j])

            area = circle_intersection_area(d, sub_radius, top_radius)
            overlap_areas[j] = area

        total_overlap = np.sum(overlap_areas)

        if total_overlap > 0:
            weights = overlap_areas / total_overlap

            max_transfer_fraction = total_overlap / (np.pi * sub_radius**2)
            transfer_factor = min(1.0, max_transfer_fraction)

            weights = weights * transfer_factor

            for j in range(len(surface_idx)):
                if weights[j] > 0:
                    contributions[j] += weights[j] * sub_force

    return contributions
