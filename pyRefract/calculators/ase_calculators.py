from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import write

from ..utils.cell_manipulation import (
    convert_to_lammps_coordinates,
    get_cna_classification,
    random_ordering,
)
from ..utils.misc import rule_of_mixtures_lattice


def volume_relaxation(
    atoms: Atoms,
    calc: any,
    atom_constraints=None,
    cell_mask: list[bool] | None = None,
) -> Atoms:
    """
    Helper function to perform volume relaxation of a structure using ASE
    optimizers.
    """

    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from ase.optimize.sciopt import SciPyFminCG

    if cell_mask is None:
        cell_mask = [True, True, True, True, True, True]

    if calc is None:
        raise ValueError("A calculator must be provided for volume relaxation.")

    atoms.calc = calc
    atoms.set_constraint(atom_constraints)

    relaxed_cell = FrechetCellFilter(atoms, mask=cell_mask)

    try:
        minimize = SciPyFminCG(relaxed_cell, trajectory=None, logfile=None)
        minimize.run(fmax=1e-5)
    except Exception:
        try:
            minimize = BFGS(relaxed_cell, trajectory=None, logfile=None)
            minimize.run(fmax=1e-5)
        except Exception as e:
            print(f"Error during BFGS relaxation: {e}")

    return atoms


def atom_relaxation(
    atoms: Atoms,
    calc,
    atom_constraints=None,
) -> Atoms:
    """
    Helper function to perform atom relaxation of a structure using ASE
    optimizers.
    """

    from ase.optimize import BFGS
    from ase.optimize.sciopt import SciPyFminCG

    if calc is None:
        raise ValueError("A calculator must be provided for atom relaxation.")

    atoms.calc = calc
    atoms.set_constraint(atom_constraints)

    try:
        minimize = SciPyFminCG(atoms, trajectory=None, logfile=None)
        minimize.run(fmax=1e-5)
    except Exception:
        try:
            minimize = BFGS(atoms, trajectory=None, logfile=None)
            minimize.run(fmax=1e-5)
        except Exception as e:
            print(f"Error during BFGS relaxation: {e}")

    return atoms


def ase_elastic_properties(
    elements: list[str],
    concentrations: list[float],
    transformation_matrix: list[list[float]] | np.ndarray,
    offset: list[float] | np.ndarray,
    calc,
    nimages: int,
    structure_type: str = "bcc",
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Calculate and return the elastic parameters and lattice parameters for the a
    given alloy composition using the ASE calculator.
    Args:
        elements (list[str]): List of elements in the alloy.
        concentrations (list[float]): Concentrations (atomic) of the elements.
        transformation_matrix (list[list[float]] | np.ndarray): Transformation
            matrix for the structure.
        offset (list[float] | np.ndarray): Offset of the atomic positions (in
            fractions of the lattice parameter) to avoid atoms moving across the
            simulation box.
        calc: ASE calculator object.
        nimages (int): Number of images for the calculation.
        structure_type (str, optional): Type of crystal structure. Defaults to
            "bcc".
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed
            structure. Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure.
            Defaults to None.

    Returns:
    tuple[dict[str, float], dict[str, float]]: A tuple containing the elastic
        parameters and lattice parameters and the standard deviation of each
        parameter.
    """
    from matscipy.elasticity import fit_elastic_constants

    concentrations = np.array(concentrations) / np.sum(concentrations)

    a = rule_of_mixtures_lattice(
        elements,
        concentrations,
        element_values=None,
    )

    if save_relaxed_structure and save_path is None:
        save_path = "relaxed_structures"
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
    elif save_relaxed_structure and save_path is not None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    alloy_name = "".join(
        f"{el}{int(conc * 100)}" for el, conc in zip(elements, concentrations)
    )

    if structure_type in ["fcc", "bcc"]:
        image_elastic_parameters = {C: [] for C in ["C11", "C12", "C44"]}
        image_lattice_parameters = {"a": []}
    elif structure_type == "hcp":
        image_elastic_parameters = {C: [] for C in ["C11", "C12", "C13", "C33", "C44"]}
        image_lattice_parameters = {"a": [], "c": []}
    else:
        raise ValueError(f"Unsupported structure type: {structure_type}")

    transformation_matrix = np.array(transformation_matrix)

    prim = bulk(name=elements[0], crystalstructure=structure_type, a=a)
    base_atoms = make_supercell(prim, transformation_matrix)
    base_atoms.translate(a * np.array(offset))

    for n in range(1, nimages + 1):
        atoms = random_ordering(base_atoms.copy(), elements, concentrations)
        relaxed_atoms = volume_relaxation(atoms, calc=calc)
        relaxed_atoms.calc = calc

        if save_relaxed_structure:
            save_file = save_dir / f"ase_elastic_{alloy_name}_{n}.vasp"
            write(save_file, relaxed_atoms, format="vasp", sort=True, direct=True)

        if check_cna:
            cna_classification = get_cna_classification(
                relaxed_atoms, structure_type=structure_type
            )
            if not cna_classification:
                continue

        # Fit the elastic constants
        elastic_tensor, _ = fit_elastic_constants(relaxed_atoms, verbose=False)

        elastic_tensor *= 160.218

        if structure_type in ["fcc", "bcc"]:
            if structure_type == "fcc":
                unit_atoms = 4
            elif structure_type == "bcc":
                unit_atoms = 2
            image_elastic_parameters["C11"].extend(
                [elastic_tensor[0, 0], elastic_tensor[1, 1], elastic_tensor[2, 2]]
            )
            image_elastic_parameters["C12"].extend(
                [elastic_tensor[1, 0], elastic_tensor[2, 0], elastic_tensor[2, 1]]
            )
            image_elastic_parameters["C44"].extend(
                [elastic_tensor[3, 3], elastic_tensor[4, 4], elastic_tensor[5, 5]]
            )
            image_lattice_parameters["a"].append(
                (unit_atoms * relaxed_atoms.get_volume() / len(relaxed_atoms))
                ** (1 / 3)
            )
        elif structure_type == "hcp":
            image_elastic_parameters["C11"].extend(
                [elastic_tensor[0, 0], elastic_tensor[1, 1]]
            )
            image_elastic_parameters["C12"].extend([elastic_tensor[1, 0]])
            image_elastic_parameters["C13"].extend(
                [elastic_tensor[2, 0], elastic_tensor[2, 1]]
            )
            image_elastic_parameters["C33"].extend([elastic_tensor[2, 2]])
            image_elastic_parameters["C44"].extend(
                [elastic_tensor[3, 3], elastic_tensor[4, 4]]
            )
            cell_par = relaxed_atoms.cell.cellpar()
            # Lattice parameters are accurate if the transformation matrix is
            # diagonal.
            image_lattice_parameters["a"].extend(
                [
                    cell_par[0] / transformation_matrix[0, 0],
                    cell_par[1] / transformation_matrix[1, 1],
                ]
            )
            image_lattice_parameters["c"].append(
                cell_par[2] / transformation_matrix[2, 2]
            )

    if not image_elastic_parameters["C11"]:
        print("No valid images passed the CNA classification.")
        return None, None

    elastic_parameters = {
        C: (np.mean(values), np.std(values))
        for C, values in image_elastic_parameters.items()
    }
    lattice_parameters = {
        param: (np.mean(values), np.std(values))
        for param, values in image_lattice_parameters.items()
    }

    return lattice_parameters, elastic_parameters


def ase_surface_energy(
    elements: list[str],
    concentrations: list[float],
    transformation_matrix: list[list[float]] | np.ndarray,
    plane: str,
    offset: list[float] | np.ndarray,
    calc,
    structure_type: str = "bcc",
    nimages: int = 3,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str | None = None,
) -> tuple[float, float]:
    """
    Calculate the surface energy for a given set of elements and concentrations
    using ASE.

    Args:
        elements (list[str]): List of element symbols.
        concentrations (list[float] | np.ndarray): List or array of element
            concentrations.
        transformation_matrix (list[list[float]] | np.ndarray): Transformation
            matrix for supercell construction.
        plane (str): Surface plane for which the surface energy is being
            calculated.
        offset (list[float] | np.ndarray): Offset for supercell construction.
        calc: Calculator object.
        structure_type (str, optional): Crystal structure type (default is
            'bcc').
        nimages (int, optional): Number of images for the calculation (default
            is 3).
        check_cna (bool, optional): Whether to check CNA classification (default
        is True).
        save_relaxed_structure (bool, optional): Whether to save the relaxed
            structure (default is False).
        save_path (str | None, optional): Path to save the relaxed structure if
            save_relaxed_structure is True (default is None).
    Returns:
        tuple[float, float]: The calculated surface energy and its standard
            deviation as a tuple.
    """
    SF = 1.602176634e4
    a = rule_of_mixtures_lattice(
        elements,
        concentrations,
        element_values=None,
    )

    if save_relaxed_structure and save_path is None:
        save_path = "relaxed_structures"
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
    elif save_relaxed_structure and save_path is not None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    alloy_name = "".join(
        f"{el}{int(conc * 100)}" for el, conc in zip(elements, concentrations)
    )

    prim = bulk(name=elements[0], crystalstructure=structure_type, a=a)
    image_surface_energies = []

    base_atoms = make_supercell(prim, transformation_matrix)
    base_atoms.translate(a * np.array(offset))
    base_atoms = convert_to_lammps_coordinates(base_atoms)

    for n in range(1, nimages + 1):
        atoms = random_ordering(base_atoms.copy(), elements, concentrations)
        relaxed_atoms = volume_relaxation(atoms, calc=calc)
        relaxed_atoms = convert_to_lammps_coordinates(relaxed_atoms)
        relaxed_atoms.calc = calc

        if save_relaxed_structure:
            save_file = save_dir / f"ase_surface-bulk_{plane}_{alloy_name}_{n}.vasp"
            write(save_file, relaxed_atoms, format="vasp", sort=True, direct=True)

        if check_cna:
            cna_classification = get_cna_classification(
                relaxed_atoms, structure_type=structure_type
            )
            if not cna_classification:
                continue

        cell = relaxed_atoms.get_cell()
        energy_0 = relaxed_atoms.get_potential_energy()

        area = 2 * np.linalg.norm(np.cross(cell[0], cell[1]))

        surface_atoms = relaxed_atoms.copy()
        surface_cell = surface_atoms.get_cell()
        surface_cell[2, 2] += 10.0

        surface_atoms.set_cell(surface_cell, scale_atoms=False)
        surface_atoms = atom_relaxation(surface_atoms, calc=calc)

        if save_relaxed_structure:
            save_file = save_dir / f"ase_surface_{plane}_{alloy_name}_{n}.vasp"
            write(save_file, surface_atoms, format="vasp", sort=True, direct=True)

        energy_f = surface_atoms.get_potential_energy()

        image_surface_energies.append((energy_f - energy_0) * SF / area)

    if not image_surface_energies:
        print("No valid images passed the CNA classification.")
        return None

    return (np.mean(image_surface_energies), np.std(image_surface_energies))
