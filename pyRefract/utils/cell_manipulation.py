import warnings

import numpy as np
from ase import Atoms
from pyscal import common_neighbor_analysis


def convert_to_lammps_coordinates(atoms: Atoms) -> Atoms:
    """
    Forces ASE to re-align cell vectors into standard LAMMPS upper-triangular
    format.
    Args:
        atoms (Atoms): ASE Atoms object to convert. The cell will be modified
        in-place.
    Returns:
        Atoms: The modified ASE Atoms object.
    """

    aligned_atoms = atoms.copy()
    cellpar = aligned_atoms.cell.cellpar()
    aligned_atoms.set_cell(cellpar, scale_atoms=True)

    return aligned_atoms


def random_ordering(
    atoms: Atoms, elements: list[str], concentrations: list[float] | np.ndarray
) -> Atoms:
    """
    Randomly assigns atom types to an ASE Atoms object.

    Args:
        atoms (Atoms): ASE Atoms object to modify.
        elements (list[str]): List of elements to assign.
        concentrations (list[float] | np.ndarray): Atomic fractions of each
            element.

    Returns:
        Atoms: The modified ASE Atoms object with randomly assigned atom types.
    """
    tatoms = len(atoms)
    natoms = [round(tatoms * c) for c in concentrations]

    # Adjust rounding errors
    diff = tatoms - sum(natoms)
    if diff != 0:
        idx = np.argmax(concentrations)
        natoms[idx] += diff

    symbols = []
    for elem, count in zip(elements, natoms):
        symbols.extend([elem] * count)

    np.random.shuffle(symbols)
    atoms.set_chemical_symbols(symbols)

    return atoms


def get_cna_classification(
    atoms: Atoms, structure_type: str = "bcc", allowed_fraction: float = 0.75
) -> bool:
    """
    Perform Common Neighbor Analysis (CNA) on an ASE Atoms object and check if
    it matches the specified structure type.

    Args:
        atoms (Atoms): ASE Atoms object to analyze.
        structure_type (str): Expected structure type for classification.
            (Default 'bcc')
        allowed_fraction (float): Minimum fraction of atoms that must match the
            expected structure type. (Default 0.75)

    Returns:
        bool: True if the CNA classification matches the specified structure
            type, False otherwise.
    """

    cna = common_neighbor_analysis(atoms)

    structure_count = cna.get(structure_type, 0)
    total_atoms = len(atoms)
    if total_atoms == 0:
        warnings.warn("The ASE Atoms object contains no atoms.")
        return False
    fraction = structure_count / total_atoms
    return fraction >= allowed_fraction
