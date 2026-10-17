import os
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import write
from lammps import lammps
from mpi4py import MPI

from ..utils.cell_manipulation import (
    convert_to_lammps_coordinates,
    get_cna_classification,
    random_ordering,
)
from ..utils.misc import (
    rule_of_mixtures_lattice,
)
from .lmp_cmds import kokkos, kokkos_no_output, lammps_header, lammps_surface_header

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_PROC_BIND"] = "spread"
os.environ["OMP_PLACES"] = "threads"


def lammps_potential(pair_style: str, filename: str, elements: list):
    """
    Create the pair_style and pair_coeff strings for LAMMPS based on the
    potential file and elements.
    Args:
        pair_style (str): The LAMMPS pair style.
        filename (str): Path to the potential file.
        elements (list): List of element symbols involved in the potential.

    Returns:
        list: A list of LAMMPS commands for setting up the potential.
    """
    from mendeleev import element

    allowed_pair_styles = ["eam/fs", "eam/alloy", "adp", "pace", "pace product"]

    # Check if the provided pair_style is supported and raise an error if not.

    if pair_style not in allowed_pair_styles:
        raise ValueError(
            f'Pair style "{pair_style}" is not supported.'
            f"Available styles: {allowed_pair_styles}"
        )

    lammps_pot = [
        f"pair_style {pair_style}",
        f"pair_coeff * * {filename} {' '.join(elements)}",
    ]

    for i, element_symbol in enumerate(elements):
        lammps_pot.append(
            f"mass {i + 1} {element(element_symbol).atomic_weight:.3f}"
            + " # {element_symbol}"
        )

    # Add neighbor list settings for LAMMPS. Using bin style since KOKKOS does not
    # support other styles.

    lammps_pot.extend(["neighbor 2.0 bin", "neigh_modify delay 10 check yes"])

    return lammps_pot


def lammps_to_ase(lmp: lammps, elements: list) -> Atoms:
    """
    Extracts an ASE Atoms object from a running LAMMPS instance in memory.

    Args:
        lmp (lammps): The active LAMMPS instance.
        elements (list): A list mapping LAMMPS atom types (int) to chemical
            symbols (str). e.g., ['Mo', 'Nb']

    Returns:
        Atoms: The extracted ASE Atoms object.
    """
    natoms = lmp.get_natoms()

    # Simulation box
    boxlo, boxhi, xy, yz, xz, periodicity, _ = lmp.extract_box()

    cell = np.array(
        [
            [boxhi[0] - boxlo[0], 0.0, 0.0],
            [xy, boxhi[1] - boxlo[1], 0.0],
            [xz, yz, boxhi[2] - boxlo[2]],
        ]
    )

    # Atom types and positions
    lmp_types = np.array([t for t in lmp.gather_atoms("type", 0, 1)])
    symbols = [elements[t - 1] for t in lmp_types]

    coords = np.array([x for x in lmp.gather_atoms("x", 1, 3)]).reshape((natoms, 3))
    origin = np.array(boxlo)
    coords -= origin

    # ASE atoms object
    atoms = Atoms(symbols=symbols, positions=coords, cell=cell, pbc=periodicity)

    return atoms


def volume_relaxation(
    lmp: lammps,
) -> lammps:
    """
    Relax simulation cell volume and atom positions using LAMMPS.

    Args:
        lmp (lammps): LAMMPS instance.

    Returns:
        lammps: LAMMPS instance after relaxation.
    """
    atomjiggle = 1.0e-5

    lmp.commands_list(
        [
            "change_box all triclinic",
            "min_style	     cg",
            "min_modify	     dmax 1.0e-2 line quadratic",
            "thermo		100",
            "thermo_style custom step pe press pxx pyy pzz pxy pxz pyz lx ly lz",
            "thermo_modify norm no",
            "fix 1 all box/relax tri 0.0",
            f"displace_atoms all random {atomjiggle} {atomjiggle} {atomjiggle} \
                {np.random.randint(1, 100000)} units box",
            "minimize 0.0 1.0e-5 100000 10000000",
            "unfix 1",
        ]
    )

    return lmp


def gsfe_relaxation(
    lmp: lammps,
) -> lammps:
    """
    Relax simulation cell and atom positions in the z-direction using LAMMPS.

    Args:
        lmp (lammps): LAMMPS instance.

    Returns:
        lammps: LAMMPS instance after relaxation.
    """

    lmp.commands_list(
        [
            "change_box all triclinic",
            "min_style	     cg",
            "min_modify	     dmax 1.0e-2 line quadratic",
            "thermo		100",
            "thermo_style custom step pe press pxx pyy pzz pxy pxz pyz lx ly lz",
            "thermo_modify norm no",
            "fix 1 all box/relax z 0.0",
            "fix 2 all setforce 0.0 0.0 NULL",
            "minimize 0.0 1.0e-5 100000 10000000",
            "unfix 1",
        ]
    )

    return lmp


def position_relaxation(
    lmp: lammps,
) -> lammps:
    """
    Relax atom positions with a fixed simulation cell using LAMMPS.

    Args:
        lmp (lammps): LAMMPS instance.

    Returns:
        lammps: LAMMPS instance after relaxation.
    """
    atomjiggle = 1.0e-5
    lmp.commands_list(
        [
            "change_box all triclinic",
            "min_style	     cg",
            "min_modify	     dmax 1.0e-2 line quadratic",
            "thermo		100",
            "thermo_style custom step pe press pxx pyy pzz pxy pxz pyz lx ly lz",
            "thermo_modify norm no",
            f"displace_atoms all random {atomjiggle} {atomjiggle} {atomjiggle} \
                {np.random.randint(1, 100000)} units box",
            "minimize 0.0 1.0e-5 100000 10000000",
        ]
    )

    return lmp


def get_deformed_pressures(
    atoms: Atoms,
    strain: float,
    direc: int,
    len0: float,
    lmp_initialize_cmds: list[str],
    elements: list[str],
    potential_cmd: list[str],
) -> tuple[float, float, float, float, float, float, float]:
    """
    Get the deformed pressures for a given LAMMPS instance and deformation
    parameters.

    Args:
        atoms: ASE Atoms object.
        strain: Strain value.
        direc: Direction of deformation.
        len0: Original length in the direction of deformation.
        lmp_initialize_cmds: List of LAMMPS initialization commands.
        elements: List of element symbols.
        potential_cmd: List of LAMMPS potential commands.
    Returns:
        tuple[float, float, float, float, float, float, float]: Deformed
        pressures.
    """
    comm = MPI.COMM_WORLD

    deformed_atoms = atoms.copy()
    cell = deformed_atoms.get_cell()

    xy = cell[1, 0]
    xz = cell[2, 0]
    yz = cell[2, 1]

    delta = strain * len0
    deltaxy = strain * xy
    deltaxz = strain * xz
    deltayz = strain * yz

    if direc == 1:
        cell[0, 0] += delta
        cell[1, 0] += deltaxy
        cell[2, 0] += deltaxz
    elif direc == 2:
        cell[1, 1] += delta
        cell[2, 1] += deltayz
    elif direc == 3:
        cell[2, 2] += delta
    elif direc == 4:
        cell[2, 1] += delta
    elif direc == 5:
        cell[2, 0] += delta
    elif direc == 6:
        cell[1, 0] += delta

    deformed_atoms.set_cell(cell, scale_atoms=True)

    lmp = lammps(name="MPI", cmdargs=lmp_initialize_cmds, comm=comm)
    lmp.commands_list(lammps_header)
    with tempfile.NamedTemporaryFile(suffix=".data", delete=True) as tmp:
        write(
            tmp.name,
            deformed_atoms,
            format="lammps-data",
            atom_style="atomic",
            specorder=elements,
        )

        lmp.commands_string(f"read_data {tmp.name}")
    lmp.commands_list(potential_cmd)
    lmp = position_relaxation(lmp)

    lmp.commands_list(
        [
            "variable pxx1 equal pxx",
            "variable pyy1 equal pyy",
            "variable pzz1 equal pzz",
            "variable pyz1 equal pyz",
            "variable pxz1 equal pxz",
            "variable pxy1 equal pxy",
        ]
    )

    variables = [
        lmp.extract_variable(v, None)
        for v in ["pxx1", "pyy1", "pzz1", "pyz1", "pxz1", "pxy1"]
    ]

    lmp.close()

    return variables


def lammps_elastic_properties(
    elements: list[str],
    concentrations: list[float],
    transformation_matrix: list[list[float]] | np.ndarray,
    offset: list[float] | np.ndarray,
    potential_cmd: list[str],
    nimages: int,
    structure_type: str = "bcc",
    lmp_initialize_cmds: list[str] = kokkos_no_output,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str | None = None,
    complete_lammps: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Calculate elastic properties using LAMMPS for a given set of elements and
    concentrations.

    Args:
        elements (list[str]): List of chemical elements.
        concentrations (list[float]): List of concentrations corresponding to
        the elements.
        transformation_matrix (list[list[float]] | np.ndarray): Transformation
        matrix for the supercell.
        offset (list[float] | np.ndarray): Initial translation offset for the
        supercell.
        potential_cmd (list[str]): LAMMPS commands for the potential.
        nimages (int): Number of images to generate for the calculations.
        structure_type (str, optional): Crystal structure type. Defaults to
        'bcc'. lmp_initialize_cmds (list[str], optional): LAMMPS initialization
        commands. Defaults to kokkos_no_output.

    Returns:
        tuple[dict[str, float], dict[str, float]]: A tuple containing two
        dictionaries. The first dictionary contains the calculated elastic
        properties (C11, C12, C44), and the second dictionary contains the
        lattice parameters (a).
    """
    comm = MPI.COMM_WORLD

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

    prim = bulk(name=elements[0], crystalstructure=structure_type, a=a)
    if structure_type in ["fcc", "bcc"]:
        image_elastic_parameters = {C: [] for C in ["C11", "C12", "C44"]}
        image_lattice_parameters = {"a": []}
    elif structure_type == "hcp":
        image_elastic_parameters = {C: [] for C in ["C11", "C12", "C13", "C33", "C44"]}
        image_lattice_parameters = {"a": [], "c": []}
    else:
        raise ValueError(f"Unsupported structure type: {structure_type}")

    base_atoms = make_supercell(prim, transformation_matrix)
    base_atoms.translate(a * np.array(offset))
    base_atoms = convert_to_lammps_coordinates(base_atoms)

    cfac = 1.0e-4
    strain = 1e-6

    for n in range(1, nimages + 1):
        lmp = lammps(name="MPI", cmdargs=lmp_initialize_cmds, comm=comm)
        lmp.commands_list(lammps_header)

        atoms = random_ordering(base_atoms.copy(), elements, concentrations)

        with tempfile.NamedTemporaryFile(suffix=".data", delete=True) as tmp:
            write(
                tmp.name,
                atoms,
                format="lammps-data",
                atom_style="atomic",
                specorder=elements,
            )

            lmp.commands_string(f"read_data {tmp.name}")
        lmp.commands_list(potential_cmd)

        # Volume relaxation for the current image.

        lmp = volume_relaxation(lmp)

        lmp.commands_list(
            [
                "variable pxx0 equal pxx",
                "variable pyy0 equal pyy",
                "variable pzz0 equal pzz",
                "variable pyz0 equal pyz",
                "variable pxz0 equal pxz",
                "variable pxy0 equal pxy",
            ]
        )

        # The relaxed pressures for the current image

        relaxed_atoms = lammps_to_ase(lmp, elements)

        if save_relaxed_structure:
            save_file = save_dir / f"lammps_elastic_{alloy_name}_{n}.vasp"
            write(save_file, relaxed_atoms, format="vasp", sort=True, direct=True)

        if check_cna:
            cna_classification = get_cna_classification(relaxed_atoms)
            if not cna_classification:
                continue

        pxx0 = lmp.extract_variable("pxx0", None)
        pyy0 = lmp.extract_variable("pyy0", None)
        pzz0 = lmp.extract_variable("pzz0", None)
        pyz0 = lmp.extract_variable("pyz0", None)
        pxz0 = lmp.extract_variable("pxz0", None)
        pxy0 = lmp.extract_variable("pxy0", None)

        lmp.close()

        # Elastic tensor as a 6x6 matrix

        elastic_tensor = np.zeros((6, 6))

        cell = relaxed_atoms.get_cell()
        lx = cell[0, 0]
        ly = cell[1, 1]
        lz = cell[2, 2]

        for direc in range(1, 7):
            if direc == 1:
                len0 = lx
            elif direc == 2 or direc == 6:
                len0 = ly
            else:
                len0 = lz

            C = {}
            pxx1, pyy1, pzz1, pyz1, pxz1, pxy1 = get_deformed_pressures(
                relaxed_atoms,
                -strain,
                direc,
                len0,
                lmp_initialize_cmds,
                elements,
                potential_cmd,
            )

            delta = -strain * len0

            C["1neg"] = -(pxx1 - pxx0) / (delta / len0) * cfac
            C["2neg"] = -(pyy1 - pyy0) / (delta / len0) * cfac
            C["3neg"] = -(pzz1 - pzz0) / (delta / len0) * cfac
            C["4neg"] = -(pyz1 - pyz0) / (delta / len0) * cfac
            C["5neg"] = -(pxz1 - pxz0) / (delta / len0) * cfac
            C["6neg"] = -(pxy1 - pxy0) / (delta / len0) * cfac

            pxx1, pyy1, pzz1, pyz1, pxz1, pxy1 = get_deformed_pressures(
                relaxed_atoms,
                strain,
                direc,
                len0,
                lmp_initialize_cmds,
                elements,
                potential_cmd,
            )

            delta = strain * len0

            C["1pos"] = -(pxx1 - pxx0) / (delta / len0) * cfac
            C["2pos"] = -(pyy1 - pyy0) / (delta / len0) * cfac
            C["3pos"] = -(pzz1 - pzz0) / (delta / len0) * cfac
            C["4pos"] = -(pyz1 - pyz0) / (delta / len0) * cfac
            C["5pos"] = -(pxz1 - pxz0) / (delta / len0) * cfac
            C["6pos"] = -(pxy1 - pxy0) / (delta / len0) * cfac

            for i in range(6):
                elastic_tensor[direc - 1, i] = (C[f"{i + 1}pos"] + C[f"{i + 1}neg"]) / 2

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
    # Average elastic and lattice parameters across all images.

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

    if complete_lammps:
        lmp.finalize()

    return lattice_parameters, elastic_parameters
