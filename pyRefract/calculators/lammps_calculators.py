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
from .lmp_cmds import kokkos_no_output, lammps_header

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

    with lammps(name="MPI", cmdargs=lmp_initialize_cmds, comm=comm) as lmp:
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
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Calculate elastic properties using LAMMPS for a given set of elements and
    concentrations.

    Args:
        elements (list[str]): List of elements in the alloy.
        concentrations (list[float]): Concentrations (atomic) of the elements.
        transformation_matrix (list[list[float]] | np.ndarray): Transformation
            matrix for the structure.
        offset (list[float] | np.ndarray): Offset of the atomic positions (in
            fractions of the lattice parameter) to avoid atoms moving across the
            simulation box.
        potential_cmd (list[str]): LAMMPS commands for the potential.
        nimages (int): Number of images for the calculation.
        structure_type (str, optional): Type of crystal structure. Defaults to
            "bcc".
        lmp_initialize_cmds (list[str], optional): LAMMPS initialization
            commands. Defaults to kokkos_no_output.
        check_cna (bool, optional): Whether to check CNA. Defaults to True.
        save_relaxed_structure (bool, optional): Whether to save the relaxed
            structure. Defaults to False.
        save_path (str | None, optional): Path to save the relaxed structure.
            Defaults to None.
        complete_lammps (bool, optional): To run the finalize LAMMPS. True if
            no other LAMMPS calculations are needed. Defaults to True.

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
        with lammps(name="MPI", cmdargs=lmp_initialize_cmds, comm=comm) as lmp:
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


def lammps_surface_energy(
    elements: list[str],
    concentrations: list[float] | np.ndarray,
    transformation_matrix: list[list[int]] | np.ndarray,
    plane: str,
    offset: list[float] | np.ndarray,
    potential_cmd: list[str],
    structure_type: str = "bcc",
    lmp_initialize_cmds: list[str] = kokkos_no_output,
    nimages: int = 3,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str | None = None,
) -> tuple[float, float]:
    """
    Calculate the surface energy for a given set of elements and concentrations
    using LAMMPS.

    Args:
        elements (list[str]): List of element symbols.
        concentrations (list[float] | np.ndarray): List or array of element
            concentrations.
        transformation_matrix (list[list[float]] | np.ndarray): Transformation
            matrix for supercell construction.
        plane (str): Surface plane for which the surface energy is being
            calculated.
        offset (list[float] | np.ndarray): Offset for supercell construction.
        potential_cmd (list[str]): LAMMPS commands for the potential.
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
        complete_lammps (bool, optional): To run the finalize LAMMPS. True if
            no other LAMMPS calculations are needed. Defaults to True.

    Returns:
        tuple[float, float]: A tuple containing the mean surface energy, and the
        standard deviation.
    """
    comm = MPI.COMM_WORLD
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
    image_surface_energy = []

    base_atoms = make_supercell(prim, transformation_matrix)
    base_atoms.translate(a * np.array(offset))
    base_atoms = convert_to_lammps_coordinates(base_atoms)

    for n in range(1, nimages + 1):
        with lammps(name="MPI", cmdargs=lmp_initialize_cmds, comm=comm) as lmp:
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
            lmp.commands_list(
                [
                    "compute peratom all pe/atom",
                    "compute eatoms all reduce sum c_peratom",
                ]
            )

            lmp = volume_relaxation(lmp)

            relaxed_atoms = lammps_to_ase(lmp, elements)

            if save_relaxed_structure:
                save_file = (
                    save_dir / f"lammps_surface-bulk_{plane}_{alloy_name}_{n}.vasp"
                )
                write(save_file, relaxed_atoms, format="vasp", sort=True, direct=True)

            if check_cna:
                cna_classification = get_cna_classification(relaxed_atoms)
                if not cna_classification:
                    continue

            lmp.commands_list(["variable lx0 equal lx", "variable ly0 equal ly"])

            lx = lmp.extract_variable("lx0", None)
            ly = lmp.extract_variable("ly0", None)
            e0 = lmp.extract_compute("eatoms", 0, 0)

            lmp.commands_string("change_box all z delta 0.0 10.0 units box")

            lmp = position_relaxation(lmp)

            ef = lmp.extract_compute("eatoms", 0, 0)

            image_surface_energy.append((ef - e0) * SF / (2 * lx * ly))

            surface_atoms = lammps_to_ase(lmp, elements)
            if save_relaxed_structure:
                save_file = save_dir / f"lammps_surface_{plane}_{alloy_name}_{n}.vasp"
                write(save_file, surface_atoms, format="vasp", sort=True, direct=True)

    if not image_surface_energy:
        print("No valid images passed the CNA classification.")
        return None

    return (np.mean(image_surface_energy), np.std(image_surface_energy))


def lammps_usf_energy(
    elements: list[str],
    concentrations: list[float] | np.ndarray,
    transformation_matrix: list[list[int]] | np.ndarray,
    plane: str,
    slip_vector: list[float] | np.ndarray,
    usf_steps: int,
    offset: list[float] | np.ndarray,
    potential_cmd: list[str],
    structure_type: str = "bcc",
    lmp_initialize_cmds: list[str] = kokkos_no_output,
    nimages: int = 3,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str | None = None,
) -> tuple[float, float]:
    """
    Calculate the unstable stacking fault (USF) energy for a given set of elements and
    concentrations using LAMMPS.

    Args:
        elements (list[str]): List of element symbols.
        concentrations (list[float] | np.ndarray): List or array of element
            concentrations.
        transformation_matrix (list[list[float]] | np.ndarray): Transformation
            matrix for supercell construction.
        plane (str): Slip plane for which the USF energy is being calculated.
        slip_vector (list[float] | np.ndarray): Slip vector for the USF calculation.
            This defines the direction and magnitude of the total slip applied to the
            crystal.
        usf_steps (int): Number of steps in the GSFE path.
        offset (list[float] | np.ndarray): Offset for supercell construction.
        potential_cmd (list[str]): LAMMPS commands for the potential.
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
        complete_lammps (bool, optional): To run the finalize LAMMPS. True if
            no other LAMMPS calculations are needed. Defaults to True.

    Returns:
        tuple[float, float]: A tuple containing the mean USF energy, and the
        standard deviation.
    """
    comm = MPI.COMM_WORLD
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
    image_usf_energy = []

    base_atoms = make_supercell(prim, transformation_matrix)
    base_atoms.translate(a * np.array(offset))
    base_atoms = convert_to_lammps_coordinates(base_atoms)

    for n in range(nimages):
        with lammps(name="MPI", cmdargs=lmp_initialize_cmds, comm=comm) as lmp:
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
            lmp.commands_list(
                [
                    "compute peratom all pe/atom",
                    "compute eatoms all reduce sum c_peratom",
                ]
            )

            lmp = volume_relaxation(lmp)

            relaxed_atoms = lammps_to_ase(lmp, elements)

            if save_relaxed_structure:
                save_file = save_dir / f"lammps_usf-bulk_{plane}_{alloy_name}_{n}.vasp"
                write(save_file, relaxed_atoms, format="vasp", sort=True, direct=True)

            if check_cna:
                cna_classification = get_cna_classification(relaxed_atoms)
                if not cna_classification:
                    continue

            lmp.commands_list(
                [
                    "variable lx0 equal lx",
                    "variable ly0 equal ly",
                    "variable xy0 equal xy",
                ]
            )

            lx = lmp.extract_variable("lx0", None)
            ly = lmp.extract_variable("ly0", None)
            xy = lmp.extract_variable("xy0", None)
            energy = [lmp.extract_compute("eatoms", 0, 0)]

            total_slip = (
                np.array([lx, 0, 0]) * slip_vector[0]
                + np.array([xy, ly, 0]) * slip_vector[1]
            )
            slip_per_step = total_slip / usf_steps

            for _ in range(usf_steps):
                lmp.commands_string(
                    f"change_box all xz delta {slip_per_step[0]} yz delta \
                        {slip_per_step[1]} units box"
                )
                lmp = gsfe_relaxation(lmp)
                energy.append(lmp.extract_compute("eatoms", 0, 0))

            sfe = (np.array(energy) - energy[0]) * SF / (lx * ly)

            usf_atoms = lammps_to_ase(lmp, elements)
            if save_relaxed_structure:
                save_file = save_dir / f"lammps_usf_{plane}_{alloy_name}_{n}.vasp"
                write(save_file, usf_atoms, format="vasp", sort=True, direct=True)

            image_usf_energy.append(np.max(sfe))

    if not image_usf_energy:
        print("No valid images passed the CNA classification.")
        return None

    return (np.mean(image_usf_energy), np.std(image_usf_energy))


def lammps_slip_plane_stress(
    elements: list[str],
    concentrations: list[float] | np.ndarray,
    transformation_matrix: list[list[int]] | np.ndarray,
    plane: str,
    slip_vector: list[float] | np.ndarray,
    offset: list[float] | np.ndarray,
    potential_cmd: list[str],
    calc,
    structure_type: str = "bcc",
    lmp_initialize_cmds: list[str] = kokkos_no_output,
    nimages: int = 3,
    check_cna: bool = True,
    save_relaxed_structure: bool = False,
    save_path: str | None = None,
) -> tuple[float, float]:
    """
    Calculate the standard deviation of the slip plane stresses for a given set of
    elements and concentrations using LAMMPS.

    Args:
        elements (list[str]): List of element symbols.
        concentrations (list[float] | np.ndarray): List or array of element
            concentrations.
        transformation_matrix (list[list[int]] | np.ndarray): Transformation
            matrix for supercell construction.
        plane (str): Slip plane for which the slip plane stresses are being calculated.
        slip_vector (list[float] | np.ndarray): Slip vector for the slip plane stress
            calculation. This is the unit vector indicating the direction of the total
            slip of the Burgers' vector.
        offset (list[float] | np.ndarray): Offset for supercell construction.
        potential_cmd (list[str]): LAMMPS commands for the potential.
        calc: ASE calculator object.
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
        complete_lammps (bool, optional): To run the finalize LAMMPS. True if
            no other LAMMPS calculations are needed. Defaults to True.

    Returns:
        float: The standard deviation of the slip plane stresses.
    """
    from ..utils.misc import count_unique_within_tolerance
    from .ase_calculators import surface_stresses

    comm = MPI.COMM_WORLD
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
    image_stresses = []

    base_atoms = make_supercell(prim, transformation_matrix)
    base_atoms.translate(a * np.array(offset))
    base_atoms = convert_to_lammps_coordinates(base_atoms)

    atompos = base_atoms.get_positions()
    tatoms = len(base_atoms)

    nz = count_unique_within_tolerance(atompos[:, 2], tol=0.25)
    layer_atoms = tatoms // nz

    for n in range(nimages):
        with lammps(name="MPI", cmdargs=lmp_initialize_cmds, comm=comm) as lmp:
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
            lmp.commands_list(
                [
                    "compute peratom all pe/atom",
                    "compute eatoms all reduce sum c_peratom",
                ]
            )

            lmp = volume_relaxation(lmp)

            relaxed_atoms = lammps_to_ase(lmp, elements)

            if save_relaxed_structure:
                save_file = save_dir / f"lammps_usf-bulk_{plane}_{alloy_name}_{n}.vasp"
                write(save_file, relaxed_atoms, format="vasp", sort=True, direct=True)

            if check_cna:
                cna_classification = get_cna_classification(relaxed_atoms)
                if not cna_classification:
                    continue

            image_slip_plane_stresses = surface_stresses(
                relaxed_atoms, slip_vector, layer_atoms, calc
            )

            image_stresses.extend(image_slip_plane_stresses.tolist())

    if not image_stresses:
        return None

    return np.std(image_stresses)
