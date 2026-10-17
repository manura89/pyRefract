kokkos = [
    "-k",
    "on",
    "g",
    "1",
    "-sf",
    "kk",
    "-pk",
    "kokkos newton on neigh half",
    "-log",
    "none",
]

kokkos_no_output = [
    "-k",
    "on",
    "g",
    "1",
    "-sf",
    "kk",
    "-pk",
    "kokkos newton on neigh half",
    "-screen",
    "none",
    "-log",
    "none",
]

lammps_header = [
    "clear",
    "units metal",
    "dimension 3",
    "boundary p p p",
    "atom_style atomic",
    "atom_modify map array",
]

lammps_surface_header = [
    "clear",
    "units metal",
    "dimension 3",
    "boundary p p sf",
    "atom_style atomic",
    "atom_modify map array",
]
