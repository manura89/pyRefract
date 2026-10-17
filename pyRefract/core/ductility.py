import gzip
import pickle

import logger
import numpy as np


class Fracture_potential:
    """
    Object to calculate and store information required to calculate fracture properties
    of an alloy based on the crystallographic and slip geometries, and alloy properties.
    This class can be used to analyze the fracture behavior of materials under different
    conditions and orientations.

    """

    def __init__(
        self,
        crack_plane: str,
        emission_plane: str,
        transformation_matrix: list[list[int]],
        rotation_matrix: list[list[float]],
        crack_extension_direction: list[float],
        slip_direction: list[float],
        burgers_direction_leading: list[float],
        crack_front: str | None = None,
        burgers_direction_trailing: list[float] | None = None,
    ):
        """
        Initializes the Fracture_potential object with crystallographic and slip
        geometries.

        Args:
            crack_plane (str): Miller indices or name of the crack plane.
            crack_front (str | None): Direction of the crack front (Optional).
            transformation_matrix (list[list[int]]): Matrix to transform unit cell to
                to the orientation of the simulation structure in lattice vectors
                [[crack front], [crack extension direction], [crack plan normal]].
            rotation_matrix (list[list[float]]): Matrix to rotate standard orientation
                to crack orientation
                [[crack extension direction], [crack plan normal], [crack front]].
            emission_plane (str): Plane of dislocation emission.
            crack_extension_direction (list[float]): Direction of crack propagation.
            slip_direction (list[float]): Direction of slip on the emission plane
                (in the emission plane, normal to the crack front).
            burgers_direction_leading (list[float]): Burgers vector direction for the
                leading partial.
            burgers_direction_trailing (list[float] | None): Burgers vector direction
                for trailing partial (optional).
        """
        self.crack_plane = crack_plane
        self.crack_front = crack_front
        self.transformation_matrix = transformation_matrix
        self.rotation_matrix = np.array(rotation_matrix)
        self.emission_plane = emission_plane
        self.crack_extension_direction = crack_extension_direction
        self.slip_direction = slip_direction
        self.burgers_direction_leading = burgers_direction_leading
        self.burgers_direction_trailing = burgers_direction_trailing

    def obtain_geometry_properties(self, Cij: np.ndarray) -> None:
        """
        Calculates angles between slip/crack directions and rotates the compliance
            matrix.

        Args:
            Cij: The 6x6 elastic stiffness tensor in Voigt notation (GPa units).
        Returns:
            None. The calculated properties are stored as attributes of the object.
        """
        from .ductility_functions import compliance_matrices, cos_angle

        # Calculate the angle between the crack plane and the emission plane
        self.theta = cos_angle(self.crack_extension_direction, self.slip_direction)

        # Calculate angles between slip direction and Burgers vectors
        self.phi = [cos_angle(self.slip_direction, self.burgers_direction_leading)]
        if self.burgers_direction_trailing is not None:
            self.phi.append(
                cos_angle(self.slip_direction, self.burgers_direction_trailing)
            )
        else:
            self.phi.append(None)

        self.A, self.Cij = compliance_matrices(
            C=Cij, rotation_matrix=self.rotation_matrix
        )

    def assign_plane_properties(
        self, plane_properties: dict[str, dict[str, float]]
    ) -> None:
        """
        Assigns surface and unstable stacking fault energies based on plane
            configurations.

        Args:
            plane_properties: Nested dictionary containing energy values for respective
                planes (J/m^2 units).

        Returns:
            None. The assigned properties are stored as attributes of the object.
        """
        self.gamma_s = plane_properties[self.crack_plane]["gamma_s"]
        self.gamma_usf = plane_properties[self.emission_plane]["gamma_usf"]
        self.e_gamma_s = plane_properties[self.emission_plane]["gamma_s"]

    def assign_properties(
        self,
        elastic_parameters: dict[str, float],
        lattice_parameters: dict[str, float],
        surface_energies: dict[str, float],
        usf_energies: dict[str, float],
        slip_plane_stresses: dict[str, float] | None = None,
        structure_type: str = "bcc",
    ) -> None:
        """
        Convenience method to calculate geometry and assign properties simultaneously.
        Args:
            elastic_parameters: Dictionary containing the elastic constants (GPa units).
            lattice_parameters: Dictionary containing the lattice parameters (Å units).
            surface_energies: Dictionary containing the surface energies (mJ/m^2 units).
            usf_energies: Dictionary containing the unstable stacking fault energies
                (mJ/m^2 units).
            slip_plane_stresses: Dictionary containing the slip plane stresses
                (MPa units).
            structure_type: Crystal structure type (default is 'bcc').
        Returns:
            None. The calculated and assigned properties are stored as attributes of the
                object.
        """
        from .ductility_functions import get_elastic_tensor

        self.obtain_geometry_properties(
            Cij=get_elastic_tensor(
                elastic_parameters,
                structure_type=structure_type,
            )
        )
        plane_properties = {
            plane: {
                "gamma_s": surface_energies.get(plane, 0.0) * 1e-3,
                "gamma_usf": usf_energies.get(plane, 0.0) * 1e-3,
            }
            for plane in [self.crack_plane, self.emission_plane]
        }
        dtau = (
            slip_plane_stresses.get(self.emission_plane, 0.0)
            if slip_plane_stresses is not None
            else 0.0
        )
        a = lattice_parameters.get("a", 0.0)
        self.assign_plane_properties(plane_properties=plane_properties)
        self.dtau = dtau
        self.a = a

    def get_aniso_coefficients(self) -> None:
        """
        Calculates roots and coefficients for anisotropic linear elastic fracture
            mechanics.
        Args:
            None. This method uses the attributes of the object to perform calculations.
        Returns:
            None. The calculated roots and coefficients are stored as attributes of the
                object.
        """
        from .ductility_functions import aniso_coefficients, aniso_roots

        self.s1, self.s2, self.p1, self.p2, self.q1, self.q2 = aniso_roots(self.A)
        self.lamda, self.sfi, self.o, self.F = aniso_coefficients(
            s1=self.s1, s2=self.s2, C=self.Cij, theta=self.theta, phi=self.phi[0]
        )

    def save_as_pickle(self, filename: str = "fracture_details.pkl.gz") -> None:
        """
        Serializes and compresses the Fracture_potential object.
        Args:
            filename: Output path for the compressed pickle file.
        Returns:
            None. The object is saved to the specified file."""
        with gzip.open(filename, "wb") as pickle_f:
            pickle.dump(self, pickle_f)

    def calculate_fracture_strength(
        self,
        emission_method: str = "Andric",
    ) -> None:
        """
        Calculates the the stress intensity factors for cleavage and emission.
        Args:
            emission_method: Method used for emission calculation. Default is 'Andric'.
        Returns:
            None. The calculated fracture strength is stored as an attribute of the
                object.
        """
        from .ductility_functions import stress_intensity_factor

        self.KIc = stress_intensity_factor(
            crack_type="cleavage",
            lamda=self.lamda,
            gamma_s=self.gamma_s,
        )

        if emission_method == "Andric":
            self.KIe = stress_intensity_factor(
                crack_type="emission_andric",
                gamma_usf=self.gamma_usf,
                gamma_s=self.e_gamma_s,
                F=self.F,
                sfi=self.sfi,
                o=self.o,
            )
        elif emission_method == "Rice":
            self.KIe = stress_intensity_factor(
                crack_type="emission_rice",
                gamma_usf=self.gamma_usf,
                gamma_s=self.e_gamma_s,
                F=self.F,
                sfi=self.sfi,
                o=self.o,
            )
        else:
            raise ValueError(
                f"Unsupported emission method: {emission_method}. Supported methods \
                    are 'Andric' and 'Rice'."
            )

    def calculate_ductility_indices(
        self,
        print_calculation: bool = False,
        dislocation_atoms: int = 10,
        emission_method: str = "Andric",
        calculate_local: bool = True,
    ) -> dict:
        """
        Calculates the ductility indices based on the alloy properties and fracture
            potential.
        Args:
            dislocation_atoms: Number of atoms swept by the emitted local dislocation.
                Default is 10.
            emission_method: Method used for emission calculation. Default is 'Andric'.
        """
        from .ductility_functions import dKIe

        # Calculate anisotropic coefficients
        self.get_aniso_coefficients()
        self.calculate_fracture_strength(
            emission_method=emission_method,
        )

        self.D = self.KIe / self.KIc

        if print_calculation:
            logger.info(
                f"Calculated fracture strength for crack plane {self.crack_plane} \
                    and emission plane {self.emission_plane}:"
            )
            logger.info(f"\tKIc: {self.KIc:.4f} MPa√m")
            logger.info(f"\tKIe: {self.KIe:.4f} MPa√m")
            logger.info("Calculated ductility indices:")
            logger.info(f"\tD: {self.D:.4f}")

        if calculate_local:
            dk = dKIe(
                dtau=self.dtau,
                a=self.a,
                F=self.F,
                dislocation_atoms=dislocation_atoms,
            )

            self.KIe_local = self.KIe - dk
            self.D_local = self.KIe_local / self.KIc

            if print_calculation:
                logger.info(f"\tD_local: {self.D_local:.4f}")
