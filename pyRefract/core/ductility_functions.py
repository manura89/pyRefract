import cmath

import numpy as np


def cos_angle(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculates the angle in degrees between two vectors.
    Args:
        A, B (np.ndarray): Vectors.
    Returns:
        float: Angle in degrees.
    """
    cos_value = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def get_elastic_tensor(C: dict[str, float], structure_type: str) -> np.ndarray:
    """
    Constructs the 6x6 elastic stiffness tensor based on crystal symmetry.

    Args:
        C: dictionary of independent elastic constants (e.g., C11, C12, C44).
        structure_type: Crystal structure type ('bcc', 'fcc', or 'hcp').

    Returns:
        np.ndarray: 6x6 elastic tensor.
    """
    if structure_type == "bcc" or structure_type == "fcc":
        return np.array(
            [
                [C["C11"], C["C12"], C["C12"], 0.0, 0.0, 0.0],
                [C["C12"], C["C11"], C["C12"], 0.0, 0.0, 0.0],
                [C["C12"], C["C12"], C["C11"], 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, C["C44"], 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, C["C44"], 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, C["C44"]],
            ]
        )
    elif structure_type == "hcp":
        return np.array(
            [
                [C["C11"], C["C12"], C["C13"], 0.0, 0.0, 0.0],
                [C["C12"], C["C11"], C["C13"], 0.0, 0.0, 0.0],
                [C["C13"], C["C13"], C["C33"], 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, C["C44"], 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, C["C44"], 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (C["C11"] - C["C12"])],
            ]
        )
    else:
        raise ValueError(
            f"Unsupported structure type: {structure_type}. Supported types are 'bcc', "
            f"'fcc', and 'hcp'."
        )


def get_KK(Q: np.ndarray) -> np.ndarray:
    """
    Calculates the transformation matrix K for the elastic tensor rotation.
    Args:
        Q (np.ndarray): 3x3 rotation matrix.
    Returns:
        np.ndarray: 6x6 transformation matrix K.
    """
    K1 = np.power(Q, 2)
    K2 = np.array(
        [
            [Q[0, 1] * Q[0, 2], Q[0, 2] * Q[0, 0], Q[0, 0] * Q[0, 1]],
            [Q[1, 1] * Q[1, 2], Q[1, 2] * Q[1, 0], Q[1, 0] * Q[1, 1]],
            [Q[2, 1] * Q[2, 2], Q[2, 2] * Q[2, 0], Q[2, 0] * Q[2, 1]],
        ]
    )

    K3 = np.array(
        [
            [Q[1, 0] * Q[2, 0], Q[1, 1] * Q[2, 1], Q[1, 2] * Q[2, 2]],
            [Q[2, 0] * Q[0, 0], Q[2, 1] * Q[0, 1], Q[2, 2] * Q[0, 2]],
            [Q[0, 0] * Q[1, 0], Q[0, 1] * Q[1, 1], Q[0, 2] * Q[1, 2]],
        ]
    )

    K4 = np.array(
        [
            [
                Q[1, 1] * Q[2, 2] + Q[1, 2] * Q[2, 1],
                Q[1, 2] * Q[2, 0] + Q[1, 0] * Q[2, 2],
                Q[1, 0] * Q[2, 1] + Q[1, 1] * Q[2, 0],
            ],
            [
                Q[2, 1] * Q[0, 2] + Q[2, 2] * Q[0, 1],
                Q[2, 2] * Q[0, 0] + Q[2, 0] * Q[0, 2],
                Q[2, 0] * Q[0, 1] + Q[2, 1] * Q[0, 0],
            ],
            [
                Q[0, 1] * Q[1, 2] + Q[0, 2] * Q[1, 1],
                Q[0, 2] * Q[1, 0] + Q[0, 0] * Q[1, 2],
                Q[0, 0] * Q[1, 1] + Q[0, 1] * Q[1, 0],
            ],
        ]
    )

    return np.block([[K1, 2 * K2], [K3, K4]])


def compliance_matrices(
    C: np.ndarray, rotation_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the rotated stiffness matrix and the reduced compliance matrix.

    Args:
        C (np.ndarray): 6x6 elastic stiffness matrix.
        rotation_matrix (np.ndarray): 3x3 rotation matrix for the crack orientation.
    Returns:
        tuple [np.ndarray, np.ndarray]: (Reduced compliance matrix A, Rotated stiffness
            matrix C_rot)
    """
    if np.shape(C) != (6, 6) or np.shape(rotation_matrix) != (3, 3):
        raise ValueError(
            "Incorrect input matrix sizes. Expected 6x6 for C and 3x3 for rotation."
        )

    if np.linalg.norm(np.eye(3) - (rotation_matrix @ rotation_matrix.T)) > 1e-7:
        raise ValueError("The rotation matrix is not orthogonal.")

    Q = rotation_matrix @ np.eye(3).T
    K = get_KK(Q)

    C_rot = K @ C @ K.T
    A1 = np.linalg.inv(C_rot)

    # Reduce matrix for generalized plane strain
    A = A1 - np.outer(A1[:, 2], A1[2, :]) / A1[2, 2]
    A = np.delete(np.delete(A, 2, axis=0), 2, axis=1)

    return A, C_rot


def aniso_roots(
    A: np.ndarray,
) -> tuple[complex, complex, complex, complex, complex, complex]:
    """
    Calculates the complex roots and parameters (s, p, q) for anisotropic fracture.
    Args:
        A (np.ndarray): 5x5 reduced compliance matrix.
    Returns:
        tuple[complex, complex, complex, complex, complex, complex]: The complex roots
            and parameters.
    """
    if np.shape(A) != (5, 5):
        raise ValueError("Incorrect size for compliance matrix. Expected 5x5.")

    S = [A[0, 0], -2 * A[0, 4], (2 * A[0, 1] + A[4, 4]), -2 * A[1, 4], A[1, 1]]
    roots = np.roots(S)

    positive_imaginary_roots = roots[roots.imag > 0.0]
    if len(positive_imaginary_roots) != 2:
        raise ValueError("Incorrect number of imaginary roots found.")

    s1, s2 = positive_imaginary_roots[0], positive_imaginary_roots[1]

    p1 = A[0, 0] * (s1**2) + A[0, 1] - A[0, 4] * s1
    p2 = A[0, 0] * (s2**2) + A[0, 1] - A[0, 4] * s2
    q1 = A[0, 1] * s1 + A[1, 1] / s1 - A[1, 4]
    q2 = A[0, 1] * s2 + A[1, 1] / s2 - A[1, 4]

    return s1, s2, p1, p2, q1, q2


def _clean_complex(arr: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    """Helper to remove floating point artifacts from complex arrays."""
    real_part = np.where(np.abs(arr.real) < tol, 0.0, arr.real)
    imag_part = np.where(np.abs(arr.imag) < tol, 0.0, arr.imag)
    return real_part + 1j * imag_part


def get_stroh(C: np.ndarray) -> np.ndarray:
    """
    Calculates the fundamental matrix N and extracts Stroh roots for elasticity.
    Args:
        C (np.ndarray): 6x6 elastic stiffness matrix.
    Returns:
        (np.ndarray): 3x3 matrix of Stroh roots (lambda).
    """
    if np.shape(C) != (6, 6):
        raise ValueError("Incorrect size for elastic matrix. Expected 6x6.")

    Q = np.array(
        [
            [C[0, 0], C[0, 5], C[0, 4]],
            [C[0, 5], C[5, 5], C[4, 5]],
            [C[0, 4], C[4, 5], C[4, 4]],
        ]
    )

    R = np.array(
        [
            [C[0, 5], C[0, 1], C[0, 3]],
            [C[5, 5], C[1, 5], C[3, 5]],
            [C[4, 5], C[1, 4], C[3, 4]],
        ]
    )

    T = np.array(
        [
            [C[5, 5], C[1, 5], C[3, 5]],
            [C[1, 5], C[1, 1], C[1, 3]],
            [C[3, 5], C[1, 3], C[3, 3]],
        ]
    )

    T_inv = np.linalg.inv(T)
    N1 = -T_inv @ R.T
    N2 = T_inv
    N3 = (R @ T_inv @ R.T) - Q

    N = np.block([[N1, N2], [N3, N1.T]])

    eigenvalues, eigenvectors = np.linalg.eig(N)
    eigenvalues = _clean_complex(eigenvalues)
    eigenvectors = _clean_complex(eigenvectors)

    positive_imaginary_root_indices = np.where(eigenvalues.imag > 0.0)[0]
    if len(positive_imaginary_root_indices) != 3:
        raise ValueError("Incorrect number of positive imaginary roots. Expected 3.")

    a_cols = eigenvectors[0:3, positive_imaginary_root_indices]
    pp = eigenvalues[positive_imaginary_root_indices]

    b1 = (R.T + pp[0] * T) @ a_cols[:, 0]
    b2 = (R.T + pp[1] * T) @ a_cols[:, 1]
    b3 = (R.T + pp[2] * T) @ a_cols[:, 2]

    AA = a_cols
    BB = np.column_stack((b1, b2, b3))

    lamda = 0.5 * np.real((AA @ np.linalg.inv(BB)) * 1j)
    return lamda


def aniso_coefficients(
    s1: complex, s2: complex, C: np.ndarray, theta: float, phi: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Calculates fracture parameters based on Stroh formalism and crack angles.
    Args:
        s1, s2 (complex): Stroh roots.
        C (np.ndarray): 6x6 elastic stiffness matrix.
        theta (float): Crack angle in degrees.
        phi (float): Crack angle in degrees.
    Returns:
        tuple containing:
        - lamda (np.ndarray): 3x3 matrix of Stroh roots.
        - sfi (np.ndarray): 3-element array related to crack orientation.
        - o (float): Scalar parameter for energy release rate.
        - F (np.ndarray): 2x2 matrix of fracture coefficients.
    """
    lamda = get_stroh(C)

    theta_radian = np.radians(theta)
    phi_radian = np.radians(phi)

    cos_theta, sin_theta = np.cos(theta_radian), np.sin(theta_radian)

    Omega = np.array(
        [[cos_theta, sin_theta, 0.0], [-sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]]
    )

    Omega2 = Omega[0:2, 0:2]

    lamdatheta = Omega @ lamda @ Omega.T
    invlamda = np.linalg.inv(lamdatheta)

    sfi = np.array([np.cos(phi_radian), 0.0, np.sin(phi_radian)])
    o = sfi @ invlamda @ sfi.T

    # Mode I & II fracture calculations
    dI = np.zeros((2, 2), dtype="complex")
    dII = np.zeros((2, 2), dtype="complex")

    dI[0, 0] = ((s1 * s2) / (s1 - s2)) * (
        s2 / cmath.sqrt(cos_theta + s2 * sin_theta)
        - s1 / cmath.sqrt(cos_theta + s1 * sin_theta)
    )
    dI[1, 1] = (1 / (s1 - s2)) * (
        s1 / cmath.sqrt(cos_theta + s2 * sin_theta)
        - s2 / cmath.sqrt(cos_theta + s1 * sin_theta)
    )
    dI[0, 1] = ((s1 * s2) / (s1 - s2)) * (
        1.0 / cmath.sqrt(cos_theta + s1 * sin_theta)
        - 1.0 / cmath.sqrt(cos_theta + s2 * sin_theta)
    )
    dI[1, 0] = dI[0, 1]

    DI = Omega2 @ np.real(dI) @ Omega2.T

    dII[0, 0] = (1 / (s1 - s2)) * (
        (s2**2) / cmath.sqrt(cos_theta + s2 * sin_theta)
        - (s1**2) / cmath.sqrt(cos_theta + s1 * sin_theta)
    )
    dII[1, 1] = (1 / (s1 - s2)) * (
        1.0 / cmath.sqrt(cos_theta + s2 * sin_theta)
        - 1.0 / cmath.sqrt(cos_theta + s1 * sin_theta)
    )
    dII[0, 1] = (1 / (s1 - s2)) * (
        s1 / cmath.sqrt(cos_theta + s1 * sin_theta)
        - s2 / cmath.sqrt(cos_theta + s2 * sin_theta)
    )
    dII[1, 0] = dII[0, 1]

    DII = Omega2 @ np.real(dII) @ Omega2.T

    F = np.array([[DII[0, 1], DI[0, 1]], [DII[1, 1], DI[1, 1]]])

    return lamda, sfi, o, F


def stress_intensity_factor(
    crack_type: str,
    lamda: np.ndarray | None = None,
    sfi: np.ndarray | None = None,
    o: float | None = None,
    F: np.ndarray | None = None,
    gamma_s: float = 0.0,
    gamma_usf: float = 0.0,
    gamma_ssf: float = 0.0,
) -> float:
    """
    Evaluates the critical stress intensity factor ($K_{IC}$ or $K_{IE}$) for cleavage
        or emission.
    Args:
        crack_type (str): Type of crack evaluation ('cleavage', 'emission_rice',
            'emission_andric', 'emission_trail').
        lamda (np.ndarray): 3x3 matrix of Stroh roots (required for cleavage).
        sfi (np.ndarray): 3-element array related to crack orientation (required for
            emission).
        o (float): Scalar parameter for energy release rate (required for emission).
        F (np.ndarray): 2x2 matrix of fracture coefficients (required for emission).
        gamma_s (float): Surface energy in J/m^2.
        gamma_usf (float): Unstable stackicing fault energy in J/m^2 (required for
            emission).
        gamma_ssf (float): Stable stackicing fault energy in J/m^2 (required for
            emission_trail).
    Returns:
        float: The calculated stress intensity factor.
    """
    missing_values = []

    # ---------------------------------------------------------
    # 1. CLEAVAGE EVALUATION
    # ---------------------------------------------------------
    if crack_type == "cleavage":
        if gamma_s <= 0.0:
            missing_values.append("gamma_s should be > 0.0")
        if lamda is None:
            missing_values.append("Missing lambda")

        if missing_values:
            raise ValueError("\n".join(missing_values))

        GIC = 2 * gamma_s
        SIF = np.sqrt(GIC / lamda[1, 1] * 1e9) * 1e-6

    # ---------------------------------------------------------
    # 2. EMISSION EVALUATION
    # ---------------------------------------------------------
    elif crack_type in ["emission_rice", "emission_andric", "emission_trail"]:
        # Base requirements for all emission types
        if gamma_usf <= 0.0:
            missing_values.append("gamma_usf should be > 0.0")

        for val, name in zip([F, o, sfi], ["F", "o", "sfi"]):
            if val is None:
                missing_values.append(f"Missing {name}")

        # Additional requirements for specific models
        if gamma_s <= 0.0:
            if crack_type in ["emission_andric", "emission_trail"]:
                missing_values.append("gamma_s should be > 0.0 (gamma_s=e_gamma_s)")
            elif crack_type == "emission_trail":
                missing_values.append("gamma_ssf should be > 0.0")

        if missing_values:
            raise ValueError("\n".join(missing_values))

        # Calculate GIE
        if crack_type == "emission_rice":
            GIE = gamma_usf
        elif crack_type == "emission_andric":
            GIE = (
                0.145 * gamma_s + 0.5 * gamma_usf
                if gamma_s >= 3.45 * gamma_usf
                else gamma_usf
            )
        elif crack_type == "emission_trail":
            GIE = (
                0.0725 * gamma_s + 0.5 * (gamma_usf - gamma_ssf)
                if gamma_s >= 6.9 * (gamma_usf - gamma_ssf)
                else gamma_usf - gamma_ssf
            )

        F12 = F[0, 1]
        KE_eff = np.sqrt(GIE * o * 1e9) * 1e-6 / sfi[0]
        SIF = abs(KE_eff) / F12

    # ---------------------------------------------------------
    # 3. UNKNOWN CRACK TYPE
    # ---------------------------------------------------------
    else:
        raise ValueError(
            f"Unknown crack_type: '{crack_type}'. Supported types are: 'cleavage', \
            'emission_rice', 'emission_andric', 'emission_trail'."
        )

    return float(SIF)


def dKIe(dtau: float, a: float, F: np.ndarray, dislocation_atoms: int) -> float:
    """
    Returns the decrease in critical stress intensity factor for emission (K_IE) in
    MPa*m^0.5 due to local stress fluctuations.

    Args:
        dtau (float): The standard deviation of local stress fluctuations in MPa.
        a (float): Lattice parameter in angstroms.
        F (np.ndarray): 2x2 matrix of angular factors relating the shear field on the
            slip planes.
        dislocation_atoms (int): Number of atoms in swept by the local dislocation loop.

    Returns:
        float: Critical stress intensity factor for emission (K_IE).
    """

    A_atom = (a**2) * np.sqrt(3 / 2) * 1e-20

    F12 = abs(F[0, 1])

    return np.power(A_atom / dislocation_atoms, 0.25) * 0.680 * dtau / F12
