from jax import numpy as jnp
import numpy as np
import scipy.linalg as la
from jitter import kinematics as jkm

def boost_matrix_2_2_x(xi):
    r"""
    [ B_x(\xi_x) = e^{\frac{\xi_x}{2} \sigma^1} = \begin{bmatrix} \cosh\left(\frac{\xi_x}{2}\right) & \sinh\left(\frac{\xi_x}{2}\right) \ \sinh\left(\frac{\xi_x}{2}\right) & \cosh\left(\frac{\xi_x}{2}\right) \end{bmatrix} ]
    """
    return jnp.array([[jnp.cosh(xi/2), jnp.sinh(xi/2)], 
                        [jnp.sinh(xi/2), jnp.cosh(xi/2)]])


def boost_matrix_2_2_y(xi):
    r"""
[ B_y(\xi_y) = e^{\frac{\xi_y}{2} \sigma^2} = \begin{bmatrix} \cosh\left(\frac{\xi_y}{2}\right) & -i \sinh\left(\frac{\xi_y}{2}\right) \ i \sinh\left(\frac{\xi_y}{2}\right) & \cosh\left(\frac{\xi_y}{2}\right) \end{bmatrix} ]
    """
    return jnp.array([[jnp.cosh(xi/2), -1j*jnp.sinh(xi/2)],
                     [1j*jnp.sinh(xi/2), jnp.cosh(xi/2)]])

def boost_matrix_2_2_z(xi):
    r"""
    [ B_z(\xi_z) = e^{\frac{\xi_z}{2} \sigma^3} = \begin{bmatrix} e^{\frac{\xi_z}{2}} & 0 \ 0 & e^{-\frac{\xi_z}{2}} \end{bmatrix} ]
    Args:
        xi (float): rapidity of the boost
    """
    return jnp.array([[jnp.exp(xi/2), 0], 
                        [0, jnp.exp(-xi/2)]])

def rotation_matrix_2_2_x(theta):
    r"""
     \begin{bmatrix} \cos\left(\frac{\theta_x}{2}\right) & -i \sin\left(\frac{\theta_x}{2}\right) \ -i \sin\left(\frac{\theta_x}{2}\right) & \cos\left(\frac{\theta_x}{2}\right) \end{bmatrix} ]
    Args:
        theta (float): rotation angle around x axis
    """
    return jnp.array([[jnp.cos(theta/2), -1j*jnp.sin(theta/2)], 
                    [-1j*jnp.sin(theta/2), jnp.cos(theta/2)]])

def rotation_matrix_2_2_y(theta):
    r"""
    [ R_y(\theta_y) = e^{-i \frac{\theta_y}{2} \sigma^2} = \begin{bmatrix} \cos\left(\frac{\theta_y}{2}\right) & -\sin\left(\frac{\theta_y}{2}\right) \ \sin\left(\frac{\theta_y}{2}\right) & \cos\left(\frac{\theta_y}{2}\right) \end{bmatrix} ]
    Args:
        theta (float): rotation angle around y axis
    """
    return jnp.array([[jnp.cos(theta/2), -jnp.sin(theta/2)], 
                    [jnp.sin(theta/2), jnp.cos(theta/2)]])

def rotation_matrix_2_2_z(theta):
    r"""
    [ R_z(\theta_z) = e^{-i \frac{\theta_z}{2} \sigma^3} = \begin{bmatrix} e^{-i \frac{\theta_z}{2}} & 0 \ 0 & e^{i \frac{\theta_z}{2}} \end{bmatrix} ]
    Args:
        theta (float): rotation angle around z axis
    """
    return jnp.array([[jnp.exp(-1j*theta/2), 0], 
                    [0, jnp.exp(1j*theta/2)]])

def build_2_2(theta, phi, xi, theta_rf, phi_rf, xi_rf):
    return rotation_matrix_2_2_z(theta) @ rotation_matrix_2_2_y(phi) @ boost_matrix_2_2_z(xi) @ rotation_matrix_2_2_z(theta_rf) @ rotation_matrix_2_2_y(phi_rf) @ boost_matrix_2_2_z(xi_rf)


def boost_matrix_4_4_z(xi):
    gamma = jnp.cosh(xi)
    beta_gamma = jnp.sinh(xi)
    return jnp.array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, gamma, beta_gamma,
        0, 0, beta_gamma, gamma
    ])

def rotation_matrix_4_4_x(theta):
    return jnp.array([
        1, 0, 0, 0,
        0, jnp.cos(theta), -jnp.sin(theta), 0,
        0, jnp.sin(theta), jnp.cos(theta), 0,
        0, 0, 0, 1
    ])

def rotation_matrix_4_4_y(theta):
    return jnp.array([
        jnp.cos(theta), 0, -jnp.sin(theta), 0,
        0, 1, 0, 0,
        jnp.sin(theta), 0, jnp.cos(theta), 0,
        0, 0, 0, 1
    ])

def rotation_matrix_4_4_z(theta):
    return jnp.array([
        jnp.cos(theta), -jnp.sin(theta), 0, 0,
        jnp.sin(theta), jnp.cos(theta), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])

def build_4_4(theta, phi, xi, theta_rf, phi_rf, xi_rf):
    return rotation_matrix_4_4_z(theta) @ rotation_matrix_4_4_y(phi) @ boost_matrix_4_4_z(xi) @ rotation_matrix_4_4_z(theta_rf) @ rotation_matrix_4_4_y(phi_rf) @ boost_matrix_4_4_z(xi_rf)


def decode_rotation_4x4(R):
    r"""decode a 4x4 rotation matrix into the 3 rotation angles

    Args:
        matrix (_type_): _description_
    """
    phi = jnp.arctan2(R[...,1,2], R[...,0,2])
    theta = jnp.arccos(R[...,2,2])
    psi = jnp.arctan2(R[...,2,1], -R[...,2,0])
    return theta, phi, psi

def decode_4_4(matrix):
    m = 1.0
    V0 = jnp.array([0, 0, 0, m])

    V = matrix @ V0
    w = jkm.time_component(V)
    p = jnp.sum(jkm.spatial_components(V)**2, axis=-1)**0.5
    gamma = w / m
    xi = jnp.arccosh(gamma)
    psi = jnp.arctan2(jkm.y_component(V), jkm.x_component(V))
    theta = jnp.arccos(jkm.z_component(V) / p)

    M_rf = boost_matrix_4_4_z(-xi) @ rotation_matrix_4_4_y(-psi) @ rotation_matrix_4_4_z(-theta) @ matrix
    theta_rf, phi_rf, psi_rf = decode_rotation_4x4(M_rf)

    return psi, theta, xi, phi_rf, theta_rf,  psi_rf

def gamma(p):
    r"""calculate gamma factor

    Args:
        p (_type_): momentum 4-vector
    """
    return p[...,0]/ jnp.sqrt(p[...,0]**2 - jnp.sum(p[...,1:]**2, axis=-1))

def rapidity(p):
    r"""calculate rapidity

    Args:
        p (_type_): momentum 4-vector
    """
    g = gamma(p)
    return 0.5 * jnp.log((g + 1) / (g - 1))

def decompose_sum_of_pauli_matrices(matrix):
    """decompose a matrix into a sum of pauli matrices

    Args:
        matrix (_type_): _description_
    """
    matrix = np.array(matrix)
    pauli_matrices = [np.array([[1,0],[0,1]]),
                      np.array([[0,1],[1,0]]),
                      np.array([[0,-1j],[1j,0]]),
                      np.array([[1,0],[0,-1]])]
    pauli_coefficients = []
    for pauli_matrix in pauli_matrices:
        pauli_coefficients.append(np.trace(matrix @ pauli_matrix))
    return pauli_coefficients

def reverse_rotation(rotation_matrix):
    """recover the rotation angles from a pure rotation matrix
    WARNING: This function has undefine behaviur for non pure rotation matrices

    Args:
        rotation_matrix (_type_): _description_
    """
    rotation_matrix = np.array(rotation_matrix)
    sum_of_angle_scaled_pauli_matirces = la.logm(rotation_matrix)
    pauli_coefficients = decompose_sum_of_pauli_matrices(sum_of_angle_scaled_pauli_matirces)

    return pauli_coefficients