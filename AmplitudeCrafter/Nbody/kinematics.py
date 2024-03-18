from jax import numpy as jnp
import numpy as np
import scipy.linalg as la
from jitter import kinematics as jkm

def boost_matrix_2_2_x(xi):
    r""" 
    Build a 2x2 boost matrix in the x-direction
    Args:
        xi (float): rapidity of the boost
    """
    return jnp.array([[jnp.cosh(xi/2), jnp.sinh(xi/2)], 
                        [jnp.sinh(xi/2), jnp.cosh(xi/2)]])

def boost_matrix_2_2_y(xi):
    r"""
    Build a 2x2 boost matrix in the y-direction
    Args:
        xi (float): rapidity of the boost
    """
    return jnp.array([[jnp.cosh(xi/2), -1j*jnp.sinh(xi/2)],
                     [1j*jnp.sinh(xi/2), jnp.cosh(xi/2)]])

def boost_matrix_2_2_z(xi):
    r"""
    Build a 2x2 boost matrix in the z-direction
    Args:
        xi (float): rapidity of the boost
    """
    return jnp.cosh(xi/2)*jnp.array([[1, 0],
                                    [0, 1]]) + jnp.sinh(xi/2)*jnp.array([[0, 1],
                                                                        [1, 0]])

def rotate_to_z_axis(v):
    """Given a vector, rotate it to the z-axis

    Args:
        v (jax.numpy.ndarray): the 4 vector to be rotated
    
    Returns:
        jax.numpy.ndarray: the rotation angles around first z and then y axis
    """ 
    v = jnp.array(v)
    psi_rf = -jnp.arctan2(jkm.y_component(v), jkm.x_component(v))
    theta_rf = jnp.arccos(jkm.z_component(v) / jkm.p(v))
    return psi_rf, -theta_rf


def rotation_matrix_2_2_x(theta):
    """Build a 2x2 rotation matrix around the x-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,2,2)
    """
    I = jnp.array([[1, 0],
                   [0, 1]])
    sgma_x = jnp.array([[0, 1],
                        [1, 0]])
    return jnp.cos(theta/2) * I - 1j*jnp.sin(theta/2)*sgma_x

def rotation_matrix_2_2_y(theta):
    """Build a 2x2 rotation matrix around the y-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,2,2)
    """
    I = jnp.array([[1, 0],
                     [0, 1]])
    sgma_y = jnp.array([[0, -1],
                        [1, 0]])
    return jnp.cos(theta/2)*I - jnp.sin(theta/2)*sgma_y

def rotation_matrix_2_2_z(theta):
    """Build a 2x2 rotation matrix around the z-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,2,2)
    """
    I = jnp.array([[1, 0], 
                   [0, 1]])
    sgma_z = jnp.array([[1,  0], 
                        [0, -1]])
    return jnp.cos(theta/2)*I - 1j*jnp.sin(theta/2)*sgma_z

def boost_matrix_4_4_z(xi):
    r"""Build a 4x4 boost matrix in the z-direction

    Args:
        xi (float): rapidity of the boost
    
    Returns:
        jax.numpy.ndarray: the 4x4 boost matrix with shape (...,4,4)
    """
    gamma = jnp.cosh(xi)
    beta_gamma = jnp.sinh(xi)
    return jnp.array([
        [1, 0, 0, 0,],
        [0, 1, 0, 0,],
        [0, 0, gamma, beta_gamma,],
        [0, 0, beta_gamma, gamma,]
    ])

def rotation_matrix_4_4_y(theta):
    """Build a 4x4 rotation matrix around the y-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,4,4)
    """
    return jnp.array([
        [jnp.cos(theta), 0, jnp.sin(theta), 0,],
        [0, 1, 0, 0,],
        [-jnp.sin(theta), 0, jnp.cos(theta), 0,],
        [0, 0, 0, 1]
    ])

def rotation_matrix_4_4_z(theta):
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta), 0, 0,],
        [jnp.sin(theta), jnp.cos(theta), 0, 0,],
        [0, 0, 1, 0,],
        [0, 0, 0, 1]
    ])

def build_2_2(psi, theta, xi, theta_rf, phi_rf, psi_rf):
    return (rotation_matrix_2_2_z(psi) @ rotation_matrix_2_2_y(theta) @ boost_matrix_2_2_z(xi) @ rotation_matrix_2_2_z(phi_rf) @ rotation_matrix_2_2_y(theta_rf) @ rotation_matrix_2_2_z(psi_rf))

def build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf):
    return (rotation_matrix_4_4_z(psi) @ rotation_matrix_4_4_y(theta) @ boost_matrix_4_4_z(xi) @ rotation_matrix_4_4_z(phi_rf) @ rotation_matrix_4_4_y(theta_rf) @ rotation_matrix_4_4_z(psi_rf))


def decode_rotation_4x4(R):
    r"""decode a 4x4 rotation matrix into the 3 rotation angles

    Args:
        matrix (_type_): _description_
    """
    phi = jnp.arctan2(R[1,2], R[0,2])
    theta = jnp.arccos(R[2,2])
    psi = jnp.arctan2(R[2,1], -R[2,0])
    return phi, theta, psi

def decode_4_4(matrix):
    m = 1.0
    V0 = jnp.array([0, 0, 0, m])

    V = matrix @ V0
    w = jkm.time_component(V)
    p = jkm.p(V)
    gamma = w / m
    xi = jnp.arccosh(gamma)

    psi = jnp.arctan2(jkm.y_component(V), jkm.x_component(V))

    cosine_input = jnp.where(abs(p) <= 1e-19, 0, jkm.z_component(V) / p)
    theta = jnp.arccos(cosine_input)

    M_rf = boost_matrix_4_4_z(-xi) @ rotation_matrix_4_4_y(-theta) @ rotation_matrix_4_4_z(-psi) @ matrix
    phi_rf, theta_rf, psi_rf = decode_rotation_4x4(M_rf[:3, :3])
    return psi, theta, xi, theta_rf, phi_rf,  psi_rf

def adjust_for_2pi_rotation(M_original_2x2, psi, theta, xi, theta_rf, phi_rf,  psi_rf):
    """Adjust the rotation angles for the 2pi rotation ambiguity

    Args:
        M_original_2x2 (jax.numpy.ndarray): the original 2x2 matrix
        psi (float): the recovered psi angle from the 4x4 matrix
        theta (float): the recovered theta angle from the 4x4 matrix
        xi (float): the recovered rapidity angle from the 4x4 matrix
        theta_rf (float): the recovered theta_rf angle from the 4x4 matrix
        phi_rf (float): the recovered phi_rf angle from the 4x4 matrix
        psi_rf (float): the recovered psi_rf angle from the 4x4 matrix
    
    Returns:
        tuple: the adjusted rotation angles
    """
    new_2x2 = build_2_2(psi, theta, xi, theta_rf, phi_rf,  psi_rf)
    if np.allclose(M_original_2x2, new_2x2):
        return psi, theta, xi, theta_rf, phi_rf,  psi_rf
    elif np.allclose(M_original_2x2, -new_2x2):
        return psi, theta, xi, theta_rf, phi_rf,  psi_rf + 2*np.pi
    else:
        raise ValueError("The matrix is not a rotation matrix")

def gamma(p):
    r"""calculate gamma factor

    Args:
        p (_type_): momentum 4-vector
    """
    return jkm.time_component(p) / jkm.mass(p)

def beta(p):	
    r"""calculate beta factor

    Args:
        p (_type_): momentum 4-vector
    """
    return jkm.p(p) / jkm.time_component(p)

def rapidity(p):
    r"""calculate rapidity

    Args:
        p (_type_): momentum 4-vector
    """
    b = beta(p)
    return 0.5 * jnp.log((b + 1) / (1 - b))