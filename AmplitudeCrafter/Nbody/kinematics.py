from jax import numpy as jnp

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

def reverse_rotation(rotation_matrix):
    """recover the rotation angles from a pure rotation matrix
    WARNING: This function has undefine behaviur for non pure rotation matrices

    Args:
        rotation_matrix (_type_): _description_
    """
    