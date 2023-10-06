from AmplitudeCrafter.Nbody.kinematics import *
from jax import numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
def test_kinematics():
    m = 139.57018 # MeV
    p = jnp.array([-400., 0, 0])
    P = jnp.array([(m**2 + jnp.sum(p**2))**0.5, *p])

    print(
        boost_matrix_2_2_x(-rapidity(P)) @ boost_matrix_2_2_x(rapidity(P))
    )

    print(
        boost_matrix_2_2_y(-rapidity(P)) @ boost_matrix_2_2_y(rapidity(P))
    )

    print(
        boost_matrix_2_2_z(-rapidity(P)) @ boost_matrix_2_2_z(rapidity(P))
    )

if __name__ == "__main__":
    test_kinematics()