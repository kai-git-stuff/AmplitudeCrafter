from AmplitudeCrafter.Nbody.kinematics import *
from jax import numpy as jnp
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
def test_kinematics():
    m = 139.57018 # MeV
    p = jnp.array([-400., 0, 0])
    P = jnp.array([(m**2 + jnp.sum(p**2))**0.5, *p])

    assert ( jnp.sum(
        abs(
            boost_matrix_2_2_x(-rapidity(P)) @ boost_matrix_2_2_x(rapidity(P))
            )
        ) < 2 + 1e-10
    )
    assert ( jnp.sum(
        abs(
        boost_matrix_2_2_y(-rapidity(P)) @ boost_matrix_2_2_y(rapidity(P))
        )
        ) < 2 + 1e-10
    )
    assert ( jnp.sum(
        abs(
        boost_matrix_2_2_z(-rapidity(P)) @ boost_matrix_2_2_z(rapidity(P))
        )
        ) < 2 + 1e-10
    )
    assert ( jnp.sum(
        abs(
        rotation_matrix_2_2_x(-1.2) @ rotation_matrix_2_2_x(1.2)
                    )
        ) < 2 + 1e-10
    )
    assert ( jnp.sum(
        abs(
        rotation_matrix_2_2_y(-1.2) @ rotation_matrix_2_2_y(1.2)
                    )
        ) < 2 + 1e-10
    )
    assert ( jnp.sum(
        abs(
        rotation_matrix_2_2_z(-1.2) @ rotation_matrix_2_2_z(1.2)
                    )
        ) < 2 + 1e-10
    )

    assert ( np.sum(
        abs(
        rotation_matrix_2_2_z(-np.pi/2) @ boost_matrix_2_2_y(rapidity(P)) @ rotation_matrix_2_2_z(np.pi/2) @ boost_matrix_2_2_x(-rapidity(P))
                    )
        ) < 2 + 1e-10
    )

if __name__ == "__main__":
    test_kinematics()