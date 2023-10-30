from AmplitudeCrafter.Nbody.kinematics import *
from AmplitudeCrafter.Nbody.lorentz import su2_lorentz_boost
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

    psi, theta, xi, theta_rf, phi_rf, psi_rf = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    boost = su2_lorentz_boost(psi, theta, xi, theta_rf, phi_rf, psi_rf)
    M = build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf)
    # M = build_4_4(1,1,1,1,1,1)
    psi, theta, xi, phi_rf, theta_rf,  psi_rf = decode_4_4(M)
    assert np.allclose(M, build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf))
    # print(psi, theta, xi, phi_rf, theta_rf,  psi_rf)

    psi, theta, xi, phi_rf, theta_rf,  psi_rf = boost.decode()
    print(psi, theta, xi, phi_rf, theta_rf,  psi_rf)
    

if __name__ == "__main__":
    test_kinematics()