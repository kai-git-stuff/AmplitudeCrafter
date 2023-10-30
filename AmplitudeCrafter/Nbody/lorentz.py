from AmplitudeCrafter.Nbody.kinematics import *


class su2_lorentz_boost:
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            theta, phi, xi, theta_rf, phi_rf, xi_rf = args
            self.M4 = build_4_4(theta, phi, xi, theta_rf, phi_rf, xi_rf)
            self.M2 = build_2_2(theta, phi, xi, theta_rf, phi_rf, xi_rf)
        M2 = kwargs.get('M2', None)
        M4 = kwargs.get('M4', None)
        if M2 is not None and M4 is not None:
            self.M2 = M2
            self.M4 = M4

    def __matmul__(self, other):
        if isinstance(other, su2_lorentz_boost):
            return su2_lorentz_boost(M2=self.M2 @ other.M2, M4=self.M4 @ other.M4)
    
    def decode(self):
        params = decode_4_4(self.M4)
        params = adjust_for_2pi_rotation(self.M2, *params)
        return params