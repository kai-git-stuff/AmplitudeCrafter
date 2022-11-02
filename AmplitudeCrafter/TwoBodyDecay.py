
from jitter.constants import spin as sp
from jitter.kinematics import wigner_capital_d
from jitter.amplitudes.dalitz_plot_function import helicity_options_nojit
class TwoBodyDecay:
    def __init__(self,p0,p1,p2):
        self.p0 = p0
        self.particles = [p1,p2]

    
    def build_decay():
        pass
    
    def get_amplitude_function(self,theta,phi, total_absolute=False, just_in_time_compile = True, decay_tree = None):

        D = {
                (l0,l1,l2): wigner_capital_d(phi, theta, 0, self.p0.spin, l0, l1-l2)
                    for l0, l1, l2 in helicity_options_nojit(self.p0.spin,
                                                            self.particles[0].spin,
                                                            self.particles[1].spin)
            }

        def f(l0,l1,l2):
            return (self.p0.spin + 1)**0.5 * D[(l0,l1,l2)]
        

        return f
