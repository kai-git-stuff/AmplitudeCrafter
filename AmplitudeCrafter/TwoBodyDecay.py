
from jitter.constants import spin as sp
from jitter.kinematics import wigner_capital_d
from jitter.amplitudes.dalitz_plot_function import helicity_options_nojit, helicity_couplings_from_ls_static, get_clebsch_dict
from AmplitudeCrafter.Resonances import read_bls, map_arguments
from AmplitudeCrafter.loading import load, write
from jax import numpy as jnp

class TwoBodyDecay:
    decays = 0
    def __init__(self,p0,p1,p2):
        self.p0 = p0
        self.particles = [p1,p2]
        self.decayNumber = TwoBodyDecay.decays
        TwoBodyDecay.decays += 1
        self.__loaded = False

    
    def build_decay(self):
        pass
    
    def load_partial_wave_couplings(self,file):

        bls_dict = load(file)
        self.mapping_dict = {}
        self.bls = read_bls(bls_dict,self.mapping_dict,f"TwoBody{self.decayNumber}"+"=>bls")
        print(self.bls)
        self.argnames = [ k for k in self.bls.values() if "_complex" in k]
        print(self.argnames)
        self.__loaded = True
        raise NotImplementedError("Not yet functional!")

    def get_amplitude_function(self,theta,phi, total_absolute=False, just_in_time_compile = True, decay_tree = None, numericArgs=False):

        D = {
                (l0,l1,l2): wigner_capital_d(phi, theta, 0, self.p0.spin, l0, l1-l2)
                    for l0, l1, l2 in helicity_options_nojit(self.p0.spin,
                                                            self.particles[0].spin,
                                                            self.particles[1].spin)
            }
        clebsch = get_clebsch_dict(10,10)
        # print(D)
        if self.__loaded:
            def getH(l1,l2,params):
                mapping_dict = self.mapping_dict.copy()
                for name,p in zip(self.argnames,params):
                    mapping_dict[name] = p
                bls = map_arguments(self.bls,mapping_dict)
                h = helicity_couplings_from_ls_static(self.p0.spin, 
                                self.particles[0].spin,
                                self.particles[1].spin,
                                l1,l2 , bls,clebsch)
        else:
            print("No Partial Wave couplings specified for TwoBody decay! Defauling to 1.")
            def getH(l1,l2,params):
                return 1.
        
        def f(params,l0,l1,l2):
            # return 1.
            h = getH(l1,l2,params)
            return h * (self.p0.spin + 1)**0.5 * D[(l0,l1,l2)]
        # print(self.argnames.copy())
        return f, [] if not self.__loaded else self.argnames.copy()
