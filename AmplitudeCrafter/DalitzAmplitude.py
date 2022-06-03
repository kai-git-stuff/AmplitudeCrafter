from jitter.amplitudes.dalitz_plot_function import DalitzDecay
from AmplitudeCrafter.loading import load
from ParticleLibrary import particle
from jitter.phasespace.DalitzPhasespace import DalitzPhaseSpace
from AmplitudeCrafter.locals import config_dir
from AmplitudeCrafter.Resonances import load_resonances, is_free
from AmplitudeCrafter.FunctionConstructor import construct_function
from jitter.fitting import FitParameter

class DalitzAmplitude:
    def __init__(self,p0:particle,p1:particle,p2:particle,p3:particle):
        self.particles = [p1,p2,p3]
        self.p0 = p0

        self.ma = p1.mass
        self.mb = p2.mass
        self.mc = p3.mass
        self.md = p0.mass

        self.phsp = DalitzPhaseSpace(self.ma,self.mb,self.mc,self.md)


    def reload(self):
        spins = [self.p0.spin] + [p.spin for p in self.particles]
        parities = [self.p0.parity] + [p.parity for p in self.particles]
        masses = [self.p0.mass] + [p.mass for p in self.particles]

        resonances = self.get_resonance_tuples()
        bls = self.get_bls()
        
        self.decay_descriptor = DalitzDecay(*masses,
                                            *spins,
                                            *parities,
                                            self.sample,
                                            resonances,bls,d=1.5/1000.,phsp = None)

    def get_bls_in(self):
        return [[r.bls_in for r in self.resonances[i]]  for i in [1,2,3]]

    def get_bls_out(self):
        return [[r.bls_out for r in self.resonances[i]]  for i in [1,2,3]]

    def load_resonances(self,f=config_dir + "decay_example.yml"):
        res, mapping_dict = load_resonances(f)
        self.resonances = res
        self.mapping_dict = mapping_dict

    def get_resonance_tuples(self):
        return [[r.tuple() for r in self.resonances[i]]  for i in [1,2,3]]
    
    def get_resonance_targs(self):
        return [[r.arguments for r in self.resonances[i]]  for i in [1,2,3]]
    
    def get_amplitude_function(self,smp):
        param_names = [p for p in self.mapping_dict.keys() if is_free(p)]
        params = [self.mapping_dict[p] for p in param_names]
        mapping_dict = self.mapping_dict
        bls_in = self.get_bls_in()
        bls_out = self.get_bls_out()
        resonances = self.get_resonance_tuples()
        resonance_args = self.get_resonance_targs()

        f = construct_function(param_names,params,mapping_dict,resonances,bls_in,bls_out,resonance_args,smp,self.phsp)

        


        


    
