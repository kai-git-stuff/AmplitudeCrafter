from jitter.amplitudes.dalitz_plot_function import DalitzDecay
from AmplitudeCrafter.loading import load
from AmplitudeCrafter.ParticleLibrary import particle
from jitter.phasespace.DalitzPhasespace import DalitzPhaseSpace
from AmplitudeCrafter.locals import config_dir
from AmplitudeCrafter.Resonances import load_resonances, is_free, needed_parameter_names
from AmplitudeCrafter.FunctionConstructor import construct_function
from jitter.fitting import FitParameter
from jitter.kinematics import two_body_momentum


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
        masses= {1:(self.mb,self.mc),2:(self.ma,self.mc),3:(self.ma,self.mb)}
        for channel,resonances_channel in self.resonances.items():
            for resonance in resonances_channel:
                resonance.p0 = two_body_momentum(self.md,*masses[channel])

    def get_resonance_tuples(self):
        return [[r.tuple() for r in self.resonances[i]]  for i in [1,2,3]]
    
    def get_resonance_targs(self):
        return [[r.arguments for r in self.resonances[i]]  for i in [1,2,3]]
    
    def dumpd(self,parameters):
        dtc = {}
        for i, resonances in self.resonances.items():
            for res in resonances:
                dtc[res.name] = res.dumpd()
        return dtc
            

    def get_amplitude_function(self,smp):
        param_names = [k for k,p in self.mapping_dict.items() if is_free(p)]
        params = [self.mapping_dict[p] for p in param_names]
        mapping_dict = self.mapping_dict
        bls_in = self.get_bls_in()
        bls_out = self.get_bls_out()
        resonance_tuples = self.get_resonance_tuples()
        resonance_args = self.get_resonance_targs()
        spins = [self.p0.spin] + [p.spin for p in self.particles]
        parities = [self.p0.parity] + [p.parity for p in self.particles]
        masses = [self.p0.mass] + [p.mass for p in self.particles]

        mapping_dict["sigma3"] = self.phsp.m2ab(smp)
        mapping_dict["sigma2"] = self.phsp.m2ac(smp)
        mapping_dict["sigma1"] = self.phsp.m2bc(smp)
        resonances = [self.resonances[i] for i in [1,2,3]]
        f,start = construct_function(masses,spins,parities,param_names,params,mapping_dict,
                                resonances,resonance_tuples,bls_in,bls_out,resonance_args,smp,self.phsp)
        return f,start

    def get_args(self):
        from AmplitudeCrafter.FunctionConstructor import map_arguments
        param_names = [k for k,p in self.mapping_dict.items() if is_free(p)]
        # translate values with _complex in to imaginary and real part
        needed_param_names = needed_parameter_names(param_names)
        mapped_args = map_arguments(needed_param_names,self.mapping_dict)
        return [FitParameter(name,value,-600.,600.) for name, value in zip(needed_param_names,mapped_args)]


        


    
