from jitter.amplitudes.dalitz_plot_function import DalitzDecay
from AmplitudeCrafter.loading import write
from AmplitudeCrafter.ParticleLibrary import particle
from jitter.phasespace.DalitzPhasespace import DalitzPhaseSpace
from AmplitudeCrafter.locals import config_dir
from AmplitudeCrafter.Resonances import load_resonances, is_free, needed_parameter_names, check_if_wanted
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
        self.__loaded = False

    @property
    def loaded(self):
        return self.__loaded

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

    def get_bls_in(self,resonances):
        return [[r.bls_in for r in self.resonances[i] if check_if_wanted(r.name,resonances)]  for i in [1,2,3]]

    def get_bls_out(self,resonances):
        return [[r.bls_out for r in self.resonances[i] if check_if_wanted(r.name,resonances)]  for i in [1,2,3]]

    def check_new(self,resonances_channel,fail=True):
        names = [res.name for res in resonances_channel]
        existing_names = [res.name for res in self.resonance_map.values()]

        if any([n in existing_names for n in names]):
            double_names = [n for n in names if n in existing_names]
            if fail:
                raise ValueError("Resoances of names %s already exist!"%double_names)
            else:
                print("WARNING: Resoances of names %s already exist!"%double_names)

    def add_resonances(self,f=config_dir + "decay_example.yml"):
        if not self.loaded:
            raise ValueError("Can only add resonances if a base set is loaded!")
        res, mapping_dict = load_resonances(f)
        for k,v in res.items():
            self.check_new(v)
            self.resonances[k].extend(v)
        self.mapping_dict.update(mapping_dict)
        masses= {1:(self.mb,self.mc),2:(self.ma,self.mc),3:(self.ma,self.mb)}
        for channel,resonances_channel in self.resonances.items():
            for resonance in resonances_channel:
                resonance.p0 = two_body_momentum(self.md,*masses[channel])
        self.__loaded = True

    @property
    def resonance_map(self):
        dtc = {}
        for channel,resonances_channel in self.resonances.items():
            for resonance in resonances_channel:
                dtc[resonance.name] = resonance
        return dtc
    
    def __repr__(self):
        string = "Dalitz Amplitude %s \nResonances:\n%s"
        if self.loaded:
            mp = self.resonance_map
            resonances = list(mp.keys())
        decay_description = "%s -> %s %s %s"%tuple([self.p0.name] + [p.name for p in self.particles])
        resonance_string ="\n".join(resonances)
        return string%(decay_description, resonance_string)
            
    def load_resonances(self,f=config_dir + "decay_example.yml"):
        res, mapping_dict = load_resonances(f)
        self.resonances = res
        self.mapping_dict = mapping_dict
        masses= {1:(self.mb,self.mc),2:(self.ma,self.mc),3:(self.ma,self.mb)}
        for channel,resonances_channel in self.resonances.items():
            for resonance in resonances_channel:
                resonance.p0 = two_body_momentum(self.md,*masses[channel])

        self.__loaded = True

    def get_resonance_tuples(self,resonances=None):
        return [[r.tuple() for r in self.resonances[i] if check_if_wanted(r.name,resonances)]  for i in [1,2,3]]
    
    def get_resonance_targs(self,resonances=None):
        return [[r.arguments for r in self.resonances[i] if check_if_wanted(r.name,resonances)]  for i in [1,2,3]]
    
    def dumpd(self,parameters):
        if not self.loaded:
            raise ValueError("Load Resonance config first, before saving!")
        dtc = {}
        mapping_dict = self.mapping_dict.copy()
        for param,name in zip(parameters,self.get_arg_names()):
            mapping_dict[name] = param
        for i, resonances in self.resonances.items():
            for res in resonances:
                dtc[res.name] = res.dumpd(mapping_dict)
        return dtc

    def dump(self,parameters,fname):
        write(self.dumpd(parameters),fname)

    def get_amplitude_function(self,smp,resonances = None):
        # resonances parameter designed to get run systematic studies later
        # so we can use the same config, but exclude or include specific resonances
        if not self.loaded:
            raise ValueError("Load Resonance config first, before building Amplitude!")
        
        if resonances is None:
            resonances = list(self.resonance_map.keys())
        if any([not isinstance(r,str) for r in resonances]):
            raise ValueError("Only string allowed for the selection of resonances!")

        param_names = [k for k,p in self.mapping_dict.items() 
                                if is_free(p) ]
        params = [self.mapping_dict[p] for p in param_names]
        mapping_dict = self.mapping_dict.copy()
        bls_in = self.get_bls_in(resonances)
        bls_out = self.get_bls_out(resonances)
        resonance_tuples = self.get_resonance_tuples(resonances)
        resonance_args = self.get_resonance_targs(resonances)
        spins = [self.p0.spin] + [p.spin for p in self.particles]
        parities = [self.p0.parity] + [p.parity for p in self.particles]
        masses = [self.p0.mass] + [p.mass for p in self.particles]

        mapping_dict["sigma3"] = self.phsp.m2ab(smp)
        mapping_dict["sigma2"] = self.phsp.m2ac(smp)
        mapping_dict["sigma1"] = self.phsp.m2bc(smp)
        resonances = [[r for r in self.resonances[i] if check_if_wanted(r.name,resonances)] for i in [1,2,3]]
        f,start = construct_function(masses,spins,parities,param_names,params,mapping_dict,
                                resonances,resonance_tuples,bls_in,bls_out,resonance_args,smp,self.phsp)
        return f,start

    def get_interference_terms(self,smp, resonances = None):
        raise NotImplementedError("To be implemented soon!")

    def get_arg_names(self):
        param_names = [k for k,p in self.mapping_dict.items() if is_free(p)]
        # translate values with _complex in to imaginary and real part
        return needed_parameter_names(param_names)

    def get_args(self):
        from AmplitudeCrafter.FunctionConstructor import map_arguments
        needed_param_names = self.get_arg_names()
        mapped_args = map_arguments(needed_param_names,self.mapping_dict)
        return [FitParameter(name,value,-600.,600.) for name, value in zip(needed_param_names,mapped_args)]


        


    
