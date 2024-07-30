from jitter.amplitudes.dalitz_plot_function import DalitzDecay
import numpy as np
from AmplitudeCrafter.loading import load, write
from AmplitudeCrafter.ParticleLibrary import particle
from jitter.phasespace.DalitzPhasespace import DalitzPhaseSpace
from AmplitudeCrafter.locals import config_dir
from AmplitudeCrafter.Resonances import check_bls, load_resonances, is_free, check_if_wanted
from AmplitudeCrafter.FunctionConstructor import construct_function
from AmplitudeCrafter.parameters import ParameterScope, parameter
from jitter.kinematics import two_body_momentum
from jitter.interface import real, imaginary, conjugate
from jitter.constants import spin as sp
from jitter.amplitudes.dalitz_plot_function import helicity_options_nojit
from jax import numpy as jnp
from multiprocessing import Pool
from AmplitudeCrafter.helpers import flatten, ensure_numeric
from decayangle.kinematics import mass_squared  
import warnings

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
        self.loaded_files = []

        # defines the parameter sccope for this function
        # do not mess with this!!
        self.__scope = {}
    @property
    def loaded(self):
        return self.__loaded
    
    @property
    def saving_name(self):
        return "+".join(self.loaded_files)
    
    def get_bls_flat(self,res):
        bls_in = flatten(self.get_bls_in([res]))
        bls_out = flatten(self.get_bls_out([res]))
        return bls_in, bls_out

    def add_file(self,f):
        self.loaded_files.append(f.replace(".yml","").replace(".yaml","").split("/")[-1])

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
                warnings.warn("WARNING: Resoances of names %s already exist!"%double_names)
    
    def check_bls(self):
        if not ParameterScope.check_scope(self.__scope):
            raise ValueError("Scope not set!")
        particles = {1:(self.particles[1],self.particles[2]),
                    2:(self.particles[0],self.particles[2]),
                    3:(self.particles[0],self.particles[1])}
        for k,v in self.resonances.items():
            p1,p2 = particles[k]
            pk = self.particles[k-1]
            for res in v:
                pR = res.to_particle()
                res.bls_in = check_bls(self.p0,pk,pR,res.bls_in,False)
                self.__mapping_dict.update({p.name: p for p in res.bls_in.values()})
                res.bls_out = check_bls(pR,p1,p2,res.bls_out,True)
                self.__mapping_dict.update({p.name: p for p in res.bls_out.values()})

    def add_resonances(self,f=config_dir + "decay_example.yml"):
        if not self.loaded:
            raise ValueError("Can only add resonances if a base set is loaded!")
        with ParameterScope(self.__scope) as scope:
            res, mapping_dict = load_resonances(f)
            for k,v in res.items():
                self.check_new(v)
                self.resonances[k].extend(v)
            self.add_file(f)
            self.__mapping_dict.update(mapping_dict)
            self.check_bls()
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
        if self.loaded:
             resonance_string ="\n".join(resonances)
        else:
            resonance_string=""
        return string%(decay_description, resonance_string)

    @property
    def active_parameters(self):
        arg_dict = {p.name:p for k,p in self.__mapping_dict.items()}
        return arg_dict

    @property
    def mapping_dict(self):
        dtc = self.__mapping_dict.copy()
        # arg_dict = self.active_parameters
        # dtc.update({k:arg_dict[v.name] for k,v in dtc.items()})
        return dtc
            
    def load_resonances(self,f=config_dir + "decay_example.yml"):
        with ParameterScope(self.__scope) as scope:
            res, mapping_dict = load_resonances(f)
            self.add_file(f)
            self.resonances = res
            self.__mapping_dict = mapping_dict
            masses= {1:(self.mb,self.mc),2:(self.ma,self.mc),3:(self.ma,self.mb)}
            for channel,resonances_channel in self.resonances.items():
                for resonance in resonances_channel:
                    resonance.p0 = two_body_momentum(self.md,*masses[channel])
            
            self.check_bls()
            self.__loaded = True
  
    def get_resonance_tuples(self,resonances=None):
        return [[r.tuple() for r in self.resonances[i] if check_if_wanted(r.name,resonances)]  for i in [1,2,3]]
    
    def get_resonance_targs(self,resonances=None):
        return [[r.arguments for r in self.resonances[i] if check_if_wanted(r.name,resonances)]  for i in [1,2,3]]
    
    def dumpd(self,parameters,fit_result=None,mapping_dict=None):
        """
        Dumps a result with updated parameters to a dictionary
        An extra entrie will be made for the fit_result, if a result is supplied
        Here errors and other fit related quantities can be supplied
        This ensures, that the resulting yml file can be read and run by the software again
        """
        if not self.loaded:
            raise ValueError("Load Resonance config first, before saving!")
        with ParameterScope(self.__scope) as scope:
            dtc = {}
            # if we dont have a mapping dict, we can load parameters into the dict
            if mapping_dict is None:
                mapping_dict = self.mapping_dict.copy()
                for value,param in zip(parameters,self.get_args(numeric=False)):
                    param.update(value)
                    mapping_dict[param.name] = param
            for i, resonances in self.resonances.items():
                for res in resonances:
                    dtc[res.name] = res.dumpd(mapping_dict)
            if fit_result is not None:
                dtc["fit_result"] = fit_result
        return dtc

    def dump(self,parameters,fname,fit_result=None,mapping_dict=None):
        parameters = [ensure_numeric(p) for p in parameters] # to get rid of numpy types and so on
        write(self.dumpd(parameters,fit_result,mapping_dict=mapping_dict),fname)

    def get_amplitude_function(self,momenta,resonances = None, numericArgs=True):
        # for definition of momenta see decayangle documentation
        # resonances parameter designed to get run systematic studies later
        # so we can use the same config, but exclude or include specific resonances
        if not self.loaded:
            raise ValueError("Load Resonance config first, before building Amplitude!")
        
        if resonances is None:
            resonances = list(self.resonance_map.keys())
        if any([not isinstance(r,str) for r in resonances]):
            raise ValueError("Only string allowed for the selection of resonances!")

        param_names = self.get_arg_names()
        params = self.get_args()

        mapping_dict = {p.name:p(numeric=True) for k,p in  self.mapping_dict.items() }

        mapping_dict["sigma1"] = mass_squared(momenta[2] + momenta[3])
        mapping_dict["sigma2"] = mass_squared(momenta[1] + momenta[3])
        mapping_dict["sigma3"] = mass_squared(momenta[1] + momenta[2])

        bls_in = self.get_bls_in(resonances)
        bls_out = self.get_bls_out(resonances)
        resonance_tuples = self.get_resonance_tuples(resonances)
        resonance_args = self.get_resonance_targs(resonances)
        spins = [self.p0.spin] + [p.spin for p in self.particles]
        masses = [self.p0.mass] + [p.mass for p in self.particles]

        resonances = {i:[r for r in self.resonances[i] if check_if_wanted(r.name,resonances)] for i in [1,2,3]}
        f,start = construct_function(masses,spins,params,mapping_dict,
                                resonances, resonance_tuples, bls_in,bls_out,resonance_args,momenta,numericArgs = numericArgs)
        return f,start

    def get_interference_terms(self,momenta,resonances1,resonances2):
        """
        Not yet properly tested, even though it should work!
        TODO: Test 
        """
        f1,start = self.get_amplitude_function(momenta,resonances=resonances1)
        f2,start = self.get_amplitude_function(momenta,resonances=resonances2)

        def interference(args,nu,*lambdas):
            return f1(args,nu,*lambdas) * conjugate(f2(args,nu,*lambdas)) + conjugate(f1(args,nu,*lambdas)) * f2(args,nu,*lambdas)

        def full_interference(args):
            return sum(
                sum(
                    interference(args,ld,la,lb,lc) 
                        for la,lb,lc in helicity_options_nojit(*[p.spin for p in self.particles])
                            ) for ld in sp.direction_options(self.p0.spin))

        return full_interference, start

    def run_function(self,args,smp,resonances = None,parallel = False):
        raise NotImplementedError()
    
    def get_args(self,numeric=False):
        if numeric is True:
            arg_dict = {p.name:p(numeric = True) for k,p in self.mapping_dict.items() if is_free(p)}
        else:
            arg_dict = {p.name:p for k,p in self.mapping_dict.items() if is_free(p)}
        return list(
            arg_dict.values()
            )
    
    def get_fit_params(self):
        arg_dict = {p.name:p(numeric=False) for k,p in self.mapping_dict.items() if is_free(p)}
        return list(
            arg_dict.values()
            )

    def get_arg_names(self):
        arg_dict = {p.name:k for k,p in self.mapping_dict.items() if is_free(p)}
        # translate values with _complex in to imaginary and real part
        return list(
            arg_dict.values()
            )

    def get_cov(self,f):
        if not self.loaded:
            raise ValueError("Covariance can only be loaded after Resoances have been loaded!")
        cov_dict = load(f)
        p_names = self.get_arg_names()
        n = len(p_names)
        cov_mat = np.empty((n,n))
        for i,p1 in enumerate(p_names):
            for j,p2 in enumerate(p_names):
                cov_mat[i,j] = cov_dict[p1][p2]
        return cov_mat

    def get_args_from_yml(self,file,raiseeException=True):
        from AmplitudeCrafter.FunctionConstructor import map_arguments
        helperAmplitude = DalitzAmplitude(self.p0,*self.particles)
        helperAmplitude.load_resonances(file)
        own_param_names = self.get_arg_names()
        if not set(helperAmplitude.get_arg_names()).issubset(set(own_param_names)):
            warnings.warn("Own: " + str(set(own_param_names)) + "/n" + f"{file} " + str(set(helperAmplitude.get_arg_names())))
            if raiseeException:
                raise ValueError(f"File {file} does not contain all needed arguments to represent the amplitude!")
            else:
                warnings.warn("Waring: " + f"File {file} does not contain all needed arguments to represent the amplitude!")
        mapping_dict = self.mapping_dict
        mapping_dict.update(helperAmplitude.mapping_dict)
        value_dict = {k:v(numeric=True) for k,v in mapping_dict.items()}
        mapped_args = map_arguments(self.get_args(),mapping_dict=value_dict,numeric=True)
        return [value for name, value in zip(own_param_names,mapped_args)]
