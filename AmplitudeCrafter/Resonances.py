from jitter import resonances
from jitter.constants import spin as sp
from AmplitudeCrafter.loading import load, write
from jitter.fitting import FitParameter
import importlib
from AmplitudeCrafter.helpers import *
from AmplitudeCrafter.ParticleLibrary import particle
from AmplitudeCrafter.parameters import specialParameter
__MINFP__ = -60000000000
__MAXFP__ =  60000000000
__SWAVE_BKG__ = "swave_bkg"

from AmplitudeCrafter.parameters import parameter, UNDERSTOOD_PARAMS, FALLBACK_PARAMETERS

def check_hit(hit,name,value):
    if hit:
        raise ValueError(f"Variable {name} matched multiple cases! \n{name}: {value}")

def analyse_value(value,name,dtc,lst):
    
    matching_signatures = [param for param in UNDERSTOOD_PARAMS if param.match(value)]

    high_level_parameters = [param for param in matching_signatures if not param.final()]

    low_level_parametes = [param for param in matching_signatures if param.final()]
        
    fallback_parameters = [param for param in FALLBACK_PARAMETERS if param.match(value)]

    if len(high_level_parameters) > 1 or len(low_level_parametes) > 1:
        raise ValueError(f"More than one parameter matches value {value} with name {name}!")
    
    if len(high_level_parameters) > 0:
        matching_parameter_type, = high_level_parameters
    elif len(low_level_parametes) > 0:
        matching_parameter_type, = low_level_parametes
    elif len(fallback_parameters) > 0:
        matching_parameter_type,  = fallback_parameters
    else:
        raise ValueError(f"No signature matches value {value} with name {name}!")

    p = matching_parameter_type(name,value)
    lst.append(p)
    dtc.update(p.dict)
    return True

def analyze_structure(parameters,parameter_dict,designation=""):
    ret_list = []
    ret_dict = {}
    for param in parameters:
        if not isinstance(param,dict):
            raise ValueError("All parameters need to have a name!")
        if len(param.keys()) != 1:
            raise(ValueError("Only one Value per name!"))
        
        name, = param.keys()
        value,  = param.values()
        new_name = designation + "=>" + name
        if isinstance(value,list):
            names,value_dict = analyze_structure(value,parameter_dict,designation=new_name)
            ret_list.append(names)
            ret_dict.update(value_dict)
            continue

        if not analyse_value(value,new_name,ret_dict,ret_list):
            raise ValueError("Can not interprete value %s with name %s!"%(value,name))
    return ret_list,ret_dict

def dump_in_dict(replace_dict,mapping_dict,designation):
    """
    Expected Structure:
    lists of dicts of lists or values ->
        recurse deeper until we find dict with only one element
    """
    if isinstance(replace_dict,list):
        for element in replace_dict:
            if not isinstance(element,dict):
                raise ValueError("Encountered a non dict entrie while dumping!")
            if len(element) != 1:
                raise ValueError("Wrong size element!")
            (k,v), = element.items()
            element[k] = dump_in_dict(v,mapping_dict,designation + "=>" + k)
        return replace_dict
    
    # we use the readout logic again, to ensure consitency
    lst = []
    analyse_value(replace_dict,designation,{},lst)
    name = lst[0].name
    return mapping_dict[name].dump()

def handle_resonance_config(config_dict:dict,name):
    parameter_dict = {}
    parameter_dict["type"] = config_dict["func"].split(".")[-1]
    parameter_dict["func"] = config_dict["func"]
    params, mapping_dict = analyze_structure(config_dict["expects"],parameter_dict,name)
    return params, mapping_dict

def check_resonance_dict(resonance_dict):
    needed_keys = [
        "expects",
        "partial waves in",
        "partial waves out",
        "func",
        "parity",
        "spin"
    ]
    return all([k in resonance_dict.keys() for k in needed_keys])

def load_resonances(f:str):
    # load Resonances based on a yml file including multiple resoances
    resonance_dict = load(f)
    global_mapping_dict = specialParameter.load_specials()
    resonances = {1:[],2:[],3:[]}
    bkg = None
    for resonance_name, resonance in resonance_dict.items():
        if not check_resonance_dict(resonance):
            continue
        params, mapping_dict = handle_resonance_config(resonance,resonance_name)
        resonance["args"] = params
        r = Resonance(resonance,mapping_dict,resonance_name)
        resonances[resonance["channel"]].append(r)
        global_mapping_dict.update(r.mapping_dict)
    return resonances, global_mapping_dict, bkg

def get_val(arg,numeric=True):
    return arg(numeric)
   
def map_arguments(args,mapping_dict=None,numeric = True):
    if numeric is False and mapping_dict is not None:
        raise ValueError("Mapping Dict can only be used to accuire numeric values!")
    if isinstance(args,list):
        # tuple is simply better here, as it is hashable and jax likes this
        return tuple([map_arguments(l,mapping_dict=mapping_dict,numeric=numeric) for l in args])
    if isinstance(args,dict):
        return {k:map_arguments(v,mapping_dict=mapping_dict,numeric=numeric) for k,v in args.items()}
    if isinstance(args,parameter):
        return args(numeric,value_dict= mapping_dict)
    raise ValueError(f"Argument {args} can not be mapped!")

def get_fit_params(args):
    return map_arguments(args,numeric=False)

def read_bls(bls_dicts,mapping_dict,name):
    dtc = {}
    for bls in bls_dicts:
        lst = []
        analyse_value(bls["coupling"],name+f"L:{bls['L']},S:{bls['S']}",mapping_dict,lst)
        dtc[(bls["L"],bls["S"])] = lst[0]
    return dtc

def dump_bls(b,mapping_dict,coupling):
    return b.dump()

def check_if_wanted(name,resonance_names):
    if resonance_names is None:
        return True
    return name in resonance_names
class Resonance:
    def __init__(self,kwargs,mapping_dict,name):
        self.kwargs = kwargs
        self.spin = kwargs["spin"]
        self.parity = kwargs["parity"]
        self.channel = kwargs["channel"]
        self.name = name

        self.args = kwargs["args"]
        self.mapping_dict = mapping_dict
        # self.data_key = [k for k,v in mapping_dict.items() if isinstance(v,specialParameter) and "sigma" in v.name][0]
        # self.data_replacement = mapping_dict[self.data_key]

        module = importlib.import_module(".".join(kwargs["func"].split(".")[:-1]))
        self.lineshape = getattr(module,kwargs["func"].split(".")[-1])
        
        self.__bls_in = read_bls(kwargs["partial waves in"],self.mapping_dict,self.name+"=>"+"bls_in")
        self.__bls_out = read_bls(kwargs["partial waves out"],self.mapping_dict,self.name+"=>"+"bls_out")

    @staticmethod
    def load_resonance(f):
        resonances, mapping_dict, bkg = load_resonances(f)
        resonance, = [r for k,v in resonances.items() for r in v]
        return resonance,mapping_dict, bkg

    def to_particle(self):
        return particle(None,self.spin,self.parity,self.name)

    def dumpd(self,mapping_dict):
        # todo not Finished yet
        dtc = self.kwargs.copy()
        del dtc["args"]
        result_dict = dump_in_dict(dtc["expects"],mapping_dict,self.name)
        dtc["partial waves in"] = [{"L":pw["L"],"S":pw["S"], "coupling":dump_bls(self.bls_in[(pw["L"],pw["S"])],mapping_dict,pw["coupling"])} for pw in self.kwargs["partial waves in"]]
        dtc["partial waves out"] = [{"L":pw["L"],"S":pw["S"], "coupling":dump_bls(self.bls_out[(pw["L"],pw["S"])],mapping_dict,pw["coupling"])} for pw in self.kwargs["partial waves out"]]
        return dtc

    @property
    def M0(self):
        return lambda *args: None

    @property
    def arguments(self):
        return self.args
    
    @property
    def bls_in(self):
        # TODO: maybe needs to be copied
        return self.__bls_in
    
    @property
    def bls_out(self):
        return self.__bls_out

    def tuple(self,s=None):
        # if s is not None:
        #     self.mapping_dict[self.data_key] = s
        #     return (self.spin,self.parity,sp.direction_options(self.spin),
        #                 self.lineshape(*map_arguments(self.args,self.mapping_dict)),
        #                 self.M0(*map_arguments(self.args,self.mapping_dict)),None,self.p0)
        return (self.spin,self.parity,sp.direction_options(self.spin),
                        self.lineshape,
                        self.M0,None,self.p0)
    
    def fixed(self):
        return not any([not p.const for p in 
                flatten(
                    self.args
                    )])

    def __repr__(self):
        M0 = self.M0(*map_arguments(self.args,self.mapping_dict))
        string = f"{self.name} - Resonance(M={M0}, S={self.spin},P={self.parity}) \n{self.arguments}\n{self.bls_in} {self.bls_out}"
        return string

if __name__=="__main__":
    from AmplitudeCrafter.locals import config_dir
    res, mapping_dict, bkg = load_resonances(config_dir + "decay_example.yml")
    for k,v in mapping_dict.items():
        print(k,v)