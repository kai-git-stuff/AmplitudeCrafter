from jitter import resonances
from jitter.constants import spin as sp
from AmplitudeCrafter.loading import load
from jitter.fitting import FitParameter

def is_free(p):
    if isinstance(p,FitParameter):
        return not p.fixed
    return False

def flatten(listoflists):
    lst = []
    def flatten_recursive(listoflists,ret_list:list):
        if isinstance(listoflists,list):
            [flatten_recursive(l,ret_list) for l in listoflists] 
            return 
        ret_list.append(listoflists)
    flatten_recursive(listoflists,lst)
    return lst


def analyse_value(value,name,dtc,lst):
    if not isinstance(value,str):
        lst.append(name)
        dtc[name] = FitParameter(name,value,-100,100,0.01)
        return True
    if "sigma" in value:
        lst.append(value)
        dtc[value] = value
    if "const" in value:
        value = value.replace("const","")
        try:
            dtc[name] = int(value)
        except ValueError:
            dtc[name] = float(value)
        lst.append(name)
        return True
    if "complex" in value:
        value = value.replace("complex(","").replace(")","")
        v1,v2 = [float(v) for v in value.split(",") ]
        n1, n2 = name + "_real", name + "_imag"
        dtc[n1] = FitParameter(n1,v1,-100,100,0.01)
        dtc[n2] = FitParameter(n2,v2,-100,100,0.01)

        lst.append(name+"_complex")
        return True
    
    return False

def analyze_structure(parameters,parameter_dict,designation=""):
    ret_list = []
    ret_dict = {}
    for param in parameters:
        if not isinstance(param,dict):
            raise ValueError("All parameters need to have a name!")
        if len(param.keys()) != 1:
            raise(ValueError("only one Value per name!"))
        
        name, = param.keys()
        value,  = param.values()
        new_name = designation + "=>" + name
        if isinstance(value,list):
            names,value_dict = analyze_structure(value,parameter_dict,designation=new_name)
            ret_list.append(names)
            ret_dict.update(value_dict)
            continue

        analyse_value(value,new_name,ret_dict,ret_list)
    return ret_list,ret_dict

def handle_resonance_config(config_dict:dict,name):
    parameter_dict = {}
    parameter_dict["type"] = config_dict["func"].split(".")[-1]
    parameter_dict["func"] = config_dict["func"]
    params, mapping_dict = analyze_structure(config_dict["expects"],parameter_dict,name)

    return params, mapping_dict

def load_resonances(f:str):
    resonance_dict = load(f)
    global_mapping_dict = {}
    resonances = {1:[],2:[],3:[]}
    for resonance_name, resonance in resonance_dict.items():
        params, mapping_dict = handle_resonance_config(resonance,resonance_name)
        resonance["args"] = params
        r = Resonance(resonance,mapping_dict)
        resonances[resonance["channel"]].append(r)
        global_mapping_dict.update(r.mapping_dict)
    return resonances, global_mapping_dict

def get_val(arg,mapping_dict):
    if "_complex" in arg:
        r, i = arg.replace("_complex","_real"), arg.replace("_complex","_imag")
        return get_val(r,mapping_dict) + 1j * get_val(i,mapping_dict)
    val = mapping_dict[arg]
    if isinstance(val,FitParameter):
        val = val()
    return val

def get_fit_parameter(arg,mapping_dict):
    if "_complex" in arg:
        r, i = arg.replace("_complex","_real"), arg.replace("_complex","_imag")
        return get_val(r,mapping_dict) , get_val(i,mapping_dict)
    val = mapping_dict[arg]
    if isinstance(val,FitParameter):
        val = val
    return val


def map_arguments(args,mapping_dict):
    if isinstance(args,list):
        return [map_arguments(l,mapping_dict) for l in args]
    if isinstance(args,dict):
        return {k:map_arguments(v,mapping_dict) for k,v in args.items()}
    return get_val(args,mapping_dict)

def get_fit_params(args,mapping_dict):
    if isinstance(args,list):
        return [get_fit_params(l,mapping_dict) for l in args]
    if isinstance(args,dict):
        return {k:get_fit_params(v,mapping_dict) for k,v in args.items()}
    return get_fit_parameter(args,mapping_dict)

def read_bls(bls_dicts,mapping_dict,name):
    dtc = {}
    for bls in bls_dicts:
        lst = []
        analyse_value(bls["coupling"],name,mapping_dict,lst)
        dtc[(bls["L"],bls["S"])] = lst[0]
    return dtc

class Resonance:
    def __init__(self,kwargs,mapping_dict):
        self.type = kwargs["type"]
        self.spin = kwargs["spin"]
        self.parity = kwargs["parity"]

        self.M0 = kwargs["M0"]
        self.d = kwargs["d"]
        self.p0 = None # todo two_body_breakup Momentum based on data and stuff

        self.args = kwargs["args"]
        self.mapping_dict = mapping_dict

        self.data_key = [k for k,v in mapping_dict.items() if isinstance(v,str) and "sigma" in v][0]

        self.lineshape = getattr(resonances,kwargs["func"].split(".")[-1])

        self.__bls_in = read_bls(kwargs["partial waves in"],self.mapping_dict,self.type+"=>"+"bls_in")
        self.__bls_out = read_bls(kwargs["partial waves out"],self.mapping_dict,self.type+"=>"+"bls_out")


    @property
    def arguments(self):
        return self.args
        return map_arguments(self.args,self.mapping_dict)
    
    @property
    def bls_in(self):
        return self.__bls_in
        return map_arguments(self.__bls_in,self.mapping_dict)
    
    @property
    def bls_out(self):
        return self.__bls_out
        return map_arguments(self.__bls_out,self.mapping_dict)

    def tuple(self,s=None):
        if s is not None:
            self.mapping_dict[self.data_key] = s
            return (self.spin,self.parity,sp.direction_options(self.spin),
                        self.lineshape(map_arguments(self.args,self.mapping_dict)),
                        self.M0,self.d,self.p0)
        return (self.spin,self.parity,sp.direction_options(self.spin),
                        self.lineshape,
                        self.M0,self.d,self.p0)
    
    def fixed(self):
        return not any([is_free(p) for p in 
                flatten(
                    get_fit_params(self.args,self.mapping_dict)
                    )])

    def __repr__(self):
        string = f"{self.type} - Resonance(M={self.M0}, S={self.spin},P={self.parity}) \n{self.arguments}\n{self.bls_in} {self.bls_out}"
        return string

if __name__=="__main__":
    from AmplitudeCrafter.locals import config_dir
    res, mapping_dict = load_resonances(config_dir + "decay_example.yml")
    for r in res[1]:
        r.mapping_dict[r.data_key] = 50
        print(r.fixed())