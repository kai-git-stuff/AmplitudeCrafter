from jitter import resonances
from jitter.constants import spin as sp
from AmplitudeCrafter.loading import load

def analyse_value(value):
    if "const" in value:
        value = value.replace("const","")
        return int(value)
    if "complex" in value:
        return "complex()"
    return value

    

def analyze_structure(parameter_list,parameter_dict,designation=""):
    ret_list = []
    ret_dict = {}
    for param in parameter_list:
        if not isinstance(param,dict):
            raise ValueError("All parameters need to have a name!")
        if len(param.keys()) != 1:
            raise(ValueError("only one Value per name!"))
        
        name, = param.keys()
        value,  = param.values()

        if isinstance(value,list):
            names,value_dict = analyze_structure(value,parameter_dict,designation=designation + "=>" + name)
            ret_list.append(names)
            ret_dict.update(value_dict)
            continue

        ret_list.append(designation + "=>" + name)
        ret_dict[designation + "=>" + name] = analyse_value(value)
    return ret_list,ret_dict

def handle_resonance_config(config_dict:dict,name):
    parameter_dict = {}
    parameter_dict["type"] = config_dict["func"].split(".")[-1]
    parameter_dict["func"] = config_dict["func"]

    params, mapping_dict = analyze_structure(config_dict["expects"],parameter_dict,name)
    return params, mapping_dict

def load_resonances(f:str):
    resonance_dict = load(f)
    resonances = []
    for resonance_name, resonance in resonance_dict.items():
        resonances.append(handle_resonance_config(resonance,resonance_name))
    print(resonances)
    return resonances

class Resonance:
    def __init__(self,**kwargs):
        self.type = kwargs["type"]
        self.spin = kwargs["spin"]
        self.parity = kwargs["parity"]

        self.M0 = kwargs["M0"]
        self.d = kwargs["d"]
        self.p0 = kwargs["p0"]

        self.lineshape = getattr(resonances,kwargs["func"])(*(kwargs["args"]))

    def tuple(self):
        return (self.spin,self.parity,sp.direction_options(self.spin),self.lineshape,self.M0,self.d,self.p0)


if __name__=="__main__":
    from locals import config_dir

    load_resonances(config_dir + "decay_example.yml")