from jitter import resonances
from jitter.constants import spin as sp
from AmplitudeCrafter.loading import load, write
from jitter.fitting import FitParameter
import importlib
from AmplitudeCrafter.ParticleLibrary import particle
__MINFP__ = -60000000000
__MAXFP__ =  60000000000
def is_free(p):
    if isinstance(p,FitParameter):
        return not p.fixed
    return False

def get_parity(L,p1,p2):
    if L % 2 != 0:
        raise ValueError("Angular momentum has to be multiple of 2!")
    return p1 * p2 * (-1)**(L//2)

def check_bls(mother:particle,daughter1:particle,daughter2:particle,bls,parity_conserved=False):
    Ls = []
    for S in sp.couple(daughter1.spin,daughter2.spin):
        for L in sp.couple(mother.spin,S):
            if get_parity(L,daughter1.parity,daughter2.parity) == mother.parity or not parity_conserved:
                Ls.append((L,S))
    minL,minS = min(Ls,key=lambda x: x[0])
    Ls_bls = [L for L,S in bls.keys()]
    Lset = set([L for L,_ in Ls])
    Sset = set([S for L,S in Ls])
    if min(Ls_bls) != minL:
        raise ValueError(f"""Lowest partial wave {(minL,minS)} not contained in LS couplings {list(bls.keys())}!
        Values {mother} -> {daughter1} {daughter2} 
        Parity{" " if parity_conserved else " not "}conserved!""")
    if not all([L in Lset for L,S in bls.keys()]):
        string = "; ".join([str(L) for L,S in bls.keys() if not S in Lset])
        raise ValueError(f"""Not all L couplings possible! {string} 
                        For decay {mother} -> {daughter1} {daughter2}""")
    if not all([S in Sset for L,S in bls.keys()]):
        string = "; ".join([str(S) for L,S in bls.keys() if not S in Sset])
        raise ValueError(f"""Not all S couplings possible! {string} 
                        For decay {mother} -> {daughter1} {daughter2}""")

def process_complex(value):
    value = value.replace("complex","")
    value = value.replace("(","")
    value = value.replace(")","")
    real, imag = value.split(",")
    return float(real) + 1j * float(imag)

def flatten(listoflists):
    lst = []
    def flatten_recursive(listoflists,ret_list:list):
        if isinstance(listoflists,list):
            [flatten_recursive(l,ret_list) for l in listoflists] 
            return 
        ret_list.append(listoflists)
    flatten_recursive(listoflists,lst)
    return lst

def get_FitParameter(name,value):
    # first detect possibility for complex
    words = value.split(" ")
    words = [word for word in words if " " not in word and len(word) > 0]
    frm = float(words[words.index("from") + 1])
    to = float(words[words.index("to") + 1])
    val = float(words[0])
    return FitParameter(name,val,frm,to,0.0001)

def check_hit(hit,name,value):
    if hit:
        raise ValueError(f"Variable {name} matched multiple cases! \n{name}: {value}")

def analyse_value(value,name,dtc,lst):
    hit = False
    if not isinstance(value,str):
        lst.append(name)
        dtc[name] = FitParameter(name,value,__MINFP__,__MAXFP__,0.01)
        return True
    try:
        val = float(value)
        lst.append(name)
        dtc[name] = FitParameter(name,val,__MINFP__,__MAXFP__,0.01)
        return True
    except:
        pass
    if "from" in value and "to" in value and not "complex" in value:
        # bounded value gets treated in separate functiuon
        lst.append(name)
        dtc[name] = get_FitParameter(name,value)
        hit =  True
    if "sigma" in value:
        check_hit(hit,name,value)
        # sigma is the key for the dalitz variables
        lst.append(value)
        dtc[value] = value
        hit =  True
    if "complex" in value:
        check_hit(hit,name,value)
        if "const" in value.split(")")[-1]:
            value = value.replace("const","")
            dtc[name] = process_complex(value)
            lst.append(name)
            return  True
        value = value.replace("complex(","").replace(")","")
        v1,v2 =value.split(",") 
        n1, n2 = name + "_real", name + "_imag"
        # send a dummy list, because we append ourselfes here
        analyse_value(v1,n1,dtc,list())
        analyse_value(v2,n2,dtc,list())
        # dtc[n1] = FitParameter(n1,v1,__MINFP__,__MAXFP__,0.01)
        # dtc[n2] = FitParameter(n2,v2,__MINFP__,__MAXFP__,0.01)
        lst.append(name+"_complex")
        return True
    if "const" in value:
        check_hit(hit,name,value)
        # if we have a constant, we expect
        value = value.replace("const","")            
        try:
            dtc[name] = int(value)
        except ValueError:
            dtc[name] = float(value)
        lst.append(name)
        hit =  True
    if value.strip() == "L":
        check_hit(hit,name,value)
        # print("Angular Momentum Variable detected!")
        dtc[name] = "L"
        dtc["L"] = None # this has to be None for now, as we need to find mistakes (None will somehwere down the line make issues, if it is not properly overwritten)
        lst.append(name)
        hit = True
    return hit

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

        if not analyse_value(value,new_name,ret_dict,ret_list):
            raise ValueError("Can not interprete value %s with name %s!"%(value,name))
    return ret_list,ret_dict

def dump_value(param,name,value,new_name,mapping_dict):
    if not isinstance(value,str):
        param[name] = mapping_dict[new_name]
    elif value.strip() == "L":
        param[name] = "L"
    elif "const" in value:
        param[name] = value
    elif "sigma" in name:
        param[name] = value
    elif "complex" in value:
        r,i =  mapping_dict[new_name+"_real"] , mapping_dict[new_name+"_imag"]
        param[name] = "complex(%s,%s)"%(r,i)
    elif "to" in value and "from" in value:
        fit_param = get_FitParameter("temp",value)
        dumping_value = "%s from %s to %s"%(mapping_dict[new_name],fit_param.lower_limit,fit_param.upper_limit)
        param[name] = dumping_value
    else:
        raise ValueError("Cant map value (%s) of type %s"%(value,type(value)))

def dump_in_dict(replace_dict,mapping_dict,designation):
    for param in replace_dict:
        if not isinstance(param,dict):
            raise ValueError("All parameters need to have a name!")
        if len(param.keys()) != 1:
            raise(ValueError("only one Value per name! %s"%param))
        
        name, = param.keys()
        value,  = param.values()

        new_name = designation + "=>" + name
        if isinstance(value,list):
            dump_in_dict(value,mapping_dict,designation=new_name)
        else:
            dump_value(param,name,value,new_name,mapping_dict)
    return True

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
        "M0",
        "d",
        "parity",
        "spin"
    ]
    return all([k in resonance_dict.keys() for k in needed_keys])

def load_resonances(f:str):
    # load Resonances based on a yml file including multiple resoances
    resonance_dict = load(f)
    global_mapping_dict = {}
    resonances = {1:[],2:[],3:[]}
    for resonance_name, resonance in resonance_dict.items():
        if not check_resonance_dict(resonance):
            print(f"Field with name {resonance_name} could not be interpreted as resonance!")
            continue
        params, mapping_dict = handle_resonance_config(resonance,resonance_name)
        resonance["args"] = params
        r = Resonance(resonance,mapping_dict,resonance_name)
        resonances[resonance["channel"]].append(r)
        global_mapping_dict.update(r.mapping_dict)
    return resonances, global_mapping_dict

def get_val(arg,mapping_dict,numeric=True):
    if "_complex" in arg:
        r, i = arg.replace("_complex","_real"), arg.replace("_complex","_imag")
        if numeric:
            return get_val(r,mapping_dict) + 1j * get_val(i,mapping_dict)
        else:
            return get_val(r,mapping_dict,numeric=False) , get_val(i,mapping_dict,numeric=False)
    val = mapping_dict[arg]
    if isinstance(val,FitParameter) and numeric:
        val = val()
    if val == "L":
        # TODO: ,aybeset these special charakters in a config somewhere?
        return mapping_dict["L"]
    return val

def get_fit_parameter(arg,mapping_dict):
    if "_complex" in arg:
        r, i = arg.replace("_complex","_real"), arg.replace("_complex","_imag")
        return get_val(r,mapping_dict) , get_val(i,mapping_dict)
    val = mapping_dict[arg]
    if isinstance(val,FitParameter):
        val = val
    return val

def needed_parameter_names(param_names):
    # this only translates all _complex values into real and imaginary
    needed_names = []
    for p in param_names:
        if "_complex" in p:
            r, i = p.replace("_complex","_real"), p.replace("_complex","_imag")
            needed_names.append(r)
            needed_names.append(i)
        else:
            needed_names.append(p)
    return needed_names

def map_arguments(args,mapping_dict,numeric = True):
    if isinstance(args,list):
        return [map_arguments(l,mapping_dict,numeric) for l in args]
    if isinstance(args,dict):
        return {k:map_arguments(v,mapping_dict,numeric) for k,v in args.items()}

    return get_val(args,mapping_dict,numeric)

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
        analyse_value(bls["coupling"],name+f"L:{bls['L']},S:{bls['S']}",mapping_dict,lst)
        dtc[(bls["L"],bls["S"])] = lst[0]
    return dtc

def dump_bls(b,mapping_dict,coupling):
    if not isinstance(coupling,str):
        return float(get_val(b,mapping_dict))
    if "const" in coupling:
        return coupling
    val = get_val(b,mapping_dict)
    if isinstance(val,complex):
        val = "complex(%s,%s)"%(val.real,val.imag)
    return val

def check_if_wanted(name,resonance_names):
    if resonance_names is None:
        return True
    return name in resonance_names
class Resonance:
    def __init__(self,kwargs,mapping_dict,name):
        self.kwargs = kwargs
        self.type = kwargs["type"]
        self.spin = kwargs["spin"]
        self.parity = kwargs["parity"]
        self.name = name

        
        self.__M0 = kwargs["M0"]
        if isinstance(self.__M0,str):
            module_M0 = importlib.import_module(".".join(self.__M0.split(".")[:-1]))
            self.__M0 = getattr(module_M0,self.__M0.split(".")[-1])
        else:
            self.__M0 = float(kwargs["M0"])
        
        self.d = kwargs["d"]
        self.p0 = None # todo two_body_breakup Momentum based on data and stuff

        self.args = kwargs["args"]
        self.mapping_dict = mapping_dict

        self.data_key = [k for k,v in mapping_dict.items() if isinstance(v,str) and "sigma" in v][0]
        self.data_replacement = mapping_dict[self.data_key]

        module = importlib.import_module(".".join(kwargs["func"].split(".")[:-1]))
        self.lineshape = getattr(module,kwargs["func"].split(".")[-1])
        
        self.__bls_in = read_bls(kwargs["partial waves in"],self.mapping_dict,self.name+"=>"+"bls_in")
        self.__bls_out = read_bls(kwargs["partial waves out"],self.mapping_dict,self.name+"=>"+"bls_out")

    @staticmethod
    def load_resonance(f):
        resonances, mapping_dict = load_resonances(f)
        resonance, = [r for k,v in resonances.items() for r in v]
        return resonance,mapping_dict

    def to_particle(self):
        return particle(self.M0(*map_arguments(self.args,self.mapping_dict)),self.spin,self.parity,self.name)

    def dumpd(self,mapping_dict):
        # todo not Finished yet
        dtc = self.kwargs.copy()
        del dtc["args"]
        mapping_dict[self.data_key] = "sigma%s"%self.kwargs["channel"]
        mapping_dict["L"] = "L"
        dump_in_dict(dtc["expects"],mapping_dict,self.name)
        dtc["partial waves in"] = [{"L":pw["L"],"S":pw["S"], "coupling":dump_bls(self.bls_in[(pw["L"],pw["S"])],mapping_dict,pw["coupling"])} for pw in self.kwargs["partial waves in"]]
        dtc["partial waves out"] = [{"L":pw["L"],"S":pw["S"], "coupling":dump_bls(self.bls_out[(pw["L"],pw["S"])],mapping_dict,pw["coupling"])} for pw in self.kwargs["partial waves out"]]
        return dtc

    @property
    def M0(self):
        if isinstance(self.__M0,float):
            return lambda *args,**kwargs : self.__M0
        return self.__M0

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
        if s is not None:
            self.mapping_dict[self.data_key] = s
            return (self.spin,self.parity,sp.direction_options(self.spin),
                        self.lineshape(*map_arguments(self.args,self.mapping_dict)),
                        self.M0(*map_arguments(self.args,self.mapping_dict)),self.d,self.p0)
        return (self.spin,self.parity,sp.direction_options(self.spin),
                        self.lineshape,
                        self.M0,self.d,self.p0)
    
    def fixed(self):
        return not any([is_free(p) for p in 
                flatten(
                    get_fit_params(self.args,self.mapping_dict)
                    )])

    def __repr__(self):
        M0 = self.M0(*map_arguments(self.args,self.mapping_dict))
        string = f"{self.type}:{self.name} - Resonance(M={M0}, S={self.spin},P={self.parity}) \n{self.arguments}\n{self.bls_in} {self.bls_out}"
        return string

if __name__=="__main__":
    from AmplitudeCrafter.locals import config_dir
    res, mapping_dict = load_resonances(config_dir + "decay_example.yml")
    for r in res[1]:
        r.mapping_dict[r.data_key] = 50
        print(r.fixed())