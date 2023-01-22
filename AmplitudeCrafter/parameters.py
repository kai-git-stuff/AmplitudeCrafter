from jitter.fitting import FitParameter
from abc import abstractmethod, ABC
from warnings import warn

def failFalse(func):
    """
    returns false whenever  a call failed
    is used as a wrapper for the match functions of parameter types
    """
    def inner(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            return False
    return inner

def find_parentesis(string:str):
    """
    helper to identify where brakcets open and close
    """
    opening = [i for i,c in enumerate(string) if c == "("]
    closing = [i for i,c in enumerate(string) if c == ")"]
    return opening, closing

def closing_index(string:str,opening_index):
    """
    helper function to get the index where a given parentesis is closed
    """
    opening, closing = find_parentesis(string)
    if opening_index not in opening:
        raise ValueError(f"Index {opening_index} not in index list for opening parentesis {opening}!s")
    n_open = opening.index(opening_index) + 1
    return closing[-(n_open)]

def findIfNamed(name, value):
    """
    name: inital name of the parameter
    value: the definition string of the parameter

    returns a new name and a value, where the value has the naming string removed if it was present
    and the new name is the corresponding name if a naming string was present
    otherwise it is the initial name
    """

    if not isinstance(value,str):
        return name, value, False
    opening, closing = find_parentesis(value)
    if "NAMED(" in value and len(closing) > 0:
        # simple check is enough, as only the name can be inside the parentesis and there may be no
        # parentesis in the name itself
        index0 = value.index("NAMED(") + len("NAMED")
        index1 = value[index0:].index(")") + index0
        if index1 != closing[-1]:
            return name,value, False
        new_name = value[index0+1:index1]
        new_value = (value[:index0] + value[index1+1:]).replace("NAMED","")
        return new_name, new_value, True
    return name, value, False

def appendName(f):
    """
    decorator that adds the name of a parameter, in case it was named explicitly
    """
    def inner(param):
        name,_,named = findIfNamed(param.name,param.initial_value)
        if named:
            return f(param) + f" NAMED({name})"
        return f(param) + ""
    return inner

def tryFloat(string:str):
    """
    helper, that casts to float if possible
    only accepts string
    """
    if not isinstance(string,str):
        raise ValueError(f"Only string allowed not {type(string)}!")
    try:
        return float(string)
    except:
        return string

class ParameterScope:
    """
    Helper class to change parameter scope in case multiple amplitudes get defined in the same file
    This class is only used by DalitzAmplitude interanly
    Can be omitted for other cases
    """
    def __init__(self,dict=None):
        if dict is None:
            self.dict = {}
        else:
            self.dict={}
        
    def __enter__(self):
        self.dict_before = parameter.setBackend(self.dict)
        return self
    def __exit__(self,*args):
        parameter.setBackend(self.dict_before)

class parameter(ABC):
    """
    parameter base class (ABC) enforces, taht all abstract classes are defined
    most important: the __new__ method controlls the namespace where parameters with the same name are
    the same parameter

    a namespace is given by the static 'parameters' dictionary
    a non global namespace is best entered using the Parameter scope helper class
    """
    parameters = dict()

    def check_exists(self):
        return hasattr(self,"name") and getattr(self,"name") in  parameter.parameters  

    @classmethod
    @abstractmethod
    def match(cls,string):
        # here we check if a string can be 
        # matched to this parameter types layout
        return False
    
    @classmethod
    @abstractmethod
    def evaluate(cls,):
        pass

    @classmethod
    @abstractmethod
    def final(cls,):
        # this method is to tell if this value allows a substructure
        return True

    @classmethod
    def clear(cls):
        cls.parameters = {}
    
    @classmethod
    def setBackend(cls,known_params:dict):
        dtc = cls.parameters
        cls.parameters = known_params
        return dtc

    @abstractmethod
    def __call__(self):
        # Calling is supposed to retrieve the underlying number itself
        raise NotImplementedError()
    
    @abstractmethod
    def __init__(self,name:str):
        if self.check_exists():
            return False
        parameter.parameters[self.name] = self
        return True
        # __init__ usually calls to class.evaluate, and then sets an internal state

    @property
    @abstractmethod
    def dict(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
        cls = type(self)
        new_obj = object.__new__(cls)
        new_obj.__dict__ = self.__dict__
        return new_obj
    
    def __new__(cls,name,value,*args,**kwargs):
        # ensure, that named parameters only exist once per name
        new_obj = object.__new__(cls)
        new_obj.initial_value = value
        name, value, named = findIfNamed(name,value) 
        new_obj.name = name
        new_obj.value_string = value            
        return parameter.parameters.get(name,new_obj)

class complexParameter(parameter):

    """
    complex number
    consists of two internal number parameters, which are combined in the correct way, whenever 
    the value of the complex is demanded
    The complex can not be updated as a whole!!
    """

    @classmethod
    @failFalse
    def match(cls,string):
        if not isinstance(string,str):
            return False
        if "complex" in string:
            if "(" in string and ")" in string:
                return True
        return False
    
    def __repr__(self):
        return "Parameter:" + " " + self.name

    @classmethod
    def final(cls):
        return False
    
    @classmethod
    def evaluate(cls,string:str):
        index0 = string.index("complex(") + len("complex")
        index1 = closing_index(string,index0)        
        if "const" in (string[:index0] + string[index1 + 1:]):
            return True, [ substring.replace("const" , "") + " const"
                        for substring in cls.strip(string)
                    ]
        else:
            return False, cls.strip(string)
    
    @property
    def param_names(self):
        # the three parameters added by this class
        return self.name , self.real.name, self.imag.name
    
    def generate_param_names(self):
        return self.name, self.name+"_REAL", self.name+"_IMAG"

    @property
    def dict(self):
        return {n:p for n,p in zip(self.param_names[1:],self.parameters)}

    @property
    def parameters(self):
        return self.real, self.imag

    def __init__(self,name,string):
        if not super().__init__(name):
            return 

        string = self.value_string
        const, (real_str, imag_str) = complexParameter.evaluate(string)
        self.const = const
        self.real_string = real_str
        self.imag_string = imag_str
        complex_name, real_name, imag_name = self.generate_param_names()
        self.real = number(real_name, real_str)
        self.imag = number(imag_name, imag_str)

    def __call__(self,numeric=True, value_dict = None):
        if numeric is False:
            return self.real(False), self.imag(False)
        return self.real(numeric=True,value_dict=value_dict) + 1j * (self.imag(numeric=True,value_dict=value_dict))

    @classmethod
    def strip(cls,string:str):
        index0 = string.index("complex(") + len("complex")
        index1 = closing_index(string,index0) 
        string = string[index0+1:index1]
        return string.split(",")
    
    def update(self,val):
        raise ValueError("Update Call on complex number!")
        # updtes do nothing on complex nubers
        pass

    @appendName
    def dump(self):
        real_string = self.real.dump()
        imag_string = self.imag.dump()

        if self.const:
            real_string = real_string.replace("const", "")
            imag_string = imag_string.replace("const", "")
        
            return f"complex({real_string}, {imag_string}) const"
        return f"complex({real_string}, {imag_string})"


def findNext(string:str,key:str):
    words = string.split()
    if key not in words:
        return string, None
    index = words.index(key)
    if index +1 >= len(words):
        raise ValueError(f"Found Key {key}, but no value was supplied: {string}!")
    wordsfiltered = [w for i,w in enumerate(words) if i != index and i != index + 1]
    return " ".join(wordsfiltered), tryFloat(words[index + 1])

def checkConst(string:str):
    if "const" in string:
        return string.replace("const",""), True
    return string, False

def checkFloat(string:str):
    try:
        float(string)
        return True
    except:
        return False

class number(parameter):
    """
    Basic numbers. 
    The main used type of parameter. 
    Checks for const, to and from
    """

    @classmethod
    @failFalse
    def match(cls,string:str):
        if not isinstance(string,str):
            return True
        string, fromValue = findNext(string,"from")
        string, toValue = findNext(string,"to")
        string, isConst = checkConst(string)
        isCastable = checkFloat(string)
        accepted = [   isConst and toValue is None and fromValue is None,
                    not isConst and toValue is not None and fromValue is not None,
                    not isConst and toValue is None and fromValue is None]
        
        return any(accepted) and isCastable
    
    def __repr__(self):
        return "Parameter:" + " " + self.name

    @classmethod
    def final(cls):
        return True
    
    @property
    def dict(self):
        return {self.name:self}

    @property
    def parameters(self):
        return self

    @classmethod
    def evaluate(cls,name:str,string:str):
        if not isinstance(string,str):
            string = str(string)
        initialString = string
        string, fromValue = findNext(string,"from")
        string, toValue = findNext(string,"to")
        string, isConst = checkConst(string)

        if not number.match(string):
            raise ValueError(f"Value '{initialString}' of parameter {name} does not match pattern of a number!")
        if isConst:
            # sometimes we want integers for constant values, as they will be indices or such
            for cast in [int,float]:
                try: 
                    val = cast(string)
                    break
                except:
                    continue
            return isConst,val
        if toValue is not None:
            if not (fromValue <= float(string) and toValue >= float(string)):
                raise ValueError(f"Parameter {name} is outside the specified interval!")
            return isConst,FitParameter(name,float(string),float(fromValue), float(toValue))
        if not isConst:
            return isConst,FitParameter(name,float(string),None, None)

    def __init__(self,name:str,string:str):
        if not super().__init__(name):
            return 
        string = self.value_string
        const, value = number.evaluate(name,string)
        self.const = const
        self.value = value
        super().__init__(name)

    
    def __call__(self,numeric=False,value_dict=None):
        if value_dict is not None:
            if numeric is True:
                return value_dict[self.name]
            raise ValueError("Value Dict supplied, but")

        if self.const is True:
            if numeric is False:
                raise ValueError("Constant parameter can not be returned as fit parameter!")
            return self.value
        if numeric:
            return self.value()
        return self.value

    def update(self,val):
        if self.const:
            raise ValueError("Can not update constant parameter!")
        else:
            if isinstance(val,FitParameter):
                val = val()
            self.value.update(val)
    @appendName
    def dump(self):
        additions = []
        if self.const:
            additions.append("const")
            return str(self(numeric=True)) + " " + " ".join(additions)
        if isinstance(self.value,FitParameter):
            if self.value.lower_limit is not None:
                additions.append("from")
                additions.append(str(self.value.lower_limit))
            if self.value.upper_limit is not None:
                additions.append("to")
                additions.append(str(self.value.upper_limit))
            return str(self(numeric=True)) + " " + " ".join(additions)
        raise ValueError("This number does not match any dump pattern!")

class specialParameter(parameter):
    """
    Idea: special parameters may only exist in one version per scope
    These parameters are defined by keywords and can be put anywhere in the parameter lists
    Parameters L and L_0 are int
    sigma1, sigma2 and sigma3 are the dalitz plot variables.
    """
    specialSymbols = ["sigma1","sigma2","sigma3","L","L_0"]
    values = {}
    @classmethod
    @failFalse
    def match(cls,string:str):
        if not isinstance(string,str):
            return False
        return string.strip() in  cls.specialSymbols
    
    @appendName
    def dump(self):
        # the value can be whatever, but the name is constant here
        return self.name

    @classmethod
    def load_specials(cls):
        return {sym: cls(sym,sym) for sym in cls.specialSymbols}
    
    @classmethod
    def final(cls):
        return True
    
    def __repr__(self):
        return "Parameter:" + " " + self.name + " " + repr(self(True))

    @property
    def dict(self):
        return {self.name:self}

    @property
    def parameters(self):
        return self

    @classmethod
    def evaluate(cls,name,string:str):
        pass

    def __init__(self,name:str,string:str):
        if not super().__init__(name):
            return 
        string = self.value_string
        self.const = True
        self.name = string
        # value will be updated, but this is not a fit parameter, so it is const
        self.value = None
    
    def __call__(self,numeric=False,value_dict=None):
        if value_dict is not None:
            if numeric is True:
                return value_dict[self.name]
            raise ValueError("Value Dict supplied, but non numeric values wanted!")
        return specialParameter.values.get(self.name,None)

    def update(self,val):
        specialParameter.values[self.name] = val

class stringParam(parameter):
    """
    The fallback parameter type
    If a string could not be interpreted it will be seen as string an treated as constant
    This may change soon, as this method will stop a proper exception to be raised in case
    of a small syntax error
    """
    @classmethod
    @failFalse
    def match(cls,string:str):
        if not isinstance(string,str):
            return False
        return True
    @appendName
    def dump(self):
        # the value can be whatever, but the name is constant here
        return self.value

    @classmethod
    def load_specials(cls):
        return {sym: cls(sym,sym) for sym in cls.specialSymbols}
    
    @classmethod
    def final(cls):
        return True
    
    def __repr__(self):
        return "Parameter:" + " " + self.name + " " + repr(self(True))

    @property
    def dict(self):
        return {self.name:self}

    @property
    def parameters(self):
        return self

    @classmethod
    def evaluate(cls,name,string:str):
        pass

    def __init__(self,name:str,string:str):
        if not super().__init__(name):
            return 
        # remove a potential const declaration
        string = self.value_string
        string, isDecalredConst = checkConst(string)
        if isDecalredConst:
            warn(f"Parameter {name} is of type string and was declared const! The explicit declaration can be omitted!")
        self.const = True
        # value will be updated, but this is not a fit parameter, so it is const
        self.value = string.strip()
        super().__init__(name)

    def __call__(self,numeric=True, value_dict=None):
        if numeric is False:
            raise ValueError("String parameters can only be constant!")
        return self.value

    def update(self,val):
        raise ValueError()

UNDERSTOOD_PARAMS = [complexParameter,number,specialParameter]

FALLBACK_PARAMETERS = [stringParam]