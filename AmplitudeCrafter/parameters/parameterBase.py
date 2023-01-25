from jitter.fitting import FitParameter
from abc import abstractmethod, ABC
from warnings import warn

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

def failFalse(func):
    """
    returns false whenever  a call failed
    is used as a wrapper for the match functions of parameter types
    """
    def inner(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            print(e)
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

def noNameString(f):
    def inner(cls,string,*args):
        name, value, _ = findIfNamed("temporaryName", string)
        return f(cls,value)
    return inner

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