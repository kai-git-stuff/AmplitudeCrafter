from jitter.fitting import FitParameter
from abc import abstractmethod

def failFalse(func):
    # returns false whenever 
    def inner(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            print(e)
            return False
    return inner

class parameter:

    parmeters = dict()

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

    @abstractmethod
    def __call__(self):
        # Calling is supposed to retrieve the underlying number itself
        raise NotImplementedError()
    
    @abstractmethod
    def __init__(self,string:str,name:str):
        # __init__ usually calls to class.evaluate, and then sets an internal state
        parameter.parmeters[name] = self
    
    @property
    @abstractmethod
    def dict(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

class complexParameter(parameter):
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
        return self.name

    @classmethod
    def final(cls):
        return False
    
    @classmethod
    def evaluate(cls,string:str):
        index0 = string.index("(")
        index1 = string.index(")")
        if "const" in (string[:index0] + string[index1 + 1:]):
            return True, [ substring.replace("const" , "") + " const"
                        for substring in cls.strip(string)
                    ]
        else:
            return False, cls.strip(string)
    
    @property
    def param_names(self):
        # the three parameters added by this class
        return self.name , self.name + "_REAL", self.name + "_IMAG"
    
    @property
    def dict(self):
        return {n:p for n,p in zip(self.param_names[1:],self.parameters)}

    @property
    def parameters(self):
        return self.real, self.imag

    def __init__(self,string,name):
        const, (real_str, imag_str) = complexParameter.evaluate(string)
        self.const = const
        self.real_string = real_str
        self.imag_string = imag_str
        self.name = name
        complex_name, real_name, imag_name = self.param_names
        self.real = number(real_str, real_name)
        self.imag = number(imag_str, imag_name)

    def __call__(self,numeric=True):
        if numeric is False:
            return self.real(False), self.imag(False)
        return self.real(numeric=True) + 1j * (self.imag(numeric=True))

    @classmethod
    def strip(cls,string:str):
        index0 = string.index("(")
        index1 = string.index(")")
        string = string[index0+1:index1]
        return string.split(",")
    
    def update(self,val):
        # updtes do nothing on complex nubers
        pass

def findNext(string:str,key:str):
    words = string.split()
    if key not in words:
        return string, None
    index = words.index(key)
    if index +1 >= len(words):
        raise ValueError(f"Found Key {key}, but no value was supplied: {string}!")
    wordsfiltered = [w for i,w in enumerate(words) if i != index and i != index + 1]
    return " ".join(wordsfiltered), float(words[index + 1])

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
        return self.name

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
    def evaluate(cls,name,string:str):
        if not isinstance(string,str):
            string = str(string)
        initialString = string
        string, fromValue = findNext(string,"from")
        string, toValue = findNext(string,"to")
        string, isConst = checkConst(string)

        if not number.match(string):
            raise ValueError(f"Value '{initialString}' of parameter {name} does not match pattern of a number!")
        if isConst:
            for cast in [int,float]:
                try: 
                    val = cast(string)
                    break
                except:
                    continue
            return isConst,val
        if toValue is not None:
            return isConst,FitParameter(name,float(string),float(fromValue), float(toValue))
        if not isConst:
            return isConst,FitParameter(name,float(string),None, None)

    def __init__(self,string,name):
        const, value = number.evaluate(name,string)
        self.const = const
        self.value = value
        self.name = name
        self.special=False
    
    def __call__(self,numeric=False):
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


class specialParameter(parameter):
    specialSymbols = ["sigma1","sigma2","sigma3","L","L_0"]
    values = {}
    @classmethod
    @failFalse
    def match(cls,string:str):
        if not isinstance(string,str):
            return False
        return string in  cls.specialSymbols
    
    @classmethod
    def final(cls):
        return True
    
    def __repr__(self):
        return self.name + " " + repr(self(True))

    @property
    def dict(self):
        return {self.name:self}

    @property
    def parameters(self):
        return self

    @classmethod
    def evaluate(cls,name,string:str):
        pass

    def __init__(self,string,name):
        self.const = True
        self.name = string
        # value will be updated, but this is not a fit parameter, so it is const
        self.value = None
    
    def __call__(self,numeric=False):
        return specialParameter.values.get(self.name,None)

    def update(self,val):
        specialParameter.values[self.name] = val


UNDERSTOOD_PARAMS = [complexParameter,number,specialParameter]