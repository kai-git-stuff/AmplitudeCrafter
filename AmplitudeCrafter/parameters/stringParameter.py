from AmplitudeCrafter.parameters.parameterBase import fail_false, parameter, append_name, check_const, no_name_string
from warnings import warn


class stringParam(parameter):
    """
    The fallback parameter type
    If a string could not be interpreted it will be seen as string an treated as constant
    This may change soon, as this method will stop a proper exception to be raised in case
    of a small syntax error
    """
    @classmethod
    @no_name_string
    @fail_false
    def match(cls,string:str):
        if not isinstance(string,str):
            return False
        return True
    @append_name
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
        string, isDecalredConst = check_const(string)
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