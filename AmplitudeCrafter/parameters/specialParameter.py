from AmplitudeCrafter.parameters.parameterBase import failFalse, parameter, appendName

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