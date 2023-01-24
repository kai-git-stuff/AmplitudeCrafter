from AmplitudeCrafter.parameters.parameterBase import failFalse, parameter, closing_index, appendName, noNameString
from AmplitudeCrafter.parameters.numberParameter import number
from jax import numpy as jnp

class sphericalComplexParameter(parameter):

    """
    complex number based on R and Phi
    consists of two internal number parameters, which are combined in the correct way, whenever 
    the value of the complex is demanded
    The complex can not be updated as a whole!!
    """

    @classmethod
    @noNameString
    @failFalse
    def match(cls,string):
        if not isinstance(string,str):
            return False
        if "complexRPhi(" in string:
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
        index0 = string.index("complexRPhi(") + len("complexRPhi")
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
        return self.name , self.R.name, self.Phi.name
    
    def generate_param_names(self):
        return self.name, self.name+"_R", self.name+"_PHI"

    @property
    def dict(self):
        return {n:p for n,p in zip(self.param_names[1:],self.parameters)}

    @property
    def parameters(self):
        return self.R, self.Phi

    def __init__(self,name,string):
        if not super().__init__(name):
            return 

        string = self.value_string
        const, (R_str, Phi_str) = type(self).evaluate(string)
        self.const = const
        self.R_string = R_str
        self.Phi_string = Phi_str
        complex_name, real_name, imag_name = self.generate_param_names()
        self.R = number(real_name, R_str)
        self.Phi = number(imag_name, Phi_str)

    def __call__(self,numeric=True, value_dict = None):
        if numeric is False:
            return self.R(False), self.Phi(False)
        return self.R(numeric=True,value_dict=value_dict) * jnp.exp(1j * (self.Phi(numeric=True,value_dict=value_dict)))

    @classmethod
    def strip(cls,string:str):
        index0 = string.index("complexRPhi(") + len("complexRPhi")
        index1 = closing_index(string,index0) 
        string = string[index0+1:index1]
        return string.split(",")
    
    def update(self,val):
        raise ValueError("Update Call on complex number!")
        # updtes do nothing on complex nubers

    @appendName
    def dump(self):
        R_string = self.R.dump()
        Phi_string = self.Phi.dump()

        if self.const:
            R_string = R_string.replace("const", "")
            Phi_string = Phi_string.replace("const", "")
        
            return f"complexRPhi({R_string}, {Phi_string}) const"
        return f"complexRPhi({R_string}, {Phi_string})"

