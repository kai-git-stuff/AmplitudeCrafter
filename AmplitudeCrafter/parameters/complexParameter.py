from AmplitudeCrafter.parameters.parameterBase import failFalse, parameter, closing_index, appendName
from AmplitudeCrafter.parameters.numberParameter import number

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
        if "complex(" in string:
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

