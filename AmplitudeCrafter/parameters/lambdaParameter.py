from AmplitudeCrafter.parameters.parameterBase import fail_false, parameter, closing_index, append_name, no_name_string, inside
from AmplitudeCrafter.parameters.numberParameter import number
from AmplitudeCrafter.parameters.complexParameter import complexParameter
from AmplitudeCrafter.parameters.shpericalComplexParameter import sphericalComplexParameter

class lambdaParameter(parameter):

    """
    parameter, which is dynamically generated from a lambda function using other parameters.
    Usage of these parameters is rarely usefull  

    Syntax: lambda(x,y,z,... : x**2 = y - z**0.8) (name1; name2; name3; name4; 0.5 form -5 to 5 NAMED(param3)) 
    we use the semicolons as dividers, to be able to define (named!!!!) parameters for a lambda parameter
    This is only useful in verry rare circumstances
    The value can not be updated as a whole!!
    """
    @classmethod
    @no_name_string
    @fail_false
    def match(cls,string):
        if "lambda" in string and not "const" in string:
            return True
        return False

    def __repr__(self):
        return "Parameter:" + " " + self.name

    @classmethod
    def final(cls):
        return False
    
    @classmethod
    def evaluate(cls,string:str,name):
        if not cls.match(string):
            raise ValueError(f"Given string {string} doe not match pattern of {cls}!" )

        lambda_string,opening,closing = inside("lambda(",string)
        parameterStrings = string[closing+1:]
        parameter_definitions = inside("(",parameterStrings)[0].split(";")

        def getParameter(n,string:str):
            understood_params = [number,complexParameter,sphericalComplexParameter]
            p = parameter.parameters.get(string.strip())
            if p is not None:
                return p
            matching_signatures = [param for param in understood_params if param.match(string)]
            high_level_matches = [param for param in matching_signatures if not param.final()]
            low_level_matches = [param for param in matching_signatures if param.final()]
            if len(high_level_matches) > 1:
                raise ValueError(f"More than one signature matches value {string}!")
            if len(high_level_matches) == 1:
                matchingType, = high_level_matches
            elif len(low_level_matches) == 1:
                matchingType, = low_level_matches
            else:
                raise ValueError(f"No matching signarture for {string}!")
            return matchingType(n,string)
        
        parameters = [getParameter(name+"_"+str(i),parameterString) for i,parameterString in enumerate(parameter_definitions)]
        f = eval("lambda " + lambda_string)

        return f,lambda_string,parameters   
 
    @property
    def param_names(self):
        # the three parameters added by this class
        return [p.name for p in self.parameters]

    @property
    def dict(self):
        dtc = {p.name:p for p in self.parameters}
        dtc.update({self.name:self})
        return dtc

    @property
    def parameters(self):
        return self.__parameters

    def __init__(self,name,string):
        if not super().__init__(name):
            return 
        self.const = True
        lambda_function, lambda_string, parmeters = type(self).evaluate(string,name) 

        self.__parameters = parmeters
        self.f = lambda_function
        self.lambda_string = lambda_string

    def args(self,numeric=True,value_dict=None):
        return[p(numeric = numeric,value_dict = value_dict) for p in self.parameters]

    def __call__(self,numeric=True, value_dict = None):
        if numeric is False:
            raise ValueError(f"Parameter of type {type(self)} can not be evaluated in a non numeric way!")
        return self.f(*self.args(numeric=True,value_dict=value_dict))
    
    def update(self,val):
        raise ValueError("Update Call on lambda genrated number!")
        # updtes do nothing on complex nubers

    @append_name
    def dump(self):
        dumped_params = ";".join([p.dump() for p in self.parameters])
        return f"lambda({self.lambda_string}) ({dumped_params})"

