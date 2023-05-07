from AmplitudeCrafter.parameters.parameterBase import fail_false, parameter, find_next, check_const, check_float, FitParameter, append_name,   no_name_string


class number(parameter):
    """
    Basic numbers. 
    The main used type of parameter. 
    Checks for const, to and from
    """

    @classmethod
    @no_name_string
    @fail_false
    def match(cls,string:str):
        if not isinstance(string,str):
            return True
        string, fromValue = find_next(string,"from")
        string, toValue = find_next(string,"to")
        string, isConst = check_const(string)
        isCastable = check_float(string)
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
        string, fromValue = find_next(string,"from")
        string, toValue = find_next(string,"to")
        string, isConst = check_const(string)

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
        const, value = number.evaluate(self.name,string)
        self.const = const
        self.value = value

    
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
    
    @append_name
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
