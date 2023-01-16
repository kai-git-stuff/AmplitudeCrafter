from jitter.fitting import FitParameter

class parameter:
    @classmethod
    @abstractmethod
    def match(string):
        # here we check if a string can be 
        # matched to this parameter types layout
        return False
    
    @classmethod
    @abstractmethod
    def evaluate():
        pass

    @classmethod
    @abstractmathod
    def final():
        # this method is to tell if this value allows a substructure
        return True
    
    # @classmethod
    # @abstractmethod
    # def strip():
    #     # here weremove the 
    #     return ""


class complexParameter(parameter):
    @classmethod
    def match(string):
        if "complex" in string:
            if "(" in string and ")" in string:
                return True
    
    @classmethod
    def final():
        return False
    
    @classmethod
    def evaluate(string:str):
        index0 = string.index("(")
        index1 = string.index(")")
        if "const" in (string[:index0] + string[index1 + 1:]):
            return [ substring.replace("const" , "") + " const"
                        for substring in complexParameter.strip()
                    ]
        else:
            return complexParameter.strip()

    @classmethod
    def strip(string:str):
        index0 = string.index("(")
        index1 = string.index(")")
        string = string[index0+1:index1]
        return string.split(",")
    

def findNext(string:str,key:str):
    words = string.split()
    if key not in words:
        return -1, None
    index = words.index(key)
    if index +1 >= len(words):
        raise ValueError(f"Found Key {key}, but no value was supplied: {string}!")
    wordsfiltered = [w for i,w in enumerate(words) if i != index and i != index + 1]
    return " ".join(wordsfiltered), words[index + 1] 


def checkConst(string:str):
    if "const" in string:
        return string.replace("const",""), True
    return False

class number(parameter):
    @classmethod
    def match(string:str):
        string, fromValue = findNext(string,"from")
        string, toValue = findNext(string,"to")
        string, isConst = checkConst(string)

        accepted = [   isConst and toValue is None and fromValue is None,
                    not isConst and toValue is not None and fromValue is not None,
                    not isConst and toValue is None and fromValue is None]
        
        return any(accepted)

        
    
    @classmethod
    def final():
        return True
    
    @classmethod
    def evaluate(name,string:str):
        initialString = string

        string, fromValue = findNext(string,"from")
        string, toValue = findNext(string,"to")
        string, isConst = checkConst(string)

        accepted = [   isConst and toValue is None and fromValue is None,
                    not isConst and toValue is not None and fromValue is not None,
                    not isConst and toValue is None and fromValue is None]
        if not any(accepted):
            raise ValueError(f"Incomplete or false input {initialString}")
        if isConst:
            return float(string)
        if toValue is not None:
            return FitParameter(name,float(string),float(fromValue), float(toValue))
        if not isConst:
            FitParameter(name,float(string),None, None)







UNDERSTOOD_PARAMS = [complexParameter,number]