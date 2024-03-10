import yaml 
from AmplitudeCrafter.locals import logger
def make_tuple(dtc):
    if isinstance(dtc,dict):
        return {k: make_tuple(v) for k,v in dtc.items()}
    if isinstance(dtc,list) or isinstance(dtc,tuple):
        return tuple([make_tuple(v) for v in dtc])
    if isinstance(dtc,str) or isinstance(dtc,float) or isinstance(dtc,int):
        return dtc
    raise ValueError(f"Only list, tuple, dict, string, float and int are allowed data types. Not {type(dtc)}")


def load(f:str):
    with open(f, "r") as stream:
        try:
            res = yaml.safe_load(stream)
            # res =  make_tuple(res)
        except yaml.YAMLError as exc:
            logger.warn(exc)
            raise ValueError("The provided yml file could not be read! Read the error printed above for more information")
    return res

def write(dtc,f:str):
    with open(f, "w") as stream:
        try:
            res = yaml.dump(dtc,stream,sort_keys=False)
        except yaml.YAMLError as exc:
            logger.warn(exc)
    return res

