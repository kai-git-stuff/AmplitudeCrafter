import yaml 

def load(f:str):
    with open(f, "r") as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError("The provided yml file could not be read! Read the error printed above for more information")
    return res

def write(dtc,f:str):
    with open(f, "w") as stream:
        try:
            res = yaml.dump(dtc,stream,sort_keys=False)
        except yaml.YAMLError as exc:
            print(exc)
    return res

