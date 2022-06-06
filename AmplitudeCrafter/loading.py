import yaml 

def load(f:str):
    with open(f, "r") as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return res

def write(dtc,f:str):
    with open(f, "w") as stream:
        try:
            res = yaml.dump(dtc,stream)
        except yaml.YAMLError as exc:
            print(exc)
    return res

