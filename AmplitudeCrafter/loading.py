import yaml 

def load(f:str):
    import yaml
    with open(f, "r") as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return res
