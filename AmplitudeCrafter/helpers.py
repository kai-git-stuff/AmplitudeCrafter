from AmplitudeCrafter.ParticleLibrary import particle
from jitter.fitting import FitParameter
from jitter.constants import spin as sp
from AmplitudeCrafter.parameters import parameter, lambdaParameter
import warnings

def is_free(p):
    return not p.const

def get_parity(L,p1,p2):
    if L % 2 != 0:
        raise ValueError("Angular momentum has to be multiple of 2!")
    return p1 * p2 * (-1)**(L//2)

def ensure_numeric(p):
    if isinstance(p,parameter):
        p = p(numeric=True)
    if isinstance(p,FitParameter):
        p = p()
    return float(p)

def check_bls(mother:particle,daughter1:particle,daughter2:particle,bls,parity_conserved=False) -> dict:
    Ls = []
    for S in sp.couple(daughter1.spin,daughter2.spin):
        for L in sp.couple(mother.spin,S):
            if get_parity(L,daughter1.parity,daughter2.parity) == mother.parity or not parity_conserved:
                Ls.append((L,S))
    minL,minS = min(Ls,key=lambda x: x[0])
    Ls_bls = [L for L,S in bls.keys()]
    Lset = set([L for L,_ in Ls])
    Sset = set([S for L,S in Ls])
    checkSet = set(Ls)
    if min(Ls_bls) != minL:
        raise ValueError(f"""Lowest partial wave {(minL,minS)} not contained in LS couplings {list(bls.keys())}!
        Values {mother} -> {daughter1} {daughter2} 
        Parity{" " if parity_conserved else " not "}conserved!""")
    if not all([L in Lset for L,S in bls.keys()]):
        string = "; ".join([str(L) for L,S in bls.keys() if not S in Lset])
        raise ValueError(f"""Not all L couplings possible! {string} 
                        For decay {mother} -> {daughter1} {daughter2}""")
    if not all([S in Sset for L,S in bls.keys()]):
        string = "; ".join([str(S) for L,S in bls.keys() if not S in Sset])
        raise ValueError(f"""Not all S couplings possible! {string} 
                        For decay {mother} -> {daughter1} {daughter2}""")
    for (L, S) in bls.keys():
        if not (L,S) in checkSet:
            raise ValueError(f"""Partial wave {(L,S)} not contained in LS couplings {Ls}!
            Values {mother} -> {daughter1} {daughter2} 
            Parity{" " if parity_conserved else " not "}conserved!""")

    if any([daughter1.mass == 0,daughter2.mass == 0]):
        # if (len(bls.keys()) + 2) >= len(checkSet):
        #     raise ValueError(f"""Not all partial waves are possible for massless daughters! 2 can be set by the others!
        #     Values {mother} -> {daughter1} {daughter2} 
        #     Parity{" " if parity_conserved else " not "}conserved!""")
        missing = checkSet - set(bls.keys())
        # if len(missing) > 2:
        #     raise ValueError(f"""Too many partial waves missing for massless daughters! 2 can be set by the others!
        #     Values {mother} -> {daughter1} {daughter2} 
        #     Parity{" " if parity_conserved else " not "}conserved!""")
        if len(missing) == 2:
            warnings.warn(f"""Two partial waves missing for massless daughters! 2 can be set by the others!
                          Default behaviour will set these two partial waves now!""")
            # here we can try to set the missing ones
            # what we will do is inject a fit parameter of the lambda type, which we will then actually also put in the output yaml
            # this is a bit of a hack, but it should work
            # also this may cause the output to fail due to the rest of this code

            # the not gamma is the resonance
            resonance = daughter1 if daughter1.mass != 0 else daughter2
            
            # we need a frozen version here
            bls_frozen = bls.copy()
            for L, S in missing:
                parameter_names = 'abcdefghijklmnop'[:len(bls_frozen.keys())]
                name = f"{resonance.name}=>autoSetBLSL:{L},S:{S}"
                lambda_string = f"lambda({', '.join(parameter_names)}: {' + '.join(parameter_names)}) ({'; '.join([param.name for param in bls_frozen.values()])})"
                print("Setting",name,lambda_string)
                bls[(L, S)] = lambdaParameter(name, lambda_string)
    return bls
    
def flatten(listoflists):
    lst = []
    def flatten_recursive(listoflists,ret_list:list):
        if isinstance(listoflists,list):
            [flatten_recursive(l,ret_list) for l in listoflists] 
            return 
        ret_list.append(listoflists)
    flatten_recursive(listoflists,lst)
    return lst