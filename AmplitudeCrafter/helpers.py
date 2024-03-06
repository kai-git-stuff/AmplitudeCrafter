from AmplitudeCrafter.ParticleLibrary import particle
from jitter.fitting import FitParameter
from jitter.constants import spin as sp
from jitter.kinematics import clebsch
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

def get_maxLS_prefactors(mother: particle, child: particle, massless_child: particle, ls: set, lmax: int, smax: int) -> dict:
    # other daughter is assumed to be photon
    def cs(L, S, m_lambda):
        return clebsch(2, 0, child.spin, m_lambda, S, 0 + m_lambda) * clebsch(L, 0, S, 1, mother.spin, 0 + m_lambda)
    prefactors_ls_max = {}
    for (L, S) in ls:
        if L == lmax:
            prefactors_ls_max[(L, S)] = 0
            continue
        prefactors_ls_max[(L, S)] = cs(L, S, 1) - cs(L, S, -1) / (cs(lmax, smax, 1) - cs(lmax, smax, -1))

    prefactors_ls_max_minus = {}
    for (L, S) in ls:
        if S == smax:
            prefactors_ls_max_minus[(L, S)] = 0
            continue
        factor = 1 / (
            (cs(lmax - 2, smax, 1) - cs(lmax - 2, smax, -1)) / (cs(lmax, smax, 1) - cs(lmax, smax, -1)) +
            (cs(lmax - 2, smax, 1) + cs(lmax - 2, smax, -1)) / (cs(lmax, smax, 1) + cs(lmax, smax, -1))
        )
        prefactors_ls_max_minus[(L, S)] = factor * (
            (cs(L, S, 1) - cs(L, S, -1)) / (cs(lmax, smax, 1) - cs(lmax, smax, -1)) +
            (cs(L, S, 1) + cs(L, S, -1)) / (cs(lmax, smax, 1) + cs(lmax, smax, -1))
        )
    return prefactors_ls_max, prefactors_ls_max_minus


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
            gamma = daughter2 if daughter1.mass != 0 else daughter1
            
            # we need a frozen version here
            bls_frozen = bls.copy()

            missing_L_sorted = sorted(list(missing), key = lambda x: x[0])
            conditions = [missing_L_sorted[-1] == (max(Lset),max(Sset)),missing_L_sorted[0] == (max(Lset)-2,max(Sset))]
            if missing_L_sorted[-1][0] != max(Lset):
                raise ValueError(f"""Only the highest and second highest parital waves can be automatically replaced. For the decay {mother} -> {daughter1} {daughter2} this would be
                                 (L,S) = ({max(Lset)}, {max(Sset)}) and ({max(Lset)-2}, {max(Sset)})""")
            
            prefactors_ls_max, prefactors_ls_max_minus = get_maxLS_prefactors(mother,resonance,gamma,bls, max(Lset), max(Sset))
            for (L, S), prefactors in zip(missing_L_sorted,[prefactors_ls_max_minus, prefactors_ls_max]):
                # we will get a lambda parameter for the missing ones
                # we calculate all the prefactors first and then get the nice lambda parameter string
                parameter_names = 'abcdefghijklmnop'[:len(bls_frozen.keys())]
                parameter_times_factor = [f"{name} * {float(prefactors[(L,S)])}" for name, (L, S) in zip(parameter_names, bls_frozen.keys())]
                name = f"{resonance.name}=>autoSetBLSL:{L},S:{S}"
                lambda_string = f"lambda({', '.join(parameter_names)}: {' + '.join(parameter_times_factor)}) ({'; '.join([param.name for param in bls_frozen.values()])})"
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