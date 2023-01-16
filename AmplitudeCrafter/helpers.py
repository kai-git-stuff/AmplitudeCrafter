from AmplitudeCrafter.ParticleLibrary import particle
from jitter.fitting import FitParameter
from jitter.constants import spin as sp

def is_free(p):
    if isinstance(p,FitParameter):
        return not p.fixed
    return False

def get_parity(L,p1,p2):
    if L % 2 != 0:
        raise ValueError("Angular momentum has to be multiple of 2!")
    return p1 * p2 * (-1)**(L//2)

def check_bls(mother:particle,daughter1:particle,daughter2:particle,bls,parity_conserved=False):
    Ls = []
    for S in sp.couple(daughter1.spin,daughter2.spin):
        for L in sp.couple(mother.spin,S):
            if get_parity(L,daughter1.parity,daughter2.parity) == mother.parity or not parity_conserved:
                Ls.append((L,S))
    minL,minS = min(Ls,key=lambda x: x[0])
    Ls_bls = [L for L,S in bls.keys()]
    Lset = set([L for L,_ in Ls])
    Sset = set([S for L,S in Ls])
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