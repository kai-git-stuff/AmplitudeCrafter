
import numpy as np
from jax import numpy as jnp
from AmplitudeCrafter.ParticleLibrary import particle
from AmplitudeCrafter.TwoBodyDecay import TwoBodyDecay
from jitter.kinematics import cos_helicity_angle, boost_to_rest, mass, perpendicular_unit_vector, spatial_components, scalar_product, azimuthal_4body_angle, wigner_capital_d
from AmplitudeCrafter.DalitzAmplitude import DalitzAmplitude
from jitter.constants import spin
"""
(E/c, p1,p2,p3)

"""

PIPE = "│"
ELBOW = "└──"
TEE = "├──"
def helicityTheta(p_Parent, P_1, P_2):
    """
    boost to P_n restframe
    calculate angle between -p_Parent and P_1
    """
    P_1_boosted = boost_to_rest(P_1, p_Parent)
    P_2_boosted = boost_to_rest(P_2, p_Parent)

    return jnp.arccos(cos_helicity_angle(P_1_boosted,P_2_boosted))

def decay_plane_vector(P_1, P_2):
    """
    calculate the unit vector defining the plane spanned by the two momenta
    """
    return perpendicular_unit_vector(spatial_components(P_1), spatial_components(P_2))

class DecayTreeNode:
    def __init__(self, name,part):
        self.name= name
        self.particle = part
        self.__partent = None
        self.__decay = None
        self.__p = None
        self.__daughters = None
        self.__smp = None

    def setP(self, p):
        if not p.shape[-1] == 4:
            raise ValueError("U stupid?")
        if not callable(self.__p): 
            self.__p = p

    @property
    def p(self):
        if callable(self.__p):
            return self.__p()
        return self.__p

    @property
    def smp(self):
        """
        Get the Dalitz Sample for the decay n -> 1 2 3
        """
        if len(self.daughters) != 3:
            raise NotImplementedError("Dalitz Sample only available for three-body decays!")
        if self.__smp is not None:
            return self.__smp
        s1 = mass(self.daughters[1].p + self.daughters[2].p)**2
        s3 = mass(self.daughters[0].p + self.daughters[1].p)**2
        return jnp.stack([jnp.array(s3),jnp.array(s1)],axis=1)
    
    @smp.setter
    def smp(self,smp):
        if len(self.daughters) != 3:
            raise NotImplementedError("Dalitz Sample only available for three-body decays!")
        self.__smp = smp

    @property
    def theta(self):
        if len(self.daughters) != 2:
            raise NotImplementedError("Theta only available for two-body decays!")
        theta, phi = self.getHelicityAngles()
        return theta
    
    @property
    def phi(self):
        if len(self.daughters) != 2:
            raise NotImplementedError("Theta only available for two-body decays!")
        theta, phi = self.getHelicityAngles()
        return phi

    @property
    def daughters(self):
        if self.__daughters is None:
            return []
        else:
            return self.__daughters
    
    @property
    def parent(self):
        return self.__partent

    @property
    def phsp(self):
        if self.decay is None:
            Warning("No decay defined!")
            return None
        return self.decay.phsp

    @parent.setter
    def parent(self,p):
        if not self in p.daughters:
            raise ValueError()
        self.__partent = p

    def setDecay(self,*nodes ):    #: list[DecayTreeNode]
        self.__p = lambda : sum(n.p for n in nodes)
        self.__daughters = nodes
        for n in nodes:
            n.parent = self
        if len(nodes) == 2:
            self.__decay = TwoBodyDecay(self.particle,*[n.particle for n in nodes])
        elif len(nodes) == 3:
            self.__decay = DalitzAmplitude(self.particle,*[n.particle for n in nodes])
        else:
            raise ValueError(f"Only tow and three body decays allowed! {nodes}\n {len(nodes)}")
        
    @property
    def decay(self):
        return self.__decay
    
    @property
    def stable(self):
        return self.decay is None
    
    def traverse(self,level=0):
        yield self, level
        level = level + 1
        for d in self.daughters:
            for f,l in d.traverse(level):
                yield f, l
    
    def __repr__(self):
        return self.name + " " + str(self.particle)

    def load_resonances(self,f):
        if len(self.daughters) != 3:
            raise NotImplementedError("Only resonances for Dalitz Decays!")
        self.decay.load_resonances(f)

    def getHelicityAngles(self):
        """
        gets the helicity angles for the decay n -> ...

        needed : boost from partent(n) -> n 

        calculate theta: angle between - P (parent (n)) and daughter(1) of n in rest frame of n

        """

        if len(self.daughters) != 2 and len(self.parent.daughters) != 3:
            raise ValueError(f"Helicity angles only defined for two- body decay not {len(self.daughters)} - body decay")

        theta = helicityTheta(self.parent.p, self.daughters[0].p, self.daughters[1].p)

        ind = self.parent.daughters.index(self)
        i,j = [ a for a in [0,1,2] if a != ind]

        p0 = boost_to_rest(self.parent.daughters[i].p,self.parent.p)
        p1 = boost_to_rest(self.parent.daughters[j].p,self.parent.p)
        p2 = boost_to_rest(self.daughters[0].p,self.parent.p)
        p3 = boost_to_rest(self.daughters[1].p,self.parent.p)

        phi = azimuthal_4body_angle(p0, p1, p2, p3)

        return theta, phi
            
    def filter(self,mask):
        if len(self.daughters) == 3:
            return self.decay.phsp.inside(self.smp)
        if len(self.daughters) == 2:
            return ( jnp.isfinite(self.theta) ) & ( jnp.isfinite(self.phi) )
        return mask
        
    def getHelicityAmplitude(self,resonances=None):

        if self.decay is None:
            return None, None, None
        fs = []
        start_params = []
        helicities = []
        if len(self.daughters) == 2 :
            if resonances is not None:
                raise NotImplementedError("You cant have resonances in two body decay!")
            theta, phi = self.getHelicityAngles()
            f, start = self.decay.get_amplitude_function(theta,phi, total_absolute=False, just_in_time_compile = False,numericArgs=False)
            hel = [self] + list(self.daughters)
            fs.append(f)
            start_params.append(start)
            helicities.append(hel)

        if len(self.daughters) == 3 :
            smp = self.smp
            f, start = self.decay.get_amplitude_function(smp,total_absolute=False, just_in_time_compile = False, numericArgs=False,resonances=resonances)
            hel = [self] + list(self.daughters)
            fs.append(f)
            start_params.append(start)
            helicities.append(hel)
        for d in self.daughters:
            if d.stable:
                # check if daughter decays
                # if not we dont need to worry any further
                continue
            # recursive generation of full amplitude
            fd, startd, helicitiesd = d.getHelicityAmplitude()
            fs.append(fd)
            start_params.append(startd)
            helicities.append(helicitiesd)
        indP = [len(par) for par in start_params]
        # print(indP)
        # print(start_params)
        # would wanna use a set, but those are not ordered
        # a dict with no values will do the same
        helicy_names = {a.name:a for hel in helicities for a in hel }

        def f(args,*hel):
            f0 = None
            nPar0 = 0
            helicity_dict = {name:h for name,h in zip(helicy_names,hel)}
            for f_,nPar, H in zip(fs,indP,helicities):
                # every function has nHel helicity arguments and nPar parameter arguments
                # we need to correctly sort them
                nPar1 = nPar0 + nPar
                p  = args[nPar0:nPar1]
                nPar0 = nPar1

                h = [helicity_dict[node.name] for node in H]
                if f0 is None:
                    f0 = f_(p,*h)
                else:
                    f0 = f_(p,*h) * f0
            return f0

        return f,[ a for par in start_params for a in par ], [ helicy_names[a] for a in helicy_names ]

    def get_helicities(self):

        return list(
            range(-self.particle.spin, self.particle.spin + 1, 2)
        )

class DecayTree:
    def __init__(self,root):
        self.root = root
    
    def traverse(self):
        for a in self.root.traverse():
            yield a
    
    def getHelicityAmplitude(self,resonances = None):
        f, args, hel = self.root.getHelicityAmplitude(resonances=resonances)
        return f, args, hel

    def draw(self):
        for n,l in self.traverse():
            prefix =  "    "*l + PIPE
            print( prefix)
            mid = "" if l == 0 else TEE
            print("    "*l + mid + " " + str(n))

    def filter(self,var):
        mask = self.root.filter(None)
        if mask is None:
            raise ValueError(f"Root {self.root} is stable")
        for n,l in self.traverse():
            mask = mask & n.filter(mask)
        return var[mask]
    
    def self_filter(self):
        """
        Filter the momenta of the particles in the decay tree
        such that we only keep the ones that are inside the phase space
        """
        mask = self.root.filter(None)
        if mask is None:
            raise ValueError(f"Root {self.root} is stable")
        for node, l in self.traverse():
            mask = mask & node.filter(mask)
        for node, l in self.traverse():
            node.setP(node.p[mask])

    def get_helicities(self):
        nodes = [ n for n,l in self.traverse()]
        helicities = [[h] for h in nodes[0].get_helicities()]

        for node in nodes[1:]:
            helicities = [h + [h_] for h in helicities for h_ in node.get_helicities()]
        return helicities
    
    def apply_spin_denysity(self,f):
        mother_spin = self.root.particle.spin

        helicities = np.array(self.get_helicities())[:,1:]

        alpha, beta, gamma = None, None, None # TODO: get these from the decay
        
        D_matrices = {(eta, nu): wigner_capital_d(alpha, beta, gamma,eta, nu) for eta in spin.direction_options(mother_spin) for nu in spin.direction_options(mother_spin)}

        def select(rho,L,L_):
            L, L_ = (L + mother_spin)//2, (L + mother_spin)//2
            return rho.reshape((mother_spin + 1,mother_spin + 1))[L, L_]

        def f_(rho, args):
            sm = 0
            for L in spin.direction_options(mother_spin):
                for L_ in spin.direction_options(mother_spin):
                    
                    sm = sm + select(rho, L, L_)  * sum(
                        jnp.conj(D_matrices[(L,nu)]) * D_matrices[(L_,nu_)] * f(args,nu,*h) * jnp.conj(f(args,nu_,*h)) 
                            for h in helicities
                                for nu in spin.direction_options(mother_spin)
                                    for nu_ in spin.direction_options(mother_spin)
                                )
            return sm
        from jitter.fitting.fitting import FitParameter
        rho_start = [
            [FitParameter(f"SpinDensity{L:.0f}{L_:.0f}",1,-1,1) for L in spin.direction_options(mother_spin)]
                for L_ in spin.direction_options(mother_spin)
            ]
        return f_, rho_start

if __name__ == "__main__":
    pass