
import numpy as np
from jax import numpy as jnp
from AmplitudeCrafter.ParticleLibrary import particle
from AmplitudeCrafter.TwoBodyDecay import TwoBodyDecay
from jitter.kinematics import cos_helicity_angle, boost_to_rest, mass, perpendicular_unit_vector, spatial_components, scalar_product
from AmplitudeCrafter.DalitzAmplitude import DalitzAmplitude

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

    return perpendicular_unit_vector(spatial_components(P_1), spatial_components(P_2))

class DecayTreeNode:
    def __init__(self, name,part):
        self.name= name
        self.particle = part
        self.__partent = None
        self.__decay = None
        self.__p = None
        self.__daughters = None

    def setP(self, p):
        if not p.shape[-1] == 4:
            raise ValueError("U stupid?")

        self.__p = p

    @property
    def p(self):
        if callable(self.__p):
            return self.__p()
        return self.__p

    @property
    def daughters(self):
        if self.__daughters is None:
            return []
        else:
            return self.__daughters
    
    @property
    def parent(self):
        return self.__partent

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

    def getHelicityAngles(self,n):
        """
        gets the helicity angles for the decay n -> ...

        needed : boost from partent(n) -> n 

        calculate theta: angle between - P (parent (n)) and daughter(1) of n in rest frame of n

        """

        if len(n.daughters) != 2:
            raise ValueError(f"Helicity angles only defined for two- body decay not {len(n.daughters)} - body decay")
        
        theta = helicityTheta(self.p, n.daughters[0].p, n.daughters[1].p)

        phi = 0.

        if self.parent is not None:
            # the plane is defined by only 2 of the angles
            phi = jnp.arcos(scalar_product(decay_plane_vector(self.parent.p.daughters[0].p, self.parent.p.daughters[1].p), 
                    decay_plane_vector(n.daughters[0].p, n.daughters[1].p)))
        return theta, phi
            
        
    def getHelicityAmplitude(self):
        if self.decay is None:
            return None, None, None
        fs = []
        start_params = []
        helicities = []
        if len(self.daughters) == 2 :
            theta = helicityTheta(self.p, *[d.p for d in self.daughters])
            phi = 0.
            f, start = self.decay.get_amplitude_function(theta,phi, total_absolute=False, just_in_time_compile = False)
            hel = [self] + list(self.daughters)
            fs.append(f)
            start_params.append(start)
            helicities.append(hel)

        if len(self.daughters) == 3 :
            s1 = mass(self.daughters[1].p + self.daughters[2].p)**2
            s3 = mass(self.daughters[0].p + self.daughters[1].p)**2
            smp = jnp.stack([jnp.array(s3),jnp.array(s1)],axis=1)
            f, start = self.decay.get_amplitude_function(smp,total_absolute=False, just_in_time_compile = False)
            hel = [self] + list(self.daughters)
            fs.append(f)
            start_params.append(start)
            helicities.append(hel)
        for d in self.daughters:
            if d.stable:
                # check if daughter decays
                # if not we dont need to worry any further
                continue
            
            fd, startd, helicitiesd = d.getHelicityAmplitude()
            fs.append(fd)
            start_params.append(startd)
            helicities.append(helicitiesd)

        nH = len([a for hel in helicities for a in hel ])
        nP = len([a for par in start_params for a in par ])

        # indH = [len(hel) for hel in helicities]
        indP = [len(par) for par in start_params]
        # TODO: organize helicities into dicts

        # would wanna use a set, but those are not ordered
        # a dict with no values will do the same
        helicy_names = {a.name:None for hel in helicities for a in hel }

        def f(args,*hel):
            f0 = None
            nPar0 = 0
            helicity_dict = {name:h for name,h in zip(helicy_names,hel)}
            for f_,nPar, H in zip(fs,indP,helicities):
                # every function has nHel helicity arguments and nPar parameter arguments
                # we need to correctly sort them
                nPar1 = nPar0 + nPar
                p  = args[nPar0:nPar1]
                h = [helicity_dict[node.name] for node in H]
                if f0 is None:
                    f0 = f_(p,*h)
                else:
                    f0 = f_(p,*h) * f0
            return f0

        return f,[ a for par in start_params for a in par ], [ a for a in helicy_names ]

class DecayTree:
    def __init__(self,root):
        self.root = root
    
    def traverse(self):
        for a in self.root.traverse():
            yield a
    
    def getHelicityAmplitude(self):
        f, args, hel = self.root.getHelicityAmplitude()
        return f, args, hel

    def draw(self):
        for n,l in self.traverse():
            prefix =  "    "*l + PIPE
            print( prefix)
            mid = "" if l == 0 else TEE
            print("    "*l + mid + " " + str(n))

if __name__ == "__main__":
    pass