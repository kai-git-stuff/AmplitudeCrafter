
import numpy as np
from jax import numpy as jnp
from AmplitudeCrafter.ParticleLibrary import particle
from AmplitudeCrafter.TwoBodyDecay import TwoBodyDecay
from jitter.kinematics import cos_helicity_angle, boost_to_rest, mass
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


class DecayTreeNode:
    def __init__(self, name,part):
        self.name= name
        self.particle = part
        self.__decay = None
        self.__p = None
        self.__daughters = None

    def setP(self, p):
        if not p.shape[-1] == 4:
            raise ValueError("U stupid?")

        self.__p = p

    @property
    def p(self):
        if isinstance(self.__p,callable):
            return self.__p()
        return self.__p

    @property
    def daughters(self):
        if self.__daughters is None:
            return []
        else:
            return self.__daughters

    def setDecay(self,*nodes ):    #: list[DecayTreeNode]
        self.__p = lambda : sum(n.p for n in nodes)
        self.__daughters = nodes
        if len(nodes) == 2:
            # print("TwoBodyDecay")
            self.__decay = TwoBodyDecay(self.particle,*[n.particle for n in nodes])
        elif len(nodes) == 3:
            # print("DalitzDecay")
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

    
    def getHelicityAngles(self,n):
        """
        gets the helicity angles for the decay n -> ...

        needed : boost from partent(n) -> n 

        calculate theta: angle between - P (parent (n)) and daughter(1) of n in rest frame of n

        """

        if len(n.daughters) != 2:
            raise ValueError(f"Helicity angles only defined for two- body decay not {len(n.daughters)} - body decay")
        
        theta = helicityTheta(self.p, n.daughters[0].p, n.daughters[1].p)
        
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
            hel = self.daughters
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

        nH = len([a for a in hel for hel in helicities])
        nP = len([a for a in par for par in start_params])

        indH = [len(hel) for hel in helicities]
        indP = [len(par) for par in start_params]

        def f(*args):
            hel = args[:nH]
            pars = args[nH:nH+nP]
            f0 = 0.

            nHel0 = 0
            nPar0 = 0
            for f_,nHel,nPar in zip(fs,indH,indP):
                # every function has nHel helicity arguments and nPar parameter arguments
                # we need to correctly sort them
                nPar1 = nPar0 + nPar
                nHel1 = nHel0 + nHel
                p  = pars[nPar0:nPar1]
                h  = hel[nHel0:nHel1]
                f0 = f_(*h,*p) * f0
            return f0

        return f,[a for a in par for par in start_params], [a for a in hel for hel in helicities]

class DecayTree:
    def __init__(self,root):
        self.root = root
    
    def traverse(self):
        for a in self.root.traverse():
            yield a
    
    def getHelicityAmplitude(self):
        return self.root.getHelicityAmplitude()

    def draw(self):
        old_l = 0
        for n,l in self.traverse():
            prefix =  "    "*l + PIPE
            # if old_l != l:
            #     prefix = "   "*(l) + ELBOW
            old_l = l
            print( prefix)
            mid = "" if l == 0 else TEE
            print("    "*l + mid + " " + str(n))
            



if __name__ == "__main__":

    a = DecayTreeNode("A",particle.get_particle("Lb"))
    b = DecayTreeNode("B",particle.get_particle("D0"))
    c = DecayTreeNode("C",particle.get_particle("Lc"))
    d = DecayTreeNode("D",particle.get_particle("K"))
    c1 = DecayTreeNode("C1",particle.get_particle("p"))
    c2 = DecayTreeNode("C2",particle.get_particle("K"))
    c3 = DecayTreeNode("C3",particle.get_particle("Pi"))

    b1 = DecayTreeNode("B1",particle.get_particle("K"))
    b2 = DecayTreeNode("B2",particle.get_particle("Pi"))

    a.setDecay(b,c,d)
    b.setDecay(b1,b2)

    c.setDecay(c1,c2,c3)
    c1.setP(np.array([100.,7,1,2][::-1],dtype=np.float64))
    c2.setP(np.array([100.,-7,-1,-2][::-1],dtype=np.float64))
    c3.setP(np.array([100.,7,1,2][::-1],dtype=np.float64))

    tree = DecayTree(a)
    tree.draw()
    exit(0)
    P1 = np.array([3.,1.,0,0][::-1],dtype=np.float64)
    P2 = np.array([100.,7,1,2][::-1],dtype=np.float64)
    print(P2)
    print(boost_to_rest(P2,P1))
    print(mass(P2))
    print(mass(boost_to_rest(P2,P1)))
