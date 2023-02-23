from jitter.constants import spin as sp
from collections import defaultdict
from AmplitudeCrafter.locals import particle_config

from particle import Particle
from AmplitudeCrafter.loading import load
import warnings
__WARN_PARTICLE__ = False
class particle:
    particles = defaultdict(list)
    particles_by_name = {}
    def __init__(self,mass,spin,parity,name,type=None,c=None,charge=None) -> None:
        self.mass = mass
        self.spin = spin
        self.parity = parity
        self.decay = None
        self.c = c
        self.charge = charge
        if type is None:
            self.type = {True:"Fermion", False:"Boson"}[sp.is_half(self.spin)]
        else:
            self.type = type
        self.name = name
        if particle.particles_by_name.get(name,None) is not None:
            if __WARN_PARTICLE__:
                warnings.warn("WARNING: Particle of Name %s already exists as %s! It will be replaced with %s"%(name,particle.particles_by_name[name],self))
        particle.particles_by_name[name] = self
        
        particle.particles[self.type].append(self)

    def __eq__(self,other):
        return all([self.mass == other.mass, 
                    self.spin == other.spin, 
                    self.parity == other.parity, 
                    self.name == other.name])

    def set_decay(self, decay):
        if not decay.p0 == self:
            raise ValueError(f"Decay {decay} has different initial particle than {self} !")
        self.decay = decay

    @staticmethod
    def get_particle(name):
        if particle.particles_by_name.get(name) is not None:
            return particle.particles_by_name[name]
        
        results = list(Particle.finditer(name))
        exact_match = list(filter(lambda x : name.strip() == x.name.strip(),results))
        if len(exact_match) == 0:
            err_string = f"""
{name} is no exact match for any known particle! 
Did you mean any of the following?
{','.join([str(res) for res in results])}"""
            raise ValueError(err_string)
        
        if len(exact_match) > 1:
            exact_match = exact_match[:1]

        return particle.from_pdg(exact_match[0])

    @staticmethod
    def load_library(path):
        for name, specifications in load(path).items():
            if isinstance(specifications,str):
                p = particle.get_particle(specifications)
                particle.particles_by_name[name] = p
                continue
            p = particle(**specifications,name=name)

    def __repr__(self):
        return f"{self.name} ({self.type}) ({self.mass}MeV, (J:P)=({self.spin}:{self.parity}))"

    def get_decay(self):
        if self.decay is None:
            return None, False

        return self.decay, True
    
    @staticmethod
    def from_pdg(pdg_particle:Particle):
        s = pdg_particle.J
        print(pdg_particle.name)
        print(s)
        print(pdg_particle.P.p,type(pdg_particle.P.p))
        if s is None:
            s = 0
        return particle(
            pdg_particle.mass,
            int(s * 2),
            int(pdg_particle.P.p),
            pdg_particle.name,
            charge=pdg_particle.charge,
            c =int( pdg_particle.C.p)
        )
particle.load_library(particle_config)