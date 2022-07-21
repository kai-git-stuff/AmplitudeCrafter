from jitter.constants import spin as sp
from collections import defaultdict
from AmplitudeCrafter.locals import particle_config

from AmplitudeCrafter.loading import load

class particle:
    particles = defaultdict(list)
    particles_by_name = {}
    def __init__(self,mass,spin,parity,name,type=None) -> None:
        self.mass = mass
        self.spin = spin
        self.parity = parity

        if type is None:
            self.type = {True:"Fermion", False:"Boson"}[sp.is_half(self.spin)]
        else:
            self.type = type
        self.name = name
        if particle.particles_by_name.get(name,None) is not None:
            pass
            print("WARNING: Particle of Name %s already exists as %s! It will be replaced with %s"%(name,particle.particles_by_name[name],self))
        particle.particles_by_name[name] = self
        
        particle.particles[self.type].append(self)

    @staticmethod
    def get_particle(name):
        return particle.particles_by_name[name]

    @staticmethod
    def load_library(path):
        for name, specifications in load(path).items():
            p = particle(**specifications,name=name)

    def __repr__(self):
        return f"{self.name} ({self.type}) ({self.mass}MeV, (J:P)=({self.spin}:{self.parity}))"

particle.load_library(particle_config)


