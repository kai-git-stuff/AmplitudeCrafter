from jitter.constants import spin as sp
from collections import defaultdict
from locals import particle_config

from AmplitudeCrafter.loading import load

class particle:
    particles = defaultdict(list)
    def __init__(self,mass,spin,parity,name,type=None) -> None:
        self.mass = mass
        self.spin = spin
        self.parity = parity

        if type is None:
            self.type = {True:"Fermion", False:"Boson"}[sp.is_half(self.spin)]
        else:
            self.type = type
        self.name = name
        
        particle.particles[self.name].append(self)

    def __repr__(self):
        num = particle.particles[self.name].index(self)

        return f"{self.name} {num} ({self.mass}MeV, (J:P)=({self.spin}:{self.parity}))"

for name, specifications in load(particle_config).items():
    p = particle(**specifications,name=name)

