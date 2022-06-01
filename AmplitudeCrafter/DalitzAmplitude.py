from jitter.amplitudes.dalitz_plot_function import DalitzDecay
from AmplitudeCrafter.loading import load
from ParticleLibrary import particle
from jitter.phasespace.DalitzPhasespace import DalitzPhaseSpace


class DalitzAmplitude:
    def __init__(self,p0:particle,p1:particle,p2:particle,p3:particle):
        self.particles = [p1,p2,p3]
        self.p0 = p0

        self.ma = p1.mass
        self.mb = p2.mass
        self.mc = p3.mass
        self.md = p0.mass

        self.phsp = DalitzPhaseSpace(self.ma,self.mb,self.mc,self.md)


    def reload(self):

        spins = [self.p0.spin] + [p.spin for p in self.particles]
        parities = [self.p0.parity] + [p.parity for p in self.particles]
        masses = [self.p0.mass] + [p.mass for p in self.particles]

        resonances = self.get_resonance_tuples()
        bls = self.get_bls()

        self.decay_descriptor = DalitzDecay(*masses,
                                            *spins,
                                            *parities,
                                            self.sample,
                                            resonances,bls,d=1.5/1000.,phsp = None)

    def get_bls(self):
        return {}

    def get_resonance_tuples(self):
        for 


