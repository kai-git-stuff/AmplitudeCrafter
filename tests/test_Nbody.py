from AmplitudeCrafter.Nbody import DecayTopology
from AmplitudeCrafter.Nbody.DecayTopology import generateTreeDefinitions, Node, TopologyGroup
from AmplitudeCrafter.ParticleLibrary import particle

from AmplitudeCrafter.Nbody.Decay import NBodyDecay

p0 = particle.get_particle("B+")
p1 = particle.get_particle("K+")
p2 = particle.get_particle("p")
p3 = particle.get_particle("p~")
p4 = particle.get_particle("pi0")

tg = TopologyGroup(p0, [p1,p2,p3,p4])
assert len(tg.trees) == 15

tg = TopologyGroup(p0, [p1,p2,p3])
assert len(tg.trees) == 3

tg = TopologyGroup(p0, [p1,p2])
assert len(tg.trees) == 1

tg = TopologyGroup(0,[1,2,3,4])
assert len(tg.trees) == 15

tg = TopologyGroup(0,[1,2,3])
assert len(tg.trees) == 3

tg = TopologyGroup(0,[1,2])
assert len(tg.trees) == 1

tg = TopologyGroup(0,[1,2,3,4,5])
assert len(tg.trees) == 105


decay = NBodyDecay(0,1,2,3,4)
print(decay.topologies)