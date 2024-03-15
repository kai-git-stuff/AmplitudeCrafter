from AmplitudeCrafter.Nbody import DecayTopology
from AmplitudeCrafter.Nbody.DecayTopology import generateTreeDefinitions, Node, TopologyGroup
from AmplitudeCrafter.ParticleLibrary import particle
from jax import numpy as jnp
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


decay = NBodyDecay(0,1,2,3,4, 5)

momenta = {   1: jnp.array([1, 0, 0, 0.9]),
              2: jnp.array([1, 0, 0.15, 0.4]),
              3: jnp.array([1, 0, 0.3, 0.3]),
              4: jnp.array([1, 0, 0.1, 0.4]),
              5: jnp.array([1, 0, 0.1, 0.8])}
all_nodes = list(tg.topologies[0].tree.inorder())
first_node = all_nodes[0]

for node in tg.filter(Node((2, 1,3))):
    # print(node.print_tree())
    print(node)
    # print(first_node.boost(node, momenta))