from AmplitudeCrafter.Nbody import DecayTopology
from AmplitudeCrafter.Nbody.DecayTopology import generateTreeDefinitions, Node, TopologyGroup

tg = TopologyGroup(0,[1,2,3,4])
assert len(tg.trees) == 15

tg = TopologyGroup(0,[1,2,3])
assert len(tg.trees) == 3

tg = TopologyGroup(0,[1,2])
assert len(tg.trees) == 1

tg = TopologyGroup(0,[1,2,3,4,5])
assert len(tg.trees) == 105

