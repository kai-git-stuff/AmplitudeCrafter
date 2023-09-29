from AmplitudeCrafter.Nbody import DecayTopology
from AmplitudeCrafter.Nbody.DecayTopology import generateTreeDefinitions, Node


# print(generateTreeDefinitions([1,2,3]))
for l,r in generateTreeDefinitions([1,2,3]):
    root = Node(0)
    root.add_daughter(l)
    root.add_daughter(r)
    print(root)