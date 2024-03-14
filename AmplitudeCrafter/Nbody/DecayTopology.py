
import numpy as np
from jax import numpy as jnp
from typing import List, Tuple, Optional, Union, Any
from functools import cached_property
from AmplitudeCrafter.ParticleLibrary import Particle
from AmplitudeCrafter.Nbody.lorentz import LorentzTrafo

class Node:
    def __init__(self, value: Union[Any, tuple]):
        self.value = value
        self.daughters = []
        self.parent = None
    
    def add_daughter(self, daughter):
        self.daughters.append(daughter)
        daughter.parent = self
    
    def __repr__(self):
        if len(self.daughters) == 0:
            return str(self.value)
        return f"( {self.value} -> " + f"{', '.join([str(d) for d in self.daughters])} )"
    
    def __str__(self):
        return self.__repr__()
    
    def print_tree(self):
        for d in self.daughters:
            d.print_tree()
        print(f"\n {self.value}" )

    def contains(self, contained_node:'Node'):
        if self.value == contained_node.value:
            if len(self.daughters) == len(contained_node.daughters):
                return all([d.contains(cd) for d,cd in zip(self.daughters, contained_node.daughters)])
            if len(contained_node.daughters) == 0:
                return True
        return False
    
    def inorder(self):
        if len(self.daughters) == 0:
            return [self]
        return [self] + [node for d in self.daughters for node in d.inorder()]
    
    def momentum(self, momenta:dict):
        """Get a particles momentum

        Args:
            momenta (dict): the momenta of the final state particles

        Returns:
            the momentum of the particle, as set by the momenta dictionary
            This expects the momenta to be jax or numpy compatible
        """
        if len(self.daughters) == 0:
            return momenta[self.value]
        return sum([d.momentum(momenta) for d in self.daughters])

    def boost(self, target: 'Node', momenta: dict):
        if self.value == target.value:
            zero = jnp.zeros_like(self.momentum(momenta))
            return LorentzTrafo(0 ,0, 0, 0, 0, 0)

class Tree:
    def __init__(self, root:Node):
        self.root = root
    
    def __repr__(self):
        return str(self.root)
    
    def contains(self, contained_tree:'Tree'):
        return self.root.contains(contained_tree.root)

def split(nodes:List[Node], split:int) -> Tuple[Tuple[Node], Tuple[Node]]:
    """
    Split a list of nodes into two lists of nodes.
    Parameters: nodes: List of nodes to split
                split: Index at which to split the list
    Returns: Tuple of lists of nodes
    """
    left = []
    right = []
    for i,n in enumerate(nodes):
        if split & (1 << i):
            left.append(n)
        else:
            right.append(n)
    return  tuple(left), tuple(right)


def generateTreeDefinitions(nodes:List[int]) -> List[Node]:
    """
    Generate all possible tree definitions for a given list of nodes.
    Parameters: nodes: List of nodes to generate tree definitions for
    Returns: List of tree definitions
    """
    trees = []
    if len(nodes) == 1:
        return [(None, None)]
    for i in range(1,1 << len(nodes) - 1):
        left, right = split(nodes, i)
        for l,r in generateTreeDefinitions(left):
            if len(left) == 1:
                lNode = Node(left[0])
            else:
                lNode = Node(left)
            if l is not None:
                lNode.add_daughter(l)
                lNode.add_daughter(r)
            for l2,r2 in generateTreeDefinitions(right):
                if len(right) == 1:
                    rNode = Node(right[0])
                else:
                    rNode = Node(right)
                if l2 is not None:
                    rNode.add_daughter(l2)
                    rNode.add_daughter(r2)
                trees.append((lNode, rNode))
    return trees

class Topology:

    def __init__(self, tree:Node):
        """
        Class to represent the topology of an N-body decay.
        Parameters: topology: List of integers representing the topology of the decay
        """
        self.__tree = tree

    @property
    def tree(self):
        """
        Returns: Tree representation of the topology
        """
        return self.__tree
    
    def __repr__(self) -> str:
        return str(self.tree)
    
    def contains(self, contained_topology:'Topology'):
        """
        Check if a given topology is contained in this topology.
        Parameters: contained_topology: Topology to check if it is contained
        Returns: True if the given topology is contained in this topology, False otherwise
        """
        return self.tree.contains(contained_topology.tree)

class TopologyGroup:
    def __init__(self, start_node:Particle, final_state_nodes:List[Particle]):
        self.start_node = start_node
        self.final_state_nodes = final_state_nodes
        self.node_numbers = {i:node for i,node in enumerate([start_node] + final_state_nodes)}
    
    @cached_property
    def intermediate_nodes(self):
        nodes = {}
        for i, node1 in enumerate(self.final_state_nodes[:-1]):
            for j_ , node2 in enumerate(self.final_state_nodes[i+1:]):
                j = i + j_ + 1
                nodes[(i, j)] = (node1, node2)
        return nodes
    
    @cached_property
    def trees(self):
        trees = generateTreeDefinitions(self.final_state_nodes)    
        trees_with_root_node = []
        for l,r in trees:
            root = Node(self.start_node)
            root.add_daughter(l)
            root.add_daughter(r)
            trees_with_root_node.append(root)
        return trees_with_root_node
    
    @cached_property
    def topologies(self):
        return [Topology(tree) for tree in self.trees]
    

    @cached_property
    def nodes(self):
        nodes = self.nodes.copy()
        nodes.update({(i, None):node for i,node in self.node_numbers.items()})
        return nodes
    
    def filter(self, contained_step: Topology):
        """
        Filter the topologies based on the number of contained steps.

        Args:
            contained_step (list): sub topology for which to filter
        """
        return [t for t in self.topologies if t.contains(contained_step)]
        