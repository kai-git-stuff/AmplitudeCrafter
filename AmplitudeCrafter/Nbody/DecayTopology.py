
import numpy as np
from typing import List, Tuple, Optional
from functools import cache
from AmplitudeCrafter.ParticleLibrary import Particle


class Node:
    def __init__(self, value):
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

class Tree:
    def __init__(self, root:Node):
        self.root = root
    
    def __repr__(self):
        return str(self.root)

def split(nodes:List[Node], split:int) -> Tuple[List[Node], List[Node]]:
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
    return  left, right


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
            lNode = Node(left)
            if l is not None:
                lNode.add_daughter(l)
                lNode.add_daughter(r)
            for l2,r2 in generateTreeDefinitions(right):
                rNode = Node(right)
                if l2 is not None:
                    rNode.add_daughter(l2)
                    rNode.add_daughter(r2)
                trees.append((lNode, rNode))
    return trees


class TopologyGroup:
    def __init__(self, start_node:Particle, final_state_nodes:List[Particle]):
        self.start_node = start_node
        self.final_state_nodes = final_state_nodes
        self.node_numbers = {i:node for i,node in enumerate([start_node] + final_state_nodes)}
    
    @property
    @cache
    def intermediate_nodes(self):
        nodes = {}
        for i, node1 in enumerate(self.final_state_nodes[:-1]):
            for j_ , node2 in enumerate(self.final_state_nodes[i+1:]):
                j = i + j_ + 1
                nodes[(i, j)] = (node1, node2)
        return nodes
    
    @property
    @cache
    def trees(self):
        trees = generateTreeDefinitions(self.final_state_nodes)    
        trees_with_root_node = []
        for l,r in trees:
            root = Node(self.start_node)
            root.add_daughter(l)
            root.add_daughter(r)
            trees_with_root_node.append(root)
        return trees_with_root_node
    
    @property
    @cache
    def topologies(self):
        return [Topology(self, tree) for tree in self.trees]
    

    @property
    @cache
    def nodes(self):
        nodes = self.nodes.copy()
        nodes.update({(i, None):node for i,node in self.node_numbers.items()})
        return nodes
        
class Topology:

    def __init__(self, group:TopologyGroup, tree:Tree):
        """
        Class to represent the topology of an N-body decay.
        Parameters: topology: List of integers representing the topology of the decay
        """
        self.group = group
        self.__tree = tree

    @property
    def tree(self):
        """
        Returns: Tree representation of the topology
        """
        return self.__tree
    

