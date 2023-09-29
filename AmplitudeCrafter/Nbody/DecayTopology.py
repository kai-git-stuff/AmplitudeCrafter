
import numpy as np
from typing import List, Tuple
from functools import cache


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
    print(1 << len(nodes))

    if len(nodes) == 1:
        return [(None, None)]
    for i in range(1,1 << len(nodes) - 1):
        left, right = split(nodes, i)
        print(left, right)
        # exit(0)
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
    def __init__(start_node:Node, final_state_nodes:List[Node]):
        self.start_node = start_node
        self.final_state_nodes = final_state_nodes
    
    @property
    @cache
    def intermediate_nodes(self):
        nodes = {}
        for i, node1 in enumerate(self.final_state_nodes[:-1]):
            for node2 in self.final_state_nodes[i+1:]:
                nodes[(node1.value, node2.value)] = Node((node1.value, node2.value))
        return nodes
    
    @property
    @cache
    def trees(self):
        trees = []
        for i in range(1 << len(self.final_state_nodes)):
            left, right = split(self.final_state_nodes, i)

        return trees
        
class Topology:

    def __init__(self, topology:List[int], nodes:List[int]):
        """
        Class to represent the topology of an N-body decay.
        Parameters: topology: List of integers representing the topology of the decay
        """
        self.topology = topology
        self.nodes = nodes

    @property
    @cache
    def tree(self):
        """
        Returns: Tree representation of the topology
        """
        nodes = [Node(i) for i in self.nodes]
    

        return Tree(nodes[0])
