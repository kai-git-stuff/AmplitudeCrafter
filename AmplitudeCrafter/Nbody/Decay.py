import numpy as np
from jax import jit, vmap
from jax import numpy as jnp


class NBodyDecay:

    @staticmethod
    def _topologies(daughters):
        """
        Recursive function to generate all possible internal topologies of an N-body decay.

        Parameters: daughters: list of integers

        Returns: list of lists of integers
        """
        particles_in_node = set(daughters)
        topoligies = []
        






    def __init__(self, *daughters):
        self.daughters = daughters

    @property
    def n_daughters(self):
        return len(self.daughters)
    
    @property
    def n_body(self):
        return self.n_daughters + 1
    
    @property
    def topologies(self):
        return self._topologies(list(range(1, self.n_daughters + 1)))
