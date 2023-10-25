import os
from AmplitudeCrafter.DecayTree import *
from jax.config import config

config.update("jax_enable_x64", True)
dir = os.path.dirname(__file__)
amplitude_file = os.path.join(dir,"resonance_configs/DKmatrix+Xi_c_2791+Ds3_2860+D2300.yml")
dump_file = os.path.join(dir,"resonance_configs/Xi_1_dump.yml")
cov_file = os.path.join(dir,"resonance_configs/resonance_configs/DKmatrix+Xi_c_2791+Ds3_2860+D2300_cov.yml")
amplitude_dump = os.path.join(dir,"ampl.npy")

def test_DecayTree():
        ####################### Amplitude building ######################
        a = DecayTreeNode("A",particle.get_particle("Lb"))
        b = DecayTreeNode("B",particle.get_particle("Lc"))
        c = DecayTreeNode("C",particle.get_particle("D0"))
        d = DecayTreeNode("D",particle.get_particle("K"))

        # OPTIONAL FURTHER DECAYS
        # c1 = DecayTreeNode("C1",particle.get_particle("p"))
        # c2 = DecayTreeNode("C2",particle.get_particle("K"))
        # c3 = DecayTreeNode("C3",particle.get_particle("Pi"))

        # b1 = DecayTreeNode("B1",particle.get_particle("K"))
        # b2 = DecayTreeNode("B2",particle.get_particle("Pi"))
        # c.setDecay(c1,c2,c3)
        # b.setDecay(b1,b2)


        a.setDecay(b,c,d)
        phsp = a.decay.phsp
        smp = phsp.rectangular_grid_sample(10,10)
        pb,pc,pd = phsp.final_state_momenta(smp[...,0],smp[...,1])

        b.setP(pb)
        c.setP(pc)
        d.setP(pd)

        a.load_resonances(amplitude_file)
        tree = DecayTree(a)
        tree.draw()

        f, start, helicities = tree.getHelicityAmplitude()
        hels = [[-1,1,0,0],
                [1,1,0,0],
                [-1,-1,0,0],
                [1,-1,0,0]]
        ampl = sum(jnp.abs(f(start,*hel))**2 for hel in hels)

        assert np.all( abs((np.array(ampl) - np.load(amplitude_dump)) /np.load(amplitude_dump)) < 1e8)

def test_Boosts():
        P1 = np.array([3.,1.,0,0][::-1],dtype=np.float64)
        P2 = np.array([100.,7,1,2][::-1],dtype=np.float64)
        print(P2)
        print(boost_to_rest(P2,P1))
        print(mass(P2))
        print(mass(boost_to_rest(P2,P1)))


if __name__ == "__main__":
        test_DecayTree()