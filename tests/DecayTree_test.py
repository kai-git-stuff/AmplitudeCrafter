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
        amplitude_file = os.path.join(dir,"resonance_configs/flat.yml")
        ####################### Amplitude building ######################
        p0 = particle(6.32397, 1, 1, "Test P0")
        p1 = particle(1.0, 1, 1, "Test P1")
        p2 = particle(2.0, 2, 1, "Test P2")
        p3 = particle(3.0, 0, 1, "Test P3")
        a = DecayTreeNode("A",p0)
        b = DecayTreeNode("B",p1)
        c = DecayTreeNode("C",p2)
        d = DecayTreeNode("D",p3)

        # b = DecayTreeNode("B",particle.get_particle("Lc"))
        # c = DecayTreeNode("C",particle.get_particle("D0"))
        # d = DecayTreeNode("D",particle.get_particle("K"))

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
        # smp = phsp.rectangular_grid_sample(10,10)
        smp = jnp.array([
                [9.55283383, 26.57159046]
        ])
        pb,pc,pd = phsp.final_state_momenta(smp[...,0],smp[...,1])

        b.setP(jnp.array([-0.47959467, -0.14835602, -0.21909796,  1.14018616]))
        c.setP(jnp.array([-0.02971613,  0.2395643 , -0.45612519,  2.06550824]))
        d.setP(jnp.array([ 0.5093108 , -0.09120829,  0.67522315,  3.1182756 ]))

        a.load_resonances(amplitude_file, bls_check=False)
        tree = DecayTree(a)
        tree.draw()

        f, start, helicities = tree.getHelicityAmplitude()
        # hels = [[-1,1,0,0],
        #         [1,1,0,0],
        #         [-1,-1,0,0],
        #         [1,-1,0,0]]
        print(helicities)
        hels = [
                [h0, h1, h2, h3]
                for h0 in [-1,1]
                for h1 in [-1,1]
                for h2 in  [-2, 0 ,2]
                for h3 in [0]
        ]
        print(hels)
        results = {
                tuple(hel): f(start,*hel)
                for hel in hels
        }
        ampl = sum(abs(a)**2 for a in results.values())
        print(results[-1,1,2,0])
        print(ampl)
        # arr_old = np.load(amplitude_dump)
        # np.save(amplitude_dump,ampl)
        # assert np.allclose(np.array(ampl), arr_old)


def test_Boosts():
        P1 = np.array([3.,1.,0,0][::-1],dtype=np.float64)
        P2 = np.array([100.,7,1,2][::-1],dtype=np.float64)
        print(P2)
        print(boost_to_rest(P2,P1))
        print(mass(P2))
        print(mass(boost_to_rest(P2,P1)))


if __name__ == "__main__":
        test_DecayTree()