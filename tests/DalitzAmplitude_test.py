from AmplitudeCrafter.DalitzAmplitude import DalitzAmplitude
from AmplitudeCrafter.ParticleLibrary import particle
import numpy as np
import os
from jax.config import config
config.update("jax_enable_x64", True)
this_dir = os.path.dirname(__file__)

def test_AmplitudeConstitency():

    amplitude_file = os.path.join(this_dir,"resonance_configs/DKmatrix+Xi_c_2791+Ds3_2860+D2300.yml")
    dump_file = os.path.join(this_dir,"resonance_configs/Xi_1_dump.yml")
    cov_file = os.path.join(this_dir,"resonance_configs/resonance_configs/DKmatrix+Xi_c_2791+Ds3_2860+D2300_cov.yml")

    amplitude_dump = os.path.join(this_dir,"ampl.npy")
    p0 = particle.get_particle("Lb")
    p1 = particle.get_particle("Lc")
    p2 = particle.get_particle("D0")
    p3 = particle.get_particle("K")

    amplitude = DalitzAmplitude(p0,p1,p2,p3)
    amplitude.load_resonances(amplitude_file)
    smp = amplitude.phsp.rectangular_grid_sample(10,10)
    f, start = amplitude.get_amplitude_function(smp)
    ampl = f(start)
    assert np.all( abs((np.array(ampl) - np.load(amplitude_dump))/np.load(amplitude_dump)) < 1e-8 )

    interference,_ = amplitude.get_interference_terms(smp,["DKmatrix"],["Ds3_2860"])
    interference = interference(start)
    assert( np.all( abs(np.imag(interference)) < 1e-18 ) )

def test_DynamicL():

    p0 = particle.get_particle("Lb")
    p1 = particle.get_particle("Lc")
    p2 = particle.get_particle("D0")
    p3 = particle.get_particle("K")
    amplitude = DalitzAmplitude(p0,p1,p2,p3)
    
    amplitude_file = "/home/kai/LHCb/AmplitudeCrafter/tests/resonance_configs/Xi_1_fixedL.yml"
    amplitude.load_resonances(amplitude_file)
    smp = amplitude.phsp.rectangular_grid_sample(10,10)
    f, start = amplitude.get_amplitude_function(smp)
    res_fix_L = f(start)

    amplitude_file = "/home/kai/LHCb/AmplitudeCrafter/tests/resonance_configs/Xi_1.yml"
    amplitude.load_resonances(amplitude_file)
    f, start = amplitude.get_amplitude_function(smp)

    res_non_fix_L = f(start)
    print(abs((res_fix_L - res_non_fix_L)/res_non_fix_L))
    assert np.all(abs((res_fix_L - res_non_fix_L)/res_non_fix_L) < 1e-10)

def test_Dmatrix():
    p0 = particle.get_particle("Lb")
    p1 = particle.get_particle("Lc")
    p2 = particle.get_particle("D0")
    p3 = particle.get_particle("K")
    amplitude = DalitzAmplitude(p0,p1,p2,p3)
    
    amplitude_file = "tests/resonance_configs/Dmatrix_test.yml"
    amplitude.load_resonances(amplitude_file)
    smp = amplitude.phsp.rectangular_grid_sample(10,10)
    f, start = amplitude.get_amplitude_function(smp)
    res = f(start)
    print(res)



if __name__=="__main__":
    test_Dmatrix()
    # test_DynamicL()
    # test_AmplitudeConstitency()
