from AmplitudeCrafter.DalitzAmplitude import DalitzAmplitude
from AmplitudeCrafter.ParticleLibrary import particle
import numpy as np
import os
from jax.config import config

config.update("jax_enable_x64", True)
dir = os.path.dirname(__file__)
def testParameter():
    p0 = particle.get_particle("Lb")
    p1 = particle.get_particle("Lc")
    p2 = particle.get_particle("D0")
    p3 = particle.get_particle("K")
    this_dir = os.path.dirname(__file__)

    amplitude_file = os.path.join(this_dir,"Xi_1.yml")
    amplitude_file_dump = os.path.join(this_dir,"Xi_1_dump.yml")

    amplitude = DalitzAmplitude(p0,p1,p2,p3)
    amplitude.load_resonances(amplitude_file)

    args = amplitude.get_args(True)
    
    print(amplitude.get_arg_names())
    args_1 = list(range(len(args)))
    mapping_dict1 = { k:p(numeric=True) for k,p in amplitude.mapping_dict.items()}
    amplitude.dump(args_1,amplitude_file_dump)
    args_2 = np.random.uniform(0,1,len(args_1))
    args = amplitude.get_args_from_yml(amplitude_file_dump)
    print(list(zip(args_1,args)))
    amplitude = DalitzAmplitude(p0,p1,p2,p3)
    amplitude.load_resonances(amplitude_file_dump)
    print([(a1,a2) for a1,a2 in zip(args,amplitude.get_args(numeric=True))])
    
    assert all([a1 == a2 for a1,a2 in zip(args,amplitude.get_args(numeric=True))])
    mapping_dict2 = { k:p(numeric=True) for k,p in amplitude.mapping_dict.items()}
    print([(v != mapping_dict2[k]) for k,v in mapping_dict1.items()])
    assert sum([(v != mapping_dict2[k]) for k,v in mapping_dict1.items()]) == len(args_1)

    amplitude_file = os.path.join(this_dir,"Xi_1.yml")
    amplitude = DalitzAmplitude(p0,p1,p2,p3)
    amplitude.load_resonances(amplitude_file)
 
if __name__ == "__main__":
    testParameter()