from AmplitudeCrafter.DalitzAmplitude import DalitzAmplitude
from AmplitudeCrafter.ParticleLibrary import particle
import numpy as np


amplitude_file = "/home/kai/LHCb/AmplitudeCrafter/tests/Xi_1.yml"

p0 = particle.get_particle("Lb")
p1 = particle.get_particle("Lc")
p2 = particle.get_particle("D0")
p3 = particle.get_particle("K")

amplitude = DalitzAmplitude(p0,p1,p2,p3)
amplitude.load_resonances(amplitude_file)
smp = amplitude.phsp.rectangular_grid_sample(10,10)
f, start = amplitude.get_amplitude_function(smp)
print(amplitude.mapping_dict)
res_non_fix_L = f(start)



amplitude_file = "/home/kai/LHCb/AmplitudeCrafter/tests/Xi_1_fixedL.yml"

p0 = particle.get_particle("Lb")
p1 = particle.get_particle("Lc")
p2 = particle.get_particle("D0")
p3 = particle.get_particle("K")

amplitude = DalitzAmplitude(p0,p1,p2,p3)
amplitude.load_resonances(amplitude_file)
smp = amplitude.phsp.rectangular_grid_sample(10,10)
f, start = amplitude.get_amplitude_function(smp)
print(amplitude.mapping_dict)
res_fix_L = f(start)


print(res_fix_L - res_non_fix_L)


print(np.sum(res_fix_L - res_non_fix_L))
assert abs(np.sum(res_fix_L - res_non_fix_L)) < 1e-10

