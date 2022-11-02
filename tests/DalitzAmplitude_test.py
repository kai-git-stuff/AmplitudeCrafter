from AmplitudeCrafter.DalitzAmplitude import DalitzAmplitude
from AmplitudeCrafter.ParticleLibrary import particle
import numpy as np
import os
from jax.config import config

config.update("jax_enable_x64", True)
dir = os.path.dirname(__file__)

amplitude_file = os.path.join(dir,"DKmatrix+Xi_c_2791+Ds3_2860+D2300.yml")
dump_file = os.path.join(dir,"Xi_1_dump.yml")
cov_file = os.path.join(dir,"DKmatrix+Xi_c_2791+Ds3_2860+D2300_cov.yml")

amplitude_dump = os.path.join(dir,"ampl.npy")
p0 = particle.get_particle("Lb")
p1 = particle.get_particle("Lc")
p2 = particle.get_particle("D0")
p3 = particle.get_particle("K")

amplitude = DalitzAmplitude(p0,p1,p2,p3)
amplitude.load_resonances(amplitude_file)
smp = amplitude.phsp.rectangular_grid_sample(10,10)
f, start = amplitude.get_amplitude_function(smp)
# print(amplitude.mapping_dict)
amplitude.dump(start,dump_file)
print("Getting COV")
cov = amplitude.get_cov(cov_file)
print("Got COV")
arg_sample = np.random.multivariate_normal(start,cov,100)
print(arg_sample.shape)
amplitudes = [f(a) for a in arg_sample]
dAmplitude = np.std(amplitudes,axis=0)
res_non_fix_L = f(start)
for v,e in zip(res_non_fix_L.flatten(),dAmplitude.flatten()):
    print(v,"+-",e)
# print(dAmplitude.shape)
# print(np.zeros(smp.shape[:-1]))
ampl = f(start)
# 
print(ampl)
print(np.array(ampl) - np.load(amplitude_dump))
# np.save(amplitude_dump,np.array(ampl))
exit(0)

amplitude_file = "/home/kai/LHCb/AmplitudeCrafter/tests/Xi_1_fixedL.yml"

p0 = particle.get_particle("Lb")
p1 = particle.get_particle("Lc")
p2 = particle.get_particle("D0")
p3 = particle.get_particle("K")

amplitude = DalitzAmplitude(p0,p1,p2,p3)
amplitude.load_resonances(amplitude_file)
smp = amplitude.phsp.rectangular_grid_sample(10,10)
f, start = amplitude.get_amplitude_function(smp)
# print(amplitude.mapping_dict)
res_fix_L = f(start)


print(res_fix_L - res_non_fix_L)


print(np.sum(res_fix_L - res_non_fix_L))
assert abs(np.sum(res_fix_L - res_non_fix_L)) < 1e-10

