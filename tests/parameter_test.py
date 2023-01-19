from AmplitudeCrafter.DalitzAmplitude import DalitzAmplitude
from AmplitudeCrafter.ParticleLibrary import particle
import numpy as np
import os
from jax.config import config

config.update("jax_enable_x64", True)
dir = os.path.dirname(__file__)
p0 = particle.get_particle("Lb")
p1 = particle.get_particle("Lc")
p2 = particle.get_particle("D0")
p3 = particle.get_particle("K")
this_dir = os.path.dirname(__file__)

amplitude_file = os.path.join(this_dir,"Xi_1.yml")
amplitude = DalitzAmplitude(p0,p1,p2,p3)
amplitude.load_resonances(amplitude_file)