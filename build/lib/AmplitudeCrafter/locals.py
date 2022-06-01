from os.path import dirname, abspath, join
import sys
main_dir = dirname(dirname(abspath(__file__)))
src_dir = dirname(abspath(__file__))
config_dir = join(src_dir,"config/")
particle_config = join(config_dir,"particles.yml")
