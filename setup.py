from setuptools import setup, find_packages
import os
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

print(find_packages())
config_path = os.path.join(os.path.dirname(__file__), "AmplitudeCrafter/config/")
# data_files = [("config/",[os.path.join(config_path, f)]) for f in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, f))]
# print(data_files)
setup(
name = "AmplitudeCrafter",
    version = "0.0.2",
    author = "Kai Habermann",
    author_email = "kai.habermann@gmx.net",
    description = ("Amplitude Crafter for LHCb using JAX by Google"),
    package_data={'AmplitudeCrafter.config': [config_path+'*.yml']},
    license = "",
    keywords = "",
    url = "",
    install_requires=[
    'jaxlib',
    'jax',
    'scipy',
    'numpy',
    'pandas',
    'jax',
    'sympy',
    'pyaml',
    'particle',
    'networkx'
    ],
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Amplitude Fitting",
    ],
)

