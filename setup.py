from setuptools import setup, find_packages
import os
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

print(find_packages())
setup(
name = "AmplitudeCrafter",
    version = "0.0.1",
    author = "Kai Habermann",
    author_email = "kai.habermann@gmx.net",
    description = ("Amplitude Crafter for LHCb using JAX by Google"),
    license = "",
    keywords = "",
    url = "",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Amplitude Fitting",
    ],
)

