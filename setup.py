from setuptools import setup, find_packages


# Collect requirements
import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="phd",
    version="0.0.1",
    author="Federico Cunico",
    author_email="federico@cunico.net",
    packages=find_packages(),
    install_requires=install_requires
    
)


