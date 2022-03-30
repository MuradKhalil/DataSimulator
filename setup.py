from setuptools import find_packages, setup
from pkg_resources import parse_requirements
from  pathlib import Path

with Path('requirements.txt').open() as reqs:
    install_requires = [str(req) for req in parse_requirements(reqs)]

setup(
    name='data_simulator',
    version='0.1.0',
    description='A short description of the project.',
    author='Murad Khalil',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'':'src'},
    python_requires='>=3.6',
    install_requires=install_requires
)