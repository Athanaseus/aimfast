import os
from setuptools import setup, find_packages

pkg = 'aimfast'
build_root = os.path.dirname(__file__)


def readme():
    """Get readme content for package long description"""
    with open(os.path.join(build_root, 'README.rst')) as f:
        return f.read()


def requirements():
    """Get package requirements"""
    with open(os.path.join(build_root, 'requirements.txt')) as f:
        return [pname.strip() for pname in f.readlines()]


setup(name=pkg,
      version="0.3.2",
      description="An Astronomical Image Fidelity Assessment Tool.",
      long_description=readme(),
      author="Athanaseus Ramaila",
      author_email="aramaila@ska.ac.za",
      packages=find_packages(),
      url="https://github.com/Athanaseus/aimfast",
      license="GNU GPL 3",
      classifiers=["Intended Audience :: Developers",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Software Development :: Libraries :: Python Modules"],
      platforms=["OS Independent"],
      install_requires=requirements(),
      tests_require=["pytest",
                     "numpy"],
      extras_require={'docs': ["sphinx-pypi-upload",
                               "numpydoc",
                               "Sphinx"]},
      scripts=['aimfast/bin/aimfast'])
