import os
from setuptools import setup, find_packages

pkg = 'aimfast'
__version__ = "1.3.2"
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
      version=__version__,
      description="An Astronomical Image Fidelity Assessment Tool.",
      long_description=readme(),
      author="Athanaseus Ramaila",
      author_email="ramaila.jat@gmail.com",
      packages=find_packages(),
      url="https://github.com/Athanaseus/aimfast",
      license="GNU GPL 3",
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                   "Programming Language :: Python :: 3.8",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Software Development :: Libraries"],
      keywords="fits dataset lsm statistics models html catalogs",
      platforms=["OS Independent"],
      install_requires=requirements(),
      tests_require=["attrs",
                     "pytest",
                     "numpy"],
      extras_require={'doc': ["sphinx-pypi-upload",
                               "numpydoc",
                               "Sphinx",
                               "furo"],
                      'aegean': ["AegeanTools"],
                      'pybdsf': ["bdsf", "matplotlib"],
                      'source_finders': ["bdsf", "matplotlib",
                                         "AegeanTools"],
                      'svg_images': ["matplotlib", "selenium"]},
      python_requires='>=3.8',
      include_package_data=True,
      scripts=['aimfast/bin/aimfast'])
