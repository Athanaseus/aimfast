import os
from setuptools import setup, find_packages

build_root = os.path.dirname(__file__)


def readme():
    """Get readme content for package long description"""
    with open(os.path.join(build_root, 'README.rst')) as f:
        return f.read()

setup(name="aimfast",
      version="0.2.1",
      description="An Astronomical Image Fidelity Assessment Tool.",
      long_description=readme(),
      author="Athanaseus Ramaila",
      author_email="aramaila@ska.ac.za",
      packages=find_packages(),
      url="https://github.com/Athanaseus/aimfast",
      license="GNU GPL 3",
      classifiers=["Intended Audience :: Developers",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 3",
                   "Topic :: Software Development :: Libraries :: Python Modules"],
      platforms=["OS Independent"],
      install_requires=["astLib",
                        "astropy==2.0.4",
                        "astro-tigger",
                        "jsonschema",
                        "mock",
                        "numpy",
                        "numpydoc",
                        "plotly",
                        "scikit-learn",
                        "scipy"],
      extras_require={'docs': ["sphinx-pypi-upload",
                               "numpydoc",
                               "Sphinx"]},
      scripts=['aimfast/bin/aimfast'])
