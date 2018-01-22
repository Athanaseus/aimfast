from setuptools import setup, find_packages

setup(name="aimfast",
      description="An Astronomical Image Fidelity Assessment Tool.",
      author="Athanaseus Ramaila",
      author_email="aramaila@ska.ac.za",
      packages=find_packages(),
      url='https://github.com/Athanaseus/aimfast',
      license="GNU GPL 3",
      classifiers=["Intended Audience :: Developers",
                   "Programming Language :: Python :: 2",
                   "Topic :: Software Development :: Libraries :: Python Modules"],
      platforms=["OS Independent"],
      install_requires=["numpy",
                        "scipy",
                        "astropy",
                        "astLib"],
      entry_points={'console_scripts':
                    ['aimfast = aimfast.aimfast:main']})
