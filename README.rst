=======
aimfast
=======
|Doc Status|
|Pypi Version|
|Build Version|
|Python Versions|

An Astronomical Image Fidelity Assessment Tool

Main website: https://aimfast.readthedocs.io

==============
Introduction
==============

Image fidelity is a measure of the accuracy of the reconstructed sky brightness distribution. A related metric, dynamic range, is a measure of the degree to which imaging artifacts around strong sources are suppressed, which in turn implies a higher fidelity of the on-source reconstruction. Moreover, the choice of image reconstruction algorithm also affects the correctness of the on-source brightness distribution. For high dynamic ranges with wide bandwidths, algorithms that model the sky spectrum as well as the average intensity can yield more accurate reconstructions.

==============
Installation
==============
Installation from source_,
working directory where source is checked out

.. code-block:: bash
  
    $ pip install .

This package is available on *PYPI*, allowing

.. code-block:: bash
  
    $ pip install aimfast

=======
License
=======

This project is licensed under the GNU General Public License v3.0 - see license_ for details.

=============
Contribute
=============

Contributions are always welcome! Please ensure that you adhere to our coding
standards pep8_.

.. |Doc Status| image:: https://readthedocs.org/projects/aimfast/badge/?version=latest
                :target: http://aimfast.readthedocs.io/en/latest
                :alt:

.. |Pypi Version| image:: https://img.shields.io/pypi/v/aimfast.svg
                  :target: https://pypi.python.org/pypi/aimfast
                  :alt:
.. |Build Version| image:: https://travis-ci.org/Athanaseus/aimfast.svg?branch=master
                  :target: https://travis-ci.org/Athanaseus/aimfast
                  :alt:

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/aimfast.svg
                     :target: https://pypi.python.org/pypi/aimfast/
                     :alt:

.. _source: https://github.com/Athanaseus/aimfast
.. _license: https://github.com/Athanaseus/aimfast/blob/master/LICENSE
.. _pep8: https://www.python.org/dev/peps/pep-0008
