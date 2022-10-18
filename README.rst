=======
aimfast
=======

|Build Version|
|Doc Status|
|Pypi Version|
|Python Versions|
|Project License|

An Astronomical Image Fidelity Assessment Tool

Main website: aimfast.rtfd.io_

==============
Introduction
==============

Image fidelity is a measure of the accuracy of the reconstructed sky brightness distribution. A related metric, dynamic range, is a measure of the degree to which imaging artifacts around strong sources are suppressed, which in turn implies a higher fidelity of the on-source reconstruction. Moreover, the choice of image reconstruction algorithm also affects the correctness of the on-source brightness distribution.

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
.. |Build Version| image:: https://github.com/Athanaseus/aimfast/actions/workflows/test_installation.yml/badge.svg
                  :target: https://github.com/Athanaseus/aimfast/actions/workflows/test_installation.yml/
                  :alt:

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/aimfast.svg
                     :target: https://pypi.python.org/pypi/aimfast/
                     :alt:

.. |Project License| image:: https://img.shields.io/badge/license-GPL-blue.svg
                     :target: https://github.com/Athanaseus/aimfast/blob/master/LICENSE
                     :alt:

.. _aimfast.rtfd.io: https://aimfast.rtfd.io
.. _source: https://github.com/Athanaseus/aimfast
.. _license: https://github.com/Athanaseus/aimfast/blob/master/LICENSE
.. _pep8: https://www.python.org/dev/peps/pep-0008
