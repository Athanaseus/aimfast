.. _source: https://github.com/Athanaseus/aimfast

=======
aimfast
=======
An Astronomical Image Fidelity Assessment Tool

============
Introduction
============

Image fidelity is a measure of the accuracy of the reconstructed sky brightness
distribution. A related metric, dynamic range, is a measure of the degree to
which imaging artifacts around strong sources are suppressed, which in turn
implies a higher fidelity of the on-source reconstruction. Moreover, the choice
of image reconstruction algorithm also affects the correctness of the on-source
brightness distribution. For high dynamic ranges with wide bandwidths, algorithms
that model the sky spectrum as well as the average intensity can yield more accurate
reconstructions.

=================
Fidelity Matrices
=================

Image dynamic range
-------------------

Dynamic range is a measure of the degree to which imaging artifacts around
strong sources are suppressed, which in turn implies a higher fidelity of
the on-source reconstruction. It is calculated by obtaining the quotient of
highest peak flux (:math:`flux_{peak}`) and the absolute of the minimum
flux (:math:`flux_{min}`) around the peak in the restored image.

.. math::

    DR = \frac{flux_{peak}}{\left | {flux_{min}} \right | }


Statistical moments of distribution
-----------------------------------

The mean and the variance provide information on the location (general value of
the residual flux) and variability (spread, dispersion) of a set of numbers,
and by doing so, provide some information on the appearance of the distribution
of residual flux in the residual image.
The mean and variance are calculated as follows respectively

.. math::

    MEAN = \frac{1}{n}\sum_{i=1}^{n}(x_{i})

and 

.. math::

    VARIANCE = \sum_{i=1}^{n}(x_{i} - \overline{x})^2

whereby

.. math::

    STD\_DEV = \sqrt{VARIANCE}

The third and fourth moments are the skewness and kurtosis respectively. The
skewness is the measure of the symmetry of the shape and kurtosis is a measure
of the flatness or peakness of a distribution. This moments are used to characterize
the residual flux after performing calibration and imaging, therefore for ungrouped
data, the r-th moment is calculated as follows:

.. math::

    m_r = \frac{1}{n}\sum_{i=1}^{n}(x_i - \overline{x})^r

The coefficient of skewness, the 3-rd moment, is obtained by

.. math::

    SKEWNESS = \frac{m_3}{{m_2}^{\frac{3}{2}}}

If there is a long tail in the positive directin, skewness will be positive,
while if there is a long tail in the negative direction, skewness will be negative.

   .. figure:: https://user-images.githubusercontent.com/16665629/35336554-7ce4953e-0121-11e8-8a14-ce1fbf3eece4.jpg
    :width: 60%
    :align: center
    :alt: alternate text
    :figclass: align-center

    Figure 1. Skewness of a distribution.

The coefficient kurtosis, the 4-th moment, is obtained by

.. math::

    KURTOSIS = \frac{m_4}{{m_2}^{2}}

Smaller values (in magnitude) indicate a flatter, more uniform distribution.

   .. figure:: https://user-images.githubusercontent.com/16665629/35336737-069c6086-0122-11e8-80e7-1e674d52c270.jpg
    :width: 60%
    :align: center
    :alt: alternate text
    :figclass: align-center

    Figure 2. Kurtosis of a distribution.

============
Installation
============

Installation from source_, working directory where source is checked out

.. code-block:: bash

    $ pip install .

Command line usage
------------------

Get the four (4) statistical moments of the residual image

.. code-block:: bash

    $ image_fidelity --residual-image cube.residual.fits

Get the dynamic range of the restored image

.. code-block:: bash
    
    $ image_fidelity --restored-image cube.image.fits -af 5

Get combination of the four (4) moments and dynamic range

.. code-block:: bash

    $ image_fidelity --residual-image cube.residual.fits --restored-image cube.image.fits -af 5

NB: Outputs will be printed on the terminal and dumped into `fidelity_results.json` file.