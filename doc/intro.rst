.. _source: https://github.com/Athanaseus/aimfast
.. _plot.ly: https://plot.ly/
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

    VARIANCE = \frac{1}{n}\sum_{i=1}^{n}(x_{i} - \overline{x})^2

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

If there is a long tail in the positive direction, skewness will be positive,
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

This package is available on *PYPI*, allowing

.. code-block:: bash
  
    $ pip install aimfast

Command line usage
------------------

Get the four (4) statistical moments of the residual image

.. code-block:: bash

    $ aimfast --residual-image cube.residual.fits

Get the dynamic range of the restored image, where argument -af is the multiplying factor of the peak source area  

.. code-block:: bash
    
    $ aimfast --restored-image cube.image.fits -af 5


NB: Outputs will be printed on the terminal and dumped into `fidelity_results.json` file.
Moreover if the source file names are distinct the output results will be
appended to the same json file.

.. code-block:: bash

    $ cat fidelity_results.json
    $ {"cube.residual.fits": {"SKEW": 0.124, "KURT": 3.825, "STDDev": 5.5e-05, "MEAN": 4.747e-07}, "cube.image.fits": {"DR": 53.868}}


Get combination of the four (4) moments and dynamic range in one step:

.. code-block:: bash

    $ aimfast --residual-image cube.residual.fits --restored-image cube.image.fits -af 5

or using sky model file (tigger lsm.html or text file):

.. code-block:: bash

    $ aimfast --residual-image cube.residual.fits --tigger-model model.lsm.html -af 5

Moreover aimfast allows you to compare two (input-output) tigger models. It returns an interactive html correlation plot, from which a `.png` file can be easily downloaded or imported to plot.ly_.

.. code-block:: bash

    $ aimfast --compare-models model1.lsm.html model2.lsm.html -af 5 -psf <size_arcsec | psf.fits> 

Where --psf-image | -psf is the Name of the point spread function file or psf size in arcsec.

The more the data points rest on the y=x (or I_out=I_in), the more correlated the two models are.

   .. figure:: https://user-images.githubusercontent.com/16665629/37516078-a82e0880-2915-11e8-8507-2002da8a6527.png
    :width: 60%
    :align: center
    :alt: alternate text
    :figclass: align-center

    Figure 3. Input-Output tigger (txt/lsm.html) model comparison
