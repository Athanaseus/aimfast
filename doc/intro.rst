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
the on-source reconstruction.
Here we determine it in three ways: Obtaining the quotient of
- highest peak flux (:math:`flux_{peak}`) and the absolute of the minimum flux (:math:`flux_{min}`) around the peak in the residual image.
- highest peak flux (:math:`flux_{peak}`) and the rms flux (:math:`flux_{local_rms}`) around the peak in the residual image.
- highest peak flux (:math:`flux_{peak}`) and the rms flux (:math:`flux_{grobal_rms}`) in the residual image.

.. math::

    DR = \frac{flux_{peak}}{\left | {flux_{min}} \right | }            (1)
    DR = \frac{flux_{peak}}{\left | {flux_{local_rms}} \right | }      (2)
    DR = \frac{flux_{peak}}{\left | {flux_{global_rms}} \right | }     (3)


Statistical moments of distribution
-----------------------------------

The mean and the variance provide information on the location (general value of
the residual flux) and variability (spread, dispersion) of a set of numbers,
and by doing so, provide some information on the appearance of the distribution
of residual flux in the residual image.
The mean and variance are calculated as follows respectively

.. math::

    MEAN = \frac{1}{n}\sum_{i=1}^{n}(x_{i})                            (4)

and 

.. math::

    VARIANCE = \frac{1}{n}\sum_{i=1}^{n}(x_{i} - \overline{x})^2       (5)

whereby

.. math::

    STD\_DEV = \sqrt{VARIANCE}                                         (6)

The third and fourth moments are the skewness and kurtosis respectively. The
skewness is the measure of the symmetry of the shape and kurtosis is a measure
of the flatness or peakness of a distribution. This moments are used to characterize
the residual flux after performing calibration and imaging, therefore for ungrouped
data, the r-th moment is calculated as follows:

.. math::

    m_r = \frac{1}{n}\sum_{i=1}^{n}(x_i - \overline{x})^r              (7)

The coefficient of skewness, the 3-rd moment, is obtained by

.. math::

    SKEWNESS = \frac{m_3}{{m_2}^{\frac{3}{2}}}                         (8)

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

    KURTOSIS = \frac{m_4}{{m_2}^{2}}                                   (9)

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

Get combination of the four (4) moments and dynamic range in one step where argument -af is the multiplying factor of the peak source area:

.. code-block:: bash

    $ aimfast --residual-image cube.residual.fits --restored-image cube.image.fits -af 5

or using sky model file (tigger lsm.html or text file):

.. code-block:: bash

    $ aimfast --residual-image cube.residual.fits --tigger-model model.lsm.html -af 5

NB: Outputs will be printed on the terminal and dumped into `fidelity_results.json` file.
Moreover if the source file names are distinct the output results will be
appended to the same json file.

.. code-block:: bash

    $ cat fidelity_results.json
    $ {"cube.residual.fits": {"SKEW": 0.124, "KURT": 3.825, "STDDev": 5.5e-05, "MEAN": 4.747e-07},
           "cube.image.fits": {"DR": 53.868}}

Additionally, normality testing of the residual image can be performed using the D’Agostino (normaltest) and
Shapiro-Wilk (shapiro) analysis, which returns a tuple result, e.g {'NORM': (123.3, 0.1)}, with the
z-score and p-value respectively.

.. code-block:: bash

    $ aimfast --residual-image cube.residual.fits --normality-model normaltest

Moreover aimfast allows you to swiftly compare two (input-output) tigger models. Currently source flux density and astrometry are examined.
It returns an interactive html correlation plots, from which a `.png` file can be easily downloaded or imported to plot.ly_.

.. code-block:: bash

    $ aimfast --compare-models model1.lsm.html:model2.lsm.html -af 5 -psf <size_arcsec | psf.fits> 

Where --psf-image | -psf is the Name of the point spread function file or psf size in arcsec.

For Flux density, the more the data points rest on the y=x (or I_out=I_in), the more correlated the two models are.

   .. figure:: https://user-images.githubusercontent.com/16665629/49431777-e9989680-f7b6-11e8-899b-cfe100f47ac7.png
    :width: 50%
    :align: center
    :alt: alternate text
    :figclass: align-center

    Figure 3. Input-Output Flux (txt/lsm.html) model comparison

For astrometry, the more sources lie on the y=0 (Delta-position axis) in the left plot and the more points with 1 sigma (blue circle) the more accurate the output source positions.

   .. figure:: https://user-images.githubusercontent.com/16665629/47504227-1f6b6680-d86c-11e8-937c-a00e2ec50d0f.png
    :width: 60%
    :align: center
    :alt: alternate text
    :figclass: align-center

    Figure 4. Input-Output Astrometry (txt/lsm.html) model comparison

Furthermore, a comparison of residuals/noise can be performed as follows: To get random residual flux measurements in a `residual1.fits` and `residual2.fits` images

.. code-block:: bash

    $ aimfast --compare-residuals residual1.fits:residual2.fits -dp 100

where -dp is the number of data points to sample. To get on source residual flux measurements in a `residual1.fits` and `residual2.fits` images

.. code-block:: bash

    $ aimfast --compare-residuals residual1.fits:residual2.fits --tigger-model model.lsm.html

where --tigger-model is the name of the tigger model lsm.html file to locate exact source residuals.
For random or on source residual noise comparisons, the plot on the left shows the residuals on image 1 and image 2 overlayed and the plot on the right shows the ratios. The colorbar shows the distance of the sources from the phase centre.

   .. figure:: https://user-images.githubusercontent.com/16665629/49431465-3fb90a00-f7b6-11e8-929a-c80633b6fe73.png
    :width: 60%
    :align: center
    :alt: alternate text
    :figclass: align-center

    Figure 5. The random/source residual-to-residual/noise ratio measurements

