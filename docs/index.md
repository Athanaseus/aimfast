## Welcome to AIMfast Page

An Astronomical Image Fidelity Assessment Tool

Introduction
============

Image fidelity is a measure of the accuracy of the reconstructed sky
brightness distribution. A related metric, dynamic range, is a measure
of the degree to which imaging artifacts around strong sources are
suppressed, which in turn implies a higher fidelity of the on-source
reconstruction. Moreover, the choice of image reconstruction algorithm
also affects the correctness of the on-source brightness distribution.
For high dynamic ranges with wide bandwidths, algorithms that model the
sky spectrum as well as the average intensity can yield more accurate
reconstructions.

Fidelity Matrices
=================

Image dynamic range
-------------------

Dynamic range is a measure of the degree to which imaging artifacts
around strong sources are suppressed, which in turn implies a higher
fidelity of the on-source reconstruction. Here we determine it in three
ways: Obtaining the quotient of - highest peak flux ($flux_{peak}$) and
the absolute of the minimum flux ($flux_{min}$) around the peak in the
residual image. - highest peak flux ($flux_{peak}$) and the rms flux
($flux_{local_rms}$) around the peak in the residual image. - highest
peak flux ($flux_{peak}$) and the rms flux ($flux_{grobal_rms}$) in the
residual image.

$$DR = \frac{flux_{peak}}{\left | {flux_{min}} \right | }            (1)
DR = \frac{flux_{peak}}{\left | {flux_{local_rms}} \right | }      (2)
DR = \frac{flux_{peak}}{\left | {flux_{global_rms}} \right | }     (3)$$

Statistical moments of distribution
-----------------------------------

The mean and the variance provide information on the location (general
value of the residual flux) and variability (spread, dispersion) of a
set of numbers, and by doing so, provide some information on the
appearance of the distribution of residual flux in the residual image.
The mean and variance are calculated as follows respectively

$$MEAN = \frac{1}{n}\sum_{i=1}^{n}(x_{i})                            (4)$$

and

$$VARIANCE = \frac{1}{n}\sum_{i=1}^{n}(x_{i} - \overline{x})^2       (5)$$

whereby

$$STD\_DEV = \sqrt{VARIANCE}                                         (6)$$

The third and fourth moments are the skewness and kurtosis respectively.
The skewness is the measure of the symmetry of the shape and kurtosis is
a measure of the flatness or peakness of a distribution. This moments
are used to characterize the residual flux after performing calibration
and imaging, therefore for ungrouped data, the r-th moment is calculated
as follows:

$$m_r = \frac{1}{n}\sum_{i=1}^{n}(x_i - \overline{x})^r              (7)$$

The coefficient of skewness, the 3-rd moment, is obtained by

$$SKEWNESS = \frac{m_3}{{m_2}^{\frac{3}{2}}}                         (8)$$
