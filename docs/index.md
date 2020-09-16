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
