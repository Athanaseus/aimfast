=======
aimfast
=======
aimfast is an Astronomical Image Fidelity Assessment Tool

============
Introduction
============

Image fidelity is a measure of the accuracy of the reconstructed sky brightness
distribution. A related metric, dynamic range, is a measure of the degree to
which imaging artifacts around strong sources are suppressed, which in turn
implies a higher fidelity of the on-source reconstruction. Moreover, the choice
of image reconstruction algorithm also affects the correctness of the on-source
brightness distribution. The CLEAN algorithm is most appropriate for
predominantly point-source dominated fields. Extended structure is better
reconstructed with multi-resolution and multi-scale algorithms. For high
dynamic ranges with wide bandwidths, algorithms that model the sky spectrum as
well as the average intensity can yield more accurate reconstructions.

=================
Fidelity Matrices
=================

Statistical moments of distribution
-----------------------------------


Image dynamic range
-------------------


============
Installation
============

Installation from [source](https://github.com/Athanaseus/aimfast),
working directory where source is checked out

```
$ pip install .
```

Command line usage
------------------

Get the four (4) statistical moments of the residual image

```
$ image_fidelity --residual-image meerkat_ddfacet-cube.residual.fits
```

Get the dynamic range of the restored image

```
$ image_fidelity --restored-image meerkat_ddfacet.cube.image.fits -af 5
```

Get combination of the four (4) moments and dynamic range

```
$ image_fidelity --residual-image meerkat_ddfacet-cube.residual.fits --restored-image meerkat_ddfacet.cube.image.fits -af 5
```

NB: Outputs will be printed on the terminal and dumped into `fidelity_results.json` file.
