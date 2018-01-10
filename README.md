# examine-image-fidelity
A Radio Astronomy Image Fidelity Examiner

# Introduction
Image fidelity is a measure of the accuracy of the reconstructed sky brightness distribution. A related metric, dynamic range, is a measure of the degree to which imaging artifacts around strong sources are suppressed, which in turn implies a higher fidelity of the on-source reconstruction. Moreover, the choice of image reconstruction algorithm also affects the correctness of the on-source brightness distribution. The CLEAN algorithm is most appropriate for predominantly point-source dominated fields. Extended structure is better reconstructed with multi-resolution and multi-scale algorithms. For high dynamic ranges with wide bandwidths, algorithms that model the sky spectrum as well as the average intensity can yield more accurate reconstructions.

# Fidelity Matrices
- Statistical moments of distribution (mean, std, skewness, kurtosis)
- Image dynamic range (peak/min_around_peak)

# Installation
Installation from [source](https://github.com/Athanaseus/examine-image-fidelity),
working directory where source is checked out
```
$ pip install .
```

# Command line usage
Get the four (4) moments of the residual image
```
$ image_fidelity --mode residual --fitsname meerkat_ddfacet-cube.residual.fits
```
Get the dynamic range of the restored image
```
$ image_fidelity --mode restored --fitsname meerkat_ddfacet.cube.image.fits --beam 6 --area_factor 5
```
NB: Outputs will be printed on the terminal and dumped into `results.json` file.

## License

This project is licensed under the GNU General Public License v3.0 - see
[license](https://github.com/Athanaseus/examine-image-fidelity/blob/master/LICENSE) for details.

## Contribute

Contributions are always welcome! Please ensure that you adhere to our coding standards
[pep8](https://www.python.org/dev/peps/pep-0008).
