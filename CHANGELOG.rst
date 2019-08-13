0.4.0
-----
- Include Mean Absolute Deviation and Root mean Square for residual stats
- Compute stats given a mask/channels or threshold to select channels
- Add a generic ploting function using plotly
- Morphology and Spectral comparisons
- Update documentation
- Update tests

0.3.0
-----
- Add a test suite
- Py3 compatible
- Importable modules
- Supports ascii/fits catalogs

0.2.1
-----
- Compare the on source/random residuals to noise

0.2.0
-----
- Compare input-output astrometry tigger models and plot correlation
- Compute correlation stats between output and input data-set

0.1.2
-----
- Use peak flux when computing Dynamic Range (DR)
- Compute three DR (local using deepest negative, local using rms, global uing rms)
- Use a label instead of a path when creating stats dictionary

0.1.1
-----
- Fix formating bug
- Add normality testing

0.1.0
-----
- Allows tigger model lsm.html or text file to determine Dynamic Range
  using residual image
- Compare input-output flux tigger models and plot correlation
