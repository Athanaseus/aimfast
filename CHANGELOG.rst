0.1.0
-----
- Allows tigger model lsm.html or text file to determine Dynamic Range
  using residual image
- Compare input-output flux tigger models and plot correlation

0.1.1
-----
- Fix formating bug
- Add normality testing

0.1.2
-----
- Use peak flux when computing Dynamic Range (DR)
- Compute three DR (local using deepest negative, local using rms, global uing rms)
- Use a label instead of a path when creating stats dictionary

0.2.0
-----
- Compare input-output astrometry tigger models and plot correlation
- Compute correlation stats between output and input data-set

0.2.1
-----
- Compare the on source/random residuals to noise

0.3.0
-----
- Add a test suite
- Py3 compatible
- Importable modules
- Supports ascii/fits catalogs

..
   NOTE: this file was not kept up to date between 0.3.0 and 1.3.4 -- real
   git tags exist for that whole range (0.3.1, 0.3.2, 0.3.3, 1.0-pre6..9,
   1.0.0, 1.2.0, 1.3.0..1.3.4) but were never entered here.

1.4.0 (pre-release)
--------------------
- Fixed three separate crashes where ``Gaussian.getShapeErr()`` returning
  ``None`` (common for real Aegean output) was unpacked unconditionally --
  in ``get_detected_sources_properties`` (both matched sources' shape
  errors) and in ``get_src_scale``.
- Fixed ``compare_models()`` silently never forwarding its ``shape_limit``
  argument to ``get_detected_sources_properties()`` -- the CLI's
  ``-sl``/``--shape-limit`` flag previously had no effect regardless of
  value.
- Fixed ``-sl``/``--shape-limit`` argparse definition missing
  ``type=float`` -- CLI-provided values were passed through as raw
  strings, crashing on the first downstream numeric comparison.
- Fixed ``get_model()``'s reference-FITS-image filename guessing (used
  only for optional phase-centre/frequency metadata) trusting an
  ``os.path.exists()`` check instead of actually verifying the guessed
  path is a valid FITS file -- added ``_resolve_phase_centre()``, a single
  helper used at all 5 call sites, that falls back safely on any failure.
- Fixed a pre-existing bug (caught by CI, not new in this pass): Aegean
  "island" table catalogues (as opposed to "component" tables) were never
  recognised by ``get_model()``'s ``.txt``/``.tab``/``.csv`` branches,
  which required shape/error columns (``a``, ``err_a``, ``b``, ``err_b``,
  ``pa``, ``err_pa``, ``err_peak_flux``) that island tables don't have --
  even though the per-source builder already treated all of them as
  optional. Relaxed the recognition gate to only require flux + position
  columns.
- Fixed ``model_1_name``/``model_2_name`` (used for flux-plot axis labels
  and position-overlay legend entries) truncating at the first ``.`` in a
  catalogue's filename -- broke for any naming convention with a literal
  decimal point (e.g. tile names like ``G312.5``), collapsing two
  differently-named catalogues sharing that prefix to the same displayed
  name. Added ``_catalog_display_name()``, which strips a known extension
  by suffix match instead.
- Fixed the flux comparison fit being unweighted ordinary least squares
  (``scipy.stats.linregress``, no mechanism to accept per-point errors) --
  added ``_weighted_linregress()`` (inverse-variance weighting), used for
  all three flux-plot types (log/inout/snr), so noisy/faint sources (and
  outliers, which tend to carry large errors) no longer pull the fit away
  from the true relation as strongly as precise, bright ones.
- Fixed the RA position offset missing a ``cos(dec)`` correction -- a
  fixed RA difference subtends a smaller true angle away from the
  celestial equator; this was previously computed as the raw, uncorrected
  coordinate difference.
- Fixed a significant units bug in ``delta_pos_angle_arc_sec`` (the true
  angular separation reported for each matched pair, and what the
  ``--compare-models`` "Cross Matching Statistics" mean/sigma offset
  figures are computed from): arcsec-scaled values were being passed into
  ``angular_dist_pos_angle``, a function whose internal trigonometry
  expects radians, producing an angle roughly 206265x too large; the
  result was also never converted to arcsec afterward. This was the actual
  root cause of implausibly large reported position-offset statistics.
- Added ``-fss``/``--flux-sigma-shade``: an optional shaded +/-1 sigma
  band around the flux comparison fit line, showing the (error-weighted)
  scatter of the data around the trend -- not the formal uncertainty on
  the fit parameters, which would be a different, narrower quantity.
- New policy starting this release: every bug fix or new function gets a
  test. Test suite grew from 27 (with 2 known-failing, pre-existing) to 41
  passing as part of this pass -- see ``aimfast/tests/test_aimfast.py``.
- Fixed the "Cross Matching Statistics" table double-converting the
  reported mean/sigma position offset (an extra ``deg2arcsec()`` applied
  to values already in arcsec).
- Fixed a crash when a matched source's position error is ``None`` (the
  common case after a Tigger save/load round-trip), and a matching crash
  for shape errors.
- Fixed the "Catalogs Overlay" RA axis not being flipped to the correct
  astronomical convention when no background restored image is supplied.
- Added a friendly warning when very few sources match between two
  catalogues, naming the two most common silent causes (tolerance,
  shape_limit) rather than leaving the user to debug it manually.
- Fixed the ``source-finder`` subcommand silently doing nothing unless
  ``--config`` was explicitly passed, even though ``-sf``/``-r``/
  ``--threshold`` are documented as standalone overrides. Now falls back
  to an auto-generated default config, matching ``--compare-images``'s
  existing behaviour.
- Added ``-od``/``--outdir`` to the ``source-finder`` subcommand: source
  finders previously always wrote output next to the input image, with no
  way to redirect it.
- Added ``--sf-threshold`` to ``--compare-images``, matching the
  standalone ``source-finder`` subcommand's existing ``--threshold``
  override.
- Widened the default cross-match ``tolerance`` (0.2" -> 1.0") and
  ``shape_limit`` (6.0" -> 16.0"). The old defaults were confirmed (by
  testing real catalogue matches with each parameter varied independently)
  to be silently filtering out genuine matches rather than reflecting
  real positional/shape differences.
- Added ``-cr``/``--combined-report``: combines the flux and position
  comparison plots into a single tabbed html report
  (``<prefix>-CrossMatchReport.html``) instead of two separate files.
- Fixed ``fitsInfo()``'s reported image centre being the raw FITS
  ``CRVAL1``/``CRVAL2`` regardless of the image's coordinate frame, so a
  Galactic-projected image (common for this project's own data) reported
  its centre as if the raw (l, b) values were already RA/Dec. Now
  converts through the image's actual WCS frame. This was the root cause
  of several downstream issues: ``--compare-residuals`` scattering its
  random sample points at the wrong sky position entirely (a chain of
  further fixes was needed to get this feature working at all -- see
  below), and a source-finder's own phase-centre metadata being wrong for
  Galactic-frame inputs.
- Fixed ``--compare-residuals`` being completely non-functional: it
  crashed on Galactic-frame images (the centre-conversion fix above),
  crashed on plain 2D FITS images (assumed a 4-axis freq/Stokes cube),
  used a Bokeh API removed since Bokeh 3.x (``plot_width``/
  ``plot_height``), and could not serialise results containing raw numpy
  scalar types to JSON. All four are fixed; verified end-to-end against
  real full-tile residual/noise maps.
- Fixed ``--compare-online``: the two originally supported catalogues,
  ``sumss`` and ``nvss``, never actually returned results regardless of
  sky position, since the bare catalogue names aren't valid Vizier
  catalogue identifiers. Fixed the underlying lookup, and added four more
  catalogue options: ``racs-low`` (887.5MHz), ``racs-mid`` (1367.5MHz),
  ``racs-high`` (1655.5MHz), and ``vlass`` (2-4GHz S-band, Dec>-40 only).
  Verified against live Vizier queries.
- Added periodic progress logging (every 30s) to the source cross-matching
  routine. Matching two large catalogues can take several minutes with no
  other output; this was previously indistinguishable from a hang.
- Flux comparison plots: a source whose flux error exceeds its own value
  (seen in practice for sources in crowded/blended source-finder islands)
  previously stretched its error-bar segment across the entire visible
  log-axis range, distorting the plot. Such points now get their own,
  independently click-to-hide legend entry ("Errors (>100%)") instead of
  being mixed into the normal error display; the underlying fit is
  unaffected either way (already down-weighted by the existing
  error-weighted regression). The "Cross Matching Statistics" table also
  now reports the count directly. New ``-hlfe``/``--hide-large-flux-errors``
  controls whether this group starts hidden or shown.
- Test suite: 61 passing (from 41 at the start of this pass).
