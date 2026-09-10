"""Tests for dynamic-range reference position and box slicing.

Covers the dr-refpos branch:
  1. the measured region is a real 2D box, not four whole image rows
  2. the reference position defaults to this image's own peak (unchanged behaviour)
  3. ref_position pins the measurement to a given sky position
  4. an unusable ref_position raises rather than silently falling back to the peak
  5. the position and peak actually used are always reported
"""
import numpy as np
import pytest
from astropy.io import fits

from aimfast import aimfast


BEAM = 10 / 3600.0   # 10" circular beam
PIX = 1 / 3600.0     # 1" pixels
N = 200


def _write(path, data):
    hdu = fits.PrimaryHDU(data[np.newaxis, np.newaxis, :, :])
    h = hdu.header
    h["CTYPE1"], h["CRVAL1"], h["CDELT1"], h["CRPIX1"] = "RA---SIN", 10.0, -PIX, N // 2
    h["CTYPE2"], h["CRVAL2"], h["CDELT2"], h["CRPIX2"] = "DEC--SIN", -30.0, PIX, N // 2
    h["CTYPE3"], h["CRVAL3"], h["CDELT3"], h["CRPIX3"] = "FREQ", 1.4e9, 1e6, 1
    h["CTYPE4"], h["CRVAL4"], h["CDELT4"], h["CRPIX4"] = "STOKES", 1, 1, 1
    h["BMAJ"] = h["BMIN"] = BEAM
    h["BPA"] = 0.0
    hdu.writeto(path, overwrite=True)
    return str(path)


@pytest.fixture
def pair(tmp_path):
    """Image A peaks at (60,60); image B peaks elsewhere and has a hole at A's peak.

    This is the peeling scenario in miniature: the brightest source moves between the
    two images, so each image nominates a different reference source unless pinned.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1e-4, (N, N))
    a[60, 60] = 1.0                       # A's peak
    b = rng.normal(0, 1e-4, (N, N))
    b[60, 60] = -0.5                      # over-subtracted hole where A peaked
    b[150, 150] = 0.2                     # B's (different) peak
    return _write(tmp_path / "a.fits", a), _write(tmp_path / "b.fits", b)


class TestBoxSlicing(object):
    def test_measured_region_is_a_box_not_whole_rows(self, pair):
        """Regression: imslice was fancy-indexed, selecting 4 full-width rows."""
        a, _ = pair
        # A negative placed far from the peak, but on one of the rows the old
        # fancy-indexing would have picked up, must NOT affect the result.
        with fits.open(a, mode="update") as hdul:
            hdul[0].data[0, 0, 60, 190] = -10.0   # same row as the peak, far away in x
        dr = aimfast.image_dynamic_range(a, a)
        # box is ~6 beams = 60 px wide, so x=190 is outside it
        assert dr["deepest_negative"] > 1e3, (
            "a negative 130 px from the peak leaked into the measurement, so the "
            "region is not a local box"
        )


class TestReferencePosition(object):
    def test_defaults_to_own_peak(self, pair):
        a, b = pair
        dr_a, dr_b = aimfast.image_dynamic_range(a, a), aimfast.image_dynamic_range(b, b)
        assert dr_a["ref_peak_flux"] == pytest.approx(1.0, rel=1e-3)
        assert dr_b["ref_peak_flux"] == pytest.approx(0.2, rel=1e-3)
        # each nominated a different position - the bug this argument addresses
        assert dr_a["ref_position"] != dr_b["ref_position"]

    def test_pinning_measures_at_the_given_position(self, pair):
        a, b = pair
        dr_a = aimfast.image_dynamic_range(a, a)
        dr_b = aimfast.image_dynamic_range(b, b, ref_position=dr_a["ref_position"])
        assert dr_b["ref_position"] == pytest.approx(dr_a["ref_position"], abs=1e-6)
        # at A's peak, B holds the hole, not B's own peak
        assert dr_b["ref_peak_flux"] == pytest.approx(0.5, rel=1e-3)

    def test_position_and_peak_are_always_reported(self, pair):
        a, _ = pair
        dr = aimfast.image_dynamic_range(a, a)
        assert "ref_position" in dr and "ref_peak_flux" in dr
        assert len(dr["ref_position"]) == 2


class TestNoSilentFallback(object):
    def test_off_image_position_raises(self, pair):
        """Falling back to the peak would silently measure somewhere else."""
        a, _ = pair
        with pytest.raises(ValueError, match="outside"):
            aimfast.image_dynamic_range(a, a, ref_position=(200.0, 60.0))

    def test_blanked_position_raises(self, tmp_path):
        from astropy.wcs import WCS

        rng = np.random.default_rng(1)
        d = rng.normal(0, 1e-4, (N, N))
        d[60, 60] = 1.0
        d[100:110, 100:110] = np.nan          # a blanked patch
        f = _write(tmp_path / "nan.fits", d)

        # ask for the exact sky position of a blanked pixel
        wcs = WCS(fits.getheader(f)).celestial
        ra, dec = wcs.wcs_pix2world(105, 105, 0)
        with pytest.raises(ValueError, match="blanked"):
            aimfast.image_dynamic_range(f, f, ref_position=(float(ra), float(dec)))

    def test_nan_in_image_does_not_break_default_path(self, tmp_path):
        """Regression: abs(data.max()) was NaN, so the peak lookup found nothing."""
        rng = np.random.default_rng(2)
        d = rng.normal(0, 1e-4, (N, N))
        d[60, 60] = 1.0
        d[10, 10] = np.nan
        f = _write(tmp_path / "nan2.fits", d)
        dr = aimfast.image_dynamic_range(f, f)
        assert dr["ref_peak_flux"] == pytest.approx(1.0, rel=1e-3)


class TestPositionParsing(object):
    """--reference-position accepts decimal degrees or sexagesimal."""

    def test_decimal_degrees(self):
        assert aimfast._parse_sky_position("355.2791,0.3096") == (355.2791, 0.3096)

    def test_sexagesimal(self):
        ra, dec = aimfast._parse_sky_position("23:41:07,+00:18:34")
        assert ra == pytest.approx(355.2792, abs=1e-3)
        assert dec == pytest.approx(0.3094, abs=1e-3)

    def test_negative_declination_between_minus_one_and_zero(self):
        """Regression: dec2deg lost the sign for '-00:mm:ss' (float('-00') >= 0)."""
        _, dec = aimfast._parse_sky_position("23:41:07,-00:55:47")
        assert dec == pytest.approx(-0.9297, abs=1e-3)

    def test_none_passes_through(self):
        assert aimfast._parse_sky_position(None) is None

    def test_bad_input_raises(self):
        with pytest.raises(ValueError, match="RA,DEC"):
            aimfast._parse_sky_position("garbage")


class TestDec2Deg(object):
    def test_sign_preserved_across_the_zero_degree_band(self):
        from aimfast.auxiliary import dec2deg

        assert dec2deg("-00:55:47") == pytest.approx(-0.929722, abs=1e-6)
        assert dec2deg("+00:18:34") == pytest.approx(0.309444, abs=1e-6)
        assert dec2deg("-00:00:30") == pytest.approx(-0.008333, abs=1e-6)
        assert dec2deg("-30:30:00") == pytest.approx(-30.5, abs=1e-6)
        assert dec2deg("30:30:00") == pytest.approx(30.5, abs=1e-6)
