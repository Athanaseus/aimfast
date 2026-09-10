"""Signed per-source residual statistics.

std() is sign-blind: a negative hole and a positive spike of the same magnitude
raise it identically, so over-subtraction is invisible to it. These tests pin that
down with a hole/spike pair and assert the signed statistics separate them.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from aimfast import aimfast


BEAM = 10 / 3600.0   # 10" circular beam
PIX = 1 / 3600.0     # 1" pixels
N = 200
SRC_PIX = (60, 60)   # where the source sits, in pixels
AMP = 0.02474        # 24.74 mJy, the real over-subtracted hole


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


def _source_radec(path):
    wcs = WCS(fits.getheader(path)).celestial
    ra, dec = wcs.wcs_pix2world(SRC_PIX[1], SRC_PIX[0], 0)
    return float(ra), float(dec)


def _skymodel(path, ra, dec):
    from Tigger.Models import ModelClasses, SkyModel

    src = SkyModel.Source(
        "S1",
        ModelClasses.Position(np.deg2rad(ra), np.deg2rad(dec)),
        ModelClasses.Polarization(0.16224, 0, 0, 0),
    )
    model = SkyModel.SkyModel(src)
    model.ra0, model.dec0 = np.deg2rad(ra), np.deg2rad(dec)
    model.save(str(path))
    return str(path)


@pytest.fixture
def triple(tmp_path):
    """clean, hole, spike - identical apart from the sign of one feature."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1e-4, (N, N))

    clean = base.copy()
    hole = base.copy()
    hole[SRC_PIX] = -AMP
    spike = base.copy()
    spike[SRC_PIX] = +AMP

    paths = (
        _write(tmp_path / "clean.fits", clean),
        _write(tmp_path / "hole.fits", hole),
        _write(tmp_path / "spike.fits", spike),
    )
    ra, dec = _source_radec(paths[0])
    return paths, _skymodel(tmp_path / "model.lsm.html", ra, dec)


def _row(res_a, res_b, skymodel):
    pairs = [[dict(label="a", path=res_a), dict(label="b", path=res_b)]]
    results = aimfast._source_residual_results(pairs, skymodel, area_factor=2.0)
    assert len(results["a"]) == 1, "expected exactly one source"
    return dict(zip(aimfast.SOURCE_RESIDUAL_FIELDS, results["a"][0]))


class TestSignedStatistics(object):
    def test_std_cannot_tell_a_hole_from_a_spike(self, triple):
        """The premise: this is why a new statistic is needed at all.

        Not bit-identical - the two images share the same noise rather than being
        mirror images, so the mean shifts slightly - but they agree to ~0.03%, while
        the minimum differs by a factor of ~80.
        """
        (clean, hole, spike), sky = triple
        h = _row(clean, hole, sky)
        s = _row(clean, spike, sky)
        assert h["res2_rms"] == pytest.approx(s["res2_rms"], rel=1e-2)
        # for contrast, the signed statistic on the same pair
        assert abs(h["res2_min"]) > 50 * abs(s["res2_min"])

    def test_min_separates_them(self, triple):
        (clean, hole, spike), sky = triple
        h = _row(clean, hole, sky)
        s = _row(clean, spike, sky)
        assert h["res2_min"] == pytest.approx(-AMP, rel=1e-6)
        assert s["res2_min"] > -1e-3          # spike leaves the minimum near the noise
        assert h["res2_min"] < s["res2_min"]

    def test_sum_neg_separates_them(self, triple):
        (clean, hole, spike), sky = triple
        h = _row(clean, hole, sky)
        s = _row(clean, spike, sky)
        # the hole contributes its full depth to the negative sum; the spike adds none
        assert h["res2_sum_neg"] < s["res2_sum_neg"] - 0.9 * AMP

    def test_clean_reference_is_unaffected(self, triple):
        """res1_* describe the first image, so they must be identical in both pairs."""
        (clean, hole, spike), sky = triple
        h = _row(clean, hole, sky)
        s = _row(clean, spike, sky)
        for key in ("res1_rms", "res1_min", "res1_sum_neg"):
            assert h[key] == pytest.approx(s[key], rel=1e-9)


class TestRowLayout(object):
    def test_field_names_match_row_width(self, triple):
        (clean, hole, _), sky = triple
        pairs = [[dict(label="a", path=clean), dict(label="b", path=hole)]]
        results = aimfast._source_residual_results(pairs, sky, area_factor=2.0)
        assert len(results["a"][0]) == len(aimfast.SOURCE_RESIDUAL_FIELDS)

    def test_existing_positions_did_not_move(self, triple):
        """Plotters index these rows by number, so 0-5 must keep their meaning."""
        assert aimfast.SOURCE_RESIDUAL_FIELDS[:6] == [
            "res1_rms", "res2_rms", "rms_ratio", "phase_centre_dist", "name", "model_flux",
        ]
