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


class TestSorting(object):
    """Catalogue order is arbitrary, so trends read as noise until sorted."""

    # rows are [res1_rms, res2_rms, ratio, dist, name, flux, min1, min2, sn1, sn2]
    ROWS = [
        [1.0, 1.0, 1.0, 5.0, "far_faint", 0.001] + [0.0] * 4,
        [1.0, 1.0, 1.0, 1.0, "near_bright", 0.100] + [0.0] * 4,
        [1.0, 1.0, 1.0, 3.0, "mid", 0.010] + [0.0] * 4,
    ]

    def test_distance_is_nearest_first(self):
        out = aimfast._sort_source_rows(self.ROWS, "distance")
        assert [r[4] for r in out] == ["near_bright", "mid", "far_faint"]

    def test_flux_is_brightest_first(self):
        out = aimfast._sort_source_rows(self.ROWS, "flux")
        assert [r[4] for r in out] == ["near_bright", "mid", "far_faint"]

    def test_none_preserves_catalogue_order(self):
        out = aimfast._sort_source_rows(self.ROWS, "none")
        assert [r[4] for r in out] == [r[4] for r in self.ROWS]

    def test_short_rows_fall_back(self):
        """_random_residual_results rows have no flux column."""
        short = [[1.0, 1.0, 1.0, 5.0, "a"], [1.0, 1.0, 1.0, 1.0, "b"]]
        assert aimfast._sort_source_rows(short, "flux") == short
        # distance is within range, so it still sorts
        assert [r[4] for r in aimfast._sort_source_rows(short, "distance")] == ["b", "a"]

    def test_does_not_mutate_input(self):
        rows = [list(r) for r in self.ROWS]
        aimfast._sort_source_rows(rows, "distance")
        assert [r[4] for r in rows] == [r[4] for r in self.ROWS]


class TestTableStats(object):
    def _stats(self, **kw):
        kw.setdefault("res1_std", [2.0, 2.0])
        kw.setdefault("res2_std", [1.0, 1.0])
        kw.setdefault("res1_min", [])
        kw.setdefault("res2_min", [])
        kw.setdefault("model_flux", [])
        return aimfast._residual_table_stats("a", "b", units="jansky", **kw)

    def test_ratio_is_res1_over_res2(self):
        """Regression: the table computed mean(res2)/mean(res1) under this label,
        the inverse of the per-source column and of the plotted green line."""
        stats = self._stats()
        row = dict(zip(stats["Stats"], stats["Value"]))
        assert row["Res1-to-Res2"] == pytest.approx(2.0)

    def test_signed_rows_absent_without_mins(self):
        assert len(self._stats()["Stats"]) == 3

    def test_signed_rows_present_with_mins(self):
        stats = self._stats(res1_min=[-0.1, -0.1], res2_min=[-0.5, -0.5])
        row = dict(zip(stats["Stats"], stats["Value"]))
        assert row["Mean deepest negative res1 (Jy)"] == pytest.approx(-0.1)
        assert row["Mean deepest negative res2 (Jy)"] == pytest.approx(-0.5)

    def test_deep_hole_counts(self):
        """One source of 1 Jy; res2 digs a 0.5 Jy hole, res1 only 0.01 Jy."""
        stats = self._stats(
            res1_min=[-0.01, -0.01], res2_min=[-0.5, -0.5], model_flux=[1.0, 1.0]
        )
        row = dict(zip(stats["Stats"], stats["Value"]))
        assert row["Sources with res1 hole > 10% of flux (of 2)"] == 0
        assert row["Sources with res2 hole > 10% of flux (of 2)"] == 2


class TestSortAxisValues(object):
    """x must carry the quantity, not the rank, or spacing means nothing."""

    DIST = [0.1, 0.5, 1.5]
    FLUX = [0.10, 0.01, 0.001]

    def test_distance_axis_is_degrees(self):
        assert aimfast._sort_axis_values("distance", self.DIST, self.FLUX) == self.DIST

    def test_flux_axis_is_jansky(self):
        assert aimfast._sort_axis_values("flux", self.DIST, self.FLUX) == self.FLUX

    def test_none_axis_is_the_index(self):
        assert aimfast._sort_axis_values("none", self.DIST, self.FLUX) == [0, 1, 2]

    def test_flux_falls_back_when_there_is_no_flux(self):
        """The random-positions path has no flux column."""
        assert aimfast._sort_axis_values("flux", self.DIST, []) == [0, 1, 2]



class TestRatioTailCounts(object):
    """A mean ratio barely moves when a few sources are wrecked; a count does."""

    def test_counts_each_direction(self):
        # ratios are res1/res2: 20, 0.05, 1.0
        names, values = aimfast._ratio_tail_counts([20.0, 0.05, 1.0], [1.0, 1.0, 1.0])
        row = dict(zip(names, values))
        assert row["Sources with ratio > 5 (res2 better, of 3)"] == 1
        assert row["Sources with ratio < 1/5 (res1 better, of 3)"] == 1
        assert row["Sources with ratio > 10 (res2 better, of 3)"] == 1
        assert row["Sources with ratio < 1/10 (res1 better, of 3)"] == 1

    def test_thresholds_are_symmetric_on_a_log_axis(self):
        """5 and 1/5 are equally far from no-change, so equal data must count equally."""
        names, values = aimfast._ratio_tail_counts([6.0, 1 / 6.0], [1.0, 1.0])
        row = dict(zip(names, values))
        assert row["Sources with ratio > 5 (res2 better, of 2)"] == 1
        assert row["Sources with ratio < 1/5 (res1 better, of 2)"] == 1

    def test_non_finite_ratios_are_dropped(self):
        names, values = aimfast._ratio_tail_counts([1.0, 1.0], [0.0, 1.0])
        assert "of 1)" in names[0]


class TestValueFormatting(object):
    def test_small_values_use_scientific_notation(self):
        out = aimfast._format_table_stats({"Stats": ["a"], "Value": [0.00018758542137220502]})
        assert out["Value"] == ["1.876e-04"]

    def test_ratios_keep_a_few_decimals(self):
        out = aimfast._format_table_stats({"Stats": ["a"], "Value": [2.96288943290710]})
        assert out["Value"] == ["2.9629"]

    def test_counts_have_no_decimal_point(self):
        out = aimfast._format_table_stats({"Stats": ["a"], "Value": [649.0]})
        assert out["Value"] == ["649"]

    def test_zero_is_plain(self):
        out = aimfast._format_table_stats({"Stats": ["a"], "Value": [0.0]})
        assert out["Value"] == ["0"]

    def test_negatives_survive(self):
        out = aimfast._format_table_stats({"Stats": ["a"], "Value": [-0.000365303]})
        assert out["Value"] == ["-3.653e-04"]
