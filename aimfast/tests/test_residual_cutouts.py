"""Side-by-side map and cutout grid for --compare-residuals."""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from aimfast import aimfast


BEAM = 10 / 3600.0
PIX = 1 / 3600.0
N = 200


def _write(path, data, crval1=10.0):
    hdu = fits.PrimaryHDU(data[np.newaxis, np.newaxis, :, :].astype(np.float32))
    h = hdu.header
    h["CTYPE1"], h["CRVAL1"], h["CDELT1"], h["CRPIX1"] = "RA---SIN", crval1, -PIX, N // 2
    h["CTYPE2"], h["CRVAL2"], h["CDELT2"], h["CRPIX2"] = "DEC--SIN", -30.0, PIX, N // 2
    h["CTYPE3"], h["CRVAL3"], h["CDELT3"], h["CRPIX3"] = "FREQ", 1.4e9, 1e6, 1
    h["CTYPE4"], h["CRVAL4"], h["CDELT4"], h["CRPIX4"] = "STOKES", 1, 1, 1
    h["BMAJ"] = h["BMIN"] = BEAM
    h["BPA"] = 0.0
    hdu.writeto(path, overwrite=True)
    return str(path)


def _skymodel(path, positions):
    from Tigger.Models import ModelClasses, SkyModel

    sources = [
        SkyModel.Source(f"S{i}", ModelClasses.Position(np.deg2rad(ra), np.deg2rad(dec)),
                        ModelClasses.Polarization(0.01, 0, 0, 0))
        for i, (ra, dec) in enumerate(positions)
    ]
    model = SkyModel.SkyModel(*sources)
    model.ra0, model.dec0 = np.deg2rad(10.0), np.deg2rad(-30.0)
    model.save(str(path))
    return str(path)


# source pixels, far enough apart that their stamps do not overlap
SOURCE_PIX = [(40, 40), (40, 150), (150, 40), (150, 150)]


@pytest.fixture
def field(tmp_path):
    """Two images on one grid; image 2 digs a deep hole at source 0 only."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1e-4, (N, N))
    b = a + rng.normal(0, 2e-5, (N, N))
    b[SOURCE_PIX[0][1], SOURCE_PIX[0][0]] = -0.05          # the damage
    p1, p2 = _write(tmp_path / "one.fits", a), _write(tmp_path / "two.fits", b)
    wcs = WCS(fits.getheader(p1)).celestial
    radec = [tuple(float(v) for v in wcs.wcs_pix2world(x, y, 0)) for x, y in SOURCE_PIX]
    return p1, p2, _skymodel(tmp_path / "model.lsm.html", radec), radec


def _pair(p1, p2):
    return [[dict(label="a", path=p1), dict(label="b", path=p2)]]


# --------------------------------------------------------------------------- helpers
class TestSameGrid(object):
    def test_identical_grids_match(self, field):
        p1, p2, _, _ = field
        a, w1 = aimfast._image_plane(p1)
        b, w2 = aimfast._image_plane(p2)
        assert aimfast._same_pixel_grid(a.shape, w1, b.shape, w2)

    def test_shifted_grid_does_not_match(self, tmp_path, field):
        p1, _, _, _ = field
        p3 = _write(tmp_path / "shifted.fits", np.zeros((N, N)), crval1=10.5)
        a, w1 = aimfast._image_plane(p1)
        c, w3 = aimfast._image_plane(p3)
        assert not aimfast._same_pixel_grid(a.shape, w1, c.shape, w3)


class TestThreshold(object):
    def test_noisier_image_sets_it(self):
        rng = np.random.default_rng(1)
        quiet, noisy = rng.normal(0, 1e-4, (400, 400)), rng.normal(0, 5e-4, (400, 400))
        t = aimfast._map_threshold(quiet, noisy)
        assert t == pytest.approx(-5 * 5e-4, rel=0.05)
        # same answer either way round
        assert aimfast._map_threshold(noisy, quiet) == pytest.approx(t)


class TestWorseMarks(object):
    def test_each_panel_marks_only_where_it_is_worse(self):
        a, b = np.zeros((40, 40)), np.zeros((40, 40))
        a[5, 5] = -1.0            # hole only in image 1
        b[25, 25] = -1.0          # hole only in image 2
        a[35, 5] = b[35, 5] = -1.0   # equal hole in both
        w1, w2 = aimfast._worse_marks(a, b, threshold=-0.5, factor=10)
        assert w1[0, 0] and not w2[0, 0]
        assert w2[2, 2] and not w1[2, 2]
        assert not w1[3, 0] and not w2[3, 0], "an equal hole is worse in neither"

    def test_block_minimum_keeps_a_hole_through_downsampling(self):
        a, b = np.zeros((100, 100)), np.zeros((100, 100))
        a[55, 55] = -1.0          # a single pixel, averaged away by a block mean
        w1, _ = aimfast._worse_marks(a, b, threshold=-0.5, factor=10)
        assert w1[5, 5]


# --------------------------------------------------------------------------- picking
def _rows(specs):
    """Rows in SOURCE_RESIDUAL_FIELDS order from (name, ra, dec, min1, min2)."""
    F = aimfast.SOURCE_RESIDUAL_FIELDS
    rows = []
    for name, ra, dec, m1, m2 in specs:
        row = [0.0] * len(F)
        row[F.index("name")] = name
        row[F.index("res1_min")], row[F.index("res2_min")] = m1, m2
        row[F.index("ra_deg")], row[F.index("dec_deg")] = ra, dec
        rows.append(row)
    return rows


class TestPickSourceSpots(object):
    def test_order_of_images_does_not_change_the_pick(self, field):
        p1, _, _, radec = field
        _, wcs = aimfast._image_plane(p1)
        specs = [("S0", *radec[0], -0.1, -5.0), ("S1", *radec[1], -3.0, -0.1),
                 ("S2", *radec[2], -0.1, -0.2)]
        fwd = aimfast._pick_source_spots(_rows(specs), (N, N), wcs, count=3)
        rev = aimfast._pick_source_spots(
            _rows([(n, ra, dec, m2, m1) for n, ra, dec, m1, m2 in specs]), (N, N), wcs, count=3)
        assert [s["name"] for s in fwd] == [s["name"] for s in rev] == ["S0", "S1", "S2"]
        assert [s["worse"] for s in fwd] == ["2", "1", "2"]
        assert [s["worse"] for s in rev] == ["1", "2", "1"]

    def test_overlapping_stamps_are_skipped(self, field):
        p1, _, _, radec = field
        _, wcs = aimfast._image_plane(p1)
        near = tuple(float(v) for v in wcs.wcs_pix2world(SOURCE_PIX[0][0] + 5, SOURCE_PIX[0][1], 0))
        spots = aimfast._pick_source_spots(
            _rows([("S0", *radec[0], 0.0, -5.0), ("close", *near, 0.0, -4.0),
                   ("S1", *radec[1], 0.0, -1.0)]), (N, N), wcs, count=3)
        assert [s["name"] for s in spots] == ["S0", "S1"]

    def test_count_is_respected(self, field):
        p1, _, _, radec = field
        _, wcs = aimfast._image_plane(p1)
        specs = [(f"S{i}", *radec[i], 0.0, -float(i + 1)) for i in range(4)]
        assert len(aimfast._pick_source_spots(_rows(specs), (N, N), wcs, count=2)) == 2


class TestPickPixelChanges(object):
    def test_finds_changes_in_both_directions(self, field):
        p1, _, _, _ = field
        _, wcs = aimfast._image_plane(p1)
        a, b = np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)
        a[50, 50] = 1.0           # source vanished from image 2
        b[150, 150] = 0.5         # source appeared in image 2
        spots = aimfast._pick_pixel_changes(a, b, wcs, count=2)
        assert [(s["x"], s["y"], s["worse"]) for s in spots] == [(50, 50, "2"), (150, 150, "1")]
        assert "2 lower by 1000.00 mJy" in spots[0]["title"]

    def test_changes_within_one_stamp_are_counted_once(self, field):
        p1, _, _, _ = field
        _, wcs = aimfast._image_plane(p1)
        a, b = np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)
        b[100, 100], b[100, 110] = -1.0, -0.9
        assert len(aimfast._pick_pixel_changes(a, b, wcs, count=5)) == 1


class TestStamps(object):
    def test_stamp_is_centred_and_padded_at_the_edge(self):
        plane = np.arange(100, dtype=np.float32).reshape(10, 10)
        st = aimfast._stamp(plane, 0, 0, half=4)
        assert st.shape == (8, 8)
        assert st[4, 4] == plane[0, 0]
        assert np.isnan(st[0, 0])

    def test_limit_follows_the_deeper_negative_but_not_below_the_floor(self):
        a, b = np.array([-2.0, 1.0]), np.array([-0.5, 9.0])
        assert aimfast._stamp_limit(a, b, floor=0.1) == pytest.approx(3.0)
        assert aimfast._stamp_limit(np.zeros(2), np.zeros(2), floor=0.1) == pytest.approx(0.1)


class TestSkyToPixel(object):
    def test_outside_raises(self, field):
        p1, _, _, _ = field
        plane, wcs = aimfast._image_plane(p1)
        with pytest.raises(ValueError, match="outside"):
            aimfast._sky_to_pixel(wcs, (200.0, 60.0), plane, p1)

    def test_blanked_raises(self, field):
        p1, _, _, radec = field
        plane, wcs = aimfast._image_plane(p1)
        plane[SOURCE_PIX[1][1], SOURCE_PIX[1][0]] = np.nan
        with pytest.raises(ValueError, match="blanked"):
            aimfast._sky_to_pixel(wcs, radec[1], plane, p1)


# --------------------------------------------------------------------------- end to end
class TestCompareResiduals(object):
    def test_catalogue_path_writes_cutouts_and_finds_the_hole(self, field, tmp_path, monkeypatch):
        p1, p2, sky, _ = field
        monkeypatch.chdir(tmp_path)
        res = aimfast.compare_residuals(_pair(p1, p2), sky, area_factor=2.0)
        assert os.path.exists("SourceResidualCutouts.html")
        spots = aimfast._pick_source_spots(res["a"], (N, N), aimfast._image_plane(p1)[1])
        assert spots[0]["name"] == "S0" and spots[0]["worse"] == "2"

    def test_rows_carry_positions(self, field, tmp_path, monkeypatch):
        p1, p2, sky, radec = field
        monkeypatch.chdir(tmp_path)
        res = aimfast._source_residual_results(_pair(p1, p2), sky, area_factor=2.0)
        F = aimfast.SOURCE_RESIDUAL_FIELDS
        got = {r[F.index("name")]: (r[F.index("ra_deg")], r[F.index("dec_deg")]) for r in res["a"]}
        assert got["S2"] == pytest.approx(radec[2], abs=1e-6)

    def test_random_path_writes_cutouts(self, field, tmp_path, monkeypatch):
        p1, p2, _, _ = field
        monkeypatch.chdir(tmp_path)
        aimfast.compare_residuals(_pair(p1, p2), points=20, fov_factor=0.9, area_factor=2.0)
        assert os.path.exists("RandomResidualCutouts.html")

    def test_combined_report_replaces_the_separate_files(self, field, tmp_path, monkeypatch):
        p1, p2, sky, _ = field
        monkeypatch.chdir(tmp_path)
        aimfast.compare_residuals(_pair(p1, p2), sky, area_factor=2.0, combined_report=True)
        assert os.path.exists("ResidualReport.html")
        assert not os.path.exists("SourceResidualCutouts.html")
        assert not os.path.exists("SourceResidualNoiseRatio.html")

    def test_different_grids_skip_cutouts_but_keep_the_noise_plot(self, field, tmp_path,
                                                                   monkeypatch, caplog):
        p1, _, sky, _ = field
        p3 = _write(tmp_path / "shifted.fits", np.random.default_rng(3).normal(0, 1e-4, (N, N)),
                    crval1=10.001)
        monkeypatch.chdir(tmp_path)
        aimfast.compare_residuals(_pair(p1, p3), sky, area_factor=2.0)
        assert not os.path.exists("SourceResidualCutouts.html")
        assert "not on the same pixel grid" in caplog.text

    def test_reference_off_image_fails_before_writing_anything(self, field, tmp_path, monkeypatch):
        p1, p2, sky, _ = field
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside"):
            aimfast.compare_residuals(_pair(p1, p2), sky, area_factor=2.0, ref_position=(200.0, 60.0))
        # the fixture's own skymodel is model.lsm.html; only plots count
        assert not [f for f in os.listdir(".") if f.endswith(".html") and not f.endswith(".lsm.html")]

    def test_reference_position_is_accepted(self, field, tmp_path, monkeypatch):
        p1, p2, sky, radec = field
        monkeypatch.chdir(tmp_path)
        aimfast.compare_residuals(_pair(p1, p2), sky, area_factor=2.0, ref_position=radec[3])
        assert "Reference position" in open("SourceResidualCutouts.html").read()

    def test_svg_writes_the_map_and_the_grid(self, field, tmp_path, monkeypatch):
        pytest.importorskip("matplotlib")
        p1, p2, sky, _ = field
        monkeypatch.chdir(tmp_path)
        aimfast._residual_cutouts(_pair(p1, p2), results=aimfast._source_residual_results(
            _pair(p1, p2), sky, area_factor=2.0), outfile="Cut.html", svg=True)
        assert os.path.exists("Cut_map.svg") and os.path.exists("Cut.svg")


class TestSvgWithoutBrowser(object):
    def test_missing_browser_does_not_abort_the_run(self, field, tmp_path, monkeypatch, caplog):
        """bokeh's SVG export needs a browser driver; its absence must not stop the
        matplotlib SVGs, which need none."""
        pytest.importorskip("matplotlib")
        p1, p2, sky, _ = field
        monkeypatch.chdir(tmp_path)

        def no_browser(*args, **kwargs):
            raise RuntimeError("Neither firefox and geckodriver nor chromium are available")

        monkeypatch.setattr(aimfast, "export_svgs", no_browser)
        aimfast.compare_residuals(_pair(p1, p2), sky, area_factor=2.0, svg=True)
        assert os.path.exists("SourceResidualCutouts.svg")
        assert os.path.exists("SourceResidualCutouts_map.svg")
        assert "Skipping SVG of the noise-ratio plot" in caplog.text


class TestSvgGridLayout(object):
    def test_reference_gets_its_own_row_and_the_grid_keeps_its_shape(self, field, tmp_path,
                                                                     monkeypatch):
        """With 18 spots and a reference: one reference row, then 3 x 6 - not 19 cells
        flowing into a 7th row."""
        pytest.importorskip("matplotlib")
        from matplotlib.figure import Figure

        grids = []
        real_add = Figure.add_gridspec

        def spy(self, nrows, ncols, **kw):
            grids.append(nrows)
            return real_add(self, nrows, ncols, **kw)

        monkeypatch.setattr(Figure, "add_gridspec", spy)
        p1, p2, _, radec = field
        plane1, _ = aimfast._image_plane(p1)
        plane2, _ = aimfast._image_plane(p2)
        spots = [dict(x=10 + i, y=10, title=f"s{i}") for i in range(aimfast.CUTOUT_COUNT)]
        ref = dict(x=100, y=100, title="ref")
        aimfast._save_cutouts_svg(plane1, plane2, spots, 1e-4, str(tmp_path / "g.svg"),
                                  reference=ref)
        assert grids == [1 + aimfast.CUTOUT_COUNT // aimfast.CUTOUT_COLUMNS]
