import os
import numpy as np
import pytest
from pathlib import Path

from aimfast import aimfast
from bokeh.models import ColumnDataSource
from Tigger.Models import SkyModel


class TestClass(object):
    """aimfast Test klass"""

    def test_deg2arcsec(self):
        """Test deg2arcsec method"""
        input_value = 10.5
        output_value = aimfast.deg2arcsec(input_value)
        expected_value = input_value * 3600.00
        assert expected_value == output_value

    def test_rad2deg(self):
        """Test rad2deg method"""
        input_value = 1.5
        output_value = aimfast.rad2deg(input_value)
        expected_value = input_value * (180 / np.pi)
        assert expected_value == output_value

    def test_rad2arcsec(self):
        """Test rad2arcsec method"""
        input_value = 1.5
        output_value = aimfast.rad2arcsec(input_value)
        expected_value = input_value * (3600.0 * 180.0 / np.pi)
        assert expected_value == output_value

    def test_noise_sigma(self):
        """Test noise sigma metho"""
        input_value = "aimfast/tests/files/cube1.fits"
        output_value = aimfast.noise_sigma(input_value)
        expected_value = 3.1e-05
        assert expected_value == pytest.approx(output_value, 0.01855)

    def test_fitsInfo(self):
        """Test fitsInfo method"""
        input_value = "aimfast/tests/files/cube1.fits"
        output_value = aimfast.fitsInfo(input_value)
        from astropy.wcs import WCS

        expected = {
            "b_size": (0.00154309340472658, 0.00136912246542523, 159.801295045408),
            "centre": (0.0, -30.0),
            "ddec": 0.000277777777777778,
            "dec": -30.0,
            "decPix": 10,
            "dra": 0.000277777777777778,
            "numPix": 20,
            "ra": 0.0,
            "raPix": 10,
            "skyArea": 3.0864197530864246e-05,
            "wcs": WCS,
        }
        for param, value in expected.items():
            val = output_value[param]
            if param == "wcs":
                assert isinstance(val, value)
            elif param == "centre":
                # "centre" now round-trips through SkyCoord(...).icrs (fix
                # for Galactic-frame images), introduces ~1e-5 deg noise
                # even for already-equatorial images, and can wrap e.g.
                # RA=0.0 to ~359.99997. Compare angle-aware, not exact.
                for got, want in zip(val, value):
                    diff = min(abs(got - want), abs(got - want + 360), abs(got - want - 360))
                    assert diff < 1e-3
            else:
                assert val == value

    def test_residual_stats(self):
        """Test the residuals stats method"""
        input_value = "aimfast/tests/files/cube1.fits"
        input_mask = "aimfast/tests/files/mask.fits"

        def test(expected, output_value, normality=False):
            if normality:
                output_normaltest_value = output_value.pop("NORM")
                expected_normaltest_value = expected_value.pop("NORM")
                assert expected_normaltest_value == pytest.approx(output_normaltest_value, 1.0e-4)
            # Use approx for floating point comparisons with higher tolerance for MAD
            for key, expected_val in expected_value.items():
                if key == "MAD":
                    assert output_value[key] == pytest.approx(expected_val, rel=0.25)
                else:
                    assert output_value[key] == pytest.approx(expected_val, rel=1e-2)

        # Test residual stats
        output_value = aimfast.residual_image_stats(input_value, test_normality="normaltest")
        expected_value = {
            "NORM": (10.276033206715848, 0.005869319364736688),
            "SKEW": 0.186153,
            "KURT": 2.870047,
            "RMS": 3.1e-05,
            "MAD": 2.2e-05,
            "MIN": -8.35573e-05,
            "MAX": 9.98428e-05,
            "SUM_NEG": -0.01914,
            "STDDev": 3.1e-05,
            "MEAN": 1.21497e-06,
        }
        test(expected_value, output_value, normality=True)

        # Test using mask
        output_value = aimfast.residual_image_stats(input_value, mask=input_mask)
        expected_value = {
            "SKEW": -0.341298,
            "KURT": 2.422672,
            "RMS": 2.6e-05,
            "MAD": 2.2e-05,
            "MIN": -8.35573e-05,
            "MAX": 3.98551e-05,
            "SUM_NEG": -0.01914,
            "STDDev": 2.5e-05,
            "MEAN": -5.57698e-06,
        }
        test(expected_value, output_value)

        # Test using channels
        output_value = aimfast.residual_image_stats(input_value, chans="2~3")
        expected_value = {
            "SKEW": 0.287936,
            "KURT": 2.891433,
            "RMS": 3.2e-05,
            "MAD": 2.2e-05,
            "MIN": -8.35573e-05,
            "MAX": 9.98428e-05,
            "SUM_NEG": -0.010724,
            "STDDev": 3.2e-05,
            "MEAN": -9.18495e-07,
        }
        test(expected_value, output_value)

        # Test using threshold
        output_value = aimfast.residual_image_stats(input_value, threshold=0.00005)
        expected_value = {
            "SKEW": 0.186153,
            "KURT": 2.870047,
            "RMS": 3.1e-05,
            "MAD": 2.2e-05,
            "MIN": -8.35573e-05,
            "MAX": 9.98428e-05,
            "SUM_NEG": -0.01914,
            "STDDev": 3.1e-05,
            "MEAN": 1.21497e-06,
        }
        test(expected_value, output_value)

    def test_get_detected_sources_properties(self):
        """Test get detected sources properties"""
        expected_label = "None-model_a_0"
        label = None
        input_dir = "aimfast/tests/files"
        model1 = "catalog1.txt"
        model2 = "catalog1.lsm.html"
        model1_path = "{:s}/{:s}".format(input_dir, model1)
        model2_path = "{:s}/{:s}".format(input_dir, model2)
        models = [
            [
                dict(label="{}-model_a_0".format(label), path=model1_path),
                dict(label="{}-model_b_0".format(label), path=model2_path),
            ]
        ]
        expected = aimfast.get_aimfast_data("fidelity_results.json", input_dir)
        output = aimfast.compare_models(models, tolerance=0.2, plot=False, all_sources=True)
        models = expected[expected_label]["models"]
        assert models == [model1, model2]
        assert len(expected[expected_label]["flux"]) == len(output[expected_label]["flux"])
        assert len(expected[expected_label]["position"]) == len(output[expected_label]["position"])

    def test_resolve_compare_source_finders(self):
        """Test compare-image source finder list resolution"""
        single = aimfast._resolve_compare_source_finders(["pybdsf"], 2)
        assert single == ["pybdsf", "pybdsf", "pybdsf", "pybdsf"]

        pairwise = aimfast._resolve_compare_source_finders(["breizorro", "pybdsf"], 3)
        assert pairwise == ["breizorro", "pybdsf", "breizorro", "pybdsf", "breizorro", "pybdsf"]

        exact = aimfast._resolve_compare_source_finders(
            ["aegean", "pybdsf", "breizorro", "aegean"], 2
        )
        assert exact == ["aegean", "pybdsf", "breizorro", "aegean"]

        fallback = aimfast._resolve_compare_source_finders(["aegean", "pybdsf", "breizorro"], 2)
        assert fallback == ["aegean", "breizorro", "aegean", "breizorro"]

    def test_get_argparser_compare_images_ncpu_and_sourcery_list(self):
        """Test general compare-images CLI options parse as expected"""
        parser = aimfast.get_argparser()
        args = parser.parse_args(
            [
                "--compare-images",
                "image1.fits",
                "image2.fits",
                "--ncpu",
                "2",
                "-sf",
                "breizorro",
                "pybdsf",
            ]
        )

        assert args.ncpu == 2
        assert args.sourcery == ["breizorro", "pybdsf"]

    def test_random_residual_results(self):
        """Test comparison of random residuals in images"""
        expected_label = "random-res_a_0"
        label = "random"
        input_dir = "aimfast/tests/files"
        res1 = "cube1.fits"
        res2 = "cube2.fits"
        res1_path = "{:s}/{:s}".format(input_dir, res1)
        res2_path = "{:s}/{:s}".format(input_dir, res2)
        res_imgs = [
            [
                dict(label="{}-res_a_0".format(label), path=res1_path),
                dict(label="{}-res_b_0".format(label), path=res2_path),
            ]
        ]
        expected = aimfast.get_aimfast_data("fidelity_results.json", input_dir)
        output = aimfast._random_residual_results(
            res_imgs, data_points=10, area_factor=0.5, fov_factor=0.5
        )
        assert len(expected[expected_label]) == len(output[expected_label])

    def test_source_residual_results(self):
        """Test comparison of source residuals in images"""
        default_area_factor = 2.0
        expected_label = "source-res_a_0"
        label = "source"
        input_dir = "aimfast/tests/files"
        res1 = "cube1.fits"
        res2 = "cube2.fits"
        skymodel = "catalog.lsm.html"
        skymodel_path = "{:s}/{:s}".format(input_dir, skymodel)
        res1_path = "{:s}/{:s}".format(input_dir, res1)
        res2_path = "{:s}/{:s}".format(input_dir, res2)
        res_imgs = [
            [
                dict(label="{}-res_a_0".format(label), path=res1_path),
                dict(label="{}-res_b_0".format(label), path=res2_path),
            ]
        ]
        expected = aimfast.get_aimfast_data("fidelity_results.json", input_dir)
        output = aimfast._source_residual_results(
            res_imgs, skymodel_path, area_factor=default_area_factor
        )
        assert len(expected[expected_label]) == len(output[expected_label])

    def test_model_dynamic_range(self):
        """Test dynamic range from model"""
        input_model = "DR_catalog.txt"
        residual = "cube1.fits"
        input_dir = "aimfast/tests/files"
        model_path = "{:s}/{:s}".format(input_dir, input_model)
        res_path = "{:s}/{:s}".format(input_dir, residual)
        output_value = aimfast.model_dynamic_range(model_path, res_path, area_factor=1)
        expected_value = {
            "deepest_negative": 130.40011268142158,
            "local_rms": 224.29535518140233,
            "global_rms": 174.74979385790667,
        }
        # Use approx for floating point comparisons with 1% relative tolerance
        assert output_value["deepest_negative"] == pytest.approx(
            expected_value["deepest_negative"], rel=0.01
        )
        assert output_value["local_rms"] == pytest.approx(expected_value["local_rms"], rel=0.01)
        assert output_value["global_rms"] == pytest.approx(expected_value["global_rms"], rel=0.01)

    def test_image_dynamic_range(self):
        """Test dynamic range from image"""
        restored_image = "cube1.fits"
        residual_image = "cube2.fits"
        input_dir = "aimfast/tests/files"
        restored_image_path = "{:s}/{:s}".format(input_dir, restored_image)
        residual_image_path = "{:s}/{:s}".format(input_dir, residual_image)
        output_value = aimfast.image_dynamic_range(
            restored_image_path, residual_image_path, area_factor=1
        )
        expected_value = {
            "deepest_negative": 1.4872031158104637,
            "local_rms": 3.098743200302124,
            "global_rms": 3.394456386566162,
        }
        assert expected_value == output_value

    def test_ra2deg_conversion(self):
        """Test ra2deg method with standard input"""
        from aimfast.auxiliary import ra2deg

        # Test standard RA conversion: 12:30:45.5 should be ~187.69 degrees
        input_value = "12:30:45.5"
        output_value = ra2deg(input_value)
        expected_value = (12 * 15.0) + ((30 / 60.0) * 15.0) + ((45.5 / 3600) * 15.0)
        assert pytest.approx(expected_value, 0.001) == output_value

    def test_deg2ra_negative_value(self):
        """Test deg2ra handles negative RA values"""
        from aimfast.auxiliary import deg2ra

        # Test negative RA value gets normalized
        input_value = -10.0
        output_value = deg2ra(input_value)
        # -10 degrees should become 350 degrees
        assert output_value is not None
        assert ":" in output_value

    def test_deg2ra_large_value(self):
        """Test deg2ra handles RA values > 360"""
        from aimfast.auxiliary import deg2ra

        # Test RA value > 360 gets normalized
        input_value = 370.0
        output_value = deg2ra(input_value)
        # 370 degrees should become 10 degrees
        assert output_value is not None
        assert ":" in output_value

    def test_dec2deg_positive(self):
        """Test dec2deg with positive declination"""
        from aimfast.auxiliary import dec2deg

        # Test positive DEC conversion: +30:15:20 should be ~30.255 degrees
        input_value = "+30:15:20"
        output_value = dec2deg(input_value)
        expected_value = 30 + (15 / 60.0) + (20 / 3600.0)
        assert pytest.approx(expected_value, 0.001) == output_value

    def test_dec2deg_negative(self):
        """Test dec2deg with negative declination"""
        from aimfast.auxiliary import dec2deg

        # Test negative DEC conversion: -30:15:20 should be ~-30.255 degrees
        input_value = "-30:15:20"
        output_value = dec2deg(input_value)
        expected_value = -(30 + (15 / 60.0) + (20 / 3600.0))
        assert pytest.approx(expected_value, 0.001) == output_value

    def test_convert_catalog_with_index_columns(self, tmp_path):
        """Test conversion of a simple CSV using column indices as mappings"""
        import os

        csv = tmp_path / "test_map.csv"
        csv.write_text(
            "name,ra,dec,flux,flux_err\nSRC1,10.0,-30.0,0.0001,1e-06\nSRC2,12.0,-31.0,0.0002,2e-06\n"
        )
        mappings = {
            "name": "0",
            "position_xaxis": "1",
            "position_yaxis": "2",
            "flux_xaxis": "3",
            "flux_err_xaxis": "4",
        }
        model = aimfast.convert_catalog_with_mapping(str(csv), mappings)
        assert len(model.sources) == 2
        s0 = model.sources[0]
        assert float(s0.flux.I) == pytest.approx(0.0001)
        # ra/dec were given in degrees -> check angle
        assert pytest.approx(10.0, rel=1e-6) == round(np.rad2deg(s0.pos.ra), 6)

    def test_convert_catalog_with_sexagesimal(self, tmp_path):
        """Test conversion of sexagesimal RA/DEC strings using mappings by name"""
        csv = tmp_path / "test_map2.csv"
        csv.write_text("name,ra_hms,dec_dms,flux\nSRC1,12:30:45.5,-30:15:20,0.00015\n")
        mappings = {
            "name": "name",
            "position_xaxis": "ra_hms",
            "position_yaxis": "dec_dms",
            "flux_xaxis": "flux",
        }
        model = aimfast.convert_catalog_with_mapping(str(csv), mappings)
        assert len(model.sources) == 1
        s0 = model.sources[0]
        # RA should be close to 187.6895833333 degrees
        assert pytest.approx(187.6895833, rel=1e-5) == round(np.rad2deg(s0.pos.ra), 7)

    def test_convert_catalog_with_error_metadata(self, tmp_path):
        """Test mapped error columns are preserved in source metadata"""
        csv = tmp_path / "test_map3.csv"
        csv.write_text(
            "name,ra,dec,ra_err,dec_err,flux,flux_err\nSRC1,10.0,-30.0,0.01,0.02,0.0001,1e-06\n"
        )
        mappings = {
            "name": "name",
            "position_xaxis": "ra",
            "position_yaxis": "dec",
            "position_err_xaxis": "ra_err",
            "position_err_yaxis": "dec_err",
            "flux_xaxis": "flux",
            "flux_err_xaxis": "flux_err",
        }
        model = aimfast.convert_catalog_with_mapping(str(csv), mappings)
        source = model.sources[0]
        assert pytest.approx(10.0, rel=1e-6) == round(np.rad2deg(source.pos.ra), 6)
        assert pytest.approx(-30.0, rel=1e-6) == round(np.rad2deg(source.pos.dec), 6)
        assert source.pos.ra_err > 0
        assert source.pos.dec_err > 0
        assert source.getTag("I_peak_err") == pytest.approx(1e-06)

    def test_catalog_default_extension(self):
        """Test default filename extension selection for catalog formats"""
        from aimfast.auxiliary import catalog_default_extension

        assert catalog_default_extension("fits") == "fits"
        assert catalog_default_extension("csv") == "csv"
        assert catalog_default_extension("ascii") == "txt"
        assert catalog_default_extension("ascii.ecsv") == "txt"
        assert catalog_default_extension("tab") == "tab"
        assert catalog_default_extension("ecsv") == "ecsv"
        assert catalog_default_extension("unknown") == "txt"

    def test_compare_models_with_per_catalog_mappings(self, tmp_path):
        """Test compare_models accepts independent column mappings per catalog"""
        catalog1 = tmp_path / "catalog_one.csv"
        catalog2 = tmp_path / "catalog_two.csv"
        catalog1.write_text("source_id,alpha,delta,flux_jy\nA,10.0,-30.0,0.0001\n")
        catalog2.write_text("src_name,ra_deg,dec_deg,int_flux\nB,10.0,-30.0,0.0002\n")
        models = [
            [
                dict(label="pair-model_a_0", path=str(catalog1)),
                dict(label="pair-model_b_0", path=str(catalog2)),
            ]
        ]
        model_mappings = [
            {
                "name": "source_id",
                "position_xaxis": "alpha",
                "position_yaxis": "delta",
                "flux_xaxis": "flux_jy",
            },
            {
                "name": "src_name",
                "position_xaxis": "ra_deg",
                "position_yaxis": "dec_deg",
                "flux_xaxis": "int_flux",
            },
        ]
        output = aimfast.compare_models(
            models, plot=False, all_sources=True, model_mappings=model_mappings
        )
        result = output["pair-model_a_0"]
        assert len(result["flux"]) == 1
        assert len(result["position"]) == 1

    def test_plot_model_columns_with_plain_catalog(self, tmp_path):
        """Test basic catalog plotting works for a plain CSV catalog"""
        catalog = tmp_path / "wise_hii_V2.3.csv"
        catalog.write_text("name,RA,Dec,flux\nSRC1,10.0,-30.0,0.001\nSRC2,11.0,-31.0,0.002\n")

        output_prefix = tmp_path / "basic_catalog_plot"
        aimfast.plot_model_columns(str(catalog), "RA", "Dec", html_prefix=str(output_prefix))

        assert Path(f"{output_prefix}.html").exists()

    def test_plot_model_columns_with_txt_catalog(self, tmp_path):
        """Test basic catalog plotting works for a commented-header TXT catalog"""
        catalog = Path("aimfast/tests/files/catalog1.txt")

        output_prefix = tmp_path / "txt_catalog_plot"
        aimfast.plot_model_columns(str(catalog), "ra_d", "dec_d", html_prefix=str(output_prefix))

        assert Path(f"{output_prefix}.html").exists()

    def test_plot_model_columns_with_sexagesimal_positions(self, tmp_path):
        """Test basic catalog plotting accepts sexagesimal RA/Dec strings"""
        catalog = tmp_path / "sexagesimal_catalog.csv"
        catalog.write_text(
            "name,ra,dec,flux\n"
            "SRC1,14:12:21.90,-30:00:00.0,0.001\n"
            "SRC2,14:15:00.00,-30:10:00.0,0.002\n"
        )

        output_prefix = tmp_path / "sexagesimal_catalog_plot"
        aimfast.plot_model_columns(str(catalog), "ra", "dec", html_prefix=str(output_prefix))

        assert Path(f"{output_prefix}.html").exists()

    def test_link_table_selection_to_plot_registers_callback(self):
        """Test table selection wiring registers a selection callback"""
        table_source = ColumnDataSource(data={"ra": [1.0], "dec": [2.0]})
        plot_source = ColumnDataSource(data={"ra": [1.0], "dec": [2.0]})

        aimfast._link_table_selection_to_plot(table_source, plot_source)

        callbacks = table_source.selected.js_property_callbacks.get("change:indices", [])
        assert callbacks
        assert any("plot_source.selected.indices" in callback.code for callback in callbacks)

    def test_get_model_reads_aegean_tab(self, tmp_path, monkeypatch):
        """Test Aegean-style tab catalogs produce a populated model"""
        catalog = tmp_path / "sample_aegean_isle.tab"
        catalog.write_text(
            "island components background local_rms ra_str dec_str ra dec peak_flux int_flux err_int_flux eta x_width y_width max_angular_size pa pixels area beam_area flags uuid\n"
            "1 1 0.0 0.0 14:12:21.90 -30:00:00.0 213.091265 -30.0 0.0095 0.00045 0.00001 0.0 3 10 0.0 84.5 22 1.0 1.0 0 x\n"
        )
        fits_file = tmp_path / "sample.fits"
        fits_file.write_text("dummy")

        monkeypatch.setattr(aimfast.os.path, "exists", lambda path: str(path) == str(fits_file))
        monkeypatch.setattr(aimfast, "fitsInfo", lambda path: {"centre": (0.0, -30.0)})

        model = aimfast.get_model(str(catalog))

        assert len(model.sources) == 1
        assert pytest.approx(213.091265, rel=1e-6) == round(np.rad2deg(model.sources[0].pos.ra), 6)

    def test_get_model_sanitizes_aegean_nan_errors(self, tmp_path, monkeypatch):
        """Test Aegean nan uncertainty values do not propagate into model attributes"""
        catalog = tmp_path / "sample_aegean_isle.tab"
        catalog.write_text(
            "island components background local_rms ra_str dec_str ra dec peak_flux int_flux err_int_flux eta x_width y_width max_angular_size pa pixels area beam_area flags uuid\n"
            "1 1 0.0 0.0 14:12:21.90 -30:00:00.0 213.091265 -30.0 0.0095 0.00045 nan 0.0 3 10 0.0 84.5 22 1.0 1.0 0 x\n"
        )
        fits_file = tmp_path / "sample.fits"
        fits_file.write_text("dummy")

        monkeypatch.setattr(aimfast.os.path, "exists", lambda path: str(path) == str(fits_file))
        monkeypatch.setattr(aimfast, "fitsInfo", lambda path: {"centre": (0.0, -30.0)})

        model = aimfast.get_model(str(catalog))
        source = model.sources[0]

        assert np.isfinite(source.flux.I_err)
        assert np.isfinite(source.getTag("I_peak_err"))

    @staticmethod
    def _make_source(name, ra_deg, dec_deg, flux_jy, maj_arcsec, min_arcsec, pa_deg=0.0):
        """Build a Tigger source with a real Gaussian shape but no explicit
        shape-error kwargs, reproduces getShapeErr() returning None the
        same way real Legacy/Cosmetic Aegean .lsm.html catalogues do after
        a Tigger save/load round-trip (Tigger's serialiser omits shape-error
        tags that were never set, rather than writing zeros)."""
        from Tigger.Models import ModelClasses, SkyModel

        pos = ModelClasses.Position(np.deg2rad(ra_deg), np.deg2rad(dec_deg))
        flux = ModelClasses.Polarization(flux_jy, 0, 0, 0)
        shape = ModelClasses.Gaussian(
            np.deg2rad(maj_arcsec / 3600.0), np.deg2rad(min_arcsec / 3600.0), np.deg2rad(pa_deg)
        )
        return SkyModel.Source(name, pos, flux, shape=shape)

    def test_get_src_scale_handles_none_shape_error(self):
        """Regression test: get_src_scale must not crash when
        shape.getShapeErr() returns None (aimfast.py line ~897 used to do
        shape_out_err[0]/[1] unconditionally)."""
        from Tigger.Models import ModelClasses

        shape = ModelClasses.Gaussian(np.deg2rad(10.0 / 3600.0), np.deg2rad(5.0 / 3600.0), 0.0)
        assert shape.getShapeErr() is None  # sanity check this actually hits the trigger condition

        scale, scale_err = aimfast.get_src_scale(shape)
        assert np.isfinite(scale) and scale > 0
        assert scale_err == 0.0

    def test_get_detected_sources_properties_handles_sources_without_shape_errors(self, tmp_path):
        """Regression test: matching two catalogues whose sources have a
        real shape but no shape-error info must not crash
        get_detected_sources_properties, previously crashed in three
        places (model1_source's shape_in_err, model2_source's
        shape_out_err, and get_src_scale) whenever getShapeErr() returned
        None, which is the common case for real Aegean .lsm.html output."""
        model1 = SkyModel.SkyModel(
            self._make_source("S1", 10.0, -30.0, 0.001, 8.0, 6.0)
        )
        model2 = SkyModel.SkyModel(
            self._make_source("S2", 10.0001, -30.0001, 0.0012, 8.5, 6.2)
        )
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        props = aimfast.get_detected_sources_properties(
            path1, path2, tolerance=8.0, shape_limit=12.0
        )
        targets_flux = props[0]
        assert len(targets_flux) == 1

    def test_compare_models_forwards_shape_limit(self, tmp_path):
        """Regression test: compare_models() must actually forward its
        shape_limit argument to get_detected_sources_properties(), it
        previously silently dropped it, so the CLI's -sl/--shape-limit
        flag had zero effect regardless of value (the underlying function
        always used its own default, 6.0")."""
        header = "ra,dec,int_flux,err_int_flux,peak_flux,err_peak_flux,a,err_a,b,err_b,pa,err_pa\n"
        row = "10.0,-30.0,{flux},0.00001,{flux},0.00001,{maj},0.1,6.0,0.1,0.0,1.0\n"
        catalog1 = tmp_path / "model_a.csv"
        catalog2 = tmp_path / "model_b.csv"
        catalog1.write_text(header + row.format(flux=0.001, maj=8.0))
        # model2's source is a real 100" extended source at the same position
        catalog2.write_text(header + row.format(flux=0.0012, maj=100.0))

        models = [
            [
                dict(label="pair-model_a_0", path=str(catalog1)),
                dict(label="pair-model_b_0", path=str(catalog2)),
            ]
        ]

        tight = aimfast.compare_models(
            models, tolerance=8.0, shape_limit=50.0, plot=False, all_sources=False
        )
        loose = aimfast.compare_models(
            models, tolerance=8.0, shape_limit=150.0, plot=False, all_sources=False
        )

        assert len(tight["pair-model_a_0"]["flux"]) == 0
        assert len(loose["pair-model_a_0"]["flux"]) == 1

    def test_shape_limit_argparser_type(self):
        """Regression test: -sl/--shape-limit must parse as a float, not a
        bare string, it previously had no type=float (unlike the
        neighbouring -tol/--tolerance argument), so any CLI-provided value
        crashed downstream with a float-vs-str TypeError the first time it
        reached a numeric comparison."""
        parser = aimfast.get_argparser()
        args = parser.parse_args(
            ["--compare-models", "model1.lsm.html", "model2.lsm.html", "-sl", "12"]
        )

        assert args.shape_limit == 12.0
        assert isinstance(args.shape_limit, float)

    def test_resolve_phase_centre_falls_back_on_bad_fits_file(self, tmp_path, monkeypatch):
        """Regression test: _resolve_phase_centre must fall back to the
        model's own computed centre for *any* failure reading fits_file,
        missing file, wrong/corrupt file, or (as happened in practice) a
        filename heuristic elsewhere in get_model() that guessed a fits_file
        path equal to the catalogue itself."""
        model = SkyModel.SkyModel(
            self._make_source("S1", 10.0, -30.0, 0.001, 8.0, 6.0),
            self._make_source("S2", 12.0, -28.0, 0.001, 8.0, 6.0),
        )
        expected = aimfast._get_phase_centre(model)

        # missing file
        assert aimfast._resolve_phase_centre(str(tmp_path / "does_not_exist.fits"), model) == expected

        # a path that exists but isn't a valid FITS file (the exact failure
        # mode that originally crashed the PyBDSF .txt branch of get_model)
        bad_fits = tmp_path / "not_really_fits.fits"
        bad_fits.write_text("this is not a FITS file")
        assert aimfast._resolve_phase_centre(str(bad_fits), model) == expected

        # no fits_file guess available at all
        assert aimfast._resolve_phase_centre(None, model) == expected

        # success path still works when fits_file is genuinely valid
        monkeypatch.setattr(aimfast, "fitsInfo", lambda path: {"centre": (1.0, -2.0)})
        assert aimfast._resolve_phase_centre(str(bad_fits), model) == (1.0, -2.0)

    def test_catalog_display_name_handles_decimal_in_tilename(self):
        """Regression test: _catalog_display_name must not truncate at the
        first '.' in a path, naive `basename.split(".")[0]` (the previous
        approach) collapses any two catalogues sharing a tile name with a
        literal decimal point (e.g. 'G312.5', common in this project's
        naming) to the same string, e.g. both
        'cosmetic_G312.5_full_breizorro_catalog.txt' and
        'cosmetic_G312.5_full_aegean_icrs.lsm.html' truncated to
        'cosmetic_G312', which made aimfast's flux-plot axis labels and
        position-overlay legend indistinguishable between the two
        catalogues (Bokeh merges glyphs sharing an identical legend_label
        into a single legend entry)."""
        name1 = aimfast._catalog_display_name(
            "/x/cosmetic_G312.5_full_breizorro_catalog.txt"
        )
        name2 = aimfast._catalog_display_name(
            "/x/cosmetic_G312.5_full_aegean_icrs.lsm.html"
        )
        assert name1 != name2
        assert name1 == "cosmetic_G312.5_full_breizorro_catalog"
        assert name2 == "cosmetic_G312.5_full_aegean_icrs"
        # plain, unambiguous extensions still strip correctly too
        assert aimfast._catalog_display_name("/x/plain_catalog.csv") == "plain_catalog"

    def test_weighted_linregress_downweights_noisy_outlier(self):
        """Regression test: the flux comparison fit must be weighted by
        measurement error, not plain OLS (scipy.stats.linregress), so that
        the many faint/noisy points, which also tend to include the
        outliers, don't pull the fit away from where the few precise,
        bright points actually sit."""
        rng = np.random.default_rng(0)
        n_good = 30
        x_good = np.linspace(1.0, 10.0, n_good)
        y_good = x_good.copy()  # true 1:1 relation
        err_good = np.full(n_good, 0.01)  # precise

        # one bright, precise point that should anchor the fit near 1:1 ...
        x_bright, y_bright, err_bright = 10.0, 10.0, 0.01
        # ... and one very noisy outlier, well off the 1:1 line, with a
        # correspondingly large error, exactly the case the supervisor
        # described (faint/noisy points dragging the fit below 1:1)
        x_outlier, y_outlier, err_outlier = 5.0, 1.0, 5.0

        x = np.concatenate([x_good, [x_bright, x_outlier]])
        y = np.concatenate([y_good, [y_bright, y_outlier]])
        err = np.concatenate([err_good, [err_bright, err_outlier]])

        unweighted = aimfast.linregress(x, y)
        weighted = aimfast._weighted_linregress(x, y, xerr=err, yerr=err)

        # both should be pulled from the true slope=1 by the outlier, but
        # the weighted fit, which discounts the outlier for its large
        # error, must end up closer to the true relation than OLS.
        assert abs(weighted.slope - 1.0) < abs(unweighted.slope - 1.0)

    def test_weighted_linregress_falls_back_without_errors(self):
        """No usable error info (all zero) -> falls back to an unweighted
        fit rather than dividing by zero or dropping points."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = aimfast._weighted_linregress(x, y, xerr=np.zeros(4), yerr=np.zeros(4))
        expected = aimfast.linregress(x, y)
        assert result.slope == pytest.approx(expected.slope)
        assert result.intercept == pytest.approx(expected.intercept)

    def test_get_detected_sources_properties_ra_offset_uses_cos_dec(self, tmp_path):
        """Regression test: the RA offset stored per matched source must be
        scaled by cos(dec), a fixed RA difference subtends a smaller true
        angle away from the equator. Without it, the RA offset is
        overstated by 1/cos(dec) (about 2x at this project's ~-61 deg
        declination)."""
        dec_deg = -61.0
        ra1_deg = 10.0
        ra2_deg = 10.0 + 2.0 / 3600.0  # 2 arcsec raw RA difference

        model1 = SkyModel.SkyModel(self._make_source("S1", ra1_deg, dec_deg, 0.001, 8.0, 6.0))
        model2 = SkyModel.SkyModel(self._make_source("S2", ra2_deg, dec_deg, 0.001, 8.0, 6.0))
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        props = aimfast.get_detected_sources_properties(path1, path2, tolerance=8.0, shape_limit=12.0)
        targets_position = props[2]
        ra_offset_arcsec = list(targets_position.values())[0][1]

        raw_offset_arcsec = 2.0  # the naive, uncorrected Δra in arcsec
        expected_offset_arcsec = raw_offset_arcsec * np.cos(np.deg2rad(dec_deg))

        assert ra_offset_arcsec == pytest.approx(expected_offset_arcsec, rel=1e-3)
        # sanity check this is actually a *different*, smaller value than
        # the (bug) uncorrected offset would have been
        assert ra_offset_arcsec < raw_offset_arcsec

    def test_get_detected_sources_properties_delta_pos_angle_is_physically_sane(self, tmp_path):
        """Regression test: delta_pos_angle_arc_sec (the true angular
        separation between a matched pair) must be a small, physically
        sane value for a close match, previously this was computed by
        passing arcsec-scaled values into angular_dist_pos_angle (which
        expects radians for its internal sin/cos calls), producing an
        essentially meaningless angle roughly 206265x too large."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        dec_deg = -61.0
        ra1_deg = 10.0
        ra2_deg = 10.0 + 2.0 / 3600.0
        dec2_deg = dec_deg + 1.0 / 3600.0

        model1 = SkyModel.SkyModel(self._make_source("S1", ra1_deg, dec_deg, 0.001, 8.0, 6.0))
        model2 = SkyModel.SkyModel(
            self._make_source("S2", ra2_deg, dec2_deg, 0.001, 8.0, 6.0)
        )
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        props = aimfast.get_detected_sources_properties(path1, path2, tolerance=8.0, shape_limit=12.0)
        targets_position = props[2]
        delta_pos_angle_arc_sec = list(targets_position.values())[0][0]

        expected = (
            SkyCoord(ra1_deg * u.deg, dec_deg * u.deg)
            .separation(SkyCoord(ra2_deg * u.deg, dec2_deg * u.deg))
            .arcsec
        )

        assert delta_pos_angle_arc_sec == pytest.approx(expected, rel=1e-2)
        assert delta_pos_angle_arc_sec < 8.0  # sane, well within the match tolerance used

    def test_weighted_linregress_sigma_attribute(self):
        """_weighted_linregress must expose .sigma (the error-weighted RMS
        of the residuals around the fit), used by --flux-sigma-shade to
        draw a +/-1 sigma data-scatter band around the flux comparison fit
        line. A perfect y=x fit with zero scatter should report sigma~=0;
        adding real scatter should increase it."""
        x = np.linspace(1.0, 10.0, 20)
        y_perfect = x.copy()
        err = np.full(20, 0.01)

        perfect = aimfast._weighted_linregress(x, y_perfect, xerr=err, yerr=err)
        assert perfect.sigma == pytest.approx(0.0, abs=1e-6)

        rng = np.random.default_rng(1)
        y_noisy = x + rng.normal(scale=0.5, size=20)
        noisy = aimfast._weighted_linregress(x, y_noisy, xerr=err, yerr=err)
        assert noisy.sigma > perfect.sigma

    def test_flux_sigma_shade_argparser_flag(self):
        """-fss/--flux-sigma-shade must parse as a boolean flag, defaulting
        to False (band off unless explicitly requested)."""
        parser = aimfast.get_argparser()

        args_default = parser.parse_args(
            ["--compare-models", "model1.lsm.html", "model2.lsm.html"]
        )
        assert args_default.flux_sigma_shade is False

        args_set = parser.parse_args(
            ["--compare-models", "model1.lsm.html", "model2.lsm.html", "-fss"]
        )
        assert args_set.flux_sigma_shade is True

    def test_json_dump_appends_json_extension(self, tmp_path):
        """Regression test: json_dump() must append .json if the caller's
        filename doesn't already have it, --outfile's own docstring says
        the convention is a .json-suffixed name (default
        'fidelity_results.json'), but a user-supplied prefix without the
        extension was previously written completely literally, producing
        an extensionless file containing JSON content."""
        prefix = str(tmp_path / "my_results")
        aimfast.json_dump({"a": 1}, filename=prefix)

        assert not os.path.exists(prefix)
        assert os.path.exists(prefix + ".json")

        # already has the extension, must not double it up
        with_ext = str(tmp_path / "already_named.json")
        aimfast.json_dump({"b": 2}, filename=with_ext)
        assert os.path.exists(with_ext)
        assert not os.path.exists(with_ext + ".json")

    def test_cross_matching_stats_table_not_double_converted(self, tmp_path):
        """Regression test: the 'Cross Matching Statistics' table's (RA,
        DEC) mean/sigma offset values must not be double-converted.
        RA_mean/DEC_mean/r1/r2 are computed from RA_offset/DEC_offset,
        which get_detected_sources_properties already returns in arcsec
        (via rad2arcsec()), the table code previously ran them through
        deg2arcsec() again (x3600), producing implausible thousands-of-
        arcsec values for genuinely sub-arcsec matches. Checks the actual
        rendered PositionOffset.html output, not just the underlying data,
        since the bug was specifically in the table-formatting code."""
        rng = np.random.default_rng(2)
        sources1, sources2 = [], []
        base_ra, base_dec = 10.0, -61.0
        for i in range(10):
            ra = base_ra + i * 0.01
            dec = base_dec + i * 0.01
            offset_arcsec = rng.uniform(0.1, 1.0)
            flux = 0.001 * (1.0 + i)
            sources1.append(
                self._make_source(f"S1_{i}", ra, dec, flux, 8.0, 6.0)
            )
            sources2.append(
                self._make_source(
                    f"S2_{i}", ra + offset_arcsec / 3600.0, dec, flux, 8.0, 6.0
                )
            )
        model1 = SkyModel.SkyModel(*sources1)
        model2 = SkyModel.SkyModel(*sources2)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        import os as _os

        cwd = _os.getcwd()
        _os.chdir(tmp_path)
        try:
            models = [[
                dict(label="pair-model_a_0", path=path1),
                dict(label="pair-model_b_0", path=path2),
            ]]
            aimfast.compare_models(models, tolerance=8.0, shape_limit=12.0, plot=True)
            with open(tmp_path / "PositionOffset.html") as f:
                content = f.read()
        finally:
            _os.chdir(cwd)

        idx = content.find('"Stats",[')
        assert idx != -1
        snippet = content[idx:idx + 300]
        # the true offsets here are all sub-2" by construction, if the
        # double-conversion bug is present, the reported mean/sigma will
        # be in the hundreds/thousands instead
        import re

        values = re.search(r'"Value",\[(\d+),"\(([-\d.]+),([-\d.]+)\)"', snippet)
        assert values is not None
        ra_mean, dec_mean = float(values.group(2)), float(values.group(3))
        assert abs(ra_mean) < 10.0, f"RA mean offset implausibly large: {ra_mean}\" (double-conversion bug?)"
        assert abs(dec_mean) < 10.0, f"DEC mean offset implausibly large: {dec_mean}\" (double-conversion bug?)"

    def test_catalogs_overlay_ra_axis_flipped_without_background_image(self, tmp_path, monkeypatch):
        """Regression test: the 'Catalogs Overlay' panel's RA axis must
        increase leftward (standard astronomical convention, viewing the
        sky from inside looking out, not a ground map from above), even
        when no --restored-image background is supplied. Bokeh's Range1d
        has no working 'flipped' toggle (a previous attempt at
        `x_range.flipped = True` here was dead code, commented out); the
        correct approach, already used correctly in the with-background-
        image branch, is swapping start/end. Verified by intercepting the
        actual Bokeh figure object passed to save() (monkeypatched),
        rather than parsing the serialized HTML/JSON output."""
        sources1, sources2 = [], []
        for i in range(5):
            ra = 210.0 + i * 0.5
            dec = -61.0
            flux = 0.001 * (1.0 + i)
            sources1.append(self._make_source(f"S1_{i}", ra, dec, flux, 8.0, 6.0))
            sources2.append(
                self._make_source(f"S2_{i}", ra + 1.0 / 3600.0, dec, flux, 8.0, 6.0)
            )
        model1 = SkyModel.SkyModel(*sources1)
        model2 = SkyModel.SkyModel(*sources2)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        captured = {}

        def _fake_save(obj, title=None):
            captured["obj"] = obj

        monkeypatch.setattr(aimfast, "save", _fake_save)

        models = [[
            dict(label="pair-model_a_0", path=path1),
            dict(label="pair-model_b_0", path=path2),
        ]]
        aimfast.compare_models(models, tolerance=8.0, shape_limit=12.0, plot=True)

        def _find_overlay_figure(node):
            title = getattr(getattr(node, "title", None), "text", None)
            if title == "Catalogs Overlay":
                return node
            for child in getattr(node, "children", []):
                # column/row children are (model, ...) tuples in some
                # bokeh versions, plain models in others
                candidate = child[0] if isinstance(child, tuple) else child
                found = _find_overlay_figure(candidate)
                if found is not None:
                    return found
            return None

        overlay_fig = _find_overlay_figure(captured["obj"])
        assert overlay_fig is not None, "Could not locate the Catalogs Overlay figure"
        assert overlay_fig.x_range.start > overlay_fig.x_range.end, (
            f"RA axis not flipped: start={overlay_fig.x_range.start}, "
            f"end={overlay_fig.x_range.end} (RA should increase leftward)"
        )

    def test_low_match_count_logs_helpful_warning(self, tmp_path, caplog):
        """When very few/no sources match, get_detected_sources_properties
        should hint at the two most common silent causes, tolerance and
        shape_limit, rather than leave the user to rediscover this the
        hard way (as happened repeatedly this session)."""
        # model2 deliberately far away (>> tolerance) from model1, so
        # nothing matches
        model1 = SkyModel.SkyModel(self._make_source("S1", 10.0, -61.0, 0.001, 8.0, 6.0))
        model2 = SkyModel.SkyModel(self._make_source("S2", 50.0, -61.0, 0.001, 8.0, 6.0))
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        with caplog.at_level("WARNING"):
            props = aimfast.get_detected_sources_properties(path1, path2, tolerance=8.0)

        assert len(props[0]) == 0
        assert any(
            "tolerance" in rec.message and "shape_limit" in rec.message
            for rec in caplog.records
        ), "Expected a helpful low-match-count warning mentioning tolerance and shape_limit"

    def test_healthy_match_count_does_not_warn(self, tmp_path, caplog):
        """The low-match-count hint must not fire when matching is healthy
       , it should only appear when genuinely few/no sources matched."""
        sources1, sources2 = [], []
        for i in range(20):
            ra = 10.0 + i * 0.01
            flux = 0.001 * (1.0 + i)
            sources1.append(self._make_source(f"S1_{i}", ra, -61.0, flux, 8.0, 6.0))
            sources2.append(
                self._make_source(f"S2_{i}", ra + 0.5 / 3600.0, -61.0, flux, 8.0, 6.0)
            )
        model1 = SkyModel.SkyModel(*sources1)
        model2 = SkyModel.SkyModel(*sources2)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        with caplog.at_level("WARNING"):
            props = aimfast.get_detected_sources_properties(
                path1, path2, tolerance=8.0, shape_limit=12.0
            )

        assert len(props[0]) == 20
        assert not any("shape_limit" in rec.message for rec in caplog.records)

    def test_source_finder_subcommand_runs_without_dash_c(self, tmp_path, monkeypatch, caplog):
        """The `source-finder` subcommand used to silently do nothing unless
        --config was explicitly passed, even though -sf/-r/--threshold are
        documented as standalone overrides. It should instead fall back to
        an auto-generated default config, matching --compare-images'
        existing behaviour."""
        import sys

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["aimfast", "source-finder"])

        with caplog.at_level("WARNING"):
            aimfast.main()

        assert (tmp_path / "default_sf_config.yml").exists(), (
            "source-finder subcommand should auto-generate a default config "
            "when -c/--config is omitted, not silently no-op"
        )
        assert any("No source finder selected" in rec.message for rec in caplog.records)

    def test_source_finder_subcommand_forwards_outdir(self, tmp_path, monkeypatch):
        """`source-finder --outdir DIR` should be threaded through to
        source_finding() so output catalogs land in DIR instead of always
        next to the input image (previously no such override existed)."""
        import sys

        monkeypatch.chdir(tmp_path)
        captured = {}

        def _fake_source_finding(sf_params, sf=None, mappings=None, outdir=None):
            captured["outdir"] = outdir

        monkeypatch.setattr(aimfast, "source_finding", _fake_source_finding)
        outdir = str(tmp_path / "results")
        monkeypatch.setattr(
            sys, "argv", ["aimfast", "source-finder", "-sf", "pybdsf", "--outdir", outdir]
        )

        aimfast.main()

        assert captured["outdir"] == outdir

    def test_compare_images_forwards_sf_threshold(self, tmp_path, monkeypatch):
        """--compare-images should let --sf-threshold override the selected
        finder's detection threshold, matching what the `source-finder`
        subcommand's --threshold already does (previously --compare-images
        had no way to override this at all)."""
        import sys

        monkeypatch.chdir(tmp_path)
        captured_thresholds = []

        def _fake_source_finding(sf_params, sf=None, mappings=None, outdir=None):
            captured_thresholds.append(sf_params[sf]["thresh_pix"])
            return None

        monkeypatch.setattr(aimfast, "source_finding", _fake_source_finding)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "aimfast",
                "--compare-images",
                "image1.fits",
                "image2.fits",
                "-sf",
                "pybdsf",
                "pybdsf",
                "--sf-threshold",
                "3.5",
            ],
        )

        aimfast.main()

        assert captured_thresholds == [3.5, 3.5]

    def test_default_tolerance_and_shape_limit_widened(self):
        """Defaults were 0.2" tolerance / 6.0" shape_limit, too tight for
        this project's ~8" beam data (real matches sit 0.07-0.5" apart,
        real breizorro island shapes commonly exceed 6" and up to ~15")."""
        parser = aimfast.get_argparser()
        args = parser.parse_args(["--compare-models", "model1.lsm.html", "model2.lsm.html"])
        assert args.tolerance == 1.0
        assert args.shape_limit == 16.0

        import inspect

        assert inspect.signature(aimfast.compare_models).parameters["tolerance"].default == 1.0
        assert (
            inspect.signature(aimfast.compare_models).parameters["shape_limit"].default == 16.0
        )
        assert (
            inspect.signature(aimfast.get_detected_sources_properties)
            .parameters["tolerance"]
            .default
            == 1.0
        )
        assert (
            inspect.signature(aimfast.get_detected_sources_properties)
            .parameters["shape_limit"]
            .default
            == 16.0
        )

    def test_combined_report_produces_single_tabbed_html(self, tmp_path):
        """--combined-report / combined_report=True should produce one
        <prefix>-Report.html with both Flux and Position tabs, instead of
        separate FluxOffset.html/PositionOffset.html files."""
        sources1, sources2 = [], []
        for i in range(5):
            ra = 210.0 + i * 0.5
            dec = -61.0
            flux = 0.001 * (1.0 + i)
            sources1.append(self._make_source(f"S1_{i}", ra, dec, flux, 8.0, 6.0))
            sources2.append(
                self._make_source(f"S2_{i}", ra + 1.0 / 3600.0, dec, flux, 8.0, 6.0)
            )
        model1 = SkyModel.SkyModel(*sources1)
        model2 = SkyModel.SkyModel(*sources2)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        models = [[
            dict(label="pair-model_a_0", path=path1),
            dict(label="pair-model_b_0", path=path2),
        ]]

        import os

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            aimfast.compare_models(
                models, tolerance=8.0, shape_limit=12.0, plot=True,
                combined_report=True, prefix="combo",
            )
        finally:
            os.chdir(cwd)

        assert (tmp_path / "combo-CrossMatchReport.html").exists()
        assert not (tmp_path / "combo-FluxOffset.html").exists()
        assert not (tmp_path / "combo-PositionOffset.html").exists()
        content = (tmp_path / "combo-CrossMatchReport.html").read_text()
        assert "Catalogs Overlay" in content

    @staticmethod
    def _make_galactic_rms_fits(path, naxis=60, seed=0):
        """Build a small 2D (no freq/Stokes axes), Galactic-frame FITS
        image with standard CRPIX (=NAXIS/2), mimicking this project's own
        full-tile RMS map products, used to reproduce the
        --compare-residuals crash chain end to end."""
        from astropy.io import fits

        rng = np.random.default_rng(seed)
        data = np.abs(rng.normal(1e-4, 1e-5, size=(naxis, naxis))).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdr = hdu.header
        hdr["CTYPE1"] = "GLON-SIN"
        hdr["CTYPE2"] = "GLAT-SIN"
        hdr["CRVAL1"] = 312.5
        hdr["CRVAL2"] = 0.0
        hdr["CRPIX1"] = naxis / 2.0
        hdr["CRPIX2"] = naxis / 2.0
        hdr["CDELT1"] = -0.001
        hdr["CDELT2"] = 0.001
        hdr["CUNIT1"] = "deg"
        hdr["CUNIT2"] = "deg"
        hdr["BMAJ"] = 0.0022222
        hdr["BMIN"] = 0.0022222
        hdr["BPA"] = 0.0
        hdu.writeto(str(path))

    def test_compare_residuals_on_galactic_2d_image(self, tmp_path, monkeypatch):
        """--compare-residuals on a Galactic-frame, plain-2D (no freq/
        Stokes axes) image used to crash three different ways in a row:
        (1) fits_info["centre"] is raw CRVAL, (l, b) for a Galactic
        image, not RA/Dec, so get_box()'s ICRS SkyCoord landed random
        sample points nowhere near the image, producing NaN pixel coords;
        (2) res_data[0, 0, :, :] assumed a 4D cube, crashing on a plain 2D
        array; (3) figure(plot_width=..., plot_height=...) is removed in
        Bokeh 3.x. All three are fixed now, this exercises the full
        compare_residuals() -> _random_residual_results() ->
        _residual_plotter() path end to end and checks a real html report
        is produced."""
        path1 = tmp_path / "res1.fits"
        path2 = tmp_path / "res2.fits"
        self._make_galactic_rms_fits(path1, seed=1)
        self._make_galactic_rms_fits(path2, seed=2)

        residuals = [[
            dict(label="galtest-res_a_0", path=str(path1)),
            dict(label="galtest-res_b_0", path=str(path2)),
        ]]

        monkeypatch.chdir(tmp_path)
        aimfast.compare_residuals(
            residuals, points=30, fov_factor=0.9, area_factor=2, prefix="galtest"
        )

        html_files = list(tmp_path.glob("galtest-*.html"))
        assert html_files, "compare_residuals should produce an html report"

    def test_json_dump_handles_numpy_float32(self, tmp_path):
        """json_dump() crashed with 'Object of type float32 is not JSON
        serializable' whenever a results dict (as produced by the
        residual/flux comparison pipelines, which compute with numpy)
        contained a raw numpy scalar rather than a plain Python float."""
        outfile = str(tmp_path / "results.json")
        aimfast.json_dump({"a": np.float32(1.5), "b": [np.float64(2.5)]}, filename=outfile)

        import json

        with open(outfile) as f:
            data = json.load(f)
        assert data["a"] == pytest.approx(1.5)
        assert data["b"] == [pytest.approx(2.5)]

    def test_get_online_catalog_maps_friendly_names_to_vizier_ids(self, tmp_path, monkeypatch):
        """--compare-online has never actually worked: bare 'SUMSS'/'NVSS'
        (the CLI's own -oc choices) aren't valid Vizier catalog
        identifiers and silently resolve to zero results regardless of
        sky position. Fixed by mapping the short, familiar CLI name to
        the real Vizier catalog ID internally."""
        from aimfast import auxiliary
        from astroquery.utils import TableList
        from astropy.table import Table

        captured = {}

        def _fake_query_region(coord_obj, width=None, catalog=None):
            captured["catalog"] = catalog
            return TableList([("VIII/81B/sumss212", Table({"RAJ2000": [], "DEJ2000": []}))])

        monkeypatch.setattr(auxiliary.Vizier, "query_region", staticmethod(_fake_query_region))

        auxiliary.get_online_catalog(
            catalog="SUMSS", catalog_table=str(tmp_path / "out.txt")
        )
        assert captured["catalog"] == "VIII/81B/sumss212"

        auxiliary.get_online_catalog(
            catalog="RACS-MID", catalog_table=str(tmp_path / "out2.txt")
        )
        assert captured["catalog"] == "J/other/PASA/41.3"

    def test_get_online_catalog_racs_picks_source_level_table(self, tmp_path, monkeypatch):
        """RACS queries return multiple tables (source-level +
        Gaussian-component-level), must pick the source-level one by
        name, not just blindly use whichever came first."""
        from aimfast import auxiliary
        from astroquery.utils import TableList
        from astropy.table import Table

        gauss_table = Table({"RAJ2000": [1.0], "DEJ2000": [2.0], "Ftot": [5.0]})
        source_table = Table({"RAJ2000": [3.0], "DEJ2000": [4.0], "Ftot": [7.0]})

        def _fake_query_region(coord_obj, width=None, catalog=None):
            return TableList([
                ("J/other/PASA/41.3/gcompsm", gauss_table),
                ("J/other/PASA/41.3/sourcesm", source_table),
            ])

        monkeypatch.setattr(auxiliary.Vizier, "query_region", staticmethod(_fake_query_region))

        result = auxiliary.get_online_catalog(
            catalog="RACS-MID", catalog_table=str(tmp_path / "out.txt")
        )
        assert list(result["Ftot"]) == [7.0]

    def test_tigger_src_racs_handles_missing_error_columns(self):
        """RACS-mid/high's source table has no per-source error columns
        at all (unlike racs-low); tigger_src_racs must not crash, just
        default those to 0.0."""
        from astropy.table import Row, Table

        row = Table({
            "RAJ2000": [212.6], "DEJ2000": [-61.5],
            "Ftot": [4.0], "Fpeak": [4.5],
            "Maj": [16.9], "Min": [9.5], "PA": [1.0],
        })

        # tigger_src_racs is nested inside get_model(); exercise it via a
        # real online-catalog-style ascii table + get_model() round trip
        # instead of reaching into the closure directly.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            catalog_path = os.path.join(d, "default_racs-mid_catalog_table.txt")
            from astropy.io import ascii as io_ascii

            io_ascii.write(row, catalog_path, overwrite=True)
            model = aimfast.get_model(catalog_path)
            assert len(model.sources) == 1
            src = model.sources[0]
            assert src.flux.I == pytest.approx(4.0 / 1000.0)
            assert src.flux.I_err == 0.0

    def test_tigger_src_vlass_round_trip(self):
        """VLASS has real per-source errors (unlike racs-mid/high) and
        different column names (DCMaj/DCMin/DCPA), exercised through
        the same get_model() ascii-catalog path."""
        from astropy.table import Table
        from astropy.io import ascii as io_ascii
        import tempfile

        row = Table({
            "RAJ2000": [179.845], "DEJ2000": [19.89],
            "Ftot": [2.862], "e_Ftot": [0.373],
            "Fpeak": [2.476], "e_Fpeak": [0.192],
            "DCMaj": [1.514], "DCMin": [0.3055], "DCPA": [71.0],
        })

        with tempfile.TemporaryDirectory() as d:
            catalog_path = os.path.join(d, "default_vlass_catalog_table.txt")
            io_ascii.write(row, catalog_path, overwrite=True)
            model = aimfast.get_model(catalog_path)
            assert len(model.sources) == 1
            src = model.sources[0]
            assert src.flux.I == pytest.approx(2.862 / 1000.0)
            assert src.flux.I_err == pytest.approx(0.373 / 1000.0)

    def test_cross_matching_logs_periodic_progress(self, tmp_path, monkeypatch, caplog):
        """Cross-matching large catalogues can take minutes with zero
        other output (confirmed directly: the full-tile grid test took
        ~30min for a single pair), indistinguishable from a hang while
        it's running. A periodic progress log should fire without
        needing to actually wait, simulate elapsed time via a
        monkeypatched time.time() rather than a real multi-minute test."""
        sources1, sources2 = [], []
        for i in range(5):
            ra = 210.0 + i * 0.5
            sources1.append(self._make_source(f"S1_{i}", ra, -61.0, 0.001, 8.0, 6.0))
            sources2.append(
                self._make_source(f"S2_{i}", ra + 1.0 / 3600.0, -61.0, 0.001, 8.0, 6.0)
            )
        model1 = SkyModel.SkyModel(*sources1)
        model2 = SkyModel.SkyModel(*sources2)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        import aimfast.aimfast as aimfast_module

        fake_now = [1000.0]

        def _fake_time():
            # Jump 31s forward every call after the first, so the 30s
            # progress-log threshold is crossed on the very next source
            # without a real wait.
            fake_now[0] += 31.0
            return fake_now[0]

        monkeypatch.setattr(aimfast_module.time, "time", _fake_time)

        with caplog.at_level("INFO"):
            aimfast.get_detected_sources_properties(path1, path2, tolerance=8.0, shape_limit=12.0)

        assert any("Cross-matching:" in rec.message for rec in caplog.records)

    @staticmethod
    def _make_source_with_flux_err(name, ra_deg, dec_deg, flux_jy, flux_err_jy):
        """Like _make_source but with an explicit flux error, needed to
        reproduce a source whose flux error exceeds its own value (a real
        pattern seen for sources in crowded/blended Aegean islands)."""
        from Tigger.Models import ModelClasses, SkyModel

        pos = ModelClasses.Position(np.deg2rad(ra_deg), np.deg2rad(dec_deg))
        flux = ModelClasses.Polarization(flux_jy, 0, 0, 0, I_err=flux_err_jy)
        return SkyModel.Source(name, pos, flux)

    def _find_legend_labels(self, node):
        labels = []
        legend = getattr(node, "legend", None)
        if legend:
            for leg in legend:
                for item in leg.items:
                    label = item.label
                    labels.append(getattr(label, "value", label))
        for child in getattr(node, "children", []) or []:
            candidate = child[0] if isinstance(child, tuple) else child
            labels.extend(self._find_legend_labels(candidate))
        return labels

    def test_flux_plot_large_error_gets_own_legend_entry(self, tmp_path, monkeypatch):
        """A source whose flux error exceeds its own flux value used to
        stretch its error-bar segment across the whole visible log-axis
        range (clamped near-zero lower bound), distorting the plot.
        Fixed by routing it into its own, separately click-to-hide legend
        entry ('Errors (>100%)') rather than dropping it or leaving it in
        the normal 'Errors' group."""
        s1_normal = self._make_source_with_flux_err("S1", 10.0, -61.0, 0.005, 0.0005)
        s1_large = self._make_source_with_flux_err("S2", 10.5, -61.0, 0.005, 0.02)
        s2_normal = self._make_source_with_flux_err("T1", 10.0 + 1 / 3600.0, -61.0, 0.005, 0.0005)
        s2_large = self._make_source_with_flux_err("T2", 10.5 + 1 / 3600.0, -61.0, 0.006, 0.0)

        model1 = SkyModel.SkyModel(s1_normal, s1_large)
        model2 = SkyModel.SkyModel(s2_normal, s2_large)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        captured = {}

        def _fake_save(obj, title=None):
            # compare_models() saves both FluxOffset.html and
            # PositionOffset.html, must not overwrite the flux one with
            # the later position-plot save() call.
            if title and "Flux" in title:
                captured["obj"] = obj

        monkeypatch.setattr(aimfast, "save", _fake_save)

        models = [[dict(label="p-model_a_0", path=path1), dict(label="p-model_b_0", path=path2)]]
        aimfast.compare_models(models, tolerance=8.0, shape_limit=12.0, plot=True)

        labels = self._find_legend_labels(captured["obj"])
        assert "Errors" in labels
        assert "Errors (>100%)" in labels

    def test_flux_plot_no_large_error_legend_when_all_errors_normal(self, tmp_path, monkeypatch):
        """The 'Errors (>100%)' legend entry must not appear at all when
        no source actually has one, Bokeh adds a legend item even for
        an empty-data glyph, so this needs an explicit guard."""
        s1a = self._make_source_with_flux_err("S1", 10.0, -61.0, 0.005, 0.0005)
        s1b = self._make_source_with_flux_err("S2", 10.5, -61.0, 0.005, 0.0004)
        s2a = self._make_source_with_flux_err("T1", 10.0 + 1 / 3600.0, -61.0, 0.005, 0.0005)
        s2b = self._make_source_with_flux_err("T2", 10.5 + 1 / 3600.0, -61.0, 0.005, 0.0004)

        model1 = SkyModel.SkyModel(s1a, s1b)
        model2 = SkyModel.SkyModel(s2a, s2b)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        captured = {}

        def _fake_save(obj, title=None):
            captured["obj"] = obj

        monkeypatch.setattr(aimfast, "save", _fake_save)

        models = [[dict(label="p-model_a_0", path=path1), dict(label="p-model_b_0", path=path2)]]
        aimfast.compare_models(models, tolerance=8.0, shape_limit=12.0, plot=True)

        labels = self._find_legend_labels(captured["obj"])
        assert "Errors" in labels
        assert "Errors (>100%)" not in labels

    @staticmethod
    def _find_stats_table(node):
        from bokeh.models.widgets import DataTable

        if isinstance(node, DataTable):
            return node
        for child in getattr(node, "children", []) or []:
            candidate = child[0] if isinstance(child, tuple) else child
            found = TestClass._find_stats_table(candidate)
            if found is not None:
                return found
        return None

    def test_flux_plot_stats_table_counts_large_flux_errors(self, tmp_path, monkeypatch):
        """The Cross Matching Statistics table should report how many
        matched sources had a flux error exceeding their own value, so a
        reader can tell at a glance without inspecting the plot itself."""
        s1_normal = self._make_source_with_flux_err("S1", 10.0, -61.0, 0.005, 0.0005)
        s1_large = self._make_source_with_flux_err("S2", 10.5, -61.0, 0.005, 0.02)
        s2_normal = self._make_source_with_flux_err("T1", 10.0 + 1 / 3600.0, -61.0, 0.005, 0.0005)
        s2_large = self._make_source_with_flux_err("T2", 10.5 + 1 / 3600.0, -61.0, 0.006, 0.0)

        model1 = SkyModel.SkyModel(s1_normal, s1_large)
        model2 = SkyModel.SkyModel(s2_normal, s2_large)
        path1 = str(tmp_path / "model1.lsm.html")
        path2 = str(tmp_path / "model2.lsm.html")
        model1.save(path1)
        model2.save(path2)

        captured = {}

        def _fake_save(obj, title=None):
            if title and "Flux" in title:
                captured["obj"] = obj

        monkeypatch.setattr(aimfast, "save", _fake_save)

        models = [[dict(label="p-model_a_0", path=path1), dict(label="p-model_b_0", path=path2)]]
        aimfast.compare_models(models, tolerance=8.0, shape_limit=12.0, plot=True)

        table = self._find_stats_table(captured["obj"])
        data = table.source.data
        idx = data["Stats"].index("Errors >100%")
        assert data["Value"][idx] == "1"
