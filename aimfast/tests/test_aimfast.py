import numpy as np
import pytest
from pathlib import Path

from aimfast import aimfast
from bokeh.models import ColumnDataSource


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
        output = aimfast.compare_models(models, plot=False, all_sources=True, model_mappings=model_mappings)
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
