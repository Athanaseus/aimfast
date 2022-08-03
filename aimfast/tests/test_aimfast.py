import pytest
import numpy as np
from aimfast import aimfast


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
        input_value = 'aimfast/tests/files/cube1.fits'
        output_value = aimfast.noise_sigma(input_value)
        expected_value = 3.1e-05
        assert expected_value == pytest.approx(output_value, 0.01855)

    def test_fitsInfo(self):
        """Test fitsInfo method"""
        input_value = 'aimfast/tests/files/cube1.fits'
        output_value = aimfast.fitsInfo(input_value)
        from astropy.wcs import WCS
        expected = {'b_size': (0.00154309340472658,
                               0.00136912246542523,
                               159.801295045408),
                    'centre': (0.0, -30.0),
                    'ddec': 0.000277777777777778,
                    'dec': -30.0,
                    'decPix': 10,
                    'dra': 0.000277777777777778,
                    'numPix': 20,
                    'ra': 0.0,
                    'raPix': 10,
                    'skyArea': 3.0864197530864246e-05,
                    'wcs': WCS}
        for param, value in expected.items():
            val = output_value[param]
            if param == 'wcs':
                assert isinstance(val, value)
            else:
                assert val == value

    def test_residual_stats(self):
        """Test the residuals stats method"""
        input_value = 'aimfast/tests/files/cube1.fits'
        input_mask = 'aimfast/tests/files/mask.fits'

        def test(expected, output_value, normality=False):
            if normality:
                output_normaltest_value = output_value.pop('NORM')
                expected_normaltest_value = expected_value.pop('NORM')
                assert expected_normaltest_value == pytest.approx(
                    output_normaltest_value, 1.0e-4)
            assert expected_value == output_value

        # Test residual stats
        output_value = aimfast.residual_image_stats(
            input_value, test_normality='normaltest')
        expected_value = {'NORM': (10.276033206715848, 0.005869319364736688),
                          'SKEW': 0.186153,
                          'KURT': 2.870047,
                          'RMS': 3.1e-05,
                          'MAD': 2.2e-05,
                          'MIN': -8.35573e-05,
                          'MAX': 9.98428e-05,
                          'SUM_NEG': -0.01914,
                          'STDDev': 3.1e-05,
                          'MEAN': 1.21497e-06}
        test(expected_value, output_value, normality=True)

        # Test using mask
        output_value = aimfast.residual_image_stats(
            input_value,
            mask=input_mask)
        expected_value = {'SKEW': -0.341298,
                          'KURT': 2.422672,
                          'RMS': 2.6e-05,
                          'MAD': 2.2e-05,
                          'MIN': -8.35573e-05,
                          'MAX': 3.98551e-05,
                          'SUM_NEG': -0.01914,
                          'STDDev': 2.5e-05,
                          'MEAN': -5.57698e-06}
        test(expected_value, output_value)

        # Test using channels
        output_value = aimfast.residual_image_stats(
            input_value,
            chans='2~3')
        expected_value = {'SKEW': 0.287936,
                          'KURT': 2.891433,
                          'RMS': 3.2e-05,
                          'MAD': 2.2e-05,
                          'MIN': -8.35573e-05,
                          'MAX': 9.98428e-05,
                          'SUM_NEG': -0.010724,
                          'STDDev': 3.2e-05,
                          'MEAN': -9.18495e-07}
        test(expected_value, output_value)

        # Test using threshold
        output_value = aimfast.residual_image_stats(
            input_value,
            threshold=0.00005)
        expected_value = {'SKEW': 0.186153,
                          'KURT': 2.870047,
                          'RMS': 3.1e-05,
                          'MAD': 2.2e-05,
                          'MIN': -8.35573e-05,
                          'MAX': 9.98428e-05,
                          'SUM_NEG': -0.01914,
                          'STDDev': 3.1e-05,
                          'MEAN': 1.21497e-06}
        test(expected_value, output_value)

    def test_get_detected_sources_properties(self):
        """Test get detected sources properties"""
        expected_label = 'None-model_a_0'
        label = None
        input_dir = 'aimfast/tests/files'
        model1 = 'catalog1.txt'
        model2 = 'catalog1.lsm.html'
        model1_path = '{:s}/{:s}'.format(input_dir, model1)
        model2_path = '{:s}/{:s}'.format(input_dir, model2)
        models = [[dict(label="{}-model_a_0".format(label), path=model1_path),
                   dict(label="{}-model_b_0".format(label), path=model2_path)]]
        expected = aimfast.get_aimfast_data('fidelity_results.json', input_dir)
        output = aimfast.compare_models(models, tolerance=0.2, plot=False,
                                        all_sources=True)
        models = expected[expected_label]['models']
        assert models == [model1, model2]
        assert(len(expected[expected_label]['flux'])
               == len(output[expected_label]['flux']))
        assert(len(expected[expected_label]['position'])
               == len(output[expected_label]['position']))

    def test_random_residual_results(self):
        """Test comparison of random residuals in images"""
        expected_label = 'random-res_a_0'
        label = 'random'
        input_dir = 'aimfast/tests/files'
        res1 = 'cube1.fits'
        res2 = 'cube2.fits'
        res1_path = '{:s}/{:s}'.format(input_dir, res1)
        res2_path = '{:s}/{:s}'.format(input_dir, res2)
        res_imgs = [[dict(label="{}-res_a_0".format(label), path=res1_path),
                    dict(label="{}-res_b_0".format(label), path=res2_path)]]
        expected = aimfast.get_aimfast_data('fidelity_results.json', input_dir)
        output = aimfast._random_residual_results(res_imgs, data_points=10,
                                                  area_factor=0.5, fov_factor=0.5)
        assert len(expected[expected_label]) == len(output[expected_label])

    def test_source_residual_results(self):
        """Test comparison of source residuals in images"""
        default_area_factor = 2.0
        expected_label = 'source-res_a_0'
        label = 'source'
        input_dir = 'aimfast/tests/files'
        res1 = 'cube1.fits'
        res2 = 'cube2.fits'
        skymodel = 'catalog.lsm.html'
        skymodel_path = '{:s}/{:s}'.format(input_dir, skymodel)
        res1_path = '{:s}/{:s}'.format(input_dir, res1)
        res2_path = '{:s}/{:s}'.format(input_dir, res2)
        res_imgs = [[dict(label="{}-res_a_0".format(label), path=res1_path),
                     dict(label="{}-res_b_0".format(label), path=res2_path)]]
        expected = aimfast.get_aimfast_data('fidelity_results.json', input_dir)
        output = aimfast._source_residual_results(res_imgs, skymodel_path,
                                                  area_factor=default_area_factor)
        assert len(expected[expected_label]) == len(output[expected_label])

    def test_model_dynamic_range(self):
        """Test dynamic range from model"""
        input_model = 'DR_catalog.txt'
        residual = 'cube1.fits'
        input_dir = 'aimfast/tests/files'
        model_path = '{:s}/{:s}'.format(input_dir, input_model)
        res_path = '{:s}/{:s}'.format(input_dir, residual)
        output_value = aimfast.model_dynamic_range(model_path, res_path,
                                                   area_factor=1)
        expected_value = {"deepest_negative"  : 130.40011268142158,
                          "local_rms"         : 224.29535518140233,
                          "global_rms"        : 174.74979385790667}
        assert expected_value == output_value

    def test_image_dynamic_range(self):
        """Test dynamic range from image"""
        restored_image = 'cube1.fits'
        residual_image = 'cube2.fits'
        input_dir = 'aimfast/tests/files'
        restored_image_path = '{:s}/{:s}'.format(input_dir, restored_image)
        residual_image_path = '{:s}/{:s}'.format(input_dir, residual_image)
        output_value = aimfast.image_dynamic_range(restored_image_path,
                                                   residual_image_path,
                                                   area_factor=1)
        expected_value = {"deepest_negative"  : 1.4872031158104637,
                          "local_rms"         : 3.098743200302124,
                          "global_rms"        : 3.394456386566162}
        assert expected_value == output_value
