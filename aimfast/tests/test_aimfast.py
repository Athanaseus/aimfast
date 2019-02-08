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
        input_value = 'aimfast/tests/files/cube.fits'
        output_value = aimfast.noise_sigma(input_value)
        expected_value = 3.1e-05
        assert expected_value == pytest.approx(output_value, 0.01855)

    def test_fitsInfo(self):
        """Test fitsInfo method"""
        input_value = 'aimfast/tests/files/cube.fits'
        output_value = aimfast.fitsInfo(input_value)
        from astLib import astWCS
        expected = {'b_size': (0.00154309340472658,
                               0.00136912246542523,
                               159.801295045408),
                    'centre': 'J2000.0,0.0deg,-30.0deg',
                    'ddec': 0.000277777777777778,
                    'dec': -30.0,
                    'decPix': 10,
                    'dra': 0.000277777777777778,
                    'numPix': 20,
                    'ra': 0.0,
                    'raPix': 10,
                    'skyArea': 3.0864197530864246e-05,
                    'wcs': astWCS.WCS}
        for param, value in expected.items():
            val = output_value[param]
            if param is 'wcs':
                assert isinstance(val, value)
            else:
                assert val == value

    def test_residual_stats(self):
        """Test the residuals stats method"""
        input_value = 'aimfast/tests/files/cube.fits'
        output_value = aimfast.residual_image_stats(
            input_value, test_normality='normaltest')
        expected_value = {'NORM': (10.276033206715848, 0.005869319364736688),
                          'SKEW': 0.186153,
                          'KURT': 2.870047,
                          'STDDev': 3.1e-05,
                          'MEAN': 1.21497e-06}
        expected_normaltest_value = expected_value.pop('NORM')
        output_normaltest_value = output_value.pop('NORM')
        assert expected_value == output_value
        assert expected_normaltest_value == pytest.approx(
            output_normaltest_value, 1.0e-4)

    def test_get_detected_sources_properties(self):
        """Test get detected sources properties"""
        expected_label = 'None-model_a_'
        label = None
        input_dir = 'aimfast/tests/files'
        model1 = 'catalog.txt'
        model2 = 'catalog.lsm.html'
        model1_path = '{:s}/{:s}'.format(input_dir, model1)
        model2_path = '{:s}/{:s}'.format(input_dir, model2)
        models = [[dict(label="{}-model_a_".format(label), path=model1_path),
                   dict(label="{}-model_b_".format(label), path=model2_path)]]
        expected = aimfast.get_aimfast_data('fidelity_results.json', input_dir)
        output = aimfast.compare_models(models, tolerance=0.000001, plot=False)
        models = expected[expected_label]['models']
        # Remove the reisdual stats
        expected.pop('cube.fits')
        assert models == [model1, model2]
        assert(len(expected[expected_label]['flux']) ==
               len(output[expected_label]['flux']))
        assert(len(expected[expected_label]['position']) ==
               len(output[expected_label]['position']))
