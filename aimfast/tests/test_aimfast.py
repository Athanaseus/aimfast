import pytest
import numpy as np
from aimfast import aimfast


class TestClass(object):
    """aimfast Test klass"""

    def test_deg2arcsec(self):
        """Test deg2arcsec method"""
        input_value = 10.5
        output_value = aimfast.deg2arcsec(input_value)
        expected_value = input_value*3600.00
        assert expected_value == output_value

    def test_rad2deg(self):
        """Test rad2deg method"""
        input_value = 1.5
        output_value = aimfast.rad2deg(input_value)
        expected_value = input_value*(180/np.pi)
        assert expected_value == output_value

    def test_rad2arcsec(self):
        """Test rad2arcsec method"""
        input_value = 1.5
        output_value = aimfast.rad2arcsec(input_value)
        expected_value = input_value*(3600.0*180.0/np.pi)
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
        expected_value = {"NORM": (10.276013230775483, 0.005869377987616255),
                          "SKEW": 0.186153,
                          "KURT": 2.870047,
                          "STDDev": 3.1e-05,
                          "MEAN": 1.215e-06}
        assert expected_value == output_value
