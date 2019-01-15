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
        input_value = 'files/image.fits'
        output_value = aimfast.noise_sigma(input_value)
        expected_value = 2.1e-05
        assert expected_value == pytest.approx(output_value, 0.01855)

    def test_fitsInfo(self):
        """Test fitsInfo method"""
        input_value = 'files/image.fits'
        output_value = aimfast.fitsInfo(input_value)
        from astLib import astWCS
        expected = {'b_size': (0.0, 0.0, 0.0),
                    'centre': 'J2000.0,0.0deg,-30.0deg',
                    'ddec': 0.000277777777777778,
                    'dec': -30.0,
                    'decPix': 2049.0,
                    'dra': 0.000277777777777778,
                    'numPix': 4096,
                    'ra': 0.0,
                    'raPix': 2049.0,
                    'skyArea': 1.2945382716049403,
                    'wcs': astWCS.WCS}
        for param, value in expected.items():
            val = output_value[param]
            if param is 'wcs':
                assert isinstance(val, value)
            else:
                assert val == value
