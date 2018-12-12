import numpy as np
from aimfast import aimfast


class TestClass(object):
    "aimfast Test klass"
    def test_deg2arcsec(self):
        "Test deg2arcsec method"
        input_value = 10.5
        output_value = aimfast.deg2arcsec(input_value)
        expected_value = input_value*3600.00
        assert expected_value == output_value

    def test_rad2deg(self):
        "Test rad2deg method"
        input_value = 1.5
        output_value = aimfast.rad2deg(input_value)
        expected_value = input_value*(180/np.pi)
        assert expected_value == output_value

    def test_rad2arcsec(self):
        "Test rad2arcsec method"
        input_value = 1.5
        output_value = aimfast.rad2arcsec(input_value)
        expected_value = input_value*(3600.0*180.0/np.pi)
        assert expected_value == output_value
