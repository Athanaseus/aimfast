"""Tests for source-finder CLI/config consistency.

Covers the four fixes in the sf-cli branch:
  1. -t/--threshold sets the DETECTION threshold for every finder
  2. the shipped config template exposes a CPU option for every finder
  3. default detection thresholds are matched across finders
  4. the dead -closest/--closest flag is gone
"""
import pathlib

import yaml

from aimfast import aimfast


TEMPLATE = pathlib.Path(aimfast.__file__).parent / "source_finder.yml"

# The key each finder uses for its DETECTION threshold, i.e. the level a source peak
# must exceed to be found at all. Note aegean's is seedclip, NOT floodclip: floodclip
# only grows an island around a peak already detected at seedclip.
DETECTION_KEY = {"pybdsf": "thresh_pix", "aegean": "seedclip", "breizorro": "threshold"}
CPU_KEY = {"pybdsf": "ncores", "aegean": "cores", "breizorro": "ncpu"}


def _template():
    with open(TEMPLATE) as fh:
        return yaml.safe_load(fh)


class TestThresholdMapping(object):
    """-t/--threshold must mean the same thing for every finder."""

    def test_threshold_sets_detection_key_for_each_finder(self):
        for finder, key in DETECTION_KEY.items():
            params, selected = aimfast.apply_sf_cli_overrides(
                _template(), sourcery=finder, threshold=7.0
            )
            assert selected == finder
            assert params[finder][key] == 7.0, (
                f"-t did not set {finder}'s detection threshold ({key})"
            )

    def test_threshold_does_not_touch_aegean_floodclip(self):
        """Regression: -t used to set floodclip, leaving detection at the default."""
        template = _template()
        original_floodclip = template["aegean"]["floodclip"]
        params, _ = aimfast.apply_sf_cli_overrides(
            template, sourcery="aegean", threshold=7.0
        )
        assert params["aegean"]["seedclip"] == 7.0
        assert params["aegean"]["floodclip"] == original_floodclip


class TestCpuMapping(object):
    """-j/--ncpu must reach every finder, and be discoverable in the template."""

    def test_ncpu_sets_cpu_key_for_each_finder(self):
        for finder, key in CPU_KEY.items():
            params, _ = aimfast.apply_sf_cli_overrides(
                _template(), sourcery=finder, ncpu=1
            )
            assert params[finder][key] == 1

    def test_template_exposes_a_cpu_option_for_every_finder(self):
        """Regression: aegean had no cores key, so config runs silently used all cores."""
        template = _template()
        for finder, key in CPU_KEY.items():
            assert key in template[finder], (
                f"{finder} block of source_finder.yml has no {key} key, so the CPU "
                "count is not discoverable from the generated config"
            )


class TestMatchedDefaults(object):
    """A default multi-finder run must compare like with like."""

    def test_default_detection_thresholds_are_matched(self):
        """Regression: breizorro defaulted to 6 sigma against the others' 5."""
        template = _template()
        thresholds = {f: float(template[f][k]) for f, k in DETECTION_KEY.items()}
        assert len(set(thresholds.values())) == 1, (
            f"default detection thresholds differ across finders: {thresholds}"
        )


class TestClosestFlagRemoved(object):
    """The dead -closest flag is gone; closest-only matching stays enforced."""

    def test_cli_no_longer_advertises_closest(self):
        source = pathlib.Path(aimfast.__file__).read_text()
        assert '"-closest"' not in source
        assert "args.closest_only" not in source

    def test_compare_models_rejects_closest_only_kwarg(self):
        """The parameter is gone rather than silently ignored."""
        import inspect

        assert "closest_only" not in inspect.signature(aimfast.compare_models).parameters
