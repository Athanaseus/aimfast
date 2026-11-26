"""A subcommand's options must reach its own parser untouched.

argparse classifies every token against the main parser before dispatching to a
subcommand, and single-dash options match by prefix there (allow_abbrev does not stop
that for "-x" style options). source-finder's -t matched -thresh, -tol, -title and
-title-size and failed as ambiguous; its -r broke once a second "-r*" option existed.
"""
import shutil
import subprocess

import pytest

from aimfast import aimfast


def _source_finder_parser():
    parser = aimfast.get_argparser()
    group = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    return group.choices["source-finder"]


def _value_for(action):
    if action.choices:
        return str(next(iter(action.choices)))
    if action.type in (int, float):
        return "1"
    return "somevalue"


class TestSourceFinderOptions(object):
    def test_dash_t_sets_the_threshold(self):
        args = aimfast.parse_args(["source-finder", "-t", "7"])
        assert float(args.sf_thresh) == 7
        assert args.subcommand == "source-finder"

    def test_dash_r_sets_the_image(self):
        assert aimfast.parse_args(["source-finder", "-r", "img.fits"]).sf_restored == "img.fits"

    def test_every_option_that_takes_a_value_parses(self):
        """Guards the whole class, not only the options that happened to break."""
        checked = 0
        for action in _source_finder_parser()._actions:
            if not action.option_strings or action.nargs == 0:
                continue
            value = _value_for(action)
            expected = action.type(value) if action.type in (int, float) else value
            for option in action.option_strings:
                got = getattr(aimfast.parse_args(["source-finder", option, value]), action.dest)
                assert got == expected, f"{option} gave {got!r}"
                checked += 1
        assert checked > 5

    def test_main_parser_defaults_are_still_present(self):
        """main() reads main-parser attributes even on the subcommand path."""
        args = aimfast.parse_args(["source-finder", "-t", "7"])
        defaults = aimfast.get_argparser().parse_args([])
        for name, value in vars(defaults).items():
            if name != "subcommand":
                assert getattr(args, name) == value, name

    def test_main_options_before_the_subcommand_still_apply(self):
        args = aimfast.parse_args(["--outfile", "x.json", "source-finder", "-t", "7"])
        assert args.outfile == "x.json" and float(args.sf_thresh) == 7


class TestWithoutSubcommand(object):
    def test_main_options_parse_as_before(self):
        args = aimfast.parse_args(["--compare-residuals", "a.fits", "b.fits"])
        assert args.noise == [["a.fits", "b.fits"]]
        assert args.subcommand is None


def test_console_script_accepts_dash_t():
    exe = shutil.which("aimfast")
    if exe is None:
        pytest.skip("aimfast console script not on PATH")
    proc = subprocess.run([exe, "source-finder", "-r", "foo.fits", "-t", "10", "--help"],
                          capture_output=True, text=True)
    assert "ambiguous option" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr
