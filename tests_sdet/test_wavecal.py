import pytest
import importlib
from unittest.mock import Mock
from astropy.io import fits

from stistools import wavecal as wavecal_module


@pytest.fixture(autouse=True)
def mock_cs4_version(monkeypatch):

    def fake_run(*args, **kwargs):
        m = Mock()
        m.stdout = "3.5.0" 
        return m

    monkeypatch.setattr(wavecal_module.subprocess, "run", fake_run)


@pytest.fixture(autouse=True)
def mock_subprocess(monkeypatch):

    def fake_call(*args, **kwargs):
        return 0

    monkeypatch.setattr(wavecal_module.subprocess, "call", fake_call)


@pytest.fixture(autouse=True)
def mock_glob(monkeypatch):

    def fake_glob(x):
        return [x] 

    monkeypatch.setattr(wavecal_module.glob, "glob", fake_glob)


@pytest.fixture(autouse=True)
def mock_fits_getval(monkeypatch):

    def fake_getval(fname, key, default=None):
        if key == "PROPAPER":
            return "G130ME1"   # triggers E-aperture branch
        return default

    monkeypatch.setattr(wavecal_module.fits, "getval", fake_getval)


def make_fits(tmp_path):
    f = tmp_path / "input.fits"

    hdu = fits.PrimaryHDU()
    hdr = hdu.header

    hdr["OPT_ELEM"] = "G230L"
    hdr["PROPAPER"] = "G130ME1"
    hdr["DQICORR"] = "COMPLETE"
    hdr["BLEVCORR"] = "COMPLETE"
    hdr["DARKCORR"] = "COMPLETE"
    hdr["FLATCORR"] = "COMPLETE"
    hdr["X2DCORR"] = "COMPLETE"

    fits.HDUList([hdu]).writeto(f)

    return str(f)


def test_wavecal_basic(tmp_path):

    infile = make_fits(tmp_path)

    status = wavecal_module.wavecal(
        input=infile,
        wavecal=infile
    )

    assert status == 0


def test_input_parsing_error(monkeypatch):

    monkeypatch.setattr(wavecal_module.glob, "glob", lambda x: [])

    status = wavecal_module.wavecal(input="missing.fits", wavecal="missing.fits")

    assert status == 2


def test_e_aperture_branch(tmp_path):

    infile = make_fits(tmp_path)

    status = wavecal_module.wavecal(
        input=infile,
        wavecal=infile
    )

    assert status == 0


def test_print_version(monkeypatch):

    called = {}

    def fake_call(*args, **kwargs):
        called["ran"] = True
        return 0

    monkeypatch.setattr(wavecal_module.subprocess, "call", fake_call)

    status = wavecal_module.wavecal(
        input="a.fits",
        wavecal="b.fits",
        print_version=True
    )

    assert status == 0
    assert called["ran"]


def test_debugfile_branch(tmp_path):

    infile = make_fits(tmp_path)

    status = wavecal_module.wavecal(
        input=infile,
        wavecal=infile,
        debugfile="debug.txt"
    )

    assert status == 0