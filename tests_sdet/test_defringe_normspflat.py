import numpy as np
from astropy.io import fits
import pytest
import importlib
import numpy as np
import os
normspflat_module = importlib.import_module("stistools.defringe.normspflat")

def make_test_flat(tmp_path, opt_elem="G750M", cenwave=9800, aperture="52X0.2", bincols=1):

    path = tmp_path / "flat.fits"
    wave = tmp_path / "wave.fits"
    bias = tmp_path / "bias.fits"
    dark = tmp_path / "dark.fits"
    pflt = tmp_path / "pflt.fits"
    lflt = tmp_path / "lflt.fits"

    fits.PrimaryHDU().writeto(wave)
    fits.PrimaryHDU().writeto(bias)
    fits.PrimaryHDU().writeto(dark)
    fits.PrimaryHDU().writeto(pflt)
    fits.PrimaryHDU().writeto(lflt)

    data = np.ones((20, 1200), dtype=float)

    hdu0 = fits.PrimaryHDU()
    hdr = hdu0.header

    hdr["ROOTNAME"] = "test"
    hdr["OPT_ELEM"] = opt_elem
    hdr["APERTURE"] = aperture
    hdr["BINAXIS1"] = bincols
    hdr["BINAXIS2"] = 1
    hdr["CENWAVE"] = cenwave
    hdr["WAVECAL"] = str(wave)

    hdr["DQICORR"] = "COMPLETE"
    hdr["BLEVCORR"] = "COMPLETE"
    hdr["BIASCORR"] = "COMPLETE"
    hdr["DARKCORR"] = "COMPLETE"
    hdr["FLATCORR"] = "COMPLETE"
    hdr["CRCORR"] = "COMPLETE"
    hdr["WAVECORR"] = "COMPLETE"
    hdr["HELCORR"] = "COMPLETE"
    hdr["X2DCORR"] = "COMPLETE"

    hdr["BIASFILE"] = str(bias)
    hdr["DARKFILE"] = str(dark)
    hdr["PFLTFILE"] = str(pflt)
    hdr["LFLTFILE"] = str(lflt)

    sci = fits.ImageHDU(data=data, name="SCI")

    fits.HDUList([hdu0, sci]).writeto(path)

    return path

@pytest.fixture
def mock_fit(monkeypatch):

    def fake_fit(x, y, **kwargs):
        return lambda x: np.ones_like(x)

    monkeypatch.setattr(
        normspflat_module,
        "fit1d",
        fake_fit
    )

@pytest.fixture
def mock_calstis(monkeypatch):

    def fake_calstis(infile, wavecal=None, outroot=None, trailer=None):

        hdr = fits.getheader(infile)
        rootname = hdr["ROOTNAME"]
        opt_elem = hdr["OPT_ELEM"]

        suffix = "sx2" if opt_elem == "G750M" else "crj"
        outname = os.path.join(outroot, f"{rootname}_{suffix}.fits")

        data = np.ones((20, 1200))

        fits.HDUList([
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(data=data)
        ]).writeto(outname)

        return 0

    monkeypatch.setattr(normspflat_module, "calstis", fake_calstis)

@pytest.fixture(autouse=True)
def mock_oref(tmp_path, monkeypatch):
    monkeypatch.setenv("oref", str(tmp_path))


def test_missing_inflat():

    with pytest.raises(FileNotFoundError):
        normspflat_module.normspflat("missing.fits")

def test_invalid_opt_elem(tmp_path):

    from astropy.io import fits

    path = tmp_path / "bad.fits"

    hdu = fits.PrimaryHDU()
    hdu.header["ROOTNAME"] = "bad"
    hdu.header["OPT_ELEM"] = "G140L"

    fits.HDUList([hdu]).writeto(path)

    with pytest.raises(ValueError):
        normspflat_module.normspflat(path)

def test_skip_calibration(tmp_path, mock_fit):

    flat = make_test_flat(tmp_path)

    normspflat_module.normspflat(flat, do_cal=False)

def test_g750m_processing(tmp_path, mock_fit, mock_calstis):

    flat = make_test_flat(tmp_path, opt_elem="G750M", cenwave=9800)

    normspflat_module.normspflat(flat)

def test_g750l_processing(tmp_path, mock_fit):

    flat = make_test_flat(tmp_path, opt_elem="G750L", cenwave=7751)

    normspflat_module.normspflat(flat, do_cal=False)

def test_call_normspflat(monkeypatch):

    called = {}

    def fake_normspflat(**kwargs):
        called["ran"] = True

    monkeypatch.setattr(
        normspflat_module,
        "normspflat",
        fake_normspflat
    )

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "input.fits"]
    )

    normspflat_module.call_normspflat()

    assert called["ran"]

