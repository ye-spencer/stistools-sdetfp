import numpy as np
import pytest
from astropy.io import fits
from stistools import poisson_err


def _make_x1d(tmp_path, instrument="STIS", detector="FUV-MAMA",
              n_orders=1, n_pix=4, name="x1d.fits"):
    phdu = fits.PrimaryHDU()
    phdu.header["INSTRUME"] = instrument
    phdu.header["DETECTOR"] = detector

    shape = (n_orders, n_pix)
    flux = np.ones(shape, dtype=np.float32)
    net = np.ones(shape, dtype=np.float32)
    gross = np.ones(shape, dtype=np.float32)
    net_err = np.full(shape, 0.1, dtype=np.float32)
    fmt = f"{n_pix}E"
    cols = fits.ColDefs([
        fits.Column(name="FLUX", array=flux, format=fmt),
        fits.Column(name="NET", array=net, format=fmt),
        fits.Column(name="GROSS", array=gross, format=fmt),
        fits.Column(name="NET_ERROR", array=net_err, format=fmt),
    ])
    table = fits.BinTableHDU.from_columns(cols)
    table.header["EXPTIME"] = 100.0

    path = tmp_path / name
    fits.HDUList([phdu, table]).writeto(path)
    return str(path)


# Statement coverage: valid FUV-MAMA x1d writes output with new columns.
def test_poisson_err_writes_output(tmp_path):
    inp = _make_x1d(tmp_path)
    out = tmp_path / "out.fits"
    poisson_err.poisson_err(inp, str(out), verbose=False)
    assert out.exists()
    with fits.open(str(out)) as hdul:
        names = hdul[1].columns.names
    for c in ("NET_ERROR_PCI_LOW", "NET_ERROR_PCI_UP",
              "ERROR_PCI_LOW", "ERROR_PCI_UP", "N_COUNTS"):
        assert c in names


# Branch coverage: NUV-MAMA detector is also accepted.
def test_poisson_err_nuv_mama(tmp_path):
    inp = _make_x1d(tmp_path, detector="NUV-MAMA", name="nuv.fits")
    out = tmp_path / "nuv_out.fits"
    poisson_err.poisson_err(inp, str(out), verbose=False)
    assert out.exists()


# Branch coverage: verbose=True takes the print branch.
def test_poisson_err_verbose(tmp_path, capsys):
    inp = _make_x1d(tmp_path, name="v.fits")
    out = tmp_path / "v_out.fits"
    poisson_err.poisson_err(inp, str(out), verbose=True)
    assert "Added" in capsys.readouterr().out


# Branch coverage: non-STIS instrument raises ValueError.
def test_poisson_err_non_stis(tmp_path):
    inp = _make_x1d(tmp_path, instrument="COS", name="cos.fits")
    with pytest.raises(ValueError):
        poisson_err.poisson_err(inp, str(tmp_path / "out.fits"), verbose=False)


# Branch coverage: CCD detector raises ValueError.
def test_poisson_err_ccd_rejected(tmp_path):
    inp = _make_x1d(tmp_path, detector="CCD", name="ccd.fits")
    with pytest.raises(ValueError):
        poisson_err.poisson_err(inp, str(tmp_path / "out.fits"), verbose=False)


# Blackbox: multiple orders are all processed.
def test_poisson_err_multi_order(tmp_path):
    inp = _make_x1d(tmp_path, n_orders=3, n_pix=5, name="multi.fits")
    out = tmp_path / "multi_out.fits"
    poisson_err.poisson_err(inp, str(out), verbose=False)
    with fits.open(str(out)) as hdul:
        assert hdul[1].data["N_COUNTS"].shape == (3, 5)
