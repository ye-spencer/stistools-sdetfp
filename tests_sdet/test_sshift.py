import numpy as np
import pytest
from astropy.io import fits
from stistools import sshift


DEFAULT_HEADER = {
    "FLATCORR": "COMPLETE",
    "TARGNAME": "TARGET-A",
    "PROPOSID": 12345,
    "OBSET_ID": "01",
    "PROPAPER": "52X0.1",
    "OPT_ELEM": "G140L",
    "CENWAVE": 1425,
    "BINAXIS1": 1,
    "BINAXIS2": 1,
    "POSTARG1": 0.0,
    "POSTARG2": 0.0,
}


@pytest.fixture
def flt_factory(tmp_path):
    counter = {"n": 0}

    def _make(overrides=None, crpix2=512.0, name=None):
        hdr = dict(DEFAULT_HEADER)
        if overrides:
            hdr.update(overrides)
        phdu = fits.PrimaryHDU()
        for k, v in hdr.items():
            phdu.header[k] = v
        img = np.zeros((4, 4), dtype=np.float32)
        ext = fits.ImageHDU(data=img)
        ext.header["CRPIX2"] = crpix2
        counter["n"] += 1
        path = tmp_path / (name or f"f{counter['n']}_flt.fits")
        fits.HDUList([phdu, ext]).writeto(path)
        return str(path)

    return _make


# Statement coverage: shiftimage with positive shift.
def test_shiftimage_positive(flt_factory, tmp_path):
    infile = flt_factory()
    out = tmp_path / "out.fits"
    sshift.shiftimage(infile, str(out), shift=1)
    assert out.exists()


# Branch coverage: shiftimage with negative shift.
def test_shiftimage_negative(flt_factory, tmp_path):
    infile = flt_factory()
    out = tmp_path / "out.fits"
    sshift.shiftimage(infile, str(out), shift=-1)
    assert out.exists()


# Branch coverage: shiftimage with zero shift.
def test_shiftimage_zero(flt_factory, tmp_path):
    infile = flt_factory()
    out = tmp_path / "out.fits"
    sshift.shiftimage(infile, str(out), shift=0)
    assert out.exists()


# Branch coverage: single-string input wrapped into list, defaults run.
def test_sshift_single_input(flt_factory, tmp_path):
    infile = flt_factory(name="x_flt.fits")
    out = tmp_path / "x_sfl.fits"
    sshift.sshift(infile, output=str(out))
    assert out.exists()


# Branch coverage: output=None default-name path (re.sub _flt -> _sfl).
def test_sshift_default_output(flt_factory, tmp_path):
    infile = flt_factory(name="y_flt.fits")
    sshift.sshift(infile)
    assert (tmp_path / "y_sfl.fits").exists()


# Branch coverage: shift list provided skips auto-calc; binaxis2!=1 print path.
def test_sshift_with_shifts_and_binned(flt_factory, tmp_path):
    infile = flt_factory(overrides={"BINAXIS2": 2}, name="b_flt.fits")
    out = tmp_path / "b_sfl.fits"
    sshift.sshift(infile, output=str(out), shifts=0)
    assert out.exists()


# Branch coverage: platescale provided uses float() branch.
def test_sshift_explicit_platescale(flt_factory, tmp_path):
    infile = flt_factory(name="p_flt.fits")
    out = tmp_path / "p_sfl.fits"
    sshift.sshift(infile, output=str(out), platescale=0.05077)
    assert out.exists()


# Branch coverage: non-COMPLETE FLATCORR raises ValueError.
def test_sshift_not_flatfielded(flt_factory):
    infile = flt_factory(overrides={"FLATCORR": "OMIT"})
    with pytest.raises(ValueError):
        sshift.sshift(infile)


# Branch coverage: non-integer shift entry raises ValueError.
def test_sshift_non_int_shift(flt_factory):
    infile = flt_factory()
    with pytest.raises(ValueError):
        sshift.sshift(infile, shifts=[1.5])


# Branch coverage: input/output count mismatch raises ValueError.
def test_sshift_output_count_mismatch(flt_factory):
    infile = flt_factory()
    with pytest.raises(ValueError):
        sshift.sshift([infile], output=["a.fits", "b.fits"])


# Branch coverage: TARGNAME mismatch raises ValueError.
def test_sshift_targname_mismatch(flt_factory, tmp_path):
    a = flt_factory(name="a_flt.fits")
    b = flt_factory(overrides={"TARGNAME": "OTHER"}, name="b_flt.fits")
    with pytest.raises(ValueError):
        sshift.sshift([a, b], output=[str(tmp_path / "a_sfl.fits"),
                                      str(tmp_path / "b_sfl.fits")])


# Branch coverage: PROPOSID/OBSET_ID mismatch raises ValueError.
def test_sshift_proposid_mismatch(flt_factory, tmp_path):
    a = flt_factory(name="a_flt.fits")
    b = flt_factory(overrides={"PROPOSID": 99999}, name="b_flt.fits")
    with pytest.raises(ValueError):
        sshift.sshift([a, b], output=[str(tmp_path / "a_sfl.fits"),
                                      str(tmp_path / "b_sfl.fits")])


# Branch coverage: PROPAPER/OPT_ELEM/CENWAVE mismatch raises ValueError.
def test_sshift_config_mismatch(flt_factory, tmp_path):
    a = flt_factory(name="a_flt.fits")
    b = flt_factory(overrides={"OPT_ELEM": "G230L"}, name="b_flt.fits")
    with pytest.raises(ValueError):
        sshift.sshift([a, b], output=[str(tmp_path / "a_sfl.fits"),
                                      str(tmp_path / "b_sfl.fits")])


# Branch coverage: BINAXIS mismatch raises ValueError.
def test_sshift_binning_mismatch(flt_factory, tmp_path):
    a = flt_factory(name="a_flt.fits")
    b = flt_factory(overrides={"BINAXIS1": 2}, name="b_flt.fits")
    with pytest.raises(ValueError):
        sshift.sshift([a, b], output=[str(tmp_path / "a_sfl.fits"),
                                      str(tmp_path / "b_sfl.fits")])


# Branch coverage: POSTARG1 mismatch raises ValueError.
def test_sshift_postarg1_mismatch(flt_factory, tmp_path):
    a = flt_factory(name="a_flt.fits")
    b = flt_factory(overrides={"POSTARG1": 1.0}, name="b_flt.fits")
    with pytest.raises(ValueError):
        sshift.sshift([a, b], output=[str(tmp_path / "a_sfl.fits"),
                                      str(tmp_path / "b_sfl.fits")])


# Branch coverage: non-integral POSTARG2 shift exceeds tolerance.
def test_sshift_non_integral_postarg2(flt_factory, tmp_path):
    a = flt_factory(name="a_flt.fits")
    b = flt_factory(overrides={"POSTARG2": 0.025}, name="b_flt.fits")
    with pytest.raises(ValueError):
        sshift.sshift([a, b], output=[str(tmp_path / "a_sfl.fits"),
                                      str(tmp_path / "b_sfl.fits")])


# Blackbox: empty list input raises ValueError.
def test_sshift_empty_list():
    with pytest.raises(ValueError):
        sshift.sshift([])
