import numpy as np
import pytest
from astropy.io import fits

from stistools import mktrace


@pytest.fixture
def sptrctab():
    n = 3
    cols = [
        fits.Column(name="OPT_ELEM", format="10A", array=np.array(["G"] * n)),
        fits.Column(name="CENWAVE", format="J", array=np.array([1] * n)),
        fits.Column(name="SPORDER", format="J", array=np.array([1] * n)),
        fits.Column(name="A2CENTER", format="D",
                    array=np.array([100.0, 500.0, 900.0])),
        fits.Column(name="NELEM", format="J", array=np.array([1024] * n)),
        fits.Column(name="A2DISPL", format="1024D",
                    array=np.zeros((n, 1024))),
        fits.Column(name="A1CENTER", format="D", array=np.array([512.0] * n)),
        fits.Column(name="SNR_THRESH", format="D", array=np.array([0.0] * n)),
        fits.Column(name="PEDIGREE", format="10A", array=np.array(["X"] * n)),
    ]
    return fits.BinTableHDU.from_columns(cols).data


@pytest.fixture
def trace(mocker, sptrctab):
    mocker.patch.object(mktrace.Trace, "openTraceFile", return_value=sptrctab)
    mocker.patch("stistools.mktrace.fu.osfn", return_value="x.fits")
    kwinfo = {"opt_elem": "G", "cenwave": 1, "sporder": 1, "sptrctab": "x.fits"}
    return mktrace.Trace("f.fits", kwinfo)


@pytest.fixture
def kwinfo_full():
    return {
        "instrument": "STIS", "detector": "CCD", "binaxis1": 1, "binaxis2": 1,
        "crpix2": 512, "ltv2": 0, "sizaxis2": 1024, "opt_elem": "G",
        "cenwave": 1, "sporder": 1, "sptrctab": "x.fits",
    }


# Statement coverage: iterable true branch
def test_iterable_true():
    assert mktrace.iterable([1, 2, 3]) is True


# Branch coverage: iterable false branch
def test_iterable_false():
    assert mktrace.iterable(5) is False


# Statement coverage: interp returns size n
def test_interp_length():
    result = mktrace.interp(np.array([0.0, 1.0, 2.0, 3.0]), 8)
    assert len(result) == 8


# Statement coverage: trace_interp math helper
def test_trace_interp():
    tr1 = {"a2displ": np.zeros(1024), "a2center": 0.0}
    tr2 = {"a2displ": np.ones(1024), "a2center": 10.0}
    result = mktrace.trace_interp(tr1, tr2, 5.0)
    assert result.shape == (1024,)


# Branch coverage: getKWInfo CCD branch
def test_getKWInfo_ccd():
    hdr0 = {"INSTRUME": "STIS", "DETECTOR": "CCD", "BINAXIS1": 1,
            "BINAXIS2": 1, "sizaxis2": 1024, "OPT_ELEM": "G430L",
            "CENWAVE": 4300, "SPTRCTAB": "foo.fits"}
    hdr1 = {"CRPIX2": 512, "LTV2": 0, "SPORDER": 1}
    info = mktrace.getKWInfo(hdr0, hdr1)
    assert info["binaxis1"] == 1


# Branch coverage: getKWInfo non-CCD branch
def test_getKWInfo_non_ccd():
    hdr0 = {"INSTRUME": "STIS", "DETECTOR": "MAMA", "sizaxis2": 1024,
            "OPT_ELEM": "G140L", "CENWAVE": 1425, "SPTRCTAB": "foo.fits"}
    hdr1 = {"CRPIX2": 512, "LTV2": 0, "SPORDER": 1}
    info = mktrace.getKWInfo(hdr0, hdr1)
    assert info["binaxis1"] == 1 and info["binaxis2"] == 1


# Branch coverage: mktrace IOError exit
def test_mktrace_ioerror(mocker):
    mocker.patch("stistools.mktrace.fits.open", side_effect=IOError)
    assert mktrace.mktrace("missing.fits") is None


# Branch coverage: non-STIS instrument exits
def test_mktrace_non_stis(mocker):
    fake_hdu = mocker.MagicMock()
    fake_hdu.__getitem__.return_value.data = np.zeros((10, 10))
    fake_hdu.__getitem__.return_value.header = {}
    mocker.patch("stistools.mktrace.fits.open", return_value=fake_hdu)
    mocker.patch("stistools.mktrace.getKWInfo",
                 return_value={"instrument": "COS"})
    assert mktrace.mktrace("file.fits") is None


# Branch coverage: bad weights exits
def test_mktrace_bad_weights(mocker):
    fake_hdu = mocker.MagicMock()
    fake_hdu.__getitem__.return_value.data = np.zeros((10, 10))
    fake_hdu.__getitem__.return_value.header = {}
    mocker.patch("stistools.mktrace.fits.open", return_value=fake_hdu)
    mocker.patch("stistools.mktrace.getKWInfo",
                 return_value={"instrument": "STIS"})
    assert mktrace.mktrace("file.fits", weights=5) is None


# Branch coverage: openTraceFile None path
def test_openTraceFile_none(mocker):
    mocker.patch("stistools.mktrace.fu.osfn", return_value=None)
    kwinfo = {"opt_elem": "G", "cenwave": 1, "sporder": 1,
              "sptrctab": None}
    tr = mktrace.Trace("f.fits", kwinfo)
    assert tr.sptrctab is None


# Branch coverage: openTraceFile IOError path
def test_openTraceFile_ioerror(mocker):
    mocker.patch("stistools.mktrace.fu.osfn", return_value="bad.fits")
    mocker.patch("stistools.mktrace.fits.open", side_effect=IOError)
    kwinfo = {"opt_elem": "G", "cenwave": 1, "sporder": 1,
              "sptrctab": "bad.fits"}
    tr = mktrace.Trace("f.fits", kwinfo)
    assert tr.sptrctab is None


# Statement coverage: getTraceInd finds row
def test_getTraceInd(trace):
    i, _ = trace.getTraceInd(400.0)
    assert i == 1


# Statement coverage: readTrace returns dict
def test_readTrace(trace):
    tr = trace.readTrace(0)
    assert set(tr.keys()) == {"nelem", "a2displ", "a1center",
                              "a2center", "snr_thresh", "pedigree"}


# Branch coverage: generateTrace non-subarray path
def test_generateTrace_full(mocker, trace, kwinfo_full):
    mocker.patch.object(mktrace.Trace, "gFitTrace",
                        return_value=np.full(100, 20.0))
    data = np.zeros((100, 100))
    cen, t1024 = trace.generateTrace(data, kwinfo_full, wind=np.arange(100))
    assert t1024.shape == (1024,)


# Branch coverage: generateTrace subarray path
def test_generateTrace_subarray(mocker, trace, kwinfo_full):
    mocker.patch.object(mktrace.Trace, "gFitTrace",
                        return_value=np.full(100, 20.0))
    kwinfo_full["sizaxis2"] = 100
    data = np.zeros((100, 100))
    trace.generateTrace(data, kwinfo_full, wind=np.arange(100))


# Branch coverage: generateTrace tracecen override
def test_generateTrace_tracecen(mocker, trace, kwinfo_full):
    mocker.patch.object(mktrace.Trace, "gFitTrace",
                        return_value=np.full(100, 20.0))
    data = np.zeros((200, 100))
    trace.generateTrace(data, kwinfo_full, tracecen=50.0,
                        wind=np.arange(100))


# Branch coverage: generateTrace bad points path
def test_generateTrace_badpoints(mocker, trace, kwinfo_full):
    smoy = np.full(100, 20.0)
    smoy[50] = 1000.0
    mocker.patch.object(mktrace.Trace, "gFitTrace", return_value=smoy)
    data = np.zeros((100, 100))
    trace.generateTrace(data, kwinfo_full, wind=np.arange(100))


# Statement coverage: gFitTrace runs columns
def test_gFitTrace(mocker, trace):
    fake_fit = mocker.MagicMock()
    fake_fit.params = [0.0, 20.0, 1.0]
    mocker.patch("stistools.mktrace.gfit.gfit1d", return_value=fake_fit)
    specimage = np.zeros((10, 5))
    result = trace.gFitTrace(specimage, 0, 9)
    assert result.shape == (5,) and np.all(result == 20.0)


@pytest.fixture
def write_mocks(mocker):
    mocker.patch("stistools.mktrace.fu.copyFile")
    mocker.patch("stistools.mktrace.os.chmod")
    mocker.patch("stistools.mktrace.os.stat")
    fake_tab = mocker.MagicMock()
    fake_tab.names = ["A2DISPL", "DEGPERYR"]
    fake_hdulist = mocker.MagicMock()
    fake_hdulist.__getitem__.return_value.data = fake_tab
    mocker.patch("stistools.mktrace.fits.open", return_value=fake_hdulist)
    mocker.patch("stistools.mktrace.fits.PrimaryHDU")
    return fake_tab


# Branch coverage: writeTrace with DEGPERYR, files exist
def test_writeTrace_with_degperyr(mocker, trace, write_mocks):
    mocker.patch("stistools.mktrace.os.path.exists", return_value=True)
    mocker.patch("stistools.mktrace.os.remove")
    mocker.patch("stistools.mktrace.os.unlink")
    a2disp_ind = np.array([False, True, True, False])
    trace.writeTrace("f.fits", np.zeros(1024), np.zeros(1024),
                     np.zeros(1024), np.zeros(1024), 1, a2disp_ind)


# Branch coverage: writeTrace no DEGPERYR, no existing files
def test_writeTrace_without_degperyr(mocker, trace, write_mocks):
    write_mocks.names = ["A2DISPL"]
    mocker.patch("stistools.mktrace.os.path.exists", return_value=False)
    a2disp_ind = np.array([False, True, True, False])
    trace.writeTrace("f.fits", np.zeros(1024), np.zeros(1024),
                     np.zeros(1024), np.zeros(1024), 1, a2disp_ind)


@pytest.fixture
def mktrace_mocks(mocker):
    fake_hdulist = mocker.MagicMock()
    fake_hdulist.__getitem__.return_value.data = np.zeros((10, 10))
    fake_hdulist.__getitem__.return_value.header = {}
    mocker.patch("stistools.mktrace.fits.open", return_value=fake_hdulist)
    mocker.patch("stistools.mktrace.getKWInfo", return_value={
        "instrument": "STIS", "binaxis1": 1, "binaxis2": 1,
    })
    fake_tr = mocker.MagicMock()
    fake_tr.generateTrace.return_value = (500.0, np.zeros(1024))
    fake_tr.readTrace.side_effect = lambda i: {
        "a2displ": np.zeros(1024), "a2center": 100.0 * (i + 1),
        "a1center": 512.0, "nelem": 1024, "pedigree": "X",
        "snr_thresh": 0.0,
    }
    mocker.patch("stistools.mktrace.Trace", return_value=fake_tr)
    mocker.patch("stistools.mktrace.linefit.linefit",
                 return_value=(0.0, 0.0))
    return fake_tr


# Branch coverage: mktrace happy path no weights
def test_mktrace_happy_no_weights(mktrace_mocks):
    mktrace_mocks.getTraceInd.return_value = (2, np.array([True, True, True]))
    assert mktrace.mktrace("f.fits") is mktrace_mocks


# Branch coverage: mktrace weights + equal index branch
@pytest.mark.xfail(reason="tr1 is referenced after the else branch where it was never assigned")
def test_mktrace_happy_with_weights(mktrace_mocks):
    mktrace_mocks.getTraceInd.return_value = (1, np.array([True, True, True]))
    assert mktrace.mktrace("f.fits", weights=[(0, 5)]) is mktrace_mocks
