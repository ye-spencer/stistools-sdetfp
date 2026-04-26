import numpy as np
from stistools.wavelen import adjust_disp, get_delta_offset1, compute_wavelengths
from astropy.io import fits

class FakeTable:
    def __init__(self, fields, names=None):
        self._fields = fields
        self.names = names or list(fields.keys())

    def field(self, name):
        return self._fields[name]

def test_adjust_disp_basic():

    coeff = np.array([1.,2.,3.,4.])

    inang = FakeTable({
        "ncoeff1": [2],
        "coeff1": [[0.1,0.2]],
        "ncoeff2": [2],
        "coeff2": [[0.3,0.4]],
    })

    adjust_disp(
        ncoeff=4,
        coeff=coeff,
        delta_offset1=1.0,
        shifta1=0.5,
        inang_info=inang,
        delta_tan=0.1,
        delta_row=2,
        binaxis1=1
    )

    assert coeff[0] != 1.0

def test_get_delta_offset1(monkeypatch):

    def fake_getTable(table, filt, exactly_one=True):
        if filt["aperture"] == "APER":
            return FakeTable({"offset1":[5.]})
        return FakeTable({"offset1":[2.]})

    monkeypatch.setattr("stistools.wavelen.gettable.getTable", fake_getTable)

    val = get_delta_offset1("table", "APER", "REF")

    assert val == 3.

def test_compute_wavelengths_basic(monkeypatch):

    phdr = fits.Header()
    hdr = fits.Header()

    phdr["OPT_ELEM"] = "G750L"
    phdr["CENWAVE"] = 7751
    phdr["APERTURE"] = "APER"
    phdr["PROAPER"] = "APER"
    phdr["SCLAMP"] = "NONE"
    phdr["DISPTAB"] = "disp.fits"
    phdr["APDESTAB"] = "apdes.fits"
    phdr["INANGTAB"] = "inang.fits"
    phdr["RA_TARG"] = 0
    phdr["DEC_TARG"] = 0

    hdr["CRPIX2"] = 1

    monkeypatch.setattr(
        "stistools.wavelen.r_util.expandFileName",
        lambda x: x
    )

    monkeypatch.setattr(
        "stistools.wavelen.radialvel.radialVel",
        lambda *a: 0
    )

    monkeypatch.setattr(
        "stistools.wavelen.evaldisp.newton",
        lambda pixels, coeff, cenwave: pixels + cenwave
    )

    monkeypatch.setattr(
        "stistools.wavelen.r_util.interpolate",
        lambda x,y,row: np.array([1.,2.,3.,4.])
    )

    def fake_getTable(name, filt, **kwargs):

        if "disp" in name:
            return FakeTable({
                "ref_aper":["REF"],
                "a2center":np.array([1,2]),
                "ncoeff":[4],
                "coeff":[[1,2,3,4]]
            })

        if "apdes" in name:
            return FakeTable({
                "offset1":[1.],
                "angle":[0.3]
            }, names=["offset1","angle"])

        if "inang" in name:
            return FakeTable({
                "ncoeff1":[1],
                "coeff1":[[0.1]],
                "ncoeff2":[0],
                "coeff2":[[0]]
            })

    monkeypatch.setattr("stistools.wavelen.gettable.getTable", fake_getTable)

    wl = compute_wavelengths((2,5), phdr, hdr, helcorr="OMIT")

    assert wl.shape == (2,5)

def test_missing_angle_column(monkeypatch, capsys):

    phdr = fits.Header()
    hdr = fits.Header()

    phdr["OPT_ELEM"] = "G750L"
    phdr["CENWAVE"] = 7751
    phdr["APERTURE"] = "APER"
    phdr["PROAPER"] = "APER"
    phdr["SCLAMP"] = "NONE"
    phdr["DISPTAB"] = "disp.fits"
    phdr["APDESTAB"] = "apdes.fits"
    phdr["INANGTAB"] = "inang.fits"
    phdr["RA_TARG"] = 0
    phdr["DEC_TARG"] = 0

    hdr["CRPIX2"] = 1

    monkeypatch.setattr(
        "stistools.wavelen.r_util.expandFileName",
        lambda x: x
    )

    monkeypatch.setattr(
        "stistools.wavelen.r_util.interpolate",
        lambda x, y, row: np.array([1., 2., 3., 4.])
    )

    monkeypatch.setattr(
        "stistools.wavelen.evaldisp.newton",
        lambda pixels, coeff, cenwave: pixels + cenwave
    )

    def fake_getTable(name, filt, **kwargs):

        if "disp" in name:
            return FakeTable({
                "ref_aper": ["REF"],
                "a2center": np.array([1]),
                "ncoeff": [4],
                "coeff": [[1.,2.,3.,4.]]
            })

        if "apdes" in name:
            # intentionally missing ANGLE column
            return FakeTable(
                {"offset1":[1.]},
                names=["offset1"]
            )

        if "inang" in name:
            return FakeTable({
                "ncoeff1":[0],
                "coeff1":[[]],
                "ncoeff2":[0],
                "coeff2":[[]]
            })

    monkeypatch.setattr(
        "stistools.wavelen.gettable.getTable",
        fake_getTable
    )

    compute_wavelengths((2,5), phdr, hdr, helcorr="OMIT")

    out = capsys.readouterr().out
    assert "ANGLE not found" in out