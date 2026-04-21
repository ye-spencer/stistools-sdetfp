import pytest
import random
import math
from astropy.io import fits
from stistools import doppinfo


@pytest.fixture
def valid_doppinfo_fits_dict():
    random.seed(42)

    primary_sci = {
        "instrume": "STIS",
        "nextend": 3,
        "ra_targ": random.uniform(0, 360),
        "dec_targ": random.uniform(-90, 90),
        "cenwave": 1700,
        "detector": "CCD",
    }

    sci_ext = {
        "expstart": 56752.1,
        "expend": 56752.2,
        "cd1_1": 0.05,
        "ltm1_1": 1.0,
    }

    spt_primary = {
        "argperig": random.uniform(0, 2 * math.pi),
        "cirveloc": random.uniform(0, 10000),
        "cosincli": random.uniform(-1, 1),
        "ecbdx3": random.uniform(-1, 1),
        "eccentry": random.uniform(0, 1),
        "eccentx2": random.uniform(-1, 1),
        "ecbdx4d3": random.uniform(-1, 1),
        "epchtime": random.uniform(0, 100000),
        "esqdx5d2": random.uniform(-1, 1),
        "fdmeanan": random.uniform(0, 2 * math.pi),
        "hsthorb": random.uniform(2000, 3000),
        "meananom": random.uniform(0, 2 * math.pi),
        "rascascn": random.uniform(0, 2 * math.pi),
        "rcargper": random.uniform(0, 10000),
        "rcascnrv": random.uniform(0, 10000),
        "sdmeanan": random.uniform(0, 2 * math.pi),
        "semilrec": random.uniform(0, 100000),
        "sineincl": random.uniform(-1, 1),
    }

    return {
        "primary_sci": primary_sci,
        "sci_ext": sci_ext,
        "spt_primary": spt_primary,
    }


@pytest.fixture
def doppinfo_factory(tmp_path):
    def _factory(fits_dict, dt=0.0, update=False, quiet=True, pass_spt=True):
        primary_hdu = fits.PrimaryHDU()
        for key, value in fits_dict["primary_sci"].items():
            primary_hdu.header[key] = value

        sci_num = fits_dict["primary_sci"]["nextend"] // 3
        hdus = [primary_hdu]
        for i in range(1, sci_num + 1):
            ext_hdu = fits.ImageHDU(name="SCI", ver=i)
            for key, value in fits_dict["sci_ext"].items():
                ext_hdu.header[key] = value
            hdus.append(ext_hdu)

        input_path = tmp_path / "test_raw.fits"
        fits.HDUList(hdus).writeto(input_path)

        spt_hdu = fits.PrimaryHDU()
        for key, value in fits_dict["spt_primary"].items():
            spt_hdu.header[key] = value
        spt_path = tmp_path / "test_spt.fits"
        spt_hdu.writeto(spt_path)

        spt_arg = str(spt_path) if pass_spt else None
        return doppinfo.Doppinfo(str(input_path), spt=spt_arg,
                                 dt=dt, update=update, quiet=quiet)

    return _factory


# Blackbox: init computes doppler params
def test_init_computes_params(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    assert d.orbitper > 0
    assert math.isfinite(d.doppzero)
    assert math.isfinite(d.doppmag)
    assert math.isfinite(d.doppmag_v)


# Blackbox: sph_rec at origin
def test_sph_rec_origin(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    assert d._sph_rec(0.0, 0.0) == pytest.approx([1.0, 0.0, 0.0])


# Blackbox: rvToPixels inverts pixelsToRv
def test_rv_pixels_inverse(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    assert d._pixelsToRv(d._rvToPixels(12.5)) == pytest.approx(12.5)


# Blackbox: get_rv returns scalar float
def test_get_rv_returns_scalar(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    rv = d._get_rv(d.obs.expstart)
    assert isinstance(rv, float)


# Blackbox: peakQuadratic symmetric returns middle
def test_peakQuadratic_symmetric(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    assert d._peakQuadratic([1.0, 0.0, 1.0], 5.0, 1.0) == 5.0


# Branch: peakQuadratic zero denominator
def test_peakQuadratic_zero_denom(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    assert d._peakQuadratic([1.0, 1.0, 1.0], 3.0, 1.0) == 3.0


# Blackbox: peakQuadratic asymmetric shifts right
def test_peakQuadratic_asymmetric(doppinfo_factory, valid_doppinfo_fits_dict):
    # y values at x=-1,0,1 for (x-1)**2 -> peak (minimum) at x = 1
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    assert d._peakQuadratic([4.0, 1.0, 0.0], 0.0, 1.0) == pytest.approx(1.0)


# Blackbox: peakQuadratic scales with spacing
def test_peakQuadratic_spacing(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    assert d._peakQuadratic([4.0, 1.0, 0.0], 10.0, 2.0) == pytest.approx(12.0)


# Branch coverage: findSptName each token
@pytest.mark.parametrize("input_name,expected", [
    ("foo_raw.fits", "foo_spt.fits"),
    ("foo_corrtag.fits", "foo_spt.fits"),
    ("foo_flt.fits", "foo_spt.fits"),
    ("foo_counts.fits", "foo_spt.fits"),
    ("foo_x1d.fits", "foo_spt.fits"),
])
def test_findSptName_tokens(doppinfo_factory, valid_doppinfo_fits_dict,
                            input_name, expected):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    d.input = input_name
    assert d._findSptName() == expected


# Branch coverage: findSptName unknown name raises
def test_findSptName_invalid(doppinfo_factory, valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    d.input = "garbage.fits"
    with pytest.raises(RuntimeError):
        d._findSptName()


# Branch coverage: quiet=False prints header
def test_init_verbose_prints(capsys, doppinfo_factory, valid_doppinfo_fits_dict):
    doppinfo_factory(valid_doppinfo_fits_dict, quiet=False)
    assert "orbitper" in capsys.readouterr().out


# Branch coverage: update=True writes keywords
def test_init_update_writes_keywords(tmp_path, doppinfo_factory,
                                     valid_doppinfo_fits_dict):
    doppinfo_factory(valid_doppinfo_fits_dict, update=True)
    with fits.open(tmp_path / "test_raw.fits") as fd:
        assert "doppzero" in fd["sci", 1].header
        assert "doppmag" in fd["sci", 1].header


# Branch coverage: printDopplerShift dt > 0
def test_printDopplerShift_dt_positive(capsys, doppinfo_factory,
                                       valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    capsys.readouterr()
    d.printDopplerShift(0.01)
    assert "time" in capsys.readouterr().out


# Branch coverage: printDopplerShift dt == 0
def test_printDopplerShift_dt_zero(capsys, doppinfo_factory,
                                   valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    capsys.readouterr()
    d.printDopplerShift(0.0)
    assert "midpoint" in capsys.readouterr().out


# Branch coverage: expstart equals expend
def test_printDopplerShift_expstart_equals_expend(capsys, doppinfo_factory,
                                                  valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    d.obs.expend = d.obs.expstart
    capsys.readouterr()
    d.printDopplerShift(0.0)
    assert "midpoint" in capsys.readouterr().out


# Branch coverage: min at expstart, max at expend
def test_printDopplerShift_min_start_max_end(mocker, capsys, doppinfo_factory,
                                             valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    expstart = d.obs.expstart
    mocker.patch.object(d, "_get_rv",
                        side_effect=lambda t: (t - expstart) * 1000.0)
    capsys.readouterr()
    d.printDopplerShift(0.0)
    assert "midpoint" in capsys.readouterr().out


# Branch coverage: max at expstart, min at expend
def test_printDopplerShift_max_start_min_end(mocker, capsys, doppinfo_factory,
                                             valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict)
    expstart = d.obs.expstart
    mocker.patch.object(d, "_get_rv",
                        side_effect=lambda t: -(t - expstart) * 1000.0)
    capsys.readouterr()
    d.printDopplerShift(0.0)
    assert "midpoint" in capsys.readouterr().out


# Branch coverage: spt=None triggers _findSptName
def test_init_spt_none_uses_findSptName(doppinfo_factory,
                                        valid_doppinfo_fits_dict):
    d = doppinfo_factory(valid_doppinfo_fits_dict, pass_spt=False)
    assert d.orbitper > 0


# Branch coverage: _updateKeywords verbose, keys added
def test_updateKeywords_verbose_added(capsys, doppinfo_factory,
                                      valid_doppinfo_fits_dict):
    doppinfo_factory(valid_doppinfo_fits_dict, update=True, quiet=False)
    assert "added" in capsys.readouterr().out


# Branch coverage: _updateKeywords verbose, keys replaced
def test_updateKeywords_verbose_updated(capsys, doppinfo_factory,
                                        valid_doppinfo_fits_dict):
    valid_doppinfo_fits_dict["sci_ext"]["orbitper"] = 1.0
    valid_doppinfo_fits_dict["sci_ext"]["doppzero"] = 1.0
    valid_doppinfo_fits_dict["sci_ext"]["doppmag"] = 1.0
    valid_doppinfo_fits_dict["sci_ext"]["doppmagv"] = 1.0
    doppinfo_factory(valid_doppinfo_fits_dict, update=True, quiet=False)
    assert "-->" in capsys.readouterr().out
