import numpy as np
import pytest
from astropy.io import fits
from stistools import observation


def _make_obs(tmp_path, phdr=None, sci=None, name="obs.fits"):
    p = {"INSTRUME": "STIS", "DETECTOR": "CCD",
         "RA_TARG": 10.0, "DEC_TARG": -20.0, "CENWAVE": 1425}
    s = {"EXPSTART": 55000.0, "EXPEND": 55000.1,
         "CD1_1": 0.05, "LTM1_1": 1.0}
    if phdr: p.update(phdr)
    if sci: s.update(sci)
    phdu = fits.PrimaryHDU()
    for k, v in p.items(): phdu.header[k] = v
    ext = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32), name="SCI", ver=1)
    for k, v in s.items(): ext.header[k] = v
    path = tmp_path / name
    fits.HDUList([phdu, ext]).writeto(path)
    return str(path)


# Branch coverage: STIS instrument returns Observation.
def test_initObservation_stis(tmp_path):
    obs = observation.initObservation(_make_obs(tmp_path), "stis", 1)
    assert isinstance(obs, observation.Observation)


# Branch coverage: non-STIS instrument raises RuntimeError.
def test_initObservation_unsupported(tmp_path):
    with pytest.raises(RuntimeError):
        observation.initObservation(_make_obs(tmp_path), "COS", 1)


# Statement coverage: __init__ sets defaults.
def test_observation_init_defaults():
    obs = observation.Observation("foo.fits")
    assert obs.input == "foo.fits"
    assert obs.sci_ext == 1
    assert obs.ra_targ is None
    assert obs.dispersion is None


# Branch coverage: getInfo on CCD uses highres_factor=1.
def test_getInfo_ccd(tmp_path):
    obs = observation.Observation(_make_obs(tmp_path))
    obs.getInfo()
    assert obs.dispersion == 0.05
    assert obs.cenwave == 1425


# Branch coverage: non-CCD detector uses highres_factor=2.
def test_getInfo_mama(tmp_path):
    obs = observation.Observation(_make_obs(tmp_path, phdr={"DETECTOR": "FUV-MAMA"}))
    obs.getInfo()
    assert obs.dispersion == 0.025


# Branch coverage: CENWAVE <= 0 raises ValueError.
def test_getInfo_bad_cenwave(tmp_path):
    obs = observation.Observation(_make_obs(tmp_path, phdr={"CENWAVE": 0}))
    with pytest.raises(ValueError):
        obs.getInfo()


# Branch coverage: zero dispersion raises ValueError.
def test_getInfo_zero_dispersion(tmp_path):
    obs = observation.Observation(_make_obs(tmp_path, sci={"CD1_1": 0.0}))
    with pytest.raises(ValueError):
        obs.getInfo()
