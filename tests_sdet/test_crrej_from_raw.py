import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from stistools import crrej_from_raw


@pytest.fixture
def crrejtab_path(tmp_path):
    p = tmp_path / "crr.fits"
    Table({
        'CRSPLIT':  np.array([1],         dtype='i2'),
        'MEANEXP':  np.array([0.0],       dtype='f'),
        'SCALENSE': np.array(['1.0'],     dtype='S8'),
        'INITGUES': np.array(['minimum'], dtype='S8'),
        'SKYSUB':   np.array(['mode'],    dtype='S4'),
        'CRSIGMAS': np.array(['4,5'],     dtype='S20'),
        'CRRADIUS': np.array([1.5],       dtype='f'),
        'CRTHRESH': np.array([0.8],       dtype='f'),
        'BADINPDQ': np.array([16],        dtype='l'),
        'CRMASK':   np.array([True],      dtype=bool),
    }).write(str(p))
    return str(p)


def _raw(path, crrejtab='N/A', exptime=0.0):
    primary = fits.PrimaryHDU()
    primary.header['CRREJTAB'] = crrejtab
    sci = fits.ImageHDU(name='SCI')
    if exptime is not None:
        sci.header['EXPTIME'] = exptime
    fits.HDUList([primary, sci, fits.ImageHDU(), fits.ImageHDU()]).writeto(str(path))
    return str(path)


@pytest.fixture
def raw_path(tmp_path, crrejtab_path):
    return _raw(tmp_path / "in_raw.fits", crrejtab_path)


@pytest.fixture(autouse=True)
def _mock_calstis(mocker):
    return mocker.patch("stistools.crrej_from_raw.calstis", return_value=0)


# Branch coverage: print_version short-circuits via sys.exit(0)
def test_print_version_exits():
    with pytest.raises(SystemExit):
        crrej_from_raw.crrej_from_raw("any.fits", print_version=True)


# Branch coverage: print_revision short-circuits via sys.exit(0)
def test_print_revision_exits():
    with pytest.raises(SystemExit):
        crrej_from_raw.crrej_from_raw("any.fits", print_revision=True)


# Statement/branch coverage: defaults pulled from CRREJTAB row
def test_defaults_from_crrejtab(raw_path):
    assert crrej_from_raw.crrej_from_raw(raw_path) == 0


# Statement/branch coverage: numeric overrides + verbose path
def test_numeric_overrides_verbose(raw_path):
    assert crrej_from_raw.crrej_from_raw(
        raw_path, scalense=2.0, initgues='median', skysub='none',
        crsigmas=4.0, crradius=2.0, crthresh=0.5, badinpdq=8,
        crmask=False, verbose=True,
    ) == 0


# Branch coverage: invalid INITGUES raises ValueError
def test_invalid_initgues(raw_path):
    with pytest.raises(ValueError, match="INITGUES"):
        crrej_from_raw.crrej_from_raw(raw_path, initgues='bogus')


# Branch coverage: invalid SKYSUB raises ValueError
def test_invalid_skysub(raw_path):
    with pytest.raises(ValueError, match="SKYSUB"):
        crrej_from_raw.crrej_from_raw(raw_path, skysub='bogus')


# Branch coverage: missing EXPTIME yields NaN meanexp -> ValueError
def test_missing_exptime(tmp_path, crrejtab_path):
    p = _raw(tmp_path / "noexp.fits", crrejtab_path, exptime=None)
    with pytest.raises(ValueError, match="EXPTIME"):
        crrej_from_raw.crrej_from_raw(p)


# Branch coverage: header CRREJTAB == "N/A" raises ValueError
def test_determine_crrejtab_na(tmp_path):
    p = _raw(tmp_path / "raw.fits")
    with pytest.raises(ValueError, match="not specified"):
        crrej_from_raw.determine_crrejtab(p)


# Branch coverage: missing CRREJTAB file raises FileNotFoundError
def test_determine_crrejtab_missing_file(tmp_path):
    p = _raw(tmp_path / "raw.fits", crrejtab=str(tmp_path / "nope.fits"))
    with pytest.raises(FileNotFoundError):
        crrej_from_raw.determine_crrejtab(p)


# Branch coverage: user-supplied crrejtab overrides header lookup
def test_determine_crrejtab_user_supplied(tmp_path, crrejtab_path):
    p = _raw(tmp_path / "raw.fits")
    assert len(crrej_from_raw.determine_crrejtab(p, crrejtab=crrejtab_path)) == 1


# Branch coverage: PREV_CRR fallback when CRREJTAB starts with $TEMPCRR (verbose on)
def test_determine_crrejtab_prev_crr_fallback(tmp_path, crrejtab_path, monkeypatch):
    monkeypatch.setattr(crrej_from_raw, 'VERBOSE', True)
    p = tmp_path / "raw.fits"
    hdr = fits.Header()
    hdr['CRREJTAB'] = f'${crrej_from_raw.ENV_VAR}/stale.fits'
    hdr['PREV_CRR'] = crrejtab_path
    fits.PrimaryHDU(header=hdr).writeto(str(p))
    assert len(crrej_from_raw.determine_crrejtab(str(p))) == 1


# Branch coverage: create_new_crr rejects mismatched key set
def test_create_new_crr_bad_keys():
    with pytest.raises(ValueError, match="not specified correctly"):
        crrej_from_raw.create_new_crr({'CRSPLIT': 1})


# Statement coverage: create_new_crr happy path
def test_create_new_crr_ok():
    t = crrej_from_raw.create_new_crr({
        'crsplit': 1, 'meanexp': 0.0, 'scalense': '1', 'initgues': 'minimum',
        'skysub': 'mode', 'crsigmas': '4', 'crradius': 1.5, 'crthresh': 0.8,
        'badinpdq': 16, 'crmask': True,
    })
    assert t['CRSPLIT'][0] == 1
