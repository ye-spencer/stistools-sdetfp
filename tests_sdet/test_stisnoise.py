import pytest
import numpy as np
from astropy.io import fits
from stistools import stisnoise


def _make_fits(tmp_path, shape=(1024, 1024), amp='A', extname='SCI', name='test.fits'):
    phdr = fits.Header()
    phdr['CCDAMP'] = amp
    phdr['TARGNAME'] = 'TARGET'
    phdr['CCDGAIN'] = 1.0
    primary = fits.PrimaryHDU(header=phdr)
    data = np.ones(shape, dtype=np.float32)
    sci = fits.ImageHDU(data=data, name=extname)
    path = str(tmp_path / name)
    fits.HDUList([primary, sci]).writeto(path)
    return path


@pytest.fixture
def flt_fits(tmp_path):
    return _make_fits(tmp_path)


@pytest.fixture
def raw_fits(tmp_path):
    return _make_fits(tmp_path, shape=(1044, 1062), name='raw.fits')


# --- helper function tests ---

# Statement coverage: _median middle of sorted array
def test_median_odd():
    assert stisnoise._median(np.array([3., 1., 2.])) == 2.


# Statement coverage: medianfilter runs three loops
def test_medianfilter_runs():
    ts = np.arange(10, dtype=np.float64)
    result = stisnoise.medianfilter(ts, 3)
    assert result.shape == (10,)


# Branch coverage: gauss positive dx
def test_gauss_positive_dx():
    assert np.isclose(stisnoise.gauss(1.0, 1.0, 1.0, 5.0), 5.0)


# Branch coverage: gauss positive dx falloff
def test_gauss_positive_dx_far():
    assert np.isclose(stisnoise.gauss(100.0, 0.0, 1.0, 5.0), 0.0)


# Branch coverage: gauss dx zero delta match
def test_gauss_zero_dx_match():
    assert stisnoise.gauss(1.0, 1.0, 0.0, 5.0) == 5.0


# Branch coverage: gauss dx zero no match
def test_gauss_zero_dx_no_match():
    assert stisnoise.gauss(2.0, 1.0, 0.0, 5.0) == 0.0


# Branch coverage: wipefilter raw pads ntime+14
def test_wipefilter_raw():
    ts = np.ones(100, dtype=np.float64)
    result = stisnoise.wipefilter(ts, 'raw', 22.0, 0., 1e10, 0.5)
    assert result.shape == (102,)


# Branch coverage: wipefilter flt pads ntime+7
def test_wipefilter_flt():
    ts = np.ones(101, dtype=np.float64)
    result = stisnoise.wipefilter(ts, 'flt', 22.0, 0., 1e10, 0.5)
    assert result.shape == (103,)


# Branch coverage: windowfilter raw pads ntime+14
def test_windowfilter_raw():
    ts = np.ones(100, dtype=np.float64)
    result = stisnoise.windowfilter(ts, 'raw', 22.0, 100., 50., 50.)
    assert result.shape == (102,)


# Branch coverage: windowfilter flt pads ntime+7
def test_windowfilter_flt():
    ts = np.ones(101, dtype=np.float64)
    result = stisnoise.windowfilter(ts, 'flt', 22.0, 100., 50., 50.)
    assert result.shape == (103,)


# Branch coverage: windowfilter tiny taper (different kernw parity)
def test_windowfilter_small_taper():
    ts = np.ones(51, dtype=np.float64)
    result = stisnoise.windowfilter(ts, 'flt', 22.0, 500., 100., 2000.)
    assert result.shape == (53,)


# --- main stisnoise tests ---

# Branch coverage: conflicting filter options
def test_stisnoise_conflict():
    with pytest.raises(ValueError):
        stisnoise.stisnoise("x.fits", boxcar=5, wipe=np.array([0., 1., 0.5]))


# Branch coverage: conflicting wipe and window
def test_stisnoise_conflict_wipe_window():
    with pytest.raises(ValueError):
        stisnoise.stisnoise("x.fits",
                            wipe=np.array([0., 1., 0.5]),
                            window=np.array([0., 1., 0.5]))


# Branch coverage: wrong extname raises
def test_stisnoise_wrong_extname(tmp_path):
    path = _make_fits(tmp_path, extname='ERR')
    with pytest.raises(RuntimeError):
        stisnoise.stisnoise(path, verbose=0)


# Branch coverage: wrong shape raises
def test_stisnoise_wrong_shape(tmp_path):
    path = _make_fits(tmp_path, shape=(100, 100))
    with pytest.raises(RuntimeError):
        stisnoise.stisnoise(path, verbose=0)


# Branch coverage: bad amp raises
def test_stisnoise_bad_amp(tmp_path):
    path = _make_fits(tmp_path, amp='X')
    with pytest.raises(RuntimeError):
        stisnoise.stisnoise(path, verbose=0)


# Branch coverage: amp A flt pipeline
def test_stisnoise_amp_a(flt_fits):
    freq, mag = stisnoise.stisnoise(flt_fits, verbose=0)
    assert freq.shape == mag.shape
    assert freq.shape[0] > 0


# Branch coverage: amp B flt pipeline
def test_stisnoise_amp_b(tmp_path):
    path = _make_fits(tmp_path, amp='B')
    freq, mag = stisnoise.stisnoise(path, verbose=0)
    assert freq.shape == mag.shape


# Branch coverage: amp C flt pipeline
def test_stisnoise_amp_c(tmp_path):
    path = _make_fits(tmp_path, amp='C')
    freq, mag = stisnoise.stisnoise(path, verbose=0)
    assert freq.shape == mag.shape


# Branch coverage: amp D flt pipeline
def test_stisnoise_amp_d(tmp_path):
    path = _make_fits(tmp_path, amp='D')
    freq, mag = stisnoise.stisnoise(path, verbose=0)
    assert freq.shape == mag.shape


# Branch coverage: raw shape pipeline
def test_stisnoise_raw_shape(raw_fits):
    freq, mag = stisnoise.stisnoise(raw_fits, verbose=0)
    assert freq.shape == mag.shape


# Branch coverage: boxcar filter branch
def test_stisnoise_boxcar(flt_fits):
    freq, mag = stisnoise.stisnoise(flt_fits, verbose=0, boxcar=3)
    assert mag.shape[0] > 0


# Branch coverage: wipe filter branch
def test_stisnoise_wipe(flt_fits):
    freq, mag = stisnoise.stisnoise(flt_fits, verbose=0,
                                    wipe=[0., 1000., 0.5])
    assert mag.shape[0] > 0


# Branch coverage: window filter branch
def test_stisnoise_window(flt_fits):
    freq, mag = stisnoise.stisnoise(flt_fits, verbose=0,
                                    window=[1000., 500., 500.])
    assert mag.shape[0] > 0


# Branch coverage: dc=0 keeps first bin
def test_stisnoise_dc_zero(flt_fits):
    freq, mag = stisnoise.stisnoise(flt_fits, verbose=0, dc=0)
    assert mag[0] != 0


# Branch coverage: dc=1 zeros first bin
def test_stisnoise_dc_one(flt_fits):
    freq, mag = stisnoise.stisnoise(flt_fits, verbose=0, dc=1)
    assert mag[0] == 0


# Branch coverage: outfile writes FITS
def test_stisnoise_outfile(flt_fits, tmp_path):
    out = str(tmp_path / 'out.fits')
    stisnoise.stisnoise(flt_fits, verbose=0, outfile=out)
    with fits.open(out) as f:
        assert len(f) == 2


# Branch coverage: verbose prints target info
def test_stisnoise_verbose(flt_fits, capsys):
    stisnoise.stisnoise(flt_fits, verbose=1)
    assert 'Target' in capsys.readouterr().out
