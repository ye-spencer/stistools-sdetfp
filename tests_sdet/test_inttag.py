import pytest
import numpy as np
from astropy.io import fits
from stistools import inttag


def _events(times, axis1=None, axis2=None):
    n = len(times)
    if axis1 is None:
        axis1 = [1] * n
    if axis2 is None:
        axis2 = [1] * n
    cols = fits.ColDefs([
        fits.Column(name='TIME', format='D', array=np.asarray(times, dtype=np.float64)),
        fits.Column(name='AXIS1', format='I', array=np.asarray(axis1, dtype=np.int16)),
        fits.Column(name='AXIS2', format='I', array=np.asarray(axis2, dtype=np.int16)),
    ])
    return fits.BinTableHDU.from_columns(cols, name='EVENTS')


def _gti(starts, stops):
    cols = fits.ColDefs([
        fits.Column(name='START', format='D', array=np.asarray(starts, dtype=np.float64)),
        fits.Column(name='STOP', format='D', array=np.asarray(stops, dtype=np.float64)),
    ])
    return fits.BinTableHDU.from_columns(cols, name='GTI')


def _write_tag(tmp_path, events_hdu, gti_hdu, with_ctype=True):
    phdr = fits.Header()
    phdr['CENTERA1'] = 512
    phdr['CENTERA2'] = 512
    phdr['SIZAXIS1'] = 4
    phdr['SIZAXIS2'] = 4
    events_hdu.header['EXPSTART'] = 58000.0
    if with_ctype:
        ks = {'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
              'CRPIX1': 512.0, 'CRPIX2': 512.0,
              'CRVAL1': 0.0, 'CRVAL2': 0.0,
              'CUNIT1': 'deg', 'CUNIT2': 'deg',
              'CD1_1': 1e-5, 'CD1_2': 0.0, 'CD2_1': 0.0, 'CD2_2': 1e-5}
    else:
        ks = {'TCTYP1': 'RA---TAN', 'TCTYP2': 'DEC--TAN',
              'TCRPX1': 512.0, 'TCRPX2': 512.0,
              'TCRVL1': 0.0, 'TCRVL2': 0.0,
              'TCUNI1': 'deg', 'TCUNI2': 'deg',
              'TC1_1': 1e-5, 'TC1_2': 0.0, 'TC2_1': 0.0, 'TC2_2': 1e-5}
    for k, v in ks.items():
        events_hdu.header[k] = v
    path = str(tmp_path / 'tag.fits')
    fits.HDUList([fits.PrimaryHDU(header=phdr), events_hdu, gti_hdu]).writeto(path)
    return path


@pytest.fixture
def tag_basic(tmp_path):
    return _write_tag(tmp_path, _events([1., 2., 3., 4., 5.]), _gti([0.], [10.]))


# Statement coverage: default inttag writes output
def test_inttag_default(tag_basic, tmp_path):
    out = str(tmp_path / 'out.fits')
    inttag.inttag(tag_basic, out, verbose=False)
    with fits.open(out) as f:
        assert len(f) == 4


# Branch coverage: allevents True ignores GTI
def test_inttag_allevents(tag_basic, tmp_path):
    out = str(tmp_path / 'out.fits')
    inttag.inttag(tag_basic, out, allevents=True, verbose=False)
    with fits.open(out) as f:
        assert len(f) == 4


# Branch coverage: highres True doubles axes
def test_inttag_highres(tag_basic, tmp_path):
    out = str(tmp_path / 'out.fits')
    inttag.inttag(tag_basic, out, highres=True, verbose=False)
    with fits.open(out) as f:
        assert f[1].data.shape == (8, 8)
        assert f[0].header['BINAXIS1'] == 1


# Branch coverage: lowres sets LORSCORR complete
def test_inttag_lowres_lorscorr(tag_basic, tmp_path):
    out = str(tmp_path / 'out.fits')
    inttag.inttag(tag_basic, out, verbose=False)
    with fits.open(out) as f:
        assert f[0].header['LORSCORR'] == 'COMPLETE'
        assert f[0].header['BINAXIS1'] == 2


# Branch coverage: verbose True prints imset
def test_inttag_verbose(tag_basic, tmp_path, capsys):
    out = str(tmp_path / 'out.fits')
    inttag.inttag(tag_basic, out, verbose=True)
    assert 'imset' in capsys.readouterr().out


# Branch coverage: rcount greater than 1
def test_inttag_rcount(tmp_path):
    path = _write_tag(tmp_path, _events([1., 2., 6., 8.]), _gti([0.], [10.]))
    out = str(tmp_path / 'out.fits')
    inttag.inttag(path, out, rcount=2, verbose=False)
    with fits.open(out) as f:
        assert f[0].header['NRPTEXP'] == 2


# Branch coverage: starttime below gti_start reset
def test_inttag_starttime_below(tag_basic, tmp_path):
    out = str(tmp_path / 'out.fits')
    inttag.inttag(tag_basic, out, starttime=-5., verbose=False)
    with fits.open(out) as f:
        assert f[0].header['NRPTEXP'] == 1


# Branch coverage: stoptime truncated at gti_stop
def test_inttag_stoptime_truncated(tag_basic, tmp_path):
    out = str(tmp_path / 'out.fits')
    inttag.inttag(tag_basic, out, starttime=2., increment=100., verbose=False)
    with fits.open(out) as f:
        assert f[0].header['NRPTEXP'] == 1


# Branch coverage: imset with no events is skipped
def test_inttag_skips_empty_imset(tmp_path, capsys):
    path = _write_tag(tmp_path, _events([1., 2.]), _gti([0.], [10.]))
    out = str(tmp_path / 'out.fits')
    inttag.inttag(path, out, rcount=2, increment=3., verbose=True)
    assert 'Skipping imset' in capsys.readouterr().out


# Branch coverage: rename TCTYP keywords
def test_inttag_rename_tctyp(tmp_path):
    path = _write_tag(tmp_path, _events([1., 2., 3.]), _gti([0.], [10.]), with_ctype=False)
    out = str(tmp_path / 'out.fits')
    inttag.inttag(path, out, verbose=False)
    with fits.open(out) as f:
        assert 'CTYPE1' in f[1].header
        assert 'CD1_1' in f[1].header


# Branch coverage: no imset events returns zeros
def test_exp_range_no_imset_events():
    events = np.rec.array([(5.,)], names='TIME', formats='>f8')
    gti = np.rec.array([(0., 10.)], names='START, STOP', formats='>f8,>f8')
    exp_time, expstart, expstop, good = inttag.exp_range(100., 200., events, gti, 58000.)
    assert exp_time == 0
    assert expstart == 58000.
    assert len(good) == 0


# Branch coverage: imset events all outside GTI
def test_exp_range_no_good_events():
    events = np.rec.array([(5.,), (6.,)], names='TIME', formats='>f8')
    gti = np.rec.array([(100., 200.)], names='START, STOP', formats='>f8,>f8')
    exp_time, _, _, good = inttag.exp_range(0., 10., events, gti, 58000.)
    assert exp_time == 0
    assert len(good) == 0


# Branch coverage: single GTI no gaps
def test_exp_range_single_gti():
    events = np.rec.array([(1.,), (4.,)], names='TIME', formats='>f8')
    gti = np.rec.array([(0., 5.)], names='START, STOP', formats='>f8,>f8')
    exp_time, _, _, good = inttag.exp_range(0., 5., events, gti, 58000.)
    assert exp_time == 5
    assert len(good) == 2


# Branch coverage: gap fully inside window
def test_exp_range_gap_inside():
    events = np.rec.array([(1.,), (4.,), (11.,), (14.,)], names='TIME', formats='>f8')
    gti = np.rec.array([(0., 5.), (10., 15.)], names='START, STOP', formats='>f8,>f8')
    exp_time, _, _, _ = inttag.exp_range(0., 20., events, gti, 58000.)
    assert exp_time == 15


# Branch coverage: gap straddles stoptime
def test_exp_range_gap_straddles_stop():
    events = np.rec.array([(1.,), (4.,)], names='TIME', formats='>f8')
    gti = np.rec.array([(0., 5.), (10., 15.)], names='START, STOP', formats='>f8,>f8')
    exp_time, _, _, _ = inttag.exp_range(0., 7., events, gti, 58000.)
    assert exp_time == 5


# Branch coverage: gap straddles starttime
def test_exp_range_gap_straddles_start():
    events = np.rec.array([(11.,), (14.,)], names='TIME', formats='>f8')
    gti = np.rec.array([(0., 5.), (10., 15.)], names='START, STOP', formats='>f8,>f8')
    exp_time, _, _, _ = inttag.exp_range(7., 20., events, gti, 58000.)
    assert exp_time == 10


# Branch coverage: gap outside window else branch
def test_exp_range_gap_outside():
    events = np.rec.array([(1.,)], names='TIME', formats='>f8')
    gti = np.rec.array([(0., 5.), (10., 15.)], names='START, STOP', formats='>f8,>f8')
    exp_time, _, _, _ = inttag.exp_range(0., 4., events, gti, 58000.)
    assert exp_time == 4


# Statement coverage: events_to_accum lowres path
def test_events_to_accum_lowres():
    events = np.rec.array([(2, 2)], names='AXIS1, AXIS2', formats='>i2,>i2')
    result = inttag.events_to_accum(events, 4, 4, highres=False)
    assert result.shape == (4, 4)
    assert result.sum() == 1


# Branch coverage: events_to_accum highres path
def test_events_to_accum_highres():
    events = np.rec.array([(2, 2)], names='AXIS1, AXIS2', formats='>i2,>i2')
    result = inttag.events_to_accum(events, 4, 4, highres=True)
    assert result.shape == (4, 4)
    assert result.sum() == 1
