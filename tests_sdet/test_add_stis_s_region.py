import os
import numpy as np
import pytest
from astropy.io import fits
from stistools import add_stis_s_region as asr


# Statement coverage: coords_from_s_region parses a POLYGON ICRS string.
def test_coords_from_s_region_polygon():
    s = "POLYGON ICRS 10.0 -20.0 11.0 -21.0"
    ra, dec = asr.coords_from_s_region(s)
    assert ra == [10.0, 11.0]
    assert dec == [-20.0, -21.0]


# Statement coverage: get_siaf_limits returns min/max of arrays.
def test_get_siaf_limits():
    x = np.array([1.0, 5.0, 3.0])
    y = np.array([-2.0, 4.0, 0.0])
    assert asr.get_siaf_limits(x, y) == (1.0, 5.0, -2.0, 4.0)


# Branch coverage: smallest_size with wcslimits[0]=None uses siaf x-bounds.
def test_smallest_size_wcs_none():
    x, y = asr.smallest_size((None, None, -3, 3), (-1, 1, -2, 2))
    assert x[0] == -1 and x[2] == 1
    assert y[0] == -2 and y[1] == 2


# Branch coverage: smallest_size with finite wcslimits clips to intersection.
def test_smallest_size_finite():
    x, y = asr.smallest_size((-5, 5, -5, 5), (-1, 1, -2, 2))
    assert x[0] == -1 and x[2] == 1
    assert y[0] == -2 and y[1] == 2


# Statement coverage: write_keyword_to_header adds new S_REGION keyword.
def test_write_keyword_new(tmp_path):
    hdu = fits.ImageHDU(data=np.zeros((2, 2)))
    hdu.header["PA_APER"] = 0.0
    asr.write_keyword_to_header(hdu, "POLYGON ICRS 1 2 3 4")
    assert hdu.header["S_REGION"] == "POLYGON ICRS 1 2 3 4"


# Branch coverage: write_keyword_to_header updates existing S_REGION.
def test_write_keyword_existing():
    hdu = fits.ImageHDU(data=np.zeros((2, 2)))
    hdu.header["PA_APER"] = 0.0
    hdu.header["S_REGION"] = "OLD"
    asr.write_keyword_to_header(hdu, "NEW")
    assert hdu.header["S_REGION"] == "NEW"


# Branch coverage: get_siaf_entry returns None when aperture not in lookup.
def test_get_siaf_entry_unknown_aperture():
    assert asr.get_siaf_entry({}, "NOT_REAL_APERTURE", "CCD") is None


# Branch coverage: get_siaf_entry returns matching SIAF entry.
def test_get_siaf_entry_found():
    fake_siaf = {"OV050X29": "fake-entry"}
    assert asr.get_siaf_entry(fake_siaf, "0.05X29", "CCD") == "fake-entry"


# Branch coverage: get_files_to_process returns sorted matching files.
def test_get_files_to_process_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "abc123_raw.fits").touch()
    (tmp_path / "abc123_tag.fits").touch()
    (tmp_path / "abc123_x1d.fits").touch()
    files = asr.get_files_to_process(["abc123"])
    assert files == ["abc123_raw.fits", "abc123_tag.fits"]


# Branch coverage: no matching files yields empty list.
def test_get_files_to_process_no_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert asr.get_files_to_process(["nope"]) == []


# Blackbox: main with None rootnames returns without error.
def test_main_none_rootnames():
    assert asr.main(None) is None


def _sci_hdulist(ctype1="RA---TAN"):
    phdu = fits.PrimaryHDU()
    sci = fits.ImageHDU(data=np.zeros((10, 20), dtype=np.float32), name="SCI")
    sci.header["EXTNAME"] = "SCI"
    sci.header["NAXIS1"] = 20
    sci.header["NAXIS2"] = 10
    sci.header["CRPIX1"] = 10.0
    sci.header["CRPIX2"] = 5.0
    sci.header["CTYPE1"] = ctype1
    sci.header["CD1_1"] = 1e-5
    sci.header["CD1_2"] = 0.0
    sci.header["CD2_1"] = 0.0
    sci.header["CD2_2"] = 1e-5
    return fits.HDUList([phdu, sci])


# Statement coverage: get_pixel_scales for SCI extension uses CD matrix.
def test_get_pixel_scales_sci():
    cdelt1, cdelt2 = asr.get_pixel_scales(_sci_hdulist())
    assert cdelt1 == pytest.approx(1e-5 * 3600.0)
    assert cdelt2 == pytest.approx(1e-5 * 3600.0)


# Branch coverage: get_pixel_scales for EVENTS extension uses TC matrix.
def test_get_pixel_scales_events():
    phdu = fits.PrimaryHDU()
    ev = fits.BinTableHDU.from_columns(
        [fits.Column(name="X", array=np.array([0]), format="E")])
    ev.header["EXTNAME"] = "EVENTS"
    ev.header["TC2_2"] = 2e-5
    ev.header["TC2_3"] = 0.0
    ev.header["TC3_2"] = 0.0
    ev.header["TC3_3"] = 2e-5
    cdelt1, cdelt2 = asr.get_pixel_scales(fits.HDUList([phdu, ev]))
    assert cdelt1 == pytest.approx(2e-5 * 3600.0)


# Statement coverage: get_wcs_limits returns the four-tuple for SCI imaging.
def test_get_wcs_limits_sci():
    limits = asr.get_wcs_limits(_sci_hdulist())
    assert len(limits) == 4
    xmin, xmax, ymin, ymax = limits
    assert xmin < 0 < xmax
    assert ymin < 0 < ymax


# Branch coverage: get_wcs_limits with CTYPE1='WAVE' takes the dispersed branch.
def test_get_wcs_limits_wave():
    limits = asr.get_wcs_limits(_sci_hdulist(ctype1="WAVE"))
    assert len(limits) == 4


# Branch coverage: get_files_to_process warns when rootname has a path.
def test_get_files_to_process_path_rootname(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = asr.get_files_to_process([str(tmp_path / "abc")])
    assert files == []


# Branch coverage: get_siaf_entry fallback loop succeeds with another detector.
def test_get_siaf_entry_fallback():
    fake_siaf = {"OF050X29": "fuv-entry"}
    assert asr.get_siaf_entry(fake_siaf, "0.05X29", "CCD") is not None


class _FakeSiafEntry:
    def closed_polygon_points(self, frame):
        x = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
        y = np.array([-1.0, 1.0, 1.0, -1.0, -1.0])
        return x, y


def _full_sci_file(tmp_path, name="abc_raw.fits"):
    phdu = fits.PrimaryHDU()
    phdu.header["DETECTOR"] = "CCD"
    phdu.header["APERTURE"] = "0.05X29"
    phdu.header["PROPAPER"] = "0.05X29"
    phdu.header["ROOTNAME"] = "abc"
    sci = fits.ImageHDU(data=np.zeros((10, 20), dtype=np.float32))
    sci.header["EXTNAME"] = "SCI"
    sci.header["EXTVER"] = 1
    sci.header["NAXIS1"] = 20
    sci.header["NAXIS2"] = 10
    sci.header["CRPIX1"] = 10.0
    sci.header["CRPIX2"] = 5.0
    sci.header["CTYPE1"] = "RA---TAN"
    sci.header["CD1_1"] = 1e-5
    sci.header["CD1_2"] = 0.0
    sci.header["CD2_1"] = 0.0
    sci.header["CD2_2"] = 1e-5
    sci.header["RA_APER"] = 100.0
    sci.header["DEC_APER"] = 20.0
    sci.header["PA_APER"] = 0.0
    path = tmp_path / name
    fits.HDUList([phdu, sci]).writeto(path)
    return str(path)


# Statement/branch coverage: add_s_region dry-run path with a known aperture.
def test_add_s_region_dry_run(tmp_path):
    f = _full_sci_file(tmp_path)
    fake_siaf = {"OV050X29": _FakeSiafEntry()}
    asr.add_s_region(f, fake_siaf, dry_run=True)
    with fits.open(f) as hdul:
        assert "S_REGION" not in hdul[1].header


# Statement/branch coverage: add_s_region writes S_REGION to header.
def test_add_s_region_writes(tmp_path):
    f = _full_sci_file(tmp_path)
    fake_siaf = {"OV050X29": _FakeSiafEntry()}
    asr.add_s_region(f, fake_siaf, dry_run=False)
    with fits.open(f) as hdul:
        assert hdul[1].header["S_REGION"].startswith("POLYGON ICRS")


# Branch coverage: add_s_region with unknown aperture writes a CIRCLE region.
def test_add_s_region_no_siaf_match(tmp_path):
    f = _full_sci_file(tmp_path)
    with fits.open(f, mode="update") as hdul:
        hdul[0].header["PROPAPER"] = "BOGUS"
        hdul[0].header["APERTURE"] = "BOGUS"
    asr.add_s_region(f, {}, dry_run=False)
    with fits.open(f) as hdul:
        assert hdul[1].header["S_REGION"].startswith("CIRCLE ICRS")


# Branch coverage: add_s_region falls back to APERTURE when PROPAPER is unknown.
def test_add_s_region_aperture_fallback(tmp_path):
    f = _full_sci_file(tmp_path)
    with fits.open(f, mode="update") as hdul:
        hdul[0].header["PROPAPER"] = "BOGUS"
    fake_siaf = {"OV050X29": _FakeSiafEntry()}
    asr.add_s_region(f, fake_siaf, dry_run=True)


# Statement coverage: main iterates files and calls add_s_region.
def test_main_runs(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    _full_sci_file(tmp_path, name="abc_raw.fits")
    mocker.patch("stistools.add_stis_s_region.pysiaf.Siaf",
                 return_value={"OV050X29": _FakeSiafEntry()})
    asr.main(["abc"], dry_run=True)


# Statement coverage: write_region_file with include_mast=False writes a .reg.
def test_write_region_file_no_mast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = _full_sci_file(tmp_path, name="abc_raw.fits")
    with fits.open(f, mode="update") as hdul:
        hdul[1].header["S_REGION"] = "POLYGON ICRS 1.0 2.0 3.0 4.0"
    asr.write_region_file(f, include_mast=False)
    reg = tmp_path / "abc.reg"
    assert reg.exists()
    text = reg.read_text()
    assert "polygon" in text
    assert "color=blue" in text


# Statement coverage: call_main parses argv and dispatches to main.
def test_call_main_invokes_main(monkeypatch, mocker):
    monkeypatch.setattr("sys.argv", ["prog", "abc", "--dry_run"])
    mock_main = mocker.patch("stistools.add_stis_s_region.main")
    asr.call_main()
    mock_main.assert_called_once_with(["abc"], dry_run=True)


def _events_hdulist(tctyp2="ANGLE"):
    phdu = fits.PrimaryHDU()
    ev = fits.BinTableHDU.from_columns(
        [fits.Column(name="X", array=np.array([0]), format="E")])
    ev.header["EXTNAME"] = "EVENTS"
    ev.header["TCRPX2"] = 4.0
    ev.header["TCRPX3"] = 3.0
    ev.header["AXLEN1"] = 8
    ev.header["AXLEN2"] = 6
    ev.header["TCTYP2"] = tctyp2
    ev.header["TC2_2"] = 1e-5
    ev.header["TC2_3"] = 0.0
    ev.header["TC3_2"] = 0.0
    ev.header["TC3_3"] = 1e-5
    return fits.HDUList([phdu, ev])


# Branch coverage: get_wcs_limits for EVENTS extension.
def test_get_wcs_limits_events():
    limits = asr.get_wcs_limits(_events_hdulist())
    assert len(limits) == 4


# Branch coverage: get_wcs_limits EVENTS with TCTYP2='WAVE' dispersed branch.
def test_get_wcs_limits_events_wave():
    limits = asr.get_wcs_limits(_events_hdulist(tctyp2="WAVE"))
    assert len(limits) == 4


# Branch coverage: write_region_file with include_mast=True uses MAST query.
def test_write_region_file_with_mast(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    f = _full_sci_file(tmp_path, name="abc_raw.fits")
    with fits.open(f, mode="update") as hdul:
        hdul[1].header["S_REGION"] = "POLYGON ICRS 1.0 2.0 3.0 4.0"

    fake_obs_module = mocker.MagicMock()
    fake_obs_module.Observations.query_criteria.return_value = {
        "s_region": ["POLYGON ICRS 5.0 6.0 7.0 8.0"]
    }
    monkeypatch.setitem(__import__("sys").modules, "astroquery.mast",
                        fake_obs_module)

    asr.write_region_file(f, include_mast=True)
    text = (tmp_path / "abc.reg").read_text()
    assert "color=green" in text
    assert "color=red" in text


