import os
import numpy as np
import pytest
import sys
from astropy.io import fits

from stistools.defringe.defringe import defringe, sdqflags, parse_args

SHAPE = (8, 8)            
SCI_VALUE = 10.0            
ERR_VALUE =  0.5            
FLAT_VALUE =  2.0                           

def _make_science_file(path, opt_elem="G750L", n_imsets=1, include_dq=True, include_err=True, sci_value=SCI_VALUE, err_value=ERR_VALUE):
    # write a minimal STIS-style FITS file

    hdulist = [fits.PrimaryHDU()]
    hdulist[0].header["OPT_ELEM"] = opt_elem
    
    for v in range(1, n_imsets + 1):
        sci_hdu = fits.ImageHDU(
            data=np.full(SHAPE, sci_value, dtype=np.float32), name="SCI")
        sci_hdu.header["EXTVER"] = v
        hdulist.append(sci_hdu)

        if include_err:
            err_hdu = fits.ImageHDU(
                data=np.full(SHAPE, err_value, dtype=np.float32), name="ERR")
            err_hdu.header["EXTVER"] = v
            hdulist.append(err_hdu)

        if include_dq:
            dq_hdu = fits.ImageHDU(
                data=np.zeros(SHAPE, dtype=np.int16), name="DQ")
            dq_hdu.header["EXTVER"] = v
            hdulist.append(dq_hdu)

    fits.HDUList(hdulist).writeto(path, overwrite=True)
    return path


def _make_fringe_primary(path, data=None):
    # write a single-HDU (primary-only) fringe flat
    if data is None:
        data = np.full(SHAPE, FLAT_VALUE, dtype=np.float32)
    fits.HDUList([fits.PrimaryHDU(data=data)]).writeto(path, overwrite=True)
    return path


def _make_fringe_imset(path, sci_data=None, dq_data=None):
    # write a multi-HDU fringe flat with SCI + DQ imset.  B4 == False path.
    if sci_data is None:
        sci_data = np.full(SHAPE, FLAT_VALUE, dtype=np.float32)
    hdulist = [fits.PrimaryHDU()]
    sci_hdu = fits.ImageHDU(data=sci_data, name="SCI")
    sci_hdu.header["EXTVER"] = 1
    hdulist.append(sci_hdu)
    dq_hdu = fits.ImageHDU(data=dq_data, name="DQ")
    dq_hdu.header["EXTVER"] = 1
    hdulist.append(dq_hdu)
    fits.HDUList(hdulist).writeto(path, overwrite=True)
    return path


def _expected_output(science_path, suffix):
    d, f = os.path.split(science_path)
    root = f.rsplit("_", 1)[0]
    return os.path.join(d, root + suffix + ".fits")


class TestRawFileWarning:

    def test_raw_suffix_prints_warning(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "test_raw.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=False)
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "_raw.fits" in out

    def test_non_raw_suffix_no_warning(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "test_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=False)
        out = capsys.readouterr().out
        assert "Warning" not in out or "_raw.fits" not in out

class TestOutputSuffix:

    def test_g750l_produces_drj_suffix(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), opt_elem="G750L", include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert result.endswith("_drj.fits")
        assert os.path.exists(result)

    def test_g750m_produces_s2d_suffix(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), opt_elem="G750M", include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert result.endswith("_s2d.fits")
        assert os.path.exists(result)

    def test_other_grating_produces_drj_suffix(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), opt_elem="G430L", include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert result.endswith("_drj.fits")

class TestSameNameRuntimeError:

    def test_science_filename_equal_to_output_raises(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_drj.fits"), opt_elem="G750L", include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        with pytest.raises(RuntimeError, match="input and output file names cannot be the same"):
            defringe(sci, flt)

    def test_error_message_content(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_drj.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        with pytest.raises(RuntimeError) as exc_info:
            defringe(sci, flt)
        assert "same" in str(exc_info.value).lower()

class TestFringeFlatStructure:

    def test_primary_hdu_fringe_flat_verbose_prints_message(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert "primary HDU" in out

    def test_primary_hdu_fringe_flat_silent_when_not_verbose(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=False)
        out = capsys.readouterr().out
        assert "primary HDU" not in out

    def test_imset_fringe_flat_with_dq_verbose_prints_dq_message(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        dq_arr = np.zeros(SHAPE, dtype=np.int16)
        flt = _make_fringe_imset(str(tmp_path / "fringe.fits"), dq_data=dq_arr)
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert "DQ were read from the first imset" in out

    def test_imset_fringe_flat_no_dq_verbose_prints_no_dq_message(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_imset(str(tmp_path / "fringe.fits"), dq_data=None)
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert "first imset" in out
        assert "and DQ" not in out

    def test_imset_fringe_flat_silent_when_not_verbose(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_imset(str(tmp_path / "fringe.fits"), dq_data=np.zeros(SHAPE, dtype=np.int16))
        defringe(sci, flt, overwrite=True, verbose=False)
        out = capsys.readouterr().out
        assert "Imset 1 done" not in out

class TestFringeDataNone:

    def test_primary_hdu_with_no_data_raises(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = str(tmp_path / "empty_fringe.fits")
        fits.HDUList([fits.PrimaryHDU(data=None)]).writeto(flt, overwrite=True)
        with pytest.raises(RuntimeError, match="no data in the fringe flat"):
            defringe(sci, flt)

class TestFringeBadPixels:

    def _bad_fringe(self, tmp_path, zero_idx=(0, 0), neg_idx=(0, 1)):
        data = np.full(SHAPE, FLAT_VALUE, dtype=np.float32)
        data[zero_idx] = 0.0
        data[neg_idx]  = -1.0
        return _make_fringe_primary(str(tmp_path / "bad_fringe.fits"), data=data)

    def test_zero_pixel_not_used_for_division(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = self._bad_fringe(tmp_path)
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            assert h[("sci", 1)].data[0, 0] == pytest.approx(SCI_VALUE)

    def test_good_pixel_still_divided(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = self._bad_fringe(tmp_path)
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            assert h[("sci", 1)].data[1, 0] == pytest.approx(SCI_VALUE / FLAT_VALUE)

    def test_bad_pixel_flagged_with_512_in_dq(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = self._bad_fringe(tmp_path)
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            dq = h[("dq", 1)].data
        assert dq[0, 0] & 512
        assert dq[0, 1] & 512

    def test_good_pixel_dq_not_flagged(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = self._bad_fringe(tmp_path)
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            assert h[("dq", 1)].data[1, 0] == 0

    def test_bad_pixel_count_printed_when_verbose(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = self._bad_fringe(tmp_path)
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert "pixels in the fringe flat were less than or equal to 0" in out

    def test_bad_pixel_count_not_printed_when_not_verbose(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = self._bad_fringe(tmp_path)
        defringe(sci, flt, overwrite=True, verbose=False)
        out = capsys.readouterr().out
        assert "pixels in the fringe flat" not in out

    def test_all_good_pixels_no_masking_message(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert "pixels in the fringe flat" not in out

    def test_fringe_dq_sdqflag_extends_mask(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        fringe_dq = np.zeros(SHAPE, dtype=np.int16)
        fringe_dq[2, 2] = 4   # bit 4 is in sdqflags (= 4 + 8 + 512)
        flt = _make_fringe_imset(str(tmp_path / "fringe.fits"), dq_data=fringe_dq)
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            dq = h[("dq", 1)].data
        assert dq[2, 2] & 4

    def test_fringe_dq_non_sdqflag_does_not_extend_mask(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        fringe_dq = np.zeros(SHAPE, dtype=np.int16)
        fringe_dq[3, 3] = 1   # bit 1 is NOT in sdqflags
        flt = _make_fringe_imset(str(tmp_path / "fringe.fits"), dq_data=fringe_dq)
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            dq = h[("dq", 1)].data
        assert not (dq[3, 3] & 512)
        assert dq[3, 3] & 1

class TestExtverLoop:

    def test_hdu_without_extver_is_skipped(self, tmp_path):
        sci = str(tmp_path / "noextver_flt.fits")
        hdulist = [fits.PrimaryHDU()]
        hdulist[0].header["OPT_ELEM"] = "G750L"
        plain = fits.ImageHDU(data=np.ones(SHAPE, dtype=np.float32) * SCI_VALUE)
        hdulist.append(plain)
        fits.HDUList(hdulist).writeto(sci, overwrite=True)

        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert os.path.exists(result)

    def test_multiple_imsets_all_corrected(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), n_imsets=2, include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            assert h[("sci", 1)].data[0, 0] == pytest.approx(SCI_VALUE / FLAT_VALUE)
            assert h[("sci", 2)].data[0, 0] == pytest.approx(SCI_VALUE / FLAT_VALUE)

class TestMissingHduWarnings:

    def test_missing_sci_hdu_prints_warning(self, tmp_path, capsys):
        # build a science file with EXTVER but no SCI HDU
        sci = str(tmp_path / "nosci_flt.fits")
        hdulist = [fits.PrimaryHDU()]
        hdulist[0].header["OPT_ELEM"] = "G750L"
        err_hdu = fits.ImageHDU(data=np.full(SHAPE, ERR_VALUE, dtype=np.float32), name="ERR")
        err_hdu.header["EXTVER"] = 1
        dq_hdu = fits.ImageHDU(data=np.zeros(SHAPE, dtype=np.int16), name="DQ")
        dq_hdu.header["EXTVER"] = 1
        hdulist += [err_hdu, dq_hdu]
        fits.HDUList(hdulist).writeto(sci, overwrite=True)

        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert 'HDU ("SCI", 1) not found' in out

    def test_missing_dq_hdu_prints_warning(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_dq=False, include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        # add ERR back so science_err is not None and the ERR path is covered
        with fits.open(sci, mode="update") as h:
            err_hdu = fits.ImageHDU(
                data=np.full(SHAPE, ERR_VALUE, dtype=np.float32), name="ERR")
            err_hdu.header["EXTVER"] = 1
            h.append(err_hdu)

        with pytest.raises(KeyError):
            defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert 'HDU ("DQ", 1) not found' in out

    def test_missing_err_hdu_prints_warning(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert 'HDU ("ERR", 1) not found' in out

class TestSciErrDivision:

    def test_sci_data_divided_by_fringe_flat(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            assert h[("sci", 1)].data[0, 0] == pytest.approx(SCI_VALUE / FLAT_VALUE)

    def test_err_data_divided_by_fringe_flat(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=True)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            assert h[("err", 1)].data[0, 0] == pytest.approx(ERR_VALUE / FLAT_VALUE)

    def test_sci_none_not_divided(self, tmp_path, capsys):
        sci = str(tmp_path / "nosci_flt.fits")
        hdulist = [fits.PrimaryHDU()]
        hdulist[0].header["OPT_ELEM"] = "G750L"
        dq_hdu = fits.ImageHDU(data=np.zeros(SHAPE, dtype=np.int16), name="DQ")
        dq_hdu.header["EXTVER"] = 1
        hdulist.append(dq_hdu)
        fits.HDUList(hdulist).writeto(sci, overwrite=True)

        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert os.path.exists(result)

    def test_err_none_not_divided(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert os.path.exists(result)

class TestDqCreation:

    def test_missing_dq_hdu_creates_zeros_array(self, tmp_path):
        sci = _make_science_file(
            str(tmp_path / "obs_flt.fits"),
            include_dq=False,
            include_err=False
        )
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))

        with pytest.raises(KeyError):
            defringe(sci, flt, overwrite=True, verbose=True)

    def test_existing_dq_hdu_not_replaced_by_zeros(self, tmp_path):
        sci = str(tmp_path / "dq_flt.fits")
        hdulist = [fits.PrimaryHDU()]
        hdulist[0].header["OPT_ELEM"] = "G750L"
        sci_hdu = fits.ImageHDU(data=np.full(SHAPE, SCI_VALUE, dtype=np.float32), name="SCI")
        sci_hdu.header["EXTVER"] = 1
        hdulist.append(sci_hdu)
        dq_pre = np.zeros(SHAPE, dtype=np.int16)
        dq_pre[3, 3] = 16   # pre-existing DQ flag
        dq_hdu = fits.ImageHDU(data=dq_pre, name="DQ")
        dq_hdu.header["EXTVER"] = 1
        hdulist.append(dq_hdu)
        fits.HDUList(hdulist).writeto(sci, overwrite=True)

        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            assert h[("dq", 1)].data[3, 3] & 16

class TestFringeDqCombine:

    def test_fringe_dq_ored_into_science_dq(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        fringe_dq = np.zeros(SHAPE, dtype=np.int16)
        fringe_dq[1, 1] = 4   # sdqflag bit
        fringe_dq[2, 2] = 1   # non-sdqflag bit
        flt = _make_fringe_imset(str(tmp_path / "fringe.fits"), dq_data=fringe_dq)
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            dq = h[("dq", 1)].data
        assert dq[1, 1] & 4
        assert dq[2, 2] & 1

    def test_no_fringe_dq_science_dq_unchanged(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        with fits.open(result) as h:
            dq = h[("dq", 1)].data
        assert np.all(dq == 0)

class TestVerboseImsetMessage:

    def test_imset_done_printed_when_verbose(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), n_imsets=2, include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert "Imset" in out and "done" in out

    def test_imset_done_not_printed_when_not_verbose(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=False)
        out = capsys.readouterr().out
        assert "Imset" not in out

class TestOverwrite:

    def test_overwrite_true_removes_and_recreates_output(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)

        # write a sentinel so we can detect replacement
        sentinel_size = 42
        with open(result, "wb") as f:
            f.write(b"X" * sentinel_size)
        assert os.path.getsize(result) == sentinel_size

        result2 = defringe(sci, flt, overwrite=True, verbose=True)
        assert result2 == result
        assert os.path.getsize(result) != sentinel_size   # file was replaced

        out = capsys.readouterr().out
        assert "Removing and recreating" in out

    def test_overwrite_false_does_not_remove_existing_output(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)

        with pytest.raises(OSError):
            defringe(sci, flt, overwrite=False, verbose=False)

    def test_overwrite_true_no_existing_file_no_remove_message(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=True)
        out = capsys.readouterr().out
        assert "Removing" not in out

class TestReturnValue:

    def test_returns_string_path(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert isinstance(result, str)

    def test_returned_path_exists(self, tmp_path):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        result = defringe(sci, flt, overwrite=True, verbose=False)
        assert os.path.exists(result)

    def test_output_saved_message_printed(self, tmp_path, capsys):
        sci = _make_science_file(str(tmp_path / "obs_flt.fits"), include_err=False)
        flt = _make_fringe_primary(str(tmp_path / "fringe.fits"))
        defringe(sci, flt, overwrite=True, verbose=False)
        out = capsys.readouterr().out
        assert "Defringed science saved to" in out

class TestSdqflags:

    def test_sdqflags_value(self):
        assert sdqflags == 4 + 8 + 512

    def test_sdqflags_has_bit_4(self):
        assert sdqflags & 4

    def test_sdqflags_has_bit_8(self):
        assert sdqflags & 8

    def test_sdqflags_has_bit_512(self):
        assert sdqflags & 512

class TestParse:
    def test_parse_args(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "sci.fits", "flat.fits"])
        args = parse_args()

        assert args.science == "sci.fits"
        assert args.fringeflat == "flat.fits"