import numpy as np
import pytest
from astropy.io import fits

from stistools.ctestis import ctestis

REF = dict(ycol=182.0, net=5000.0, sky=150.0, mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D')

REF_FLUXC =  2536.7134   # ± 0.01
REF_DMAGC = -0.01582849  # ± 1e-6
REF_DYC   =  0.00707993  # ± 1e-6


def _make_fits(path, mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D'):
    h0 = fits.PrimaryHDU()
    h0.header['TEXPSTRT'] = mjd
    h0.header['BINAXIS2'] = ybin
    h0.header['CCDGAIN']  = gain
    h0.header['CCDAMP']   = amp
    h1 = fits.ImageHDU()
    h1.header['NCOMBINE'] = nread
    fits.HDUList([h0, h1]).writeto(str(path), overwrite=True)
    return str(path)

class TestReturnStructure:

    def test_returns_three_tuple(self):
        result = ctestis(**REF)
        assert len(result) == 3

    def test_scalar_input_returns_scalar_outputs(self):
        fluxc, dmagc, dyc = ctestis(**REF)
        assert np.ndim(fluxc) == 0
        assert np.ndim(dmagc) == 0
        assert np.ndim(dyc)   == 0

    def test_array_input_returns_array_outputs(self):
        fluxc, dmagc, dyc = ctestis(
            ycol=np.array([182., 200.]),
            net =np.array([5000., 3000.]),
            sky =np.array([150., 100.]),
            mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D')
        assert fluxc.shape == (2,)
        assert dmagc.shape == (2,)
        assert dyc.shape   == (2,)

    def test_list_input_converted_to_array(self):
        """np.asarray coercion: list inputs should work identically to arrays."""
        fluxc_list, dmagc_list, dyc_list = ctestis(
            ycol=[182.], net=[5000.], sky=[150.],
            mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D')
        fluxc_ref, dmagc_ref, dyc_ref = ctestis(**REF)
        assert fluxc_list[0] == pytest.approx(fluxc_ref, rel=1e-5)

class TestReferenceValues:

    def test_fluxc_reference(self):
        fluxc, _, _ = ctestis(**REF)
        assert fluxc == pytest.approx(REF_FLUXC, rel=1e-4)

    def test_dmagc_reference(self):
        _, dmagc, _ = ctestis(**REF)
        assert dmagc == pytest.approx(REF_DMAGC, rel=1e-4)

    def test_dyc_reference(self):
        _, _, dyc = ctestis(**REF)
        assert dyc == pytest.approx(REF_DYC, rel=1e-4)

    def test_fluxc_greater_than_net_over_nread(self):
        fluxc, _, _ = ctestis(**REF)
        cts = REF['net'] * REF['gain'] / REF['nread']
        assert fluxc > cts

    def test_dmagc_is_negative(self):
        _, dmagc, _ = ctestis(**REF)
        assert dmagc < 0

    def test_dyc_is_positive_for_amp_d(self):
        _, _, dyc = ctestis(**REF)
        assert dyc > 0

class TestStisimage:

    def test_stisimage_matches_manual_params(self, tmp_path):
        path = _make_fits(tmp_path / "obs_crj.fits")
        fluxc_fits, dmagc_fits, dyc_fits = ctestis(182., 5000., 150., stisimage=path)
        fluxc_man, dmagc_man, dyc_man = ctestis(**REF)
        assert fluxc_fits == pytest.approx(fluxc_man, rel=1e-6)
        assert dmagc_fits == pytest.approx(dmagc_man, rel=1e-6)
        assert dyc_fits   == pytest.approx(dyc_man,   rel=1e-6)

    def test_stisimage_reads_mjd_from_header(self, tmp_path):
        path_a = _make_fits(tmp_path / "a_crj.fits", mjd=50893.30)
        path_b = _make_fits(tmp_path / "b_crj.fits", mjd=52000.00)
        fa, _, _ = ctestis(182., 5000., 150., stisimage=str(path_a))
        fb, _, _ = ctestis(182., 5000., 150., stisimage=str(path_b))
        assert fa != pytest.approx(fb, rel=1e-3)

    def test_stisimage_reads_nread_from_header(self, tmp_path):
        path_a = _make_fits(tmp_path / "a_crj.fits", nread=2)
        path_b = _make_fits(tmp_path / "b_crj.fits", nread=4)
        fa, _, _ = ctestis(182., 5000., 150., stisimage=str(path_a))
        fb, _, _ = ctestis(182., 5000., 150., stisimage=str(path_b))
        assert fa != pytest.approx(fb, rel=1e-3)

    def test_stisimage_reads_ybin_from_header(self, tmp_path):
        path_a = _make_fits(tmp_path / "a_crj.fits", ybin=1)
        path_b = _make_fits(tmp_path / "b_crj.fits", ybin=2)
        fa, _, _ = ctestis(182., 5000., 150., stisimage=str(path_a))
        fb, _, _ = ctestis(182., 5000., 150., stisimage=str(path_b))
        assert fa != pytest.approx(fb, rel=1e-3)

    def test_stisimage_reads_amp_from_header(self, tmp_path):
        path_d = _make_fits(tmp_path / "d_crj.fits", amp='D')
        path_a = _make_fits(tmp_path / "a_crj.fits", amp='A')
        fd, _, _ = ctestis(182., 5000., 150., stisimage=str(path_d))
        fa, _, _ = ctestis(182., 5000., 150., stisimage=str(path_a))
        assert fd != pytest.approx(fa, rel=1e-3)

    def test_stisimage_amp_stripped_and_uppercased(self, tmp_path):
        h0 = fits.PrimaryHDU()
        h0.header['TEXPSTRT'] = 50893.30
        h0.header['BINAXIS2'] = 1
        h0.header['CCDGAIN'] = 1.0
        h0.header['CCDAMP'] = ' d '  
        h1 = fits.ImageHDU(); h1.header['NCOMBINE'] = 2
        path = str(tmp_path / "pad_crj.fits")
        fits.HDUList([h0, h1]).writeto(path, overwrite=True)
        fluxc_pad, _, _ = ctestis(182., 5000., 150., stisimage=path)
        fluxc_ref, _, _ = ctestis(**REF)
        assert fluxc_pad == pytest.approx(fluxc_ref, rel=1e-6)

    def test_stisimage_gain_4_replaced_with_4_08(self, tmp_path):
        path_4   = _make_fits(tmp_path / "g4_crj.fits",  gain=4.0)
        path_408 = _make_fits(tmp_path / "g408_crj.fits", gain=4.08)
        f4, _, _ = ctestis(182., 5000., 150., stisimage=str(path_4))
        f408, _, _ = ctestis(182., 5000., 150., stisimage=str(path_408))
        assert f4 == pytest.approx(f408, rel=1e-6)

    def test_stisimage_overrides_manual_mjd(self, tmp_path):
        path = _make_fits(tmp_path / "obs_crj.fits", mjd=50893.30)
        f_file, _, _ = ctestis(182., 5000., 150., stisimage=str(path), mjd=99999.0)
        f_manual, _, _ = ctestis(**REF)
        assert f_file == pytest.approx(f_manual, rel=1e-6)

class TestMissingParams:

    @pytest.mark.parametrize("missing_param", ["mjd", "nread", "ybin", "gain"])
    def test_missing_param_raises_value_error(self, missing_param):
        params = dict(REF)
        params[missing_param] = None
        with pytest.raises(ValueError, match=missing_param):
            ctestis(**params)

    def test_error_message_names_missing_param(self):
        with pytest.raises(ValueError, match="mjd"):
            ctestis(182., 5000., 150., mjd=None, nread=2, ybin=1, gain=1.0)

    def test_all_params_present_no_error(self):
        fluxc, dmagc, dyc = ctestis(**REF)
        assert fluxc > 0

class TestInvalidAmp:

    def test_invalid_amp_raises_value_error(self):
        with pytest.raises(ValueError, match="Amplifier"):
            ctestis(**dict(REF, amp='Z'))

    def test_error_message_contains_bad_amp(self):
        with pytest.raises(ValueError, match="E"):
            ctestis(**dict(REF, amp='E'))

    @pytest.mark.parametrize("valid_amp", ['A', 'B', 'C', 'D'])
    def test_all_valid_amps_no_error(self, valid_amp):
        fluxc, _, _ = ctestis(**dict(REF, amp=valid_amp))
        assert fluxc > 0

    def test_lowercase_amp_uppercased_before_check(self):
        fluxc, _, _ = ctestis(**dict(REF, amp='a'))
        assert fluxc > 0

    def test_lowercase_invalid_amp_also_raises(self):
        with pytest.raises(ValueError):
            ctestis(**dict(REF, amp='z'))

class TestGain4Replacement:

    def test_gain_4_produces_same_result_as_gain_4_08(self):
        f4,   d4,   y4   = ctestis(**dict(REF, gain=4.0))
        f408, d408, y408 = ctestis(**dict(REF, gain=4.08))
        assert f4 == pytest.approx(f408,  rel=1e-6)
        assert d4 == pytest.approx(d408,  rel=1e-6)
        assert y4 == pytest.approx(y408,  rel=1e-6)

    def test_gain_4_differs_from_gain_1(self):
        f4,  _, _ = ctestis(**dict(REF, gain=4.0))
        f1,  _, _ = ctestis(**dict(REF, gain=1.0))
        assert f4 != pytest.approx(f1, rel=1e-3)

    def test_gain_4_prints_arrow_in_no_stisimage_path(self, capsys):
        ctestis(**dict(REF, gain=4.0))
        out = capsys.readouterr().out
        assert "--> 4.08" in out

    def test_gain_not_4_prints_no_arrow(self, capsys):
        ctestis(**dict(REF, gain=1.0))
        out = capsys.readouterr().out
        assert "--> 4.08" not in out

    def test_gain_other_value_not_replaced(self):
        f2, _, _ = ctestis(**dict(REF, gain=2.0))
        f1, _, _ = ctestis(**dict(REF, gain=1.0))
        assert f2 != pytest.approx(f1, rel=1e-3)

class TestSx2Buffer:

    def test_sx2_true_removes_38_rows_from_ycol(self):
        f_sx2,   _, _ = ctestis(**dict(REF, sx2=True))
        f_minus, _, _ = ctestis(**dict(REF, ycol=REF['ycol'] - 38))
        assert f_sx2 == pytest.approx(f_minus, rel=1e-6)

    def test_sx2_false_no_buffer_removal(self):
        f_sx2,  _, _ = ctestis(**dict(REF, sx2=True))
        f_norm, _, _ = ctestis(**dict(REF, sx2=False))
        assert f_sx2 != pytest.approx(f_norm, rel=1e-4)

    def test_sx2_in_stisimage_name_removes_38_rows(self, tmp_path):
        path = _make_fits(tmp_path / "obs_sx2.fits")
        f_sx2, _, _ = ctestis(182., 5000., 150., stisimage=str(path))
        f_ref, _, _ = ctestis(144., 5000., 150.,mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D')
        assert f_sx2 == pytest.approx(f_ref, rel=1e-6)

    def test_no_stisimage_name_no_buffer_removal(self, tmp_path):
        import tempfile, os
        td = tempfile.mkdtemp(prefix='ctestis_normal_')
        path = _make_fits(os.path.join(td, 'obs_crj.fits'))
        assert '_sx2' not in path, f"Test path must not contain '_sx2', got: {path}"
        f_crj, _, _ = ctestis(182., 5000., 150., stisimage=path)
        f_ref, _, _ = ctestis(**REF)
        assert f_crj == pytest.approx(f_ref, rel=1e-6)

    def test_sx2_true_and_sx2_in_filename_consistent(self, tmp_path):
        """Both sx2=True and '_sx2' in name produce the same result."""
        path = _make_fits(tmp_path / "obs_sx2.fits")
        f_name, _, _ = ctestis(182., 5000., 150., stisimage=str(path))
        f_flag, _, _ = ctestis(**dict(REF, sx2=True))
        assert f_name == pytest.approx(f_flag, rel=1e-6)

class TestAmpYcolDirection:

    def test_amp_b_and_d_produce_same_result(self):
        fb, _, _ = ctestis(**dict(REF, amp='B'))
        fd, _, _ = ctestis(**dict(REF, amp='D'))
        assert fb == pytest.approx(fd, rel=1e-6)

    def test_amp_a_and_c_produce_same_result(self):
        fa, _, _ = ctestis(**dict(REF, amp='A'))
        fc, _, _ = ctestis(**dict(REF, amp='C'))
        assert fa == pytest.approx(fc, rel=1e-6)

    def test_amp_d_differs_from_amp_a(self):
        fa, _, _ = ctestis(**dict(REF, amp='A'))
        fd, _, _ = ctestis(**dict(REF, amp='D'))
        assert fa != pytest.approx(fd, rel=1e-3)

    def test_amp_a_ycol_dir_uses_1024_formula(self):
        fa, _, _ = ctestis(**dict(REF, amp='A'))
        fd_eq, _, _ = ctestis(**dict(REF, amp='D', ycol=1024. - 182.))
        assert fa == pytest.approx(fd_eq, rel=1e-6)

    def test_amp_c_ycol_dir_matches_amp_a(self):
        fa, da, ya = ctestis(**dict(REF, amp='A'))
        fc, dc, yc = ctestis(**dict(REF, amp='C'))
        assert fa == pytest.approx(fc, rel=1e-6)
        assert da == pytest.approx(dc, rel=1e-6)
        assert ya == pytest.approx(yc, rel=1e-6)

    def test_amp_b_ycol_dir_matches_amp_d(self):
        fb, db, yb = ctestis(**dict(REF, amp='B'))
        fd, dd, yd = ctestis(**dict(REF, amp='D'))
        assert fb == pytest.approx(fd, rel=1e-6)

class TestPrintBranch:

    def test_scalar_input_triggers_scalar_print_format(self, capsys):
        ctestis(**REF)
        out = capsys.readouterr().out
        assert "lcts: -0.67595399" in out

    def test_array_input_triggers_array_print_format(self, capsys):
        ctestis(
            ycol=np.array([182., 200.]),
            net =np.array([5000., 3000.]),
            sky =np.array([150., 100.]),
            mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D')
        out = capsys.readouterr().out
        assert "[" in out
        assert "75." in out

    def test_scalar_print_contains_net_sky_ycol_summary(self, capsys):
        ctestis(**REF)
        out = capsys.readouterr().out
        assert "net:" in out
        assert "sky:" in out
        assert "ycol:" in out
        assert "fluxc:" in out
        assert "dmagc:" in out
        assert "dyc:" in out

    def test_array_print_contains_net_sky_ycol_summary(self, capsys):
        ctestis(
            ycol=np.array([182.]),
            net =np.array([5000.]),
            sky =np.array([150.]),
            mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D')
        out = capsys.readouterr().out
        assert "net:" in out
        assert "fluxc:" in out

class TestPrintedOutput:

    def test_prints_mjd_nread_ybin_gain_amp_block(self, capsys):
        ctestis(**REF)
        out = capsys.readouterr().out
        assert "mjd:" in out
        assert "nread:" in out
        assert "ybin:" in out
        assert "gain:" in out
        assert "amp:" in out

    def test_prints_param_summary_for_manual_mode(self, capsys):
        ctestis(**REF)
        out = capsys.readouterr().out
        assert "MJD" in out
        assert "NREAD" in out
        assert "YBIN" in out
        assert "GAIN" in out

    def test_stisimage_does_not_print_param_summary(self, tmp_path, capsys):
        path = _make_fits(tmp_path / "obs_crj.fits")
        ctestis(182., 5000., 150., stisimage=str(path))
        out = capsys.readouterr().out
        assert "MJD   = " not in out

    def test_tt0_printed_in_output(self, capsys):
        ctestis(**REF)
        out = capsys.readouterr().out
        assert "tt0:" in out

    def test_cti_printed_in_output(self, capsys):
        ctestis(**REF)
        out = capsys.readouterr().out
        assert "cti:" in out

class TestClamping:

    def test_net_zero_clamps_cts_to_1(self):
        """np.maximum(net*gain/nread, 1): net=0 → cts=1, not 0."""
        fluxc, _, _ = ctestis(**dict(REF, net=0.))
        # cts=1 → lcts = log(1) - 8.5 = -8.5 (very negative → high cti)
        assert fluxc > 0

    def test_net_zero_and_positive_give_different_results(self):
        f0, _, _ = ctestis(**dict(REF, net=0.))
        f5, _, _ = ctestis(**dict(REF, net=5000.))
        assert f0 != pytest.approx(f5, rel=1e-3)

    def test_negative_sky_clamped_to_zero_same_as_zero_sky(self):
        """np.maximum(sky*gain/nread, 0): sky<0 → bck=0, same as sky=0."""
        f_neg, _, _ = ctestis(**dict(REF, sky=-100.))
        f_zero, _, _ = ctestis(**dict(REF, sky=0.))
        assert f_neg == pytest.approx(f_zero, rel=1e-6)

    def test_sky_zero_differs_from_sky_positive(self):
        f_zero, _, _ = ctestis(**dict(REF, sky=0.))
        f_pos,  _, _ = ctestis(**dict(REF, sky=150.))
        assert f_zero != pytest.approx(f_pos, rel=1e-3)

class TestEquationProperties:

    def test_lower_ycol_means_larger_correction(self):
        f_low,  _, _ = ctestis(**dict(REF, ycol=100.))   # far from readout
        f_high, _, _ = ctestis(**dict(REF, ycol=900.))   # near readout
        assert f_low > f_high

    def test_higher_net_counts_means_smaller_relative_correction(self):
        f_bright, d_bright, _ = ctestis(**dict(REF, net=50000.))
        f_faint,  d_faint,  _ = ctestis(**dict(REF, net=500.))
        assert d_bright > d_faint

    def test_later_mjd_larger_correction(self):
        f_early, _, _ = ctestis(**dict(REF, mjd=50000.))
        f_late,  _, _ = ctestis(**dict(REF, mjd=54000.))
        assert f_late > f_early

    def test_fluxc_always_positive(self):
        for net in [10., 500., 5000., 50000.]:
            fluxc, _, _ = ctestis(**dict(REF, net=net))
            assert fluxc > 0, f"fluxc not positive for net={net}"

    def test_dyc_scales_with_distance_from_centre(self):
        _, _, dyc_512 = ctestis(**dict(REF, ycol=512.))
        _, _, dyc_0   = ctestis(**dict(REF, ycol=0.))
        assert dyc_0 == pytest.approx(2 * dyc_512, rel=1e-5)

    def test_array_results_match_scalar_results(self):
        f_arr, d_arr, y_arr = ctestis(
            ycol=np.array([182.]),
            net =np.array([5000.]),
            sky =np.array([150.]),
            mjd=50893.30, nread=2, ybin=1, gain=1.0, amp='D')
        f_sc, d_sc, y_sc = ctestis(**REF)
        assert f_arr[0] == pytest.approx(f_sc, rel=1e-6)
        assert d_arr[0] == pytest.approx(d_sc, rel=1e-6)
        assert y_arr[0] == pytest.approx(y_sc, rel=1e-6)

    def test_nread_greater_means_larger_fractional_correction(self):
        _, d_n1, _ = ctestis(**dict(REF, nread=1))
        _, d_n4, _ = ctestis(**dict(REF, nread=4))
        assert d_n4 < d_n1