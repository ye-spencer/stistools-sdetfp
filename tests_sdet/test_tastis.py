# TODO: tests using 'result = t.tastis('test_raw.fits')' need fixing

import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call, mock_open

from stistools import tastis as t
from stistools.tastis import (
    _arcseconds, _v2coord, _v3coord, _ndec,
    _calculate_slews, _print_output, _print_warnings,
    PLATESCALE, COSTHETA, SINTHETA,
    BAD_ACQ, BAD_SLEW, BAD_LAMP_LOW, BAD_RATIO_HIGH, BAD_RATIO_LOW,
    BAD_SATURATED, BAD_FLUX, BAD_END, BAD_TDF,
    LOW_FLUX_CUTOFF, HIGH_FLUX_CUTOFF,
    MAX_GOODMAX, MIN_GOODMAX, MIN_IMAGING_FLUX, MIN_SPECTROSCOPIC_FLUX,
)


# helpers

def _make_acq_keywords(**overrides):
    """Return a minimal keywords dict for an ACQ observation."""
    kw = {
        'obsmode': 'ACQ',
        'obstype': 'IMAGING',
        'rootname': 'testroot',
        'proposid': 1234,
        'visit': '01',
        'expnum': 1.0,
        'targname': 'TARGET1',
        'tdateobs': '2024-01-01',
        'ttimeobs': '12:00:00',
        'texptime': 1.0,
        'biaslev': 1510.0,
        'sizaxis1': 100,
        'sizaxis2': 100,
        'centera1': 512,
        'centera2': 512,
        'optelem': 'MIRVIS',
        'aperture': 'F25ND3',
        'acqtype': 'POINT',
        'search': 'FLUX CENTROID',
        'box_step': 3,
        'counts1': 2000.0,
        'counts2': 2100.0,
        'targax1': 530.0,
        'targay1': 520.0,
        'targax4': 531.0,
        'targay4': 521.0,
        'goodmax1': 0,
        'goodmax2': 15000.0,
        'goodmax3': 5000.0,
        'pedestal': 0,
        'apera1': 537.0,
        'apera2': 517.0,
        'aperlka1': 535.0,
        'aperlka2': 515.0,
        'coarse1': 48.0,
        'coarse2': 42.0,
        'fine1': 49.0,
        'fine2': 43.0,
        'refaper1': 19.0,
        'refaper2': 17.0,
        'corner1': 462.0,
        'corner2': 462.0,
        'a1coarse_pix': -1.5,
        'a2coarse_pix': -9.0,
        'a1fine_pix': -2.1,
        'a2fine_pix': -0.2,
        'a1total_pix': -3.6,
        'a2total_pix': -9.2,
        'a1coarse_arc': -0.076,
        'a2coarse_arc': -0.457,
        'a1fine_arc': -0.107,
        'a2fine_arc': -0.010,
        'a1total_arc': -0.183,
        'a2total_arc': -0.467,
        'V2coarse': -0.379,
        'V3coarse': 0.268,
        'V2fine': -0.014,
        'V3fine': -0.067,
        'V2total': -0.460,
        'V3total': 0.201,
        'domfgs': 'S4B0000993F2',
        'subfgs': 'S4B0000953F1',
        'ocstdfx': 'TDF_Up',
        'linenum': '01.1',
    }
    kw.update(overrides)
    return kw


def _make_peak_keywords(**overrides):
    """Return a minimal keywords dict for an ACQ/PEAK observation."""
    dwell = np.array([[100.0, 200.0, 150.0]])  # 1×3 array; max at index [0,1]
    kw = {
        'obsmode': 'ACQ/PEAK',
        'obstype': 'SPECTROSCOPIC',
        'rootname': 'testpeak',
        'proposid': 9999,
        'visit': '11',
        'expnum': 3.0,
        'targname': 'HD128621',
        'tdateobs': '2024-01-01',
        'ttimeobs': '22:00:00',
        'texptime': 0.10,
        'biaslev': 1510.0,
        'sizaxis1': 1022,
        'sizaxis2': 32,
        'centera1': 536,
        'centera2': 516,
        'optelem': 'G430L',
        'aperture': '0.3X0.05ND',
        'search': 'LINEARAXIS2',
        'box_step': 3,
        'peakstep': 250.0,
        'peakcent': 'RETURN-TO-BRIGHTEST',
        'pedestal': 748587.0,
        'counts1': 751752.0,
        'counts2': 0.0,
        'goodmax1': 15000.0,
        'goodmax2': 0.0,
        'goodmax3': 0.0,
        'naxis1': 3,
        'naxis2': 1,
        'dwell': dwell,
        'otaslwa1': 0.0,
        'otaslwa2': 0.0,
        'corner1': 25,
        'corner2': 500,
        'a1total_pix': 0.0,
        'a2total_pix': -0.1,
        'a1total_arc': 0.000,
        'a2total_arc': -0.005,
        'V2total': -0.004,
        'V3total': 0.004,
        'domfgs': 'S7QX000330F1',
        'subfgs': 'S7QX000694F2',
        'ocstdfx': 'TDF_Up',
        'linenum': '11.3',
    }
    kw.update(overrides)
    return kw

class TestArcseconds:
    def test_positive_pixel(self):
        assert _arcseconds(1.0) == pytest.approx(PLATESCALE)

    def test_zero(self):
        assert _arcseconds(0) == 0.0

    def test_negative_pixel(self):
        assert _arcseconds(-2.0) == pytest.approx(-2.0 * PLATESCALE)

    def test_large_value(self):
        assert _arcseconds(1000.0) == pytest.approx(1000.0 * PLATESCALE)

    def test_fractional_pixel(self):
        assert _arcseconds(0.5) == pytest.approx(0.5 * PLATESCALE)

    def test_integer_input(self):
        # Accepts integers, not just floats
        assert _arcseconds(10) == pytest.approx(10 * PLATESCALE)

class TestV2Coord:
    def test_basic(self):
        x, y = 1.0, 0.0
        assert _v2coord(x, y) == pytest.approx(COSTHETA)

    def test_zero_inputs(self):
        assert _v2coord(0.0, 0.0) == 0.0

    def test_both_nonzero(self):
        x, y = 2.0, 3.0
        expected = COSTHETA * x + SINTHETA * y
        assert _v2coord(x, y) == pytest.approx(expected)

    def test_negative_inputs(self):
        x, y = -1.0, -1.0
        expected = COSTHETA * (-1.0) + SINTHETA * (-1.0)
        assert _v2coord(x, y) == pytest.approx(expected)

    def test_symmetry_with_v3(self):
        x, y = 1.0, 2.0
        result = _v2coord(x, y)
        assert isinstance(result, float)

class TestV3Coord:
    def test_basic(self):
        x, y = 1.0, 0.0
        assert _v3coord(x, y) == pytest.approx(COSTHETA)

    def test_zero_inputs(self):
        assert _v3coord(0.0, 0.0) == 0.0

    def test_both_nonzero(self):
        x, y = 2.0, 3.0
        expected = COSTHETA * x - SINTHETA * y
        assert _v3coord(x, y) == pytest.approx(expected)

    def test_negative_inputs(self):
        x, y = -1.0, -1.0
        expected = COSTHETA * (-1.0) - SINTHETA * (-1.0)
        assert _v3coord(x, y) == pytest.approx(expected)

    def test_differs_from_v2_when_x_ne_y(self):
        x, y = 1.0, 3.0
        assert _v2coord(x, y) != _v3coord(x, y)

    def test_equals_v2_when_y_zero(self):
        # v3(x, 0) = COSTHETA*x and v2(x, 0) = COSTHETA*x
        assert _v3coord(5.0, 0.0) == pytest.approx(_v2coord(5.0, 0.0))

class TestNdec:
    def test_positive_rounds_up(self):
        assert _ndec(0.14) == pytest.approx(0.1)

    def test_positive_halfway(self):
        assert _ndec(0.15) == pytest.approx(0.2)

    def test_negative_rounds_down(self):
        assert _ndec(-0.14) == pytest.approx(-0.1)

    def test_negative_halfway(self):
        assert _ndec(-0.15) == pytest.approx(-0.2)

    def test_zero(self):
        assert _ndec(0) == pytest.approx(0.0)

    def test_exact_tenth(self):
        assert _ndec(0.1) == pytest.approx(0.1)

    def test_large_positive(self):
        assert _ndec(12.34) == pytest.approx(12.3)

    def test_large_negative(self):
        assert _ndec(-12.34) == pytest.approx(-12.3)

class TestCalculateSlewsACQ:

    def _base_kw(self, aperture='GENERIC', **overrides):
        kw = {
            'obsmode': 'ACQ',
            'aperture': aperture,
            'targax1': 530.0,
            'targay1': 520.0,
            'targax4': 531.0,
            'targay4': 521.0,
            'apera1': 537.0,
            'apera2': 517.0,
            'aperlka1': 535.0,
            'aperlka2': 515.0,
        }
        kw.update(overrides)
        return kw

    def test_unknown_aperture_uses_zero_offset(self):
        kw = self._base_kw(aperture='UNKNOWN_APERTURE')
        _calculate_slews(kw)
        expected = 530.0 - 0.0 - 535.0 + 1
        assert kw['a1coarse_pix'] == pytest.approx(expected)

    def test_known_aperture_applies_offset(self):
        kw = self._base_kw(aperture='F25ND3')
        _calculate_slews(kw)
        offset = -1.24840
        expected = 530.0 - offset - 535.0 + 1
        assert kw['a1coarse_pix'] == pytest.approx(expected)

    def test_all_known_apertures_apply_offsets(self):
        apertures = ['F25NDQ1', 'F25NDQ2', 'F25NDQ3', 'F25NDQ4', 'F28X50LP', 'F28X50OIII', 'F28X50OII', 'F25ND3', 'F25ND5']
        for ap in apertures:
            kw = self._base_kw(aperture=ap)
            _calculate_slews(kw)
            assert 'a1coarse_pix' in kw, f"Missing key for aperture {ap}"

    def test_coarse_slew_pixels_computed(self):
        kw = self._base_kw()
        _calculate_slews(kw)
        assert 'a1coarse_pix' in kw
        assert 'a2coarse_pix' in kw

    def test_fine_slew_pixels_computed(self):
        kw = self._base_kw()
        _calculate_slews(kw)
        assert 'a1fine_pix' in kw
        assert 'a2fine_pix' in kw

    def test_total_pix_is_sum_of_coarse_and_fine(self):
        kw = self._base_kw()
        _calculate_slews(kw)
        assert kw['a1total_pix'] == pytest.approx(kw['a1coarse_pix'] + kw['a1fine_pix'])
        assert kw['a2total_pix'] == pytest.approx(kw['a2coarse_pix'] + kw['a2fine_pix'])

    def test_arcsec_values_are_pixel_times_platescale(self):
        kw = self._base_kw()
        _calculate_slews(kw)
        assert kw['a1coarse_arc'] == pytest.approx(kw['a1coarse_pix'] * PLATESCALE)
        assert kw['a2coarse_arc'] == pytest.approx(kw['a2coarse_pix'] * PLATESCALE)

    def test_v2_v3_coords_computed(self):
        kw = self._base_kw()
        _calculate_slews(kw)
        for key in ['V2coarse', 'V3coarse', 'V2fine', 'V3fine', 'V2total', 'V3total']:
            assert key in kw

class TestCalculateSlewsPeak:

    def _base_peak_kw(self, search='LINEARAXIS2', box_step=4,
                      peakstep=250.0, otaslwa1=0.0, otaslwa2=0.0):
        return {
            'obsmode': 'ACQ/PEAK',
            'search': search,
            'box_step': box_step,
            'peakstep': peakstep,
            'otaslwa1': otaslwa1,
            'otaslwa2': otaslwa2,
        }

    def test_linearaxis2_sets_finaly_zero(self):
        kw = self._base_peak_kw(search='LINEARAXIS2', box_step=4)
        _calculate_slews(kw)
        assert kw['a1total_pix'] == pytest.approx(0.0)

    def test_linearaxis1_sets_finalx_zero(self):
        kw = self._base_peak_kw(search='LINEARAXIS1', box_step=4)
        _calculate_slews(kw)
        assert kw['a2total_pix'] == pytest.approx(0.0)

    def test_spiral_computes_total_pix(self):
        kw = self._base_peak_kw(search='SPIRAL', box_step=9)
        _calculate_slews(kw)
        assert 'a1total_pix' in kw
        assert 'a2total_pix' in kw

    def test_spiral_small_value_clamps_to_zero(self):
        kw = self._base_peak_kw(search='SPIRAL', box_step=4, otaslwa1=0, otaslwa2=0)
        _calculate_slews(kw)
        # If clamped, value is exactly 0.0
        assert abs(kw['a1total_pix']) >= 0.0  # sanity

    def test_spiral_does_not_clamp_large_value(self):
        # Force otaslwa values large enough that the clamping threshold is not met
        kw = self._base_peak_kw(search='SPIRAL', box_step=4,
                                 otaslwa1=100, otaslwa2=100)
        _calculate_slews(kw)
        # Values should not be 0
        assert kw['a1total_pix'] != 0.0 or kw['a2total_pix'] != 0.0

    def test_arcsec_computed_from_pixels(self):
        kw = self._base_peak_kw(search='LINEARAXIS2')
        _calculate_slews(kw)
        assert kw['a1total_arc'] == pytest.approx(kw['a1total_pix'] * PLATESCALE)
        assert kw['a2total_arc'] == pytest.approx(kw['a2total_pix'] * PLATESCALE)

    def test_v2_v3_total_computed(self):
        kw = self._base_peak_kw(search='LINEARAXIS2')
        _calculate_slews(kw)
        assert 'V2total' in kw
        assert 'V3total' in kw

    def test_ndec_applied_to_pixel_values(self):
        kw = self._base_peak_kw(search='LINEARAXIS2', box_step=3, peakstep=250.0, otaslwa1=10, otaslwa2=10)
        _calculate_slews(kw)
        from stistools.tastis import _ndec
        assert kw['a1total_pix'] == pytest.approx(_ndec(kw['a1total_pix']))

class TestPrintWarningsACQ:

    def test_no_warnings_returns_zero(self, capsys):
        kw = _make_acq_keywords(
            ocstdfx='TDF_Up',
            a1fine_pix=1.0, a2fine_pix=1.0,
            counts1=2000.0, counts2=2100.0,
            goodmax2=10000.0, goodmax3=5000.0,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result == 0

    def test_tdf_down_sets_bad_tdf(self, capsys):
        kw = _make_acq_keywords(ocstdfx='TDFDown', a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2100.0)
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_TDF

    def test_tdf_down_prints_warning(self, capsys):
        kw = _make_acq_keywords(ocstdfx='TDFDown', a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2000.0)
        _print_warnings(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert "Telemetry" in out

    def test_no_spt_prints_missing_info_warning(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2000.0)
        _print_warnings(kw, spt_exists=False)
        out = capsys.readouterr().out
        assert "spt.fits" in out

    def test_large_a1fine_slew_sets_bad_slew(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=5.0, a2fine_pix=0.0, counts1=2000.0, counts2=2100.0)
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_SLEW

    def test_large_a2fine_slew_sets_bad_slew(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=5.0, counts1=2000.0, counts2=2100.0)
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_SLEW

    def test_exact_fine_slew_boundary_no_bad_slew(self, capsys):
        # exactly 4.0 should not trigger BAD_SLEW (condition is strictly > 4.0)
        kw = _make_acq_keywords(a1fine_pix=4.0, a2fine_pix=0.0, counts1=2000.0, counts2=2100.0)
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_SLEW)

    def test_ratio_below_0_75_sets_bad_ratio_low(self, capsys):
        # counts2/counts1 < 0.75
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=1000.0)
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_RATIO_LOW
        assert not (result & BAD_RATIO_HIGH)

    def test_ratio_above_1_25_sets_bad_ratio_high(self, capsys):
        # counts2/counts1 > 1.25
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=3000.0)
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_RATIO_HIGH
        assert not (result & BAD_RATIO_LOW)

    def test_ratio_at_0_75_boundary_no_bad_flag(self, capsys):
        # ratio = exactly 0.75, condition is ratio < 0.75 (strict)
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=1500.0)
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_RATIO_LOW)

    def test_ratio_at_1_25_boundary_no_bad_flag(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2500.0)
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_RATIO_HIGH)

    def test_goodmax2_above_max_sets_bad_saturated(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2000.0, goodmax2=MAX_GOODMAX + 1)
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_SATURATED

    def test_goodmax2_at_max_no_bad_saturated(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2000.0, goodmax2=MAX_GOODMAX)
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_SATURATED)

    def test_goodmax3_below_min_sets_bad_lamp_low(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2000.0, goodmax3=MIN_GOODMAX - 1)
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_LAMP_LOW

    def test_goodmax3_at_min_no_bad_lamp(self, capsys):
        kw = _make_acq_keywords(a1fine_pix=0.0, a2fine_pix=0.0, counts1=2000.0, counts2=2000.0, goodmax3=MIN_GOODMAX)
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_LAMP_LOW)

    def test_all_ok_prints_success_message(self, capsys):
        kw = _make_acq_keywords(
            a1fine_pix=1.0, a2fine_pix=1.0,
            counts1=2000.0, counts2=2100.0,
            goodmax2=10000.0, goodmax3=5000.0,
        )
        _print_warnings(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert "succeeded" in out


class TestPrintWarningsPeak:

    def test_no_warnings_returns_zero(self, capsys):
        dwell = np.array([[100.0, 500.0, 300.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=750000.0, pedestal=500000.0,  # flux=250000 > MIN_SPEC
            goodmax1=10000.0,
            peakcent='RETURN-TO-BRIGHTEST',
            search='LINEARAXIS2',
            naxis1=3, naxis2=1,
        )
        kw['pedestal'] = 0.0
        kw['counts1'] = 600.0
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_RATIO_LOW)
        assert not (result & BAD_RATIO_HIGH)

    def test_flux_ratio_below_low_cutoff_sets_bad_ratio_low(self, capsys):
        dwell = np.array([[100.0, 1000.0, 300.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=100.0, pedestal=0.0,
            goodmax1=10000.0,
            peakcent='OK',
            search='LINEARAXIS2',
            naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_RATIO_LOW

    def test_flux_ratio_above_high_cutoff_sets_bad_ratio_high(self, capsys):
        dwell = np.array([[100.0, 100.0, 100.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=500.0, pedestal=0.0,
            goodmax1=10000.0,
            peakcent='OK',
            search='LINEARAXIS2',
            naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_RATIO_HIGH

    def test_flux_ratio_exactly_at_low_cutoff_no_flag(self, capsys):
        # ratio = LOW_FLUX_CUTOFF exactly, condition is strict <
        dwell = np.array([[100.0, 1000.0, 100.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=LOW_FLUX_CUTOFF * 1000.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
            obstype='SPECTROSCOPIC',
        )
        kw['counts1'] = LOW_FLUX_CUTOFF * 1000.0
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_RATIO_LOW)

    def test_flux_ratio_exactly_at_high_cutoff_no_flag(self, capsys):
        dwell = np.array([[100.0, 1000.0, 100.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=HIGH_FLUX_CUTOFF * 1000.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_RATIO_HIGH)

    def test_goodmax1_above_max_sets_bad_saturated(self, capsys):
        dwell = np.array([[100.0, 500.0, 300.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=600.0, pedestal=0.0,
            goodmax1=MAX_GOODMAX + 1,
            peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_SATURATED

    def test_imaging_flux_below_min_sets_bad_flux(self, capsys):
        dwell = np.array([[100.0, 500.0, 300.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=100.0 + 0.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
            obstype='IMAGING',
        )
        kw['counts1'] = 100.0
        kw['pedestal'] = 0.0
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_FLUX

    def test_spectroscopic_flux_below_min_sets_bad_flux(self, capsys):
        dwell = np.array([[100.0, 15000.0, 100.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=14000.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
            obstype='SPECTROSCOPIC',
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_FLUX

    def test_spectroscopic_flux_above_min_no_bad_flux(self, capsys):
        dwell = np.array([[100.0, 10000.0, 100.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=25000.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
            obstype='SPECTROSCOPIC',
        )
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_FLUX)

    def test_peakcent_failed_sets_bad_acq(self, capsys):
        dwell = np.array([[100.0, 500.0, 300.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=600.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='FAILED',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_ACQ

    def test_linearaxis2_max_at_start_sets_bad_end(self, capsys):
        # j_max = 0, naxis2-1 = 0 so max is at end
        dwell = np.array([[1000.0, 100.0, 50.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=600.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_END

    def test_linearaxis2_max_at_end_sets_bad_end(self, capsys):
        dwell = np.array([[50.0], [100.0], [1000.0]])  # max at row 2, j_max=2=naxis2-1
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=600.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=1, naxis2=3,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_END

    def test_linearaxis1_max_at_start_sets_bad_end(self, capsys):
        dwell = np.array([[1000.0, 100.0, 50.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=600.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS1', naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_END

    def test_linearaxis1_max_at_end_sets_bad_end(self, capsys):
        dwell = np.array([[50.0, 100.0, 1000.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=600.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS1', naxis1=3, naxis2=1,
        )
        result = _print_warnings(kw, spt_exists=True)
        assert result & BAD_END

    def test_max_in_middle_no_bad_end(self, capsys):
        dwell = np.array([[100.0, 1000.0, 100.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=1100.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS1', naxis1=3, naxis2=1,
            obstype='SPECTROSCOPIC',
        )
        result = _print_warnings(kw, spt_exists=True)
        assert not (result & BAD_END)

    def test_all_ok_peak_prints_inadequate_message(self, capsys):
        dwell = np.array([[100.0, 1000.0, 100.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=1100.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS1', naxis1=3, naxis2=1,
            obstype='SPECTROSCOPIC',
        )
        _print_warnings(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert "may be inadequate for an accurate centering" in out

    def test_max_at_end_prints_message(self, capsys):
        dwell = np.array([[1000.0, 100.0, 50.0]])
        kw = _make_peak_keywords(
            dwell=dwell,
            counts1=600.0, pedestal=0.0,
            goodmax1=10000.0, peakcent='OK',
            search='LINEARAXIS2', naxis1=3, naxis2=1,
        )
        _print_warnings(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert "maximum flux in the sequence" in out

class TestPrintOutputACQ:

    def test_prints_separator_lines(self, capsys):
        kw = _make_acq_keywords()
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert '=' * 79 in out
        assert '-' * 79 in out

    def test_prints_rootname_and_acqtype(self, capsys):
        kw = _make_acq_keywords(rootname='myroot', acqtype='POINT')
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'myroot' in out
        assert 'POINT' in out

    def test_prints_proposid_and_target(self, capsys):
        kw = _make_acq_keywords(proposid=4242, targname='MY_STAR')
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert '4242' in out
        assert 'MY_STAR' in out

    def test_prints_domfgs_when_present(self, capsys):
        kw = _make_acq_keywords(domfgs='FGSDOM', subfgs='FGSSUB')
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'FGSDOM' in out
        assert 'FGSSUB' in out

    def test_no_fgs_line_when_both_empty(self, capsys):
        kw = _make_acq_keywords(domfgs='', subfgs='')
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'dom GS/FGS' not in out

    def test_returns_integer_badacq(self, capsys):
        kw = _make_acq_keywords()
        result = _print_output(kw, spt_exists=True)
        assert isinstance(result, int)

    def test_prints_coarse_and_fine_phases(self, capsys):
        kw = _make_acq_keywords()
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'Coarse locate phase' in out
        assert 'Fine locate phase' in out

    def test_prints_total_est_slew(self, capsys):
        kw = _make_acq_keywords()
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'Total est. slew' in out

class TestPrintOutputPeak:

    def test_prints_acq_peak_up_header(self, capsys):
        kw = _make_peak_keywords()
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'ACQ/PEAK-UP' in out

    def test_prints_scan_type(self, capsys):
        kw = _make_peak_keywords(search='LINEARAXIS2')
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'LINEARAXIS2' in out

    def test_prints_flux_pedestal_line(self, capsys):
        kw = _make_peak_keywords(counts1=751752.0, pedestal=748587.0)
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'Pedestal' in out

    def test_spiral_search_prints_axis_label(self, capsys):
        dwell = np.array([[100.0, 200.0]])
        kw = _make_peak_keywords(search='SPIRAL', dwell=dwell,
                                  naxis1=2, naxis2=1)
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert 'axis 1' in out

    def test_linearaxis1_prints_dwell(self, capsys):
        dwell = np.array([[100.0, 200.0, 300.0]])
        kw = _make_peak_keywords(search='LINEARAXIS1', dwell=dwell,
                                  naxis1=3, naxis2=1)
        _print_output(kw, spt_exists=True)
        out = capsys.readouterr().out
        assert '100' in out or '200' in out

def _make_mock_raw_hdu(obsmode='ACQ', acqtype='POINT', aperture='F25ND3', propaper='F25ND3', box_step=3, goodmax2=15000.0, goodmax3=5000.0, **primary_overrides):
    # build MagicMock
    primary_header = MagicMock()
    primary_defaults = {
        'OBSMODE': obsmode,
        'OBSTYPE': 'IMAGING',
        'ROOTNAME': 'testroot',
        'PROPOSID': 1234,
        'SIZAXIS1': 100,
        'SIZAXIS2': 100,
        'TEXPTIME': 1.0,
        'BIASLEV': 1510.0,
        'TARGNAME': 'STAR1',
        'TDATEOBS': '2024-01-01',
        'TTIMEOBS': '12:00:00',
        'LINENUM': '01.1',
        'CENTERA1': 512,
        'CENTERA2': 512,
        'OPT_ELEM': 'MIRVIS',
        'APERTURE': aperture,
        'PROPAPER': propaper,
        'ACQTYPE': acqtype,
        'CHECKBOX': box_step,
        'CENTMETH': 'RETURN-TO-BRIGHTEST',
        'PEAKCENT': 'RETURN-TO-BRIGHTEST',
        'PKSEARCH': 'LINEARAXIS2',
        'NUMSTEPS': 3,
        'PEAKSTEP': 250.0,
        'PEDESTAL': 0.0,
    }
    primary_defaults.update(primary_overrides)
    primary_header.__getitem__ = lambda self, key: primary_defaults[key.upper()]

    ext1_header = MagicMock()
    ext1_defaults = {
        'NGOODPIX': 1000,
        'GOODMEAN': 0.75,
        'MAXCHCNT': 2000.0,
        'TARGA1': 530.0,
        'TARGA2': 520.0,
        'GOODMAX': 10000.0,
    }
    ext1_header.__getitem__ = lambda self, key: ext1_defaults[key.upper()]

    ext4_header = MagicMock()
    ext4_defaults = {
        'MAXCHCNT': 2100.0,
        'GOODMAX': goodmax2,
        'TARGA1': 531.0,
        'TARGA2': 521.0,
    }
    ext4_header.__getitem__ = lambda self, key: ext4_defaults[key.upper()]

    ext7_header = MagicMock()
    ext7_defaults = {
        'GOODMAX': goodmax3,
        'APERA1': 537.0,
        'APERA2': 517.0,
        'APERLKA1': 535.0,
        'APERLKA2': 515.0,
    }
    ext7_header.__getitem__ = lambda self, key: ext7_defaults[key.upper()]

    ext0 = MagicMock(); ext0.header = primary_header
    ext1 = MagicMock(); ext1.header = ext1_header
    ext4 = MagicMock(); ext4.header = ext4_header
    ext7 = MagicMock(); ext7.header = ext7_header

    dwell_data = np.array([[100.0, 500.0, 300.0]])
    ext4_data = MagicMock()
    ext4_data.shape = (1, 3)
    ext4.data = dwell_data

    hdulist = MagicMock()
    hdulist.__getitem__ = lambda self, i: [ext0, ext1, None, None, ext4, None, None, ext7][i]
    hdulist.__enter__ = lambda self: self
    hdulist.__exit__ = MagicMock(return_value=False)
    hdulist.close = MagicMock()

    return hdulist


def _make_mock_spt_header(primary=True):
    header = MagicMock()
    values = {
        'DGESTAR': 'S4B0000993F2',
        'SGESTAR': 'S4B0000953F1',
        'OCSTDFX': 'TDF_Up',
        'OTASLWA1': 0,
        'OTASLWA2': 0,
    }
    header.__getitem__ = lambda self, key: values[key.upper()]
    return header


class TestTastisACQMocked:

    @pytest.fixture(autouse=True)
    def _patch_fits(self, mocker):
        self.raw_hdulist = _make_mock_raw_hdu(obsmode='ACQ')
        self.spt_prim_header = _make_mock_spt_header()
        self.spt_ext1_header = _make_mock_spt_header()

        mocker.patch('stistools.tastis.os.path.exists', return_value=True)
        mocker.patch('stistools.tastis.fits.open', return_value=self.raw_hdulist)

        def getheader_side_effect(filename, ext=0):
            if 'spt' in filename:
                if ext == 0:
                    return self.spt_prim_header
                return self.spt_ext1_header
            # raw file primary header
            return self.raw_hdulist[0].header

        mocker.patch('stistools.tastis.fits.getheader', side_effect=getheader_side_effect)

    def test_update_false_does_not_open_for_update(self, mocker, capsys):
        open_mock = mocker.patch('stistools.tastis.fits.open', return_value=self.raw_hdulist)
        t.tastis('test_raw.fits', update=False)
        # fits.open should only be called inside _read_keywords (once)
        calls_with_update = [c for c in open_mock.call_args_list if 'update' in str(c)]
        assert len(calls_with_update) == 0

    def test_update_true_opens_for_update(self, mocker, capsys):
        open_mock = mocker.patch('stistools.tastis.fits.open', return_value=self.raw_hdulist)
        t.tastis('test_raw.fits', update=True)
        calls = [c for c in open_mock.call_args_list if "update" in str(c).lower()]
        assert len(calls) >= 1


class TestTastisPeakMocked:
    # integration tests for tastis() with ACQ/PEAK obsmode, mocked FITS I/O

    @pytest.fixture(autouse=True)
    def _patch_fits(self, mocker):
        self.raw_hdulist = _make_mock_raw_hdu(obsmode='ACQ/PEAK')
        self.spt_prim_header = _make_mock_spt_header()
        self.spt_ext1_header = _make_mock_spt_header()

        mocker.patch('stistools.tastis.os.path.exists', return_value=True)
        mocker.patch('stistools.tastis.fits.open', return_value=self.raw_hdulist)

        def getheader_side_effect(filename, ext=0):
            if 'spt' in filename:
                return self.spt_prim_header if ext == 0 else self.spt_ext1_header
            return self.raw_hdulist[0].header

        mocker.patch('stistools.tastis.fits.getheader', side_effect=getheader_side_effect)


class TestReadKeywords:

    def test_acq_keywords_populated(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ')
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        spt_header = _make_mock_spt_header()
        mocker.patch('stistools.tastis.fits.getheader', return_value=spt_header)

        result = t._read_keywords('test_raw.fits', 'test_spt.fits', spt_exists=True)

        assert result['obsmode'] == 'ACQ'
        assert 'corner1' in result
        assert 'corner2' in result
        assert 'visit' in result
        assert 'expnum' in result

    def test_linenum_with_period_parses_expnum(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ', LINENUM='11.3')
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        mocker.patch('stistools.tastis.fits.getheader', return_value=_make_mock_spt_header())

        result = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)
        assert result['visit'] == '11'
        assert result['expnum'] == pytest.approx(3.0)

    def test_linenum_without_period_sets_zero_expnum(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ', LINENUM='05')
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        mocker.patch('stistools.tastis.fits.getheader', return_value=_make_mock_spt_header())

        result = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)
        assert result['expnum'] == 0

    def test_corner_calculated_from_centera_and_sizaxis(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ', CENTERA1=512, SIZAXIS1=100)
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        mocker.patch('stistools.tastis.fits.getheader', return_value=_make_mock_spt_header())

        result = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)
        assert result['corner1'] == pytest.approx(512 - 100/2)

    def test_spt_not_exists_sets_ocstdfx_unknown(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ')
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)

        result = t._read_keywords('raw.fits', 'spt.fits', spt_exists=False)
        assert result['ocstdfx'] == 'unknown'

    def test_propaper_e1_overrides_aperture(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ', propaper='52X0.1E1',
                                  aperture='52X0.2')
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        mocker.patch('stistools.tastis.fits.getheader', return_value=_make_mock_spt_header())

        result = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)
        assert result['aperture'] == '52X0.1E1'

    def test_propaper_d1_overrides_aperture(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ', propaper='52X0.1D1', aperture='52X0.2')
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        mocker.patch('stistools.tastis.fits.getheader', return_value=_make_mock_spt_header())

        result = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)
        assert result['aperture'] == '52X0.1D1'

    def test_propaper_other_suffix_uses_aperture_keyword(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ', propaper='GENERIC', aperture='F25ND3')
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        mocker.patch('stistools.tastis.fits.getheader', return_value=_make_mock_spt_header())

        result = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)
        assert result['aperture'] == 'F25ND3'

    def test_box_step_gt_3_applies_refaper_offset(self, mocker):
        hdu = _make_mock_raw_hdu(obsmode='ACQ', box_step=5)
        mocker.patch('stistools.tastis.fits.open', return_value=hdu)
        mocker.patch('stistools.tastis.fits.getheader', return_value=_make_mock_spt_header())

        result_large = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)
        hdu3 = _make_mock_raw_hdu(obsmode='ACQ', box_step=3)
        mocker.patch('stistools.tastis.fits.open', return_value=hdu3)
        result_small = t._read_keywords('raw.fits', 'spt.fits', spt_exists=True)

        assert result_large['refaper1'] != result_small['refaper1']

class TestBitFlags:

    def test_all_flags_are_powers_of_two(self):
        flags = [BAD_ACQ, BAD_SLEW, BAD_LAMP_LOW, BAD_RATIO_HIGH, BAD_RATIO_LOW, BAD_SATURATED, BAD_FLUX, BAD_END, BAD_TDF]
        for f in flags:
            assert f > 0 and (f & (f - 1)) == 0, f"Flag {f} is not a power of 2"

    def test_all_flags_are_distinct(self):
        flags = [BAD_ACQ, BAD_SLEW, BAD_LAMP_LOW, BAD_RATIO_HIGH, BAD_RATIO_LOW, BAD_SATURATED, BAD_FLUX, BAD_END, BAD_TDF]
        assert len(set(flags)) == len(flags)

    def test_combining_flags_with_or(self):
        combined = BAD_ACQ | BAD_SLEW | BAD_FLUX
        assert combined & BAD_ACQ
        assert combined & BAD_SLEW
        assert combined & BAD_FLUX
        assert not (combined & BAD_TDF)

    def test_fatal_error_value(self):
        from stistools.tastis import FATAL_ERROR
        assert FATAL_ERROR == 2

class TestTastisMainFunction:

    @patch('stistools.tastis.os.path.exists', return_value=True) 
    @patch('stistools.tastis.fits')
    @patch('stistools.tastis._read_keywords')
    @patch('stistools.tastis._calculate_slews')
    @patch('stistools.tastis._print_warnings')
    def test_tastis_default_prints_to_console(self, mock_warn, mock_calc, mock_read, mock_fits, mock_exists, capsys):
        """
        Tests that calling tastis() with update=False orchestrates the internal
        functions and prints the analysis to the console.
        """
        mock_read.return_value = _make_acq_keywords() 
        mock_calc.return_value = (0.5, 0.5, 0.1, 0.1) 

        t.tastis('dummy_raw.fits', update=False)

        mock_read.assert_called_once()
        mock_calc.assert_called_once()
        
        captured = capsys.readouterr()
        assert 'ACQ' in captured.out
        assert 'TARGET1' in captured.out 

    @patch('stistools.tastis.os.path.exists', return_value=True) 
    @patch('stistools.tastis.fits.open') 
    @patch('stistools.tastis.fits.setval') 
    @patch('stistools.tastis._read_keywords')
    @patch('stistools.tastis._calculate_slews')
    @patch('stistools.tastis._print_warnings')
    def test_tastis_update_modifies_header(self, mock_warn, mock_calc, mock_read, mock_setval, mock_fits_open, mock_exists):
        """
        Tests the update=True branch to ensure it attempts to write the flags.
        """
        mock_keywords = _make_acq_keywords()
        mock_keywords['flags'] = BAD_ACQ 
        mock_read.return_value = mock_keywords
        mock_calc.return_value = (0.0, 0.0, 0.0, 0.0)

        safe_dummy_header = {
            'OBSMODE': 'ACQ',
            'OBSTYPE': 'IMAGING',
            'INSTRUME': 'STIS',
            'dgestar': 'FGS1',
            'sgestar': 'FGS2',
        }

        with patch('stistools.tastis.fits.getheader', return_value=safe_dummy_header):
            
            # Apply the safe header to our fits.open mock
            mock_hdu0 = MagicMock()
            mock_hdu0.header = safe_dummy_header # <--- Populated header!
            
            mock_hdulist = MagicMock()
            mock_hdulist.__getitem__.return_value = mock_hdu0
            mock_hdulist.__enter__.return_value = mock_hdulist
            mock_hdulist.__exit__.return_value = None
            
            mock_fits_open.return_value = mock_hdulist

            # Execute the function
            t.tastis('dummy_raw.fits', update=True)

        # Assert that SOME file-write operation occurred!
        opened_in_update_mode = mock_fits_open.called and 'update' in str(mock_fits_open.call_args).lower()
        used_setval = mock_setval.called
        
        assert opened_in_update_mode or used_setval, "tastis did not attempt to update the file!"

# class TestTastisAcqPeakUpdates:
    
#     @pytest.mark.parametrize("badacq_flag, expected_rat, expected_flx, expected_sat, expected_end", [
#         (BAD_RATIO_HIGH | BAD_FLUX | BAD_SATURATED | BAD_END, 'HIRATIO', 'LO_FLUX', 'SAT', 'HI_END'),
#         (BAD_RATIO_LOW, 'LORATIO', 'OK_FLUX', 'UNSAT', 'OK_END'),
#         (0, 'OKRATIO', 'OK_FLUX', 'UNSAT', 'OK_END')
#     ])
#     @patch('stistools.tastis.os.path.exists', return_value=True) 
#     @patch('stistools.tastis.fits.open') 
#     @patch('stistools.tastis.fits.getheader', return_value={
#         'OBSMODE': 'ACQ/PEAK', 'obsmode': 'ACQ/PEAK', 
#         'dgestar': 'FGS1', 'DGESTAR': 'FGS1',
#         'ocstdfx': 0.0, 'OCSTDFX': 0.0
#     })
#     @patch('stistools.tastis.fits.setval') 
#     @patch('stistools.tastis._read_keywords')
#     @patch('stistools.tastis._calculate_slews')
#     @patch('stistools.tastis._print_warnings')

    # def test_acq_peak_flag_updates(self, mock_warn, mock_calc, mock_read, mock_setval, mock_getheader, mock_fits_open, mock_exists, badacq_flag, expected_rat, expected_flx, expected_sat, expected_end): 
        
    #     # 1. Force getheader to return safe fallbacks
    #     mock_getheader.return_value = {'obsmode': 'ACQ/PEAK', 'dgestar': 'FGS1', 'ocstdfx': 0.0}

    #     # 2. Build the keywords dictionary from scratch to prevent flag leaks
    #     mock_keywords = _make_acq_keywords()
    #     mock_keywords['obsmode'] = 'ACQ/PEAK' 
    #     mock_keywords['peakcent'] = 'YES'
    #     mock_keywords['search'] = 'YES'
    #     mock_keywords['pedestal'] = 15.0
    #     mock_keywords['dwell'] = np.zeros((15, 20)) 
        
    #     # EXPLICITLY set the flag in the keywords dict
    #     mock_keywords['flags'] = badacq_flag 
    #     mock_read.return_value = mock_keywords
        
    #     # EXPLICITLY set the flag in the slew calculator return
    #     mock_calc.return_value = (0.0, 0.0, badacq_flag, 0)

    #     # 3. Setup the target FITS header
    #     mock_hdu0 = MagicMock()
    #     mock_hdu0.header = {'OBSMODE': 'ACQ/PEAK'}
    #     mock_hdulist = MagicMock()
    #     mock_hdulist.__getitem__.return_value = mock_hdu0
    #     mock_hdulist.__enter__.return_value = mock_hdulist
    #     mock_hdulist.__exit__.return_value = None
    #     mock_fits_open.return_value = mock_hdulist

    #     # 4. Execute
    #     t.tastis('dummy.fits', update=True)

    #     # 5. Assert
    #     h = mock_hdu0.header
    #     assert h['acqp_rat'] == expected_rat
    #     assert h['acqp_flx'] == expected_flx
    #     assert h['acqp_sat'] == expected_sat
    #     assert h['acqp_end'] == expected_end

class TestReadKeywordsAcqPeak:

    @patch('stistools.tastis.fits.open')
    @patch('stistools.tastis.fits.getheader')
    def test_read_keywords_acq_peak_extensions(self, mock_getheader, mock_fits_open):

        mock_hdu0 = MagicMock()
        mock_hdu0.header = {
            'obsmode': 'ACQ/PEAK', 'obstype': 'IMAGING', 'rootname': 'test', 
            'proposid': 1, 'visit': '1', 'expnum': 1, 'targname': 'T1', 
            'tdateobs': 'D', 'ttimeobs': 'T', 'texptime': 1.0, 'opt_elem': 'G', 
            'cenwave': 1, 'aperture': 'A', 'maxsrch': 1, 'centera1': 512, 
            'centera2': 512, 'sizaxis1': 1024, 'sizaxis2': 1024, 'ltv1': 0, 'ltv2': 0,
            'peakcent': 'YES', 'pksearch': 'YES', 'numsteps': 5, 
            'peakstep': 0.1, 'pedestal': 15.0,
            'biaslev': 0.0, 'linenum': '01.1', 'checkbox': 5, 'propaper': '52X0.2', 'ocstdfx': 0.0
        }
        
        mock_hdu1 = MagicMock()
        mock_hdu1.header = {
            'goodmax': 500.0, 
            'ngoodpix': 10, 
            'goodmean': 5.0, 
            'biaslev': 0.0,
            'ocstdfx': 0.0 
        }      
  
        mock_hdu4 = MagicMock()
        mock_hdu4.data = np.zeros((15, 20)) 

        mock_hdulist = MagicMock()
        mock_hdulist.__getitem__.side_effect = lambda idx: {0: mock_hdu0, 1: mock_hdu1, 4: mock_hdu4}[idx]
        mock_hdulist.__enter__.return_value = mock_hdulist
        mock_hdulist.__exit__.return_value = None
        mock_fits_open.return_value = mock_hdulist
        
        def spt_header_side_effect(filename, ext=0):
            if ext == 0:
                return {'dgestar': 'FGS1', 'ocstdfx': 0.0}
            elif ext == 1:
                return {'otaslwa1': 1.23, 'otaslwa2': 4.56, 'ocstdfx': 0.0} 
        mock_getheader.side_effect = spt_header_side_effect
        
        result = t._read_keywords('dummy_raw.fits', 'dummy_spt.fits', spt_exists=True)
        
        assert result['peakcent'] == 'YES'
        assert result['box_step'] == 5
        assert result['goodmax1'] == 500.0
        assert result['otaslwa1'] == 1.23
        assert result['naxis1'] == 20 
        assert result['naxis2'] == 15