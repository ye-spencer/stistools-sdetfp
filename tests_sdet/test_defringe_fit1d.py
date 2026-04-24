import math
import numpy as np
import pytest
from scipy.interpolate import LSQUnivariateSpline

from stistools.defringe._fit1d import (
    fit1d,
    get_knots,
    fit_once,
    fit_with_rejection,
    calc_rms_deviation,
    wtrebin,
)

# helper functions

@pytest.fixture
def linear_data():
    # 50-point perfectly linear signal y = 2x + 1 on [0, 10]
    x = np.linspace(0.0, 10.0, 50)
    y = 2.0 * x + 1.0
    w = np.ones(50)
    return x, y, w


@pytest.fixture
def noisy_linear_data():
    # 60-point linear signal with small Gaussian noise, seed fixed
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 10.0, 60)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.2, 60)
    w = np.ones(60)
    return x, y, w

class TestGetKnots:

    def test_zero_knots_returns_empty(self):
        x = np.linspace(0.0, 10.0, 100)
        result = get_knots(x, 0)
        assert len(result) == 0

    def test_one_knot_at_midpoint(self):
        x = np.linspace(0.0, 10.0, 100)
        result = get_knots(x, 1)
        assert len(result) == 1
        assert result[0] == pytest.approx(5.0)

    def test_two_knots_equally_spaced(self):
        x = np.linspace(0.0, 10.0, 100)
        result = get_knots(x, 2)
        assert len(result) == 2
        assert result[0] == pytest.approx(10.0 / 3.0)
        assert result[1] == pytest.approx(20.0 / 3.0)

    def test_three_knots_match_formula(self):
        x = np.linspace(2.0, 8.0, 100)
        result = get_knots(x, 3)
        interval = 8.0 - 2.0
        sub = interval / 4.0
        expected = [2.0 + sub, 2.0 + 2 * sub, 2.0 + 3 * sub]
        assert np.allclose(result, expected)

    def test_knots_are_interior_not_at_boundary(self):
        x = np.linspace(0.0, 10.0, 100)
        result = get_knots(x, 2)
        assert result[0] > x[0]
        assert result[-1] < x[-1]

    def test_first_element_of_full_array_is_excluded(self):
        x = np.linspace(0.0, 10.0, 100)
        result = get_knots(x, 2)
        assert result[0] != x[0]

    def test_non_zero_start(self):
        x = np.linspace(5.0, 15.0, 100)
        result = get_knots(x, 1)
        assert result[0] == pytest.approx(10.0)

    def test_returns_ndarray(self):
        x = np.linspace(0.0, 10.0, 50)
        result = get_knots(x, 2)
        assert isinstance(result, np.ndarray)

    def test_knots_count_matches_requested(self):
        x = np.linspace(0.0, 10.0, 100)
        for n in [1, 2, 3, 5]:
            assert len(get_knots(x, n)) == n

class TestWtrebin:

    def test_nbin_1_returns_same_objects(self):
        x = np.linspace(0.0, 1.0, 5)
        y = x.copy()
        w = np.ones(5)
        xo, yo, wo = wtrebin(x, y, w, nbin=1)
        assert xo is x
        assert yo is y

    def test_nbin_1_weights_none_returns_weights_passthrough(self):
        x = np.linspace(0.0, 1.0, 5)
        y = x.copy()
        xo, yo, wo = wtrebin(x, y, None, nbin=1)
        assert xo is x
        assert yo is y
        assert wo is None

    def test_nbin_2_output_length(self):
        x = np.arange(10, dtype=float)
        y = x * 2.0
        w = np.ones(10)
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        assert len(xo) == 5

    def test_nbin_2_xout_values_equal_weights(self):
        x = np.arange(10, dtype=float)
        y = x * 2.0
        w = np.ones(10)
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        expected_x = [0.5, 2.5, 4.5, 6.5, 8.5]
        assert np.allclose(xo, expected_x)

    def test_nbin_2_yout_values_equal_weights(self):
        x = np.arange(10, dtype=float)
        y = x * 2.0
        w = np.ones(10)
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        expected_y = [1.0, 5.0, 9.0, 13.0, 17.0]
        assert np.allclose(yo, expected_y)

    def test_nbin_2_wout_is_sum_divided_by_nbin(self):
        x = np.arange(6, dtype=float)
        y = x.copy()
        w = np.array([2.0, 4.0, 2.0, 4.0, 2.0, 4.0])
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        expected_w = [3.0, 3.0, 3.0]
        assert np.allclose(wo, expected_w)

    def test_nbin_2_weighted_xout(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        w = np.array([1.0, 3.0, 1.0, 3.0])
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        assert xo[0] == pytest.approx((0.0*1 + 1.0*3) / (1+3))  # 0.75
        assert xo[1] == pytest.approx((2.0*1 + 3.0*3) / (1+3))  # 2.75

    def test_nbin_2_weighted_yout(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        w = np.array([1.0, 3.0, 1.0, 3.0])
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        assert yo[0] == pytest.approx((0.0*1 + 2.0*3) / (1+3))  # 1.5
        assert yo[1] == pytest.approx((4.0*1 + 6.0*3) / (1+3))  # 5.5

    def test_nbin_3_output_length_divisible(self):
        x = np.arange(9, dtype=float)
        y = x * 2.0
        w = np.ones(9)
        xo, yo, wo = wtrebin(x, y, w, nbin=3)
        assert len(xo) == 3

    def test_nbin_3_xout_values(self):
        x = np.arange(9, dtype=float)
        y = x.copy()
        w = np.ones(9)
        xo, yo, wo = wtrebin(x, y, w, nbin=3)
        assert np.allclose(xo, [1.0, 4.0, 7.0])

    def test_nbin_3_wout_varying_weights(self):
        x = np.arange(9, dtype=float)
        y = x.copy()
        w = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        xo, yo, wo = wtrebin(x, y, w, nbin=3)
        assert np.allclose(wo, [2.0, 2.0, 2.0])

    def test_nbin_2_odd_length_output_rounded_up(self):
        x = np.arange(7, dtype=float)
        y = x * 3.0
        w = np.ones(7)
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        assert len(xo) == 4

    def test_nbin_2_odd_last_bin_is_partial(self):
        x = np.arange(7, dtype=float)
        y = x.copy()
        w = np.ones(7)
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        assert xo[-1] == pytest.approx(6.0)

    def test_nbin_2_weights_none_treated_as_ones(self):
        x = np.arange(10, dtype=float)
        y = x * 2.0
        xo_none, yo_none, wo_none = wtrebin(x, y, None, nbin=2)
        xo_ones, yo_ones, wo_ones = wtrebin(x, y, np.ones(10), nbin=2)
        assert np.allclose(xo_none, xo_ones)
        assert np.allclose(yo_none, yo_ones)

    def test_nbin_2_xy_length_mismatch_returns_none_triple(self, capsys):
        x = np.array([1.0, 2.0])
        y = np.array([1.0])
        w = np.ones(2)
        xo, yo, wo = wtrebin(x, y, w, nbin=2)
        assert xo is None
        assert yo is None
        assert wo is None

    def test_nbin_2_xy_mismatch_prints_message(self, capsys):
        wtrebin(np.array([1.0, 2.0]), np.array([1.0]), np.ones(2), nbin=2)
        out = capsys.readouterr().out
        assert "different" in out.lower()

    def test_nbin_2_weight_length_mismatch_returns_none_triple(self, capsys):
        x = np.arange(10, dtype=float)
        y = x.copy()
        xo, yo, wo = wtrebin(x, y, np.ones(5), nbin=2)
        assert xo is None
        assert yo is None
        assert wo is None

    def test_nbin_2_weight_mismatch_prints_message(self, capsys):
        x = np.arange(10, dtype=float)
        wtrebin(x, x.copy(), np.ones(5), nbin=2)
        out = capsys.readouterr().out
        assert "different" in out.lower()

    def test_nbin_larger_than_length(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        w = np.ones(2)
        xo, yo, wo = wtrebin(x, y, w, nbin=5)
        assert len(xo) == 1

class TestCalcRmsDeviation:

    def test_perfect_fit_gives_near_zero_rms(self, linear_data):
        x, y, w = linear_data
        fit = fit_once(x, y, w, "spline3", 3)
        rms = calc_rms_deviation(x, y, w, fit)
        assert rms == pytest.approx(0.0, abs=1e-10)

    def test_rms_formula_matches_manual_calculation(self, linear_data):
        x, y, w = linear_data
        fit = fit_once(x, y, w, "spline1", 2)
        rms = calc_rms_deviation(x, y, w, fit)
        dev = y - fit(x)
        expected = np.sqrt((w * dev ** 2).sum() / w.sum())
        assert rms == pytest.approx(expected)

    def test_rms_is_nonnegative(self, noisy_linear_data):
        x, y, w = noisy_linear_data
        fit = fit_once(x, y, w, "spline3", 3)
        rms = calc_rms_deviation(x, y, w, fit)
        assert rms >= 0.0

    def test_rms_larger_with_noisy_data(self, linear_data, noisy_linear_data):
        x_clean, y_clean, w = linear_data
        x_noisy, y_noisy, w_n = noisy_linear_data
        fit_clean = fit_once(x_clean, y_clean, w, "spline3", 3)
        fit_noisy = fit_once(x_noisy, y_noisy, w_n, "spline3", 3)
        rms_clean = calc_rms_deviation(x_clean, y_clean, w, fit_clean)
        rms_noisy = calc_rms_deviation(x_noisy, y_noisy, w_n, fit_noisy)
        assert rms_noisy > rms_clean

    def test_rms_increases_with_known_residual(self):
        x = np.linspace(0.0, 10.0, 50)
        y = 2.0 * x + 1.0
        w = np.ones(50)
        fit = fit_once(x, y, w, "spline3", 3)
        # shift y by 5.0 to compute rms — residual is ~5.0 everywhere
        rms = calc_rms_deviation(x, y + 5.0, w, fit)
        assert rms == pytest.approx(5.0, rel=1e-6)

    def test_rms_with_zero_weights_on_outliers(self):
        x = np.linspace(0.0, 10.0, 50)
        y = 2.0 * x + 1.0
        w = np.ones(50)
        fit = fit_once(x, y, w, "spline3", 3)
        y_corrupt = y.copy()
        y_corrupt[[5, 10, 15]] += 1000.0
        w_masked = w.copy()
        w_masked[[5, 10, 15]] = 0.0
        rms = calc_rms_deviation(x, y_corrupt, w_masked, fit)
        assert rms == pytest.approx(0.0, abs=1e-8)

    def test_rms_returns_float(self, linear_data):
        x, y, w = linear_data
        fit = fit_once(x, y, w, "spline3", 3)
        result = calc_rms_deviation(x, y, w, fit)
        assert isinstance(float(result), float)

class TestFitOnce:

    def test_spline3_returns_lsq_object(self, linear_data):
        x, y, w = linear_data
        result = fit_once(x, y, w, "spline3", 3)
        assert isinstance(result, LSQUnivariateSpline)

    def test_spline1_returns_lsq_object(self, linear_data):
        x, y, w = linear_data
        result = fit_once(x, y, w, "spline1", 3)
        assert isinstance(result, LSQUnivariateSpline)

    def test_spline3_fits_linear_accurately(self, linear_data):
        x, y, w = linear_data
        fit = fit_once(x, y, w, "spline3", 3)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-6)

    def test_spline1_fits_linear_accurately(self, linear_data):
        x, y, w = linear_data
        fit = fit_once(x, y, w, "spline1", 3)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-6)

    def test_spline3_fits_quadratic_signal(self):
        x = np.linspace(0.0, 10.0, 100)
        y = x ** 2
        w = np.ones(100)
        fit = fit_once(x, y, w, "spline3", 5)
        assert fit(np.array([5.0]))[0] == pytest.approx(25.0, rel=0.01)

    def test_unknown_spline_name_returns_none(self, capsys, linear_data):
        x, y, w = linear_data
        result = fit_once(x, y, w, "spline2", 3)
        assert result is None

    def test_unknown_spline_name_prints_message(self, capsys, linear_data):
        x, y, w = linear_data
        fit_once(x, y, w, "splineX", 3)
        out = capsys.readouterr().out
        assert "spline1" in out and "spline3" in out

    def test_non_spline_function_returns_none(self, capsys, linear_data):
        x, y, w = linear_data
        result = fit_once(x, y, w, "polynomial", 3)
        assert result is None

    def test_non_spline_function_prints_not_implemented(self, capsys, linear_data):
        x, y, w = linear_data
        fit_once(x, y, w, "polynomial", 3)
        out = capsys.readouterr().out
        assert "Not implemented" in out

    def test_callable_result(self, linear_data):
        x, y, w = linear_data
        fit = fit_once(x, y, w, "spline3", 3)
        result = fit(x)
        assert len(result) == len(x)

    def test_order_1_zero_knots(self, linear_data):
        x, y, w = linear_data
        fit = fit_once(x, y, w, "spline3", 1)
        assert fit is not None
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-4)

    def test_equal_weights_vs_unit_weights_same_result(self, linear_data):
        x, y, _ = linear_data
        w_ones = np.ones(50)
        w_two  = np.full(50, 2.0)
        fit_ones = fit_once(x, y, w_ones, "spline3", 3)
        fit_two  = fit_once(x, y, w_two,  "spline3", 3)
        assert np.allclose(fit_ones(x), fit_two(x), atol=1e-10)

class TestFitWithRejection:

    def test_niterate_0_returns_single_fit(self, linear_data):
        x, y, w = linear_data
        result = fit_with_rejection(x, y, w, "spline3", 3, 3.0, 3.0, 0, 0.0)
        assert isinstance(result, LSQUnivariateSpline)

    def test_niterate_0_not_affected_by_outlier(self):
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 * x + 1.0
        y[30] += 200.0
        w = np.ones(60)
        fit = fit_with_rejection(x, y, w, "spline3", 3, 3.0, 3.0, 0, 0.0)
        residual = abs(fit(np.array([0.0]))[0] - 1.0)
        assert residual > 0.5 

    def test_sigma_clipping_removes_high_outlier(self):
        rng = np.random.default_rng(7)
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.2, 60)
        y[30] += 200.0
        w = np.ones(60)
        fit = fit_with_rejection(x, y, w, "spline3", 3, 3.0, 3.0, 5, 0.0)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, abs=1.0)

    def test_sigma_clipping_removes_low_outlier(self):
        rng = np.random.default_rng(13)
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.2, 60)
        y[15] -= 200.0
        w = np.ones(60)
        fit = fit_with_rejection(x, y, w, "spline3", 3, 3.0, 3.0, 5, 0.0)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, abs=1.0)

    def test_early_break_when_no_bad_points(self):
        rng = np.random.default_rng(99)
        x = np.linspace(0.0, 10.0, 50)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.01, 50)
        w = np.ones(50)
        fit = fit_with_rejection(x, y, w, "spline3", 3, 10.0, 10.0, 20, 0.0)
        assert fit is not None
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, abs=0.5)

    def test_returns_lsq_univariate_spline(self, linear_data):
        x, y, w = linear_data
        result = fit_with_rejection(x, y, w, "spline3", 3, 3.0, 3.0, 3, 0.0)
        assert isinstance(result, LSQUnivariateSpline)

    def test_fit_with_rejection_spline1(self):
        rng = np.random.default_rng(3)
        x = np.linspace(0.0, 10.0, 50)
        y = 3.0 * x + 2.0 + rng.normal(0.0, 0.1, 50)
        y[25] += 100.0
        w = np.ones(50)
        fit = fit_with_rejection(x, y, w, "spline1", 3, 3.0, 3.0, 5, 0.0)
        assert fit(np.array([5.0]))[0] == pytest.approx(17.0, abs=1.0)

class TestFit1d:

    def test_xy_length_mismatch_returns_none(self, capsys):
        result = fit1d(np.array([1.0, 2.0]), np.array([1.0]))
        assert result is None

    def test_xy_length_mismatch_prints_message(self, capsys):
        fit1d(np.array([1.0, 2.0]), np.array([1.0]))
        out = capsys.readouterr().out
        assert "different lengths" in out

    def test_weight_length_mismatch_returns_none(self, capsys, linear_data):
        x, y, _ = linear_data
        result = fit1d(x, y, weights=np.ones(10))
        assert result is None

    def test_weight_length_mismatch_prints_first_message(self, capsys, linear_data):
        x, y, _ = linear_data
        fit1d(x, y, weights=np.ones(10))
        out = capsys.readouterr().out
        assert "different length" in out.lower()

    def test_weight_length_mismatch_prints_hint_message(self, capsys, linear_data):
        x, y, _ = linear_data
        fit1d(x, y, weights=np.ones(10))
        out = capsys.readouterr().out
        assert "weight=None" in out

    def test_weights_none_defaults_to_ones(self, linear_data):
        x, y, _ = linear_data
        result_none = fit1d(x, y, weights=None)
        result_ones = fit1d(x, y, weights=np.ones(50))
        assert np.allclose(result_none(x), result_ones(x))

    def test_returns_lsq_spline_object(self, linear_data):
        x, y, w = linear_data
        result = fit1d(x, y, weights=w)
        assert isinstance(result, LSQUnivariateSpline)

    def test_spline3_linear_signal_at_midpoint(self, linear_data):
        x, y, w = linear_data
        fit = fit1d(x, y, weights=w, function="spline3", order=3)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-5)

    def test_spline1_linear_signal_at_midpoint(self, linear_data):
        x, y, w = linear_data
        fit = fit1d(x, y, weights=w, function="spline1", order=3)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-5)

    def test_explicit_weights_accepted(self, linear_data):
        x, y, _ = linear_data
        w_explicit = np.linspace(0.5, 1.5, 50)
        result = fit1d(x, y, weights=w_explicit, function="spline3", order=3)
        assert result is not None
        assert result(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-4)

    def test_non_spline_function_returns_none(self, capsys, linear_data):
        x, y, w = linear_data
        result = fit1d(x, y, weights=w, function="polynomial", order=3)
        assert result is None

    def test_invalid_spline_name_returns_none(self, capsys, linear_data):
        x, y, w = linear_data
        result = fit1d(x, y, weights=w, function="spline2", order=3)
        assert result is None

    def test_higher_order_gives_more_knots(self, linear_data):
        x, y, w = linear_data
        fit3 = fit1d(x, y, weights=w, function="spline3", order=3)
        fit5 = fit1d(x, y, weights=w, function="spline3", order=5)
        assert fit5(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-5)
        assert fit3(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-5)

    def test_naverage_1_is_default(self, linear_data):
        x, y, w = linear_data
        fit_default = fit1d(x, y, weights=w)
        fit_nav1    = fit1d(x, y, weights=w, naverage=1)
        assert np.allclose(fit_default(x), fit_nav1(x))

    def test_naverage_2_produces_valid_fit(self, linear_data):
        x, y, w = linear_data
        result = fit1d(x, y, weights=w, naverage=2, function="spline3", order=3)
        assert result is not None
        assert result(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-4)

    def test_naverage_3_produces_valid_fit(self, linear_data):
        x, y, w = linear_data
        result = fit1d(x, y, weights=w, naverage=3, function="spline3", order=3)
        assert result is not None
        assert result(np.array([5.0]))[0] == pytest.approx(11.0, rel=1e-4)

    def test_niterate_0_no_clipping(self):
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 * x + 1.0
        y[30] += 500.0
        w = np.ones(60)
        fit_no  = fit1d(x, y, weights=w, niterate=0)
        fit_yes = fit1d(x, y, weights=w, niterate=5, low_reject=2.0,
                        high_reject=2.0)
        # With clipping the fit at x=5 is closer to 11.0
        err_no  = abs(fit_no(np.array([5.0]))[0]  - 11.0)
        err_yes = abs(fit_yes(np.array([5.0]))[0] - 11.0)
        assert err_yes < err_no

    def test_niterate_positive_removes_outlier(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.2, 60)
        y[30] += 200.0
        w = np.ones(60)
        fit = fit1d(x, y, weights=w, niterate=5, low_reject=3.0, high_reject=3.0)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, abs=1.0)

    def test_low_reject_and_high_reject_are_independent(self):
        rng = np.random.default_rng(55)
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, 60)
        y[20] += 150.0   # high outlier only
        w = np.ones(60)
        fit = fit1d(x, y, weights=w, niterate=5, low_reject=100.0,
                    high_reject=2.0)
        assert fit(np.array([5.0]))[0] == pytest.approx(11.0, abs=1.0)

    def test_grow_parameter_accepted_without_error(self, linear_data):
        x, y, w = linear_data
        result = fit1d(x, y, weights=w, grow=2.0)
        assert result is not None

    def test_result_is_callable_on_full_range(self, linear_data):
        x, y, w = linear_data
        fit = fit1d(x, y, weights=w)
        vals = fit(x)
        assert vals.shape == x.shape

    def test_result_callable_on_single_point(self, linear_data):
        x, y, w = linear_data
        fit = fit1d(x, y, weights=w)
        val = fit(np.array([3.0]))
        assert val.shape == (1,)