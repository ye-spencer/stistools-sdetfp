import pytest
from stistools import wx2d

from astropy.io import fits 
import numpy as np

def image_array():
    return np.ones((8, 8))

# Blackbox: invalid algorithm
def test_wx2d_invalid_algorithm():
    with pytest.raises(ValueError):
        wx2d.wx2d("input.fits", "output.fits", algorithm="invalid")

# Blackbox: invalid psf_width
@pytest.mark.xfail(reason="Known Failure - invalid psf_width")
def test_wx2d_invalid_psf_width():
    with pytest.raises(ValueError):
        wx2d.wx2d("input.fits", "output.fits", psf_width=-1)

# Blackbox: kd algorithm with subsampled
def test_wx2d_kd_subsampled_fails():
    with pytest.raises(ValueError):
        wx2d.wx2d("input.fits", "output.fits", algorithm="kd", subsampled="output.fits")

# Blackbox: kd algorithm with convolved
@pytest.mark.xfail(reason="Known Failure - kd algorithm with convolved")
def test_wx2d_kd_convolved_fails():
    with pytest.raises(ValueError):
        wx2d.wx2d("input.fits", "output.fits", algorithm="kd", convolved="output.fits")

# Blackbox: invalid input file
def test_wx2d_invalid_input_file():
    with pytest.raises(FileNotFoundError):
        wx2d.wx2d("invalid.fits", "output.fits")

# Blackbox: even sized input array
@pytest.mark.xfail(reason="Known Failure - even sized input array")
def test_polynomial_linear():
    x = np.array([0, 1, 2, 3])
    y = np.ones((4, 2))
    
    n = 4
    z = 2.0

    assert wx2d.polynomial(x, y, z, n) == 2.0

# Blackbox: odd sized input array
def test_polynomial_odd():
    x = np.array([0, 1, 2])
    y = np.ones((3, 2))
    
    n = 3
    z = 1.5

    assert np.array_equal(wx2d.polynomial(x, y, z, n), np.array([1., 1.]))

# Blackbox: 1-d array
def test_inv_haar_1d():
    image = np.array([1, 2, 3, 4])
    result = wx2d.inv_haar(image)
    assert np.array_equal(result, np.array([1, -1, 3, -1]))

# Blackbox: float 1-d array
def test_inv_haar_1d_float():
    image = np.array([1., 2., 3., 4.])
    result = wx2d.inv_haar(image)
    assert np.array_equal(result, np.array([1.5, -0.5, 3.5, -0.5]))

# Branch coverage: 2-d image input
def test_inv_haar_2d():
    image = np.array([[2., 4.], [6., 8.]])
    result = wx2d.inv_haar(image)
    assert np.array_equal(result, np.array([[4., 6.], [-2., -2.]]))

# Blackbox: smallest valid pair
def test_inv_haar_min_pair():
    image = np.array([4., 2.])
    result = wx2d.inv_haar(image)
    assert np.array_equal(result, np.array([3., 1.]))

# Statement coverage: in-place mutation return
def test_inv_haar_returns_input_array():
    image = np.array([10., 0., 0., 10.])
    result = wx2d.inv_haar(image)
    assert result is image
    assert np.array_equal(image, np.array([5., 5., 5., -5.]))

# Statement coverage: result shape and dtype
def test_inv_avg_interp_shape_and_dtype():
    image = np.ones((5, 3), dtype=np.float64)
    result = wx2d.inv_avg_interp(3, image)
    assert result.shape == (5, 3)
    assert result.dtype == np.float32

# Branch coverage: constant image zeros
def test_inv_avg_interp_constant_image():
    image = np.ones((6, 2), dtype=np.float64)
    result = wx2d.inv_avg_interp(3, image)
    assert np.all(result == 0)

# Branch coverage: boundary rows untouched
def test_inv_avg_interp_boundary_untouched(mocker):
    mocker.patch("stistools.wx2d.polynomial", return_value=np.array([1.0]))
    image = np.ones((5, 1), dtype=np.float64)
    result = wx2d.inv_avg_interp(3, image)
    assert np.all(result[0] == 0)
    assert np.all(result[4] == 0)

# Blackbox: computed row values
def test_inv_avg_interp_computed_values(mocker):
    mocker.patch("stistools.wx2d.polynomial", return_value=np.array([0.0]))
    image = np.arange(5, dtype=np.float64).reshape(5, 1)
    result = wx2d.inv_avg_interp(3, image)
    expected = np.array([[0.], [-1.], [-4.], [-7.], [0.]], dtype=np.float32)
    assert np.array_equal(result, expected)

# Blackbox: polynomial call args
def test_inv_avg_interp_polynomial_args(mocker):
    mock_poly = mocker.patch("stistools.wx2d.polynomial", return_value=np.zeros(2))
    image = np.ones((5, 2), dtype=np.float64)
    wx2d.inv_avg_interp(3, image)
    assert mock_poly.call_count == 3
    args = mock_poly.call_args_list[0].args
    assert np.array_equal(args[0], np.array([0., 1., 2., 3.]))
    assert args[2] == 1.5
    assert args[3] == 4

# Branch coverage: empty a2displ early return
def test_bin_traces_empty():
    a2displ = np.array([])
    result = wx2d.bin_traces(a2displ, 2, 0.0)
    assert result is a2displ

# Branch coverage: binaxis1 == 1 early return
def test_bin_traces_binaxis1_one():
    a2displ = np.ones((2, 10), dtype=np.float32)
    result = wx2d.bin_traces(a2displ, 1, 0.0)
    assert result is a2displ

# Branch coverage: binaxis1 == 2 newlen 511
def test_bin_traces_binaxis1_two():
    a2displ = np.ones((1, 1024), dtype=np.float32)
    result = wx2d.bin_traces(a2displ, 2, 0.0)
    assert result.shape == (1, 511)
    assert np.allclose(result, 1.0)

# Branch coverage: binaxis1 == 4 newlen 255
def test_bin_traces_binaxis1_four():
    a2displ = np.ones((1, 1024), dtype=np.float32)
    result = wx2d.bin_traces(a2displ, 4, 0.0)
    assert result.shape == (1, 255)
    assert np.allclose(result, 1.0)

# Branch coverage: binaxis1 == 8 newlen 127
def test_bin_traces_binaxis1_eight():
    a2displ = np.ones((1, 1024), dtype=np.float32)
    result = wx2d.bin_traces(a2displ, 8, 0.0)
    assert result.shape == (1, 127)
    assert np.allclose(result, 1.0)

# Blackbox: averaging across bin
def test_bin_traces_averages_values():
    trace = np.arange(1024, dtype=np.float32)
    a2displ = np.array([trace])
    result = wx2d.bin_traces(a2displ, 2, 0.25)
    expected = np.arange(511, dtype=np.float32) * 2 + 0.5
    assert np.allclose(result[0], expected)

# Statement coverage: multiple traces binned
def test_bin_traces_multiple_traces():
    a2displ = np.array([np.ones(1024, dtype=np.float32),
                        np.full(1024, 3.0, dtype=np.float32)])
    result = wx2d.bin_traces(a2displ, 2, 0.25)
    assert result.shape == (2, 511)
    assert np.allclose(result[0], 1.0)
    assert np.allclose(result[1], 3.0)

# Branch coverage: unsupported binaxis1 else branch
@pytest.mark.xfail(reason="Known Failure - else branch broadcasting bug")
def test_bin_traces_other_binaxis():
    a2displ = np.ones((1, 9), dtype=np.float32)
    result = wx2d.bin_traces(a2displ, 3, 0.0)
    assert result.shape == (1, 9)

# Branch coverage: array input early return
def test_get_trace_array_input():
    tracefile = np.array([0.1, 0.2, 0.3])
    a2center, a2displ = wx2d.get_trace(tracefile, None, None)
    assert a2center == [0.]
    assert a2displ[0] is tracefile

# Branch coverage: invalid LTM1_1 value
def test_get_trace_invalid_ltm():
    phdr = fits.Header()
    hdr = fits.Header()
    hdr["ltm1_1"] = 0.3
    with pytest.raises(ValueError):
        wx2d.get_trace("trace.fits", phdr, hdr)

# Blackbox: getTable called correctly
def test_get_trace_gettable_args(mocker):
    trace_info = mocker.MagicMock()
    trace_info.field.side_effect = lambda name: {
        "a2center": np.array([1.]),
        "a2displ": np.array([[0.]]),
    }[name]
    mock_get = mocker.patch("stistools.wx2d.gettable.getTable", return_value=trace_info)
    mocker.patch("stistools.wx2d.gettable.rotateTrace")

    phdr = fits.Header()
    phdr["opt_elem"] = "G230L"
    phdr["cenwave"] = 2375
    hdr = fits.Header()
    hdr["ltm1_1"] = 1.0

    wx2d.get_trace("trace.fits", phdr, hdr)
    call = mock_get.call_args
    assert call.args[0] == "trace.fits"
    assert call.args[1] == {"opt_elem": "G230L", "cenwave": 2375}
    assert call.kwargs == {"sortcol": "a2center", "at_least_one": True}

# Blackbox: a2center zero indexing
def test_get_trace_binaxis1_one_skips_bin(mocker):
    trace_info = mocker.MagicMock()
    trace_info.field.side_effect = lambda name: {
        "a2center": np.array([1., 2.]),
        "a2displ": np.array([[0.1], [0.2]]),
    }[name]
    mocker.patch("stistools.wx2d.gettable.getTable", return_value=trace_info)
    mocker.patch("stistools.wx2d.gettable.rotateTrace")
    mock_bin = mocker.patch("stistools.wx2d.bin_traces")

    phdr = fits.Header()
    hdr = fits.Header()
    hdr["ltm1_1"] = 1.0

    a2center, a2displ = wx2d.get_trace("trace.fits", phdr, hdr)
    assert np.array_equal(a2center, np.array([0., 1.]))
    mock_bin.assert_not_called()

# Branch coverage: binaxis1>1 calls bin_traces
def test_get_trace_binaxis1_two_calls_bin(mocker):
    trace_info = mocker.MagicMock()
    trace_info.field.side_effect = lambda name: {
        "a2center": np.array([1., 2.]),
        "a2displ": np.array([[0.1], [0.2]]),
    }[name]
    mocker.patch("stistools.wx2d.gettable.getTable", return_value=trace_info)
    mocker.patch("stistools.wx2d.gettable.rotateTrace")
    mock_bin = mocker.patch("stistools.wx2d.bin_traces", return_value="binned")

    phdr = fits.Header()
    hdr = fits.Header()
    hdr["ltm1_1"] = 0.5
    hdr["ltv1"] = 2.0

    _, a2displ = wx2d.get_trace("trace.fits", phdr, hdr)
    mock_bin.assert_called_once()
    args = mock_bin.call_args.args
    assert args[1] == 2
    assert args[2] == 2.0
    assert a2displ == "binned"

# Branch coverage: empty traces skip binning
def test_get_trace_empty_traces_skip_bin(mocker):
    trace_info = mocker.MagicMock()
    trace_info.field.side_effect = lambda name: {
        "a2center": np.array([]),
        "a2displ": np.array([]),
    }[name]
    mocker.patch("stistools.wx2d.gettable.getTable", return_value=trace_info)
    mocker.patch("stistools.wx2d.gettable.rotateTrace")
    mock_bin = mocker.patch("stistools.wx2d.bin_traces")

    phdr = fits.Header()
    hdr = fits.Header()
    hdr["ltm1_1"] = 0.5

    wx2d.get_trace("trace.fits", phdr, hdr)
    mock_bin.assert_not_called()

# Branch coverage: trace None uses sptrctab
def test_trace_name_none_uses_sptrctab():
    phdr = fits.Header({"SPTRCTAB": "trace.fits"})
    assert wx2d.trace_name(None, phdr) == "trace.fits"

# Branch coverage: trace None missing sptrctab
def test_trace_name_none_missing_key_raises():
    phdr = fits.Header()
    with pytest.raises(ValueError):
        wx2d.trace_name(None, phdr)

# Branch coverage: trace str returned expanded
def test_trace_name_string_returns_expanded():
    assert wx2d.trace_name("mytrace.fits", fits.Header()) == "mytrace.fits"

# Blackbox: string path expandFileName called
def test_trace_name_string_calls_expand(mocker):
    mock_expand = mocker.patch("stistools.wx2d.r_util.expandFileName",
                               return_value="/expanded/trace.fits")
    result = wx2d.trace_name("$lref/trace.fits", fits.Header())
    mock_expand.assert_called_once_with("$lref/trace.fits")
    assert result == "/expanded/trace.fits"

# Branch coverage: trace array returned unchanged
def test_trace_name_array_returned_unchanged():
    trace = np.array([1.0, 2.0, 3.0])
    result = wx2d.trace_name(trace, fits.Header())
    assert result is trace

# Blackbox: sptrctab value expanded
def test_trace_name_sptrctab_calls_expand(mocker):
    mock_expand = mocker.patch("stistools.wx2d.r_util.expandFileName",
                               return_value="/expanded/from_hdr.fits")
    phdr = fits.Header({"SPTRCTAB": "$lref/from_hdr.fits"})
    result = wx2d.trace_name(None, phdr)
    mock_expand.assert_called_once_with("$lref/from_hdr.fits")
    assert result == "/expanded/from_hdr.fits"

# Branch coverage: empty a2displ returns zeros
def test_interpolate_trace_empty_returns_zeros():
    result = wx2d.interpolate_trace(np.array([]), np.array([]), 512.0, 1024)
    assert result.shape == (1024,)
    assert result.dtype == np.float32
    assert np.all(result == 0)

# Branch coverage: empty skips r_util.interpolate
def test_interpolate_trace_empty_skips_interp(mocker):
    mock_interp = mocker.patch("stistools.wx2d.r_util.interpolate")
    wx2d.interpolate_trace(np.array([]), np.array([]), 100.0, 5)
    mock_interp.assert_not_called()

# Blackbox: delegates to r_util.interpolate
def test_interpolate_trace_delegates(mocker):
    mock_interp = mocker.patch("stistools.wx2d.r_util.interpolate", return_value="trace")
    a2center = np.array([100., 200.])
    a2displ = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = wx2d.interpolate_trace(a2center, a2displ, 150.0, 999)
    mock_interp.assert_called_once_with(a2center, a2displ, 150.0)
    assert result == "trace"

# Statement coverage: result shape and dtype
def test_extract_i16_shape_and_dtype():
    image = np.ones((4, 5), dtype=np.int16)
    locn = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    result = wx2d.extract_i16(image, locn, 2)
    assert result.shape == (5,)
    assert result.dtype == np.int16

# Blackbox: bitwise OR not sum
def test_extract_i16_bitwise_or():
    image = np.array([[0], [1], [1], [1], [0]], dtype=np.int16)
    locn = np.array([2.0])
    result = wx2d.extract_i16(image, locn, 4)
    assert result[0] == 1

# Blackbox: OR combines distinct bits
def test_extract_i16_or_distinct_bits():
    image = np.array([[0], [0], [4], [8], [0]], dtype=np.int16)
    locn = np.array([2.0])
    result = wx2d.extract_i16(image, locn, 2)
    assert result[0] == 12

# Branch coverage: s_low below 1 skipped
def test_extract_i16_low_boundary_skipped():
    image = np.full((5, 3), 7, dtype=np.int16)
    locn = np.array([0.0, 0.0, 0.0])
    result = wx2d.extract_i16(image, locn, 4)
    assert np.array_equal(result, np.zeros(3, dtype=np.int16))

# Branch coverage: s_high above range skipped
def test_extract_i16_high_boundary_skipped():
    image = np.full((5, 3), 7, dtype=np.int16)
    locn = np.array([4.0, 4.0, 4.0])
    result = wx2d.extract_i16(image, locn, 4)
    assert np.array_equal(result, np.zeros(3, dtype=np.int16))

# Branch coverage: mixed valid and skipped columns
def test_extract_i16_mixed_columns():
    image = np.array([[1, 1], [2, 2], [4, 4], [8, 8], [16, 16]], dtype=np.int16)
    locn = np.array([2.0, 0.0])
    result = wx2d.extract_i16(image, locn, 2)
    assert result[0] == 12
    assert result[1] == 0

# Statement coverage: result shape and dtype
def test_extract_shape_and_dtype():
    image = np.ones((4, 5), dtype=np.float32)
    locn = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    result = wx2d.extract(image, locn, 2)
    assert result.shape == (5,)
    assert result.dtype == np.float32

# Blackbox: integer-aligned sum no fractions
def test_extract_integer_aligned():
    image = np.array([[0.], [20.], [30.], [40.], [0.]], dtype=np.float32)
    locn = np.array([1.5])
    result = wx2d.extract(image, locn, 2)
    assert np.isclose(result[0], 50.0)

# Blackbox: fractional edges both contribute
def test_extract_fractional_edges():
    image = np.array([[0.], [10.], [20.], [30.], [0.]], dtype=np.float32)
    locn = np.array([2.0])
    result = wx2d.extract(image, locn, 2)
    assert np.isclose(result[0], 40.0)

# Blackbox: dhigh weights image at s_high
def test_extract_dhigh_uses_s_high():
    image = np.array([[0.], [0.], [0.], [10.], [0.]], dtype=np.float32)
    locn = np.array([2.0])
    result = wx2d.extract(image, locn, 2)
    assert np.isclose(result[0], 5.0)

# Blackbox: dlow weights image at s_low-1
def test_extract_dlow_uses_s_low_minus_1():
    image = np.array([[0.], [10.], [0.], [0.], [0.]], dtype=np.float32)
    locn = np.array([2.0])
    result = wx2d.extract(image, locn, 2)
    assert np.isclose(result[0], 5.0)

# Branch coverage: s_low below 1 skipped
def test_extract_low_boundary_skipped():
    image = np.full((5, 3), 7.0, dtype=np.float32)
    locn = np.array([0.0, 0.0, 0.0])
    result = wx2d.extract(image, locn, 4)
    assert np.array_equal(result, np.zeros(3, dtype=np.float32))

# Branch coverage: s_high above range skipped
def test_extract_high_boundary_skipped():
    image = np.full((5, 3), 7.0, dtype=np.float32)
    locn = np.array([4.0, 4.0, 4.0])
    result = wx2d.extract(image, locn, 4)
    assert np.array_equal(result, np.zeros(3, dtype=np.float32))

# Branch coverage: mixed valid and skipped columns
def test_extract_mixed_columns():
    image = np.array([[0., 0.], [20., 20.], [30., 30.], [40., 40.], [0., 0.]],
                     dtype=np.float32)
    locn = np.array([1.5, 0.0])
    result = wx2d.extract(image, locn, 2)
    assert np.isclose(result[0], 50.0)
    assert result[1] == 0.0

# Statement coverage: constant image rms extraction
def test_extract_err_constant():
    image = np.full((10, 3), 2.0, dtype=np.float32)
    locn = np.array([5.0, 5.0, 5.0])
    result = wx2d.extract_err(image, locn, 4)
    assert result.shape == (3,)
    assert result.dtype == np.float32
    assert np.allclose(result, 2.0)

# Blackbox: root-mean-square math
def test_extract_err_rms_math():
    image = np.array([[0.], [3.], [3.], [3.], [0.]], dtype=np.float32)
    locn = np.array([1.5])
    result = wx2d.extract_err(image, locn, 2)
    assert np.isclose(result[0], 3.0)

# Branch coverage: low edge skips column
def test_extract_err_low_edge_skip():
    image = np.ones((10, 1), dtype=np.float32)
    locn = np.array([0.0])
    result = wx2d.extract_err(image, locn, 4)
    assert result[0] == 0

# Branch coverage: high edge skips column
def test_extract_err_high_edge_skip():
    image = np.ones((10, 1), dtype=np.float32)
    locn = np.array([10.0])
    result = wx2d.extract_err(image, locn, 4)
    assert result[0] == 0

# Blackbox: mixed skip and extract
def test_extract_err_mixed():
    image = np.full((10, 3), 3.0, dtype=np.float32)
    locn = np.array([5.0, 0.0, 10.0])
    result = wx2d.extract_err(image, locn, 4)
    assert np.allclose(result, [3.0, 0.0, 0.0])

# Statement coverage: SCI output dtype float32
def test_apply_trace_sci_dtype(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mocker.patch("stistools.wx2d.extract", return_value=np.zeros(3, dtype=np.float32))
    image = np.ones((4, 3), dtype=np.float32)
    result = wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2)
    assert result.dtype == np.float32

# Branch coverage: DQ output dtype int16
def test_apply_trace_dq_dtype(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mocker.patch("stistools.wx2d.extract_i16", return_value=np.zeros(3, dtype=np.int16))
    image = np.ones((4, 3), dtype=np.int16)
    result = wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2, extname="DQ")
    assert result.dtype == np.int16

# Statement coverage: output rows divided by subdiv
def test_apply_trace_output_shape(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mocker.patch("stistools.wx2d.extract", return_value=np.zeros(3, dtype=np.float32))
    image = np.ones((6, 3), dtype=np.float32)
    result = wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 3)
    assert result.shape == (2, 3)

# Branch coverage: SCI branch calls extract
def test_apply_trace_sci_calls_extract(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mock_extract = mocker.patch("stistools.wx2d.extract", return_value=np.zeros(3, dtype=np.float32))
    image = np.ones((4, 3), dtype=np.float32)
    wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2, extname="SCI")
    assert mock_extract.call_count == 2

# Branch coverage: ERR branch calls extract_err
def test_apply_trace_err_calls_extract_err(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mock_err = mocker.patch("stistools.wx2d.extract_err", return_value=np.zeros(3, dtype=np.float32))
    image = np.ones((4, 3), dtype=np.float32)
    wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2, extname="ERR")
    assert mock_err.call_count == 2

# Branch coverage: DQ branch calls extract_i16
def test_apply_trace_dq_calls_extract_i16(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mock_dq = mocker.patch("stistools.wx2d.extract_i16", return_value=np.zeros(3, dtype=np.int16))
    image = np.ones((4, 3), dtype=np.int16)
    wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2, extname="DQ")
    assert mock_dq.call_count == 2

# Blackbox: locn uses subdiv factor
def test_apply_trace_locn_uses_subdiv(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mock_extract = mocker.patch("stistools.wx2d.extract", return_value=np.zeros(3, dtype=np.float32))
    image = np.ones((4, 3), dtype=np.float32)
    wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2)
    locn_arg = mock_extract.call_args_list[0].args[1]
    assert np.allclose(locn_arg, np.full(3, 0.5))

# Blackbox: interpolate_trace called with offset+i
def test_apply_trace_interpolate_trace_offset(mocker):
    mock_interp = mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mocker.patch("stistools.wx2d.extract", return_value=np.zeros(3, dtype=np.float32))
    image = np.ones((4, 3), dtype=np.float32)
    wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2, offset=5.0)
    assert mock_interp.call_args_list[0].args[2] == 5.0
    assert mock_interp.call_args_list[1].args[2] == 6.0

# Blackbox: shifta2 shifts locn
def test_apply_trace_shifta2_shifts_locn(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    mock_extract = mocker.patch("stistools.wx2d.extract", return_value=np.zeros(3, dtype=np.float32))
    image = np.ones((4, 3), dtype=np.float32)
    wx2d.apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]), 2, shifta2=1.0)
    locn_arg = mock_extract.call_args_list[0].args[1]
    assert np.allclose(locn_arg, np.full(3, 2.5))


# --- kd_apply_trace tests ---

# Statement coverage: output same shape as input
def test_kd_apply_trace_output_shape(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    image = np.ones((4, 3), dtype=np.float32)
    result = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]))
    assert result.shape == (4, 3)

# Statement coverage: output dtype float32
def test_kd_apply_trace_output_dtype(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    image = np.ones((4, 3), dtype=np.float32)
    result = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]))
    assert result.dtype == np.float32

# Blackbox: constant image preserved (coefficients sum to 1)
def test_kd_apply_trace_constant_image(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(3))
    image = np.full((5, 3), 7.0, dtype=np.float32)
    result = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0., 0., 0.]]))
    assert np.allclose(result, 7.0, atol=1e-5)

# Blackbox: center pixel n0 coefficient 0.77 when s=0
def test_kd_apply_trace_center_coefficient(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(1))
    image = np.zeros((5, 1), dtype=np.float32)
    image[2, 0] = 1.0
    result = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0.]]))
    assert np.isclose(result[2, 0], 0.77)

# Blackbox: interpolate_trace called with offset+i
def test_kd_apply_trace_interpolate_trace_offset(mocker):
    mock_interp = mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(2))
    image = np.ones((3, 2), dtype=np.float32)
    wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0., 0.]]), offset=4.0)
    assert mock_interp.call_args_list[0].args[2] == 4.0
    assert mock_interp.call_args_list[1].args[2] == 5.0

# Branch coverage: negative indices clamped to 0 (row 0 boundary)
def test_kd_apply_trace_lower_index_clamp(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(1))
    image = np.zeros((5, 1), dtype=np.float32)
    image[1, 0] = 1.0
    result = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0.]]))
    assert np.isclose(result[0, 0], 0.165)

# Branch coverage: indices >= total clamped to total-1 (last row boundary)
def test_kd_apply_trace_upper_index_clamp(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(1))
    image = np.zeros((5, 1), dtype=np.float32)
    image[3, 0] = 1.0
    result = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0.]]))
    assert np.isclose(result[4, 0], 0.165)

# Blackbox: b and c coefficients applied when s != 0
def test_kd_apply_trace_nonzero_s(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.array([0.5]))
    image = np.zeros((5, 1), dtype=np.float32)
    image[3, 0] = 1.0
    result = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0.]]))
    # i=2: y=2.5, nint_y=2, s=0.5, n=2; np1 index hits row 3
    # result = 0.165*1 + 0.66*1*0.5 + 0.34*1*0.25 = 0.165+0.33+0.085 = 0.58
    assert np.isclose(result[2, 0], 0.58, atol=1e-5)

# Blackbox: shifta2 shifts interpolation row
def test_kd_apply_trace_shifta2(mocker):
    mocker.patch("stistools.wx2d.interpolate_trace", return_value=np.zeros(1))
    image = np.zeros((5, 1), dtype=np.float32)
    image[2, 0] = 1.0
    result_no_shift = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0.]]))
    result_shifted = wx2d.kd_apply_trace(image, np.array([0.]), np.array([[0.]]), shifta2=1.0)
    assert not np.allclose(result_no_shift, result_shifted)


# --- kd_resampling tests ---

# Branch coverage: original_nrows == nrows no padding
def test_kd_resampling_full_size(mocker):
    mocker.patch("stistools.wx2d.kd_apply_trace",
                 return_value=np.ones((4, 3), dtype=np.float32))
    mocker.patch("stistools.wx2d.apply_trace",
                 return_value=np.full((4, 3), 2.0, dtype=np.float32))
    img = np.zeros((4, 3), dtype=np.float32)
    errimg = np.zeros((4, 3), dtype=np.float32)
    result, err_result = wx2d.kd_resampling(
        img, errimg, 4, 4, 3, (0, 4),
        np.array([0.]), np.array([[0., 0., 0.]]), 0., 0.)
    assert result.shape == (4, 3)
    assert np.allclose(result, 1.0)
    assert np.allclose(err_result, 2.0)

# Branch coverage: original_nrows > nrows pads with zeros
def test_kd_resampling_subset_padded(mocker):
    mocker.patch("stistools.wx2d.kd_apply_trace",
                 return_value=np.ones((2, 3), dtype=np.float32))
    mocker.patch("stistools.wx2d.apply_trace",
                 return_value=np.full((2, 3), 5.0, dtype=np.float32))
    img = np.zeros((2, 3), dtype=np.float32)
    errimg = np.zeros((2, 3), dtype=np.float32)
    result, err_result = wx2d.kd_resampling(
        img, errimg, 5, 2, 3, (1, 3),
        np.array([0.]), np.array([[0., 0., 0.]]), 0., 0.)
    assert result.shape == (5, 3)
    assert np.allclose(result[0], 0.0)
    assert np.allclose(result[1:3], 1.0)
    assert np.allclose(result[3:], 0.0)
    assert np.allclose(err_result[1:3], 5.0)
    assert np.allclose(err_result[0], 0.0)
    assert np.allclose(err_result[3:], 0.0)

# Blackbox: image2 oversampled by subdiv=8
def test_kd_resampling_image2_subdiv(mocker):
    mocker.patch("stistools.wx2d.kd_apply_trace",
                 return_value=np.zeros((2, 3), dtype=np.float32))
    mock_apply = mocker.patch("stistools.wx2d.apply_trace",
                              return_value=np.zeros((2, 3), dtype=np.float32))
    img = np.zeros((2, 3), dtype=np.float32)
    errimg = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    wx2d.kd_resampling(img, errimg, 2, 2, 3, (0, 2),
                       np.array([0.]), np.array([[0., 0., 0.]]), 0., 0.)
    image2 = mock_apply.call_args.args[0]
    assert image2.shape == (16, 3)
    for j in range(8):
        assert np.array_equal(image2[j::8, :], errimg)

# Blackbox: kd_apply_trace receives img and passthrough args
def test_kd_resampling_kd_apply_trace_args(mocker):
    mock_kd = mocker.patch("stistools.wx2d.kd_apply_trace",
                           return_value=np.zeros((2, 3), dtype=np.float32))
    mocker.patch("stistools.wx2d.apply_trace",
                 return_value=np.zeros((2, 3), dtype=np.float32))
    img = np.ones((2, 3), dtype=np.float32)
    errimg = np.zeros((2, 3), dtype=np.float32)
    a2center = np.array([0.])
    a2displ = np.array([[0., 0., 0.]])
    wx2d.kd_resampling(img, errimg, 2, 2, 3, (0, 2),
                       a2center, a2displ, 3.5, 1.5)
    args = mock_kd.call_args.args
    assert args[0] is img
    assert args[1] is a2center
    assert args[2] is a2displ
    assert args[3] == 3.5
    assert args[4] == 1.5

# Blackbox: apply_trace called with subdiv=8 and extname="ERR"
def test_kd_resampling_apply_trace_args(mocker):
    mocker.patch("stistools.wx2d.kd_apply_trace",
                 return_value=np.zeros((2, 3), dtype=np.float32))
    mock_apply = mocker.patch("stistools.wx2d.apply_trace",
                              return_value=np.zeros((2, 3), dtype=np.float32))
    img = np.zeros((2, 3), dtype=np.float32)
    errimg = np.zeros((2, 3), dtype=np.float32)
    wx2d.kd_resampling(img, errimg, 2, 2, 3, (0, 2),
                       np.array([0.]), np.array([[0., 0., 0.]]), 4.0, 2.0)
    args = mock_apply.call_args.args
    assert args[3] == 8
    assert args[4] == 4.0
    assert args[5] == 2.0
    assert args[6] == "ERR"

# Statement coverage: result dtype float32
def test_kd_resampling_result_dtype(mocker):
    mocker.patch("stistools.wx2d.kd_apply_trace",
                 return_value=np.zeros((2, 3), dtype=np.float32))
    mocker.patch("stistools.wx2d.apply_trace",
                 return_value=np.zeros((2, 3), dtype=np.float32))
    img = np.zeros((2, 3), dtype=np.float32)
    errimg = np.zeros((2, 3), dtype=np.float32)
    result, err_result = wx2d.kd_resampling(
        img, errimg, 4, 2, 3, (0, 2),
        np.array([0.]), np.array([[0., 0., 0.]]), 0., 0.)
    assert result.dtype == np.float32
    assert err_result.dtype == np.float32


# --- wx2d_imset tests ---

@pytest.fixture
def imset_ft():
    sci = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.float32), name="SCI")
    err = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.float32), name="ERR")
    dq = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.int16), name="DQ")
    return fits.HDUList([fits.PrimaryHDU(), sci, err, dq])


def _out(tmp_path, name="output.fits"):
    path = str(tmp_path / name)
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)
    return path


# Branch coverage: rows None processes full image
def test_wx2d_imset_rows_none(mocker, tmp_path, imset_ft):
    out = _out(tmp_path)
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    wx2d.wx2d_imset(imset_ft, 1, out, None, None, "kd", "trace.fits", 3, 1, 1.0, None, None, None)
    with fits.open(out) as f:
        assert len(f) == 4  # primary + SCI + ERR + DQ

# Branch coverage: rows clamped to image bounds
def test_wx2d_imset_rows_clamped(mocker, tmp_path, imset_ft):
    out = _out(tmp_path)
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.int16))
    wx2d.wx2d_imset(imset_ft, 1, out, None, None, "kd", "trace.fits", 3, 1, 1.0, (1, 4), None, None)
    with fits.open(out) as f:
        assert len(f) == 4

# Branch coverage: subset rows adds history when imset==1
def test_wx2d_imset_adds_history(mocker, tmp_path, imset_ft):
    out = _out(tmp_path)
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.int16))
    wx2d.wx2d_imset(imset_ft, 1, out, None, None, "kd", "trace.fits", 3, 1, 1.0, (1, 4), None, None)
    assert "rows from" in str(imset_ft[0].header.get("HISTORY", ""))

# Branch coverage: algorithm kd calls kd_resampling
def test_wx2d_imset_kd_algorithm(mocker, tmp_path, imset_ft):
    out = _out(tmp_path)
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mock_kd = mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    wx2d.wx2d_imset(imset_ft, 1, out, None, None, "kd", "trace.fits", 3, 1, 1.0, None, None, None)
    mock_kd.assert_called_once()

# Branch coverage: algorithm wavelet calls wavelet_resampling
def test_wx2d_imset_wavelet_algorithm(mocker, tmp_path, imset_ft):
    out = _out(tmp_path)
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mock_wv = mocker.patch("stistools.wx2d.wavelet_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    wx2d.wx2d_imset(imset_ft, 1, out, None, None, "wavelet", "trace.fits", 3, 1, 1.0, None, None, None)
    mock_wv.assert_called_once()

# Branch coverage: wavelengths None skips compute_wavelengths
def test_wx2d_imset_wavelengths_none_skips(mocker, tmp_path, imset_ft):
    out = _out(tmp_path)
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    mock_wl = mocker.patch("stistools.wx2d.wavelen.compute_wavelengths")
    wx2d.wx2d_imset(imset_ft, 1, out, None, None, "kd", "trace.fits", 3, 1, 1.0, None, None, None)
    mock_wl.assert_not_called()

# Branch coverage: wavelengths set writes to wavelength file
def test_wx2d_imset_wavelengths_written(mocker, tmp_path, imset_ft):
    out = _out(tmp_path, "sci.fits")
    wl_file = _out(tmp_path, "wl.fits")
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    mocker.patch("stistools.wx2d.wavelen.compute_wavelengths", return_value=np.zeros((5, 4)))
    wx2d.wx2d_imset(imset_ft, 1, out, wl_file, "PERFORM", "kd", "trace.fits", 3, 1, 1.0, None, None, None)
    with fits.open(wl_file) as f:
        assert f[0].header.get("helcorr") == "COMPLETE"

# Branch coverage: helcorr None reads from primary header
def test_wx2d_imset_helcorr_from_header(mocker, tmp_path, imset_ft):
    out = _out(tmp_path, "sci.fits")
    wl_file = _out(tmp_path, "wl.fits")
    imset_ft[0].header["helcorr"] = "PERFORM"
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    mocker.patch("stistools.wx2d.wavelen.compute_wavelengths", return_value=np.zeros((5, 4)))
    wx2d.wx2d_imset(imset_ft, 1, out, wl_file, None, "kd", "trace.fits", 3, 1, 1.0, None, None, None)
    with fits.open(wl_file) as f:
        assert f[0].header.get("helcorr") == "COMPLETE"

# Branch coverage: sclamp nonNONE forces helcorr OMIT
def test_wx2d_imset_sclamp_forces_omit(mocker, tmp_path, imset_ft):
    out = _out(tmp_path, "sci.fits")
    wl_file = _out(tmp_path, "wl.fits")
    imset_ft[0].header["sclamp"] = "D2"
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    mocker.patch("stistools.wx2d.wavelen.compute_wavelengths", return_value=np.zeros((5, 4)))
    wx2d.wx2d_imset(imset_ft, 1, out, wl_file, "PERFORM", "kd", "trace.fits", 3, 1, 1.0, None, None, None)
    with fits.open(wl_file) as f:
        assert f[0].header.get("helcorr") == "OMIT"

# Branch coverage: helcorr not PERFORM sets OMIT in wavelength header
def test_wx2d_imset_helcorr_omit_in_header(mocker, tmp_path, imset_ft):
    out = _out(tmp_path, "sci.fits")
    wl_file = _out(tmp_path, "wl.fits")
    mocker.patch("stistools.wx2d.get_trace", return_value=(np.array([0.]), np.array([[0., 0., 0., 0.]])))
    mocker.patch("stistools.wx2d.kd_resampling", return_value=(imset_ft[1].data, imset_ft[2].data))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((5, 4), dtype=np.int16))
    mocker.patch("stistools.wx2d.wavelen.compute_wavelengths", return_value=np.zeros((5, 4)))
    wx2d.wx2d_imset(imset_ft, 1, out, wl_file, "OMIT", "kd", "trace.fits", 3, 1, 1.0, None, None, None)
    with fits.open(wl_file) as f:
        assert f[0].header.get("helcorr") == "OMIT"


# --- stis_psf tests ---

# Statement coverage: center x==0 returns 1.0
def test_stis_psf_center():
    assert wx2d.stis_psf(0., 1.) == 1.0

# Blackbox: exponent is -2 not -1
def test_stis_psf_exponent():
    result = wx2d.stis_psf(1., 1.)
    assert np.isclose(result, 0.25)

# Blackbox: width a scales x
def test_stis_psf_width_scales():
    assert np.isclose(wx2d.stis_psf(2., 2.), wx2d.stis_psf(1., 1.))

# Branch coverage: negative x gives same as positive
def test_stis_psf_symmetric():
    assert wx2d.stis_psf(-1., 1.) == wx2d.stis_psf(1., 1.)

# Blackbox: larger x reduces value
def test_stis_psf_monotone_decreasing():
    assert wx2d.stis_psf(2., 1.) < wx2d.stis_psf(1., 1.)

# Blackbox: larger a increases value at fixed x
def test_stis_psf_wider_psf_higher_value():
    assert wx2d.stis_psf(1., 2.) > wx2d.stis_psf(1., 1.)


# --- wavelet_resampling tests ---

@pytest.fixture
def wv_setup():
    nrows, ncols = 3, 4
    img = np.ones((nrows, ncols), dtype=np.float32)
    errimg = np.ones((nrows, ncols), dtype=np.float32)
    hdu = fits.ImageHDU(data=img.copy(), name="SCI")
    a2center = np.array([0.])
    a2displ = np.zeros((1, ncols), dtype=np.float32)
    return hdu, img, errimg, a2center, a2displ


# Statement coverage: returns two float32 arrays of correct shape
def test_wavelet_resampling_shape_and_dtype(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    result, err_result = wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 0., None, None)
    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    assert err_result.shape == (3, 4)

# Branch coverage: subsampled None skips file write
def test_wavelet_resampling_subsampled_none_skips(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    mock_open = mocker.patch("stistools.wx2d.fits.open")
    wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 0., None, None)
    mock_open.assert_not_called()

# Branch coverage: subsampled set writes to file
def test_wavelet_resampling_subsampled_written(mocker, tmp_path, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    sub_file = str(tmp_path / "sub.fits")
    fits.HDUList([fits.PrimaryHDU()]).writeto(sub_file)
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 0., sub_file, None)
    with fits.open(sub_file) as f:
        assert len(f) == 2  # primary + SCI

# Branch coverage: psf_width 0 skips convolution
def test_wavelet_resampling_no_psf_skips_convolve(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mock_conv = mocker.patch("stistools.wx2d.convolve.convolve")
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 0., None, None)
    mock_conv.assert_not_called()

# Branch coverage: psf_width > 0 calls convolve per column
def test_wavelet_resampling_psf_calls_convolve(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mock_conv = mocker.patch("stistools.wx2d.convolve.convolve",
                             return_value=np.zeros(6, dtype=np.float32))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 1.3, None, None)
    assert mock_conv.call_count == 4  # once per column

# Branch coverage: convolved None skips convolved file write
def test_wavelet_resampling_convolved_none_skips(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mocker.patch("stistools.wx2d.convolve.convolve",
                 return_value=np.zeros(6, dtype=np.float32))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    mock_open = mocker.patch("stistools.wx2d.fits.open")
    wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 1.3, None, None)
    mock_open.assert_not_called()

# Branch coverage: convolved set writes convolved file
def test_wavelet_resampling_convolved_written(mocker, tmp_path, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    cnv_file = str(tmp_path / "cnv.fits")
    fits.HDUList([fits.PrimaryHDU()]).writeto(cnv_file)
    mocker.patch("stistools.wx2d.convolve.convolve",
                 return_value=np.zeros(6, dtype=np.float32))
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 1.3, None, cnv_file)
    with fits.open(cnv_file) as f:
        assert len(f) == 2  # primary + SCI

# Branch coverage: original_nrows > nrows pads result to full size
def test_wavelet_resampling_padded_result(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    result, err_result = wx2d.wavelet_resampling(
        hdu, img, errimg, 5, 3, 4, (1, 4), a2center, a2displ, 0., 0., 1, 3, 2, 0., None, None)
    assert result.shape == (5, 4)
    assert err_result.shape == (5, 4)

# Branch coverage: original_nrows == nrows uses direct apply_trace shape
def test_wavelet_resampling_full_image_shape(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mocker.patch("stistools.wx2d.apply_trace", return_value=np.zeros((3, 4), dtype=np.float32))
    result, err_result = wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 0., None, None)
    assert result.shape == (3, 4)

# Blackbox: apply_trace called with SCI and ERR extnames
def test_wavelet_resampling_apply_trace_extnames(mocker, wv_setup):
    hdu, img, errimg, a2center, a2displ = wv_setup
    mock_at = mocker.patch("stistools.wx2d.apply_trace",
                           return_value=np.zeros((3, 4), dtype=np.float32))
    wx2d.wavelet_resampling(
        hdu, img, errimg, 3, 3, 4, (0, 3), a2center, a2displ, 0., 0., 1, 3, 2, 0., None, None)
    extnames = [call.args[6] for call in mock_at.call_args_list]
    assert "SCI" in extnames
    assert "ERR" in extnames


# --- wx2d (top-level) tests ---

@pytest.fixture
def wx2d_input(tmp_path):
    """Minimal 1-imset input FITS file."""
    phdr = fits.Header()
    phdr["nextend"] = 3
    phdr["X2DCORR"] = "PERFORM"
    sci = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.float32), name="SCI")
    err = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.float32), name="ERR")
    dq = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.int16), name="DQ")
    path = str(tmp_path / "input.fits")
    fits.HDUList([fits.PrimaryHDU(header=phdr), sci, err, dq]).writeto(path)
    return path


# Branch coverage: psf_width=0 clears convolved before kd check
def test_wx2d_psf_zero_clears_convolved():
    # kd + convolved normally raises ValueError, but psf_width=0 clears convolved first
    with pytest.raises(FileNotFoundError):
        wx2d.wx2d("invalid.fits", "out.fits", algorithm="kd",
                  convolved="cnv.fits", psf_width=0.)

# Branch coverage: ltm2_2 != 1.0 raises RuntimeError
def test_wx2d_binned_y_raises(tmp_path):
    sci_hdr = fits.Header()
    sci_hdr["ltm2_2"] = 2.0
    sci = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.float32), header=sci_hdr, name="SCI")
    err = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.float32), name="ERR")
    dq = fits.ImageHDU(data=np.zeros((5, 4), dtype=np.int16), name="DQ")
    phdr = fits.Header()
    phdr["nextend"] = 3
    phdr["X2DCORR"] = "PERFORM"
    path = str(tmp_path / "binned.fits")
    fits.HDUList([fits.PrimaryHDU(header=phdr), sci, err, dq]).writeto(path)
    with pytest.raises(RuntimeError):
        wx2d.wx2d(path, str(tmp_path / "out.fits"), trace=np.array([0.]))

# Branch coverage: trace array adds array history message
def test_wx2d_trace_array_history(mocker, wx2d_input, tmp_path):
    mocker.patch("stistools.wx2d.wx2d_imset")
    out = str(tmp_path / "out.fits")
    wx2d.wx2d(wx2d_input, out, trace=np.array([0., 0., 0., 0.]))
    with fits.open(out) as f:
        assert any("trace array" in str(h) for h in f[0].header["HISTORY"])

# Branch coverage: trace string adds filename history message
def test_wx2d_trace_string_history(mocker, wx2d_input, tmp_path):
    mocker.patch("stistools.wx2d.wx2d_imset")
    mocker.patch("stistools.wx2d.trace_name", return_value="mytrace.fits")
    out = str(tmp_path / "out.fits")
    wx2d.wx2d(wx2d_input, out, trace="mytrace.fits")
    with fits.open(out) as f:
        assert any("trace file" in str(h) for h in f[0].header["HISTORY"])

# Branch coverage: wavelengths set creates wavelengths output file
def test_wx2d_wavelengths_file_created(mocker, wx2d_input, tmp_path):
    mocker.patch("stistools.wx2d.wx2d_imset")
    out = str(tmp_path / "out.fits")
    wl = str(tmp_path / "wl.fits")
    wx2d.wx2d(wx2d_input, out, wavelengths=wl, trace=np.array([0.]))
    with fits.open(wl) as f:
        assert f[0].header.get("filename") == "wl.fits"

# Branch coverage: subsampled set creates subsampled output file
def test_wx2d_subsampled_file_created(mocker, wx2d_input, tmp_path):
    mocker.patch("stistools.wx2d.wx2d_imset")
    out = str(tmp_path / "out.fits")
    sub = str(tmp_path / "sub.fits")
    wx2d.wx2d(wx2d_input, out, trace=np.array([0.]), subsampled=sub)
    with fits.open(sub) as f:
        assert f[0].header.get("filename") == "sub.fits"

# Branch coverage: convolved set with psf_width>0 creates convolved file
def test_wx2d_convolved_file_created(mocker, wx2d_input, tmp_path):
    mocker.patch("stistools.wx2d.wx2d_imset")
    out = str(tmp_path / "out.fits")
    cnv = str(tmp_path / "cnv.fits")
    wx2d.wx2d(wx2d_input, out, trace=np.array([0.]), convolved=cnv, psf_width=1.3)
    with fits.open(cnv) as f:
        assert f[0].header.get("filename") == "cnv.fits"

# Blackbox: wx2d_imset called once per imset
def test_wx2d_imset_loop_count(mocker, tmp_path):
    phdr = fits.Header()
    phdr["nextend"] = 6
    phdr["X2DCORR"] = "PERFORM"
    mk = lambda name: fits.ImageHDU(data=np.zeros((5, 4), dtype=np.float32), name=name)
    mk_dq = lambda: fits.ImageHDU(data=np.zeros((5, 4), dtype=np.int16), name="DQ")
    path = str(tmp_path / "two_imsets.fits")
    fits.HDUList([fits.PrimaryHDU(header=phdr),
                  mk("SCI"), mk("ERR"), mk_dq(),
                  mk("SCI"), mk("ERR"), mk_dq()]).writeto(path)
    mock_imset = mocker.patch("stistools.wx2d.wx2d_imset")
    wx2d.wx2d(path, str(tmp_path / "out.fits"), trace=np.array([0.]))
    assert mock_imset.call_count == 2

# Blackbox: helcorr uppercased before passing to wx2d_imset
def test_wx2d_helcorr_uppercased(mocker, wx2d_input, tmp_path):
    mock_imset = mocker.patch("stistools.wx2d.wx2d_imset")
    wx2d.wx2d(wx2d_input, str(tmp_path / "out.fits"),
              helcorr="perform", trace=np.array([0.]))
    assert mock_imset.call_args.args[4] == "PERFORM"
