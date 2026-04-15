import pytest
import numpy as np

from stistools import r_util

# Blackbox: unix style works
def test_expandFileNameUnixStyle(mocker):
    mocker.patch.dict("os.environ", {"lref": "/data/reference"})
    assert r_util.expandFileName("$lref/name_dqi.fits") == "/data/reference/name_dqi.fits"

# Blackbox: iraf style
@pytest.mark.xfail(reason="Known Failure")
def test_expandFileNameIrafStyle(mocker):
    mocker.patch.dict("os.environ", {"lref": "/data/reference"})
    assert r_util.expandFileName("lref$name_dqi.fits") == "/data/reference/name_dqi.fits"

# Blackbox: no env var characters
def test_expandFileNameNoEnvVar():
    assert r_util.expandFileName("name_dqi.fits") == "name_dqi.fits"

# Whitebox: attempt to hit the n/a case
def test_expandFileNameNotApplicable():
    assert r_util.expandFileName("n/a") == "n/a"

# Blackbox: single x value
def test_interpolate_single_x():
    x = np.array([1])
    values = np.array([1])
    xp = 1
    assert r_util.interpolate(x, values, xp) == 1

# Blackbox: x out of bounds low
def test_interpolate_x_out_of_bounds_low():
    x = np.array([1, 2])
    values = np.array([1, 2])
    xp = 0.5
    assert r_util.interpolate(x, values, xp) == 1

# Blackbox: x out of bounds high
def test_interpolate_x_out_of_bounds_high():
    x = np.array([1, 2])
    values = np.array([1, 2])
    xp = 2.5
    assert r_util.interpolate(x, values, xp) == 2

# Blackbox: x in bounds
def test_interpolate_x_in_bounds():
    x = np.array([1, 2])
    values = np.array([1, 2])
    xp = 1.5
    assert r_util.interpolate(x, values, xp) == 1.5

# Blackbox: empty array case
@pytest.mark.xfail(reason="Known Failure - empty array")
def test_interpolate_empty_array():
    x = np.array([])
    values = np.array([])
    xp = 1
    assert r_util.interpolate(x, values, xp) == None

# Blackbox: x equals not first element
def test_interpolate_x_equals_not_first_element():
    x = np.array([1, 1.5, 2, 2.5, 3])
    values = np.array([1, 1.5, 2, 2.5, 3])
    xp = 2
    assert r_util.interpolate(x, values, xp) == 2

# Whitebox: two equal x values
def test_interpolate_two_equal_x_values():
    x = np.array([0.5, 0.5, 1, 1.5])
    values = np.array([0.5, 0.5, 1, 1.5])
    xp = 0.5
    assert r_util.interpolate(x, values, xp) == 0.5

# Blackbox: detects not sorted x values
@pytest.mark.xfail(reason="Known Failure - not sorted x")
def test_interpolate_not_sorted_x():
    x = np.array([1, 3, 2])
    values = np.array([1, 2, 3])
    xp = 2
    with pytest.raises(Exception):
        r_util.interpolate(x, values, xp)