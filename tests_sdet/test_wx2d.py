import pytest
from stistools import wx2d

def test_wx2d_invalid_algorithm():
    with pytest.raises(ValueError):
        wx2d.wx2d("input.fits", "output.fits", algorithm="invalid")