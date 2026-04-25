import numpy as np
from stistools.defringe._findloc import find_loc
from stistools.defringe import _findloc

class DummyPoly:
    class C:
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.c1 = self.C(1.0)
        self.c2 = self.C(0.0) 


class DummyFitter:
    def __call__(self, model, x, y):
        return DummyPoly()


def test_find_loc_zero_curvature(monkeypatch):
    monkeypatch.setattr(_findloc.fitting, "LinearLSQFitter", lambda: DummyFitter())

    img = np.ones((100, 100))
    img[50:55, :] = 100
    loc = find_loc(img)

    assert loc is None

def test_find_loc_peak():
    img = np.ones((100, 100))
    img[50:55, :] = 100
    loc = find_loc(img)

    assert loc is not None
    assert 40 < loc < 70


def test_find_loc_flat_profile():
    img = np.ones((100, 100))
    loc = find_loc(img)

    assert isinstance(loc, float)


def test_find_loc_negative_values_branch():
    img = np.ones((100, 100))

    img[45:50, :] = 100
    img[30:35, :] = -20

    loc = find_loc(img)
    assert loc is not None


def test_find_loc_custom_params():
    img = np.ones((120, 120))
    img[60:65, :] = 200

    loc = find_loc(img, low_frac=0.1, high_frac=0.9)
    assert loc is not None