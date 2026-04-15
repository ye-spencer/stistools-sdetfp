from stistools import evaldisp

# statement coverage of newton()
def test_newton():
    coeff = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cenwave = 500.0
    x = 499.0
    wl = evaldisp.newton(x, coeff, cenwave)
    assert wl == 500.0

# statement coverage of evalDisp()
def test_evalDisp():
    coeff = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    wl = 500.0
    pix_number = evaldisp.evalDisp(coeff, wl)
    assert pix_number == 499.0
