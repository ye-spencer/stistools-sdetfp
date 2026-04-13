import pytest
from stistools.orbit import HSTOrbit
import random
import math
from astropy.io import fits

@pytest.fixture
def valid_orbit_fits_dict(tmp_path):
    # Create a dummy FITS file with real looking values
    hdu = fits.PrimaryHDU()

    # Randomly generate values for the header
    random.seed(42)

    argperig = random.uniform(0, 2*math.pi)
    cirveloc = random.uniform(0, 10000)
    cosincli = random.uniform(-1, 1)
    ecbdx3 = random.uniform(-1, 1)
    eccentry = random.uniform(0, 1)
    eccentx2 = random.uniform(-1, 1)
    ecbdx4d3 = random.uniform(-1, 1)
    epchtime = random.uniform(0, 100000)
    esqdx5d2 = random.uniform(-1, 1)
    fdmeanan = random.uniform(0, 2*math.pi)
    hsthorb = random.uniform(0, 100000)
    meananom = random.uniform(0, 2*math.pi)
    rascascn = random.uniform(0, 2*math.pi)
    rcargper = random.uniform(0, 10000)
    rcascnrv = random.uniform(0, 10000)
    sdmeanan = random.uniform(0, 2*math.pi)
    semilrec = random.uniform(0, 100000)
    sineincl = random.uniform(-1, 1)

    orbit_dict = {
        "argperig": argperig,
        "cirveloc": cirveloc,
        "cosincli": cosincli,
        "ecbdx3": ecbdx3,
        "eccentry": eccentry,
        "eccentx2": eccentx2,
        "ecbdx4d3": ecbdx4d3,
        "epchtime": epchtime,
        "esqdx5d2": esqdx5d2,
        "fdmeanan": fdmeanan,
        "hsthorb": hsthorb,
        "meananom": meananom,
        "rascascn": rascascn,
        "rcargper": rcargper,
        "rcascnrv": rcascnrv,
        "sdmeanan": sdmeanan,
        "semilrec": semilrec,
        "sineincl": sineincl
    }

    return orbit_dict

@pytest.fixture
def orbit_fits_factory(tmp_path):
    def _orbit_fits(orbit_dict):
        hdu = fits.PrimaryHDU()
        for key, value in orbit_dict.items():
            hdu.header[key] = value
        hdu.writeto(tmp_path / "test.spt")
        return tmp_path / "test.spt"
    return _orbit_fits

# Blackbox: test with valid input in fits file
def test_orbit_init_valid(orbit_fits_factory, valid_orbit_fits_dict):

    valid_orbit_fits = orbit_fits_factory(valid_orbit_fits_dict)
    
    orb = HSTOrbit(valid_orbit_fits)

    assert orb.orb["argperig"] == valid_orbit_fits_dict["argperig"]
    assert orb.orb["cirveloc"] == valid_orbit_fits_dict["cirveloc"]
    assert orb.orb["cosincli"] == valid_orbit_fits_dict["cosincli"]
    assert orb.orb["ecbdx3"] == valid_orbit_fits_dict["ecbdx3"]
    assert orb.orb["eccentry"] == valid_orbit_fits_dict["eccentry"]
    assert orb.orb["eccentx2"] == valid_orbit_fits_dict["eccentx2"]
    assert orb.orb["ecbdx4d3"] == valid_orbit_fits_dict["ecbdx4d3"]
    assert orb.orb["epchtime"] == valid_orbit_fits_dict["epchtime"]
    assert orb.orb["esqdx5d2"] == valid_orbit_fits_dict["esqdx5d2"]
    assert orb.orb["fdmeanan"] == valid_orbit_fits_dict["fdmeanan"]
    assert orb.orb["hsthorb"] == valid_orbit_fits_dict["hsthorb"]
    assert orb.orb["meananom"] == valid_orbit_fits_dict["meananom"]
    assert orb.orb["rascascn"] == valid_orbit_fits_dict["rascascn"]
    assert orb.orb["rcargper"] == valid_orbit_fits_dict["rcargper"]
    assert orb.orb["rcascnrv"] == valid_orbit_fits_dict["rcascnrv"]
    assert orb.orb["sdmeanan"] == valid_orbit_fits_dict["sdmeanan"]
    assert orb.orb["semilrec"] == valid_orbit_fits_dict["semilrec"]
    assert orb.orb["sineincl"] == valid_orbit_fits_dict["sineincl"]

# Blackbox: test without required keywords
def test_orbit_init_invalid(orbit_fits_factory):
    invalid_orbit_fits = orbit_fits_factory({})
    with pytest.raises(KeyError):
        orb = HSTOrbit(invalid_orbit_fits)

# Blackbox: test with non-existent file
def test_orbit_init_nonexistent():
    with pytest.raises(FileNotFoundError):
        orb = HSTOrbit("nonexistent.spt")

# Blackbox: test getOrbitper
def test_orbit_getOrbitper(orbit_fits_factory, valid_orbit_fits_dict):
    valid_orbit_fits = orbit_fits_factory(valid_orbit_fits_dict)
    orb = HSTOrbit(valid_orbit_fits)
    assert orb.getOrbitper() == 2. * valid_orbit_fits_dict["hsthorb"]

# Blackbox: test getPos with valid input at time 0.0
def test_orbit_getPosTimeAtZero(orbit_fits_factory, valid_orbit_fits_dict):
    valid_orbit_fits = orbit_fits_factory(valid_orbit_fits_dict)
    orb = HSTOrbit(valid_orbit_fits)
    x_hst, v_hst = orb.getPos(0.0)
    assert isinstance(x_hst, list)
    assert len(x_hst) == 3
    assert isinstance(v_hst, list)
    assert len(v_hst) == 3
    assert pytest.approx(x_hst[0]) == -24.677526808946286
    assert pytest.approx(x_hst[1]) == 26.50903772942742
    assert pytest.approx(x_hst[2]) == 10.23932033557949
    assert pytest.approx(v_hst[0]) == -362194.80209929636
    assert pytest.approx(v_hst[1]) == -884999.0542985148
    assert pytest.approx(v_hst[2]) == 56680.94815155411

# Blackbox: test getPos with valid input at time 1.0
def test_orbit_getPosTimeAtOne(orbit_fits_factory, valid_orbit_fits_dict):
    valid_orbit_fits = orbit_fits_factory(valid_orbit_fits_dict)
    orb = HSTOrbit(valid_orbit_fits)
    x_hst, v_hst = orb.getPos(1.0)
    assert isinstance(x_hst, list)
    assert len(x_hst) == 3
    assert isinstance(v_hst, list)
    assert len(v_hst) == 3
    assert pytest.approx(x_hst[0]) == -7.2428454733125465
    assert pytest.approx(x_hst[1]) == -44.247287264250815
    assert pytest.approx(x_hst[2]) == 0.604601656014065
    assert pytest.approx(v_hst[0]) == 1562359.257013531
    assert pytest.approx(v_hst[1]) == -221569.65868519255
    assert pytest.approx(v_hst[2]) == 99947.18825066954