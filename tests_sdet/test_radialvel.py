import numpy as np
import pytest
from stistools import radialvel


# Statement coverage: radialVel returns a finite scalar.
def test_radialVel_returns_scalar():
    v = radialvel.radialVel(180.0, 0.0, 55000.0)
    assert isinstance(v, float)
    assert np.isfinite(v)


# Blackbox: radial velocity magnitude is bounded by Earth orbital speed (~30 km/s).
def test_radialVel_bounded():
    v = radialvel.radialVel(45.0, 30.0, 55000.0)
    assert abs(v) < 35.0


# Blackbox: opposite directions give opposite-signed radial velocities.
def test_radialVel_opposite_directions():
    v1 = radialvel.radialVel(0.0, 0.0, 55000.0)
    v2 = radialvel.radialVel(180.0, 0.0, 55000.0)
    assert v1 == pytest.approx(-v2)


# Statement coverage: earthVel returns a length-3 float64 vector.
def test_earthVel_shape():
    vel = radialvel.earthVel(55000.0)
    assert vel.shape == (3,)
    assert vel.dtype == np.float64


# Blackbox: Earth's orbital speed is ~29.78 km/s.
def test_earthVel_magnitude():
    vel = radialvel.earthVel(55000.0)
    speed = np.linalg.norm(vel)
    assert 29.0 < speed < 30.5


# Statement coverage: precess of a 1-D unit vector returns a unit vector.
def test_precess_unit_vector():
    target = np.array([1.0, 0.0, 0.0])
    out = radialvel.precess(55000.0, target)
    assert out.shape == (3,)
    assert np.linalg.norm(out) == pytest.approx(1.0)


# Branch coverage: precess at REFDATE is approximately the identity.
def test_precess_at_refdate():
    target = np.array([0.5, 0.5, np.sqrt(0.5)])
    out = radialvel.precess(radialvel.REFDATE, target)
    assert np.allclose(out, target)


# Blackbox: precess accepts a (3, n) matrix and preserves shape.
def test_precess_matrix_shape():
    target = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    out = radialvel.precess(55000.0, target)
    assert out.shape == (3, 3)
