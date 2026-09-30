import numpy as np
import pytest
from ptg_model.utils import smooth_pw, stim, sens
from ptg_model.parameters import STIM


@pytest.fixture
def simple_endpoints():
    endpoints = np.array([[0, 1, 2], [0, 1, 0]])
    return endpoints


def test_smooth_pw_scalar_output(simple_endpoints):
    x = 0.5
    val = smooth_pw(x, simple_endpoints)
    assert np.isscalar(val) or np.size(val) == 1
    assert np.isfinite(val)


def test_smooth_pw_vectorized_behavior(simple_endpoints):
    x_vals = np.linspace(0, 2, 20)
    y_vals = [smooth_pw(x, simple_endpoints) for x in x_vals]
    assert np.all(np.isfinite(y_vals))
    assert np.ptp(y_vals) > 0, "Output range should not be zero"


@pytest.mark.parametrize("param", ["c", "p", "d"])
def test_stim_behavior(param):
    x_vals = np.linspace(-5, 5, 100)
    out = stim(x_vals, param)
    assert out.shape == x_vals.shape
    assert np.all(np.isfinite(out))

    if param in ["c", "p"]:
        assert np.any(np.abs(out) > 0), f"stim({param}) returned all zeros unexpectedly"


def test_stim_rejects_unknown_parameter():
    with pytest.raises(KeyError):
        stim(0.0, "x")


@pytest.mark.parametrize("param", ["c", "p", "d"])
def test_stim_zero_at_reference(param):
    assert stim(0.0, param) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("param", ["c", "p", "d"])
def test_stim_odd_symmetry(param):
    x = np.linspace(0, 50, 200)
    np.testing.assert_allclose(stim(-x, param), -stim(x, param), atol=1e-12)


@pytest.mark.parametrize("param", ["c", "p", "d"])
def test_stim_increasing(param):
    c1, c2, k, _ = STIM[param]
    out = stim(np.linspace(c1 - 5 / k, c2 + 5 / k, 1000), param)
    assert np.all(np.diff(out) > 0)


@pytest.mark.parametrize("param", ["c", "p", "d"])
def test_stim_saturates(param):
    c1, c2, k, l = STIM[param]
    assert stim(c2 + 20 / k, param) == pytest.approx(l)
    assert stim(c1 - 20 / k, param) == pytest.approx(-l)


def test_stim_flat_near_reference():
    assert abs(stim(0.01, "c")) < 1e-3


def test_sens_scalar_output():
    val = sens(1.0, 1.0)
    assert np.isscalar(val)
    assert np.isfinite(val)


def test_sens_array_output():
    c_vals = np.linspace(0.5, 2.0, 10)
    d_vals = np.linspace(0.5, 2.0, 10)
    out = sens(c_vals, d_vals)
    assert out.shape == c_vals.shape
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)
    assert np.isclose(
        out[5], 1.0, atol=0.2
    ), "sens not approximately normalized at midrange"


def test_sens_monotonic_trend():
    c_vals = np.linspace(0, 10, 50)
    d_vals = np.linspace(0, 10, 50)
    out = sens(c_vals, d_vals)
    diffs = np.diff(out)
    assert np.all(
        diffs >= -1e-6
    ), "sens should not decrease with higher calcium/calcitriol"
