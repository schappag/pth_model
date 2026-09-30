import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ptg_model.model import deriv, N_STATES
from ptg_model.steady_state import steady_state
from ptg_model.parameters import C_OPT, P_OPT, D_OPT, CA_MAX, P_MIN


@pytest.fixture
def y0():
    return steady_state(pth=40.0)


def test_steady_state_shape(y0):
    assert y0.shape == (N_STATES,)
    assert np.all(np.isfinite(y0))


def test_healthy_steady_state_is_stationary(y0):
    dydt = deriv(0.0, y0, y0[0] + y0[1], C_OPT, P_OPT, D_OPT)
    assert dydt.shape == (N_STATES,)
    assert np.allclose(dydt, 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "c, p, d, gfr",
    [(0.9 * C_OPT, P_OPT, D_OPT, 1.0), (C_OPT, 1.3 * P_OPT, 0.6 * D_OPT, 0.2)],
)
def test_patient_steady_state_is_stationary(c, p, d, gfr):
    y = steady_state(c, p, d, 80.0, gfr)
    dydt = deriv(0.0, y, y[0] + y[1], c, p, d, gfr)
    assert np.allclose(dydt[:4], 0.0, atol=1e-8)


def test_callable_inputs_match_constants(y0):
    s0 = y0[0] + y0[1]
    y = y0 * 1.01
    a = deriv(1.0, y, s0, C_OPT, P_OPT, D_OPT, 1.0)
    b = deriv(1.0, y, s0, lambda t: C_OPT, lambda t: P_OPT, lambda t: D_OPT, lambda t: 1.0)
    np.testing.assert_array_equal(a, b)


def test_hypocalcemia_raises_pth(y0):
    sol = solve_ivp(
        deriv, (0, 1), y0, args=(y0[0] + y0[1], 0.9 * C_OPT, P_OPT, D_OPT), method="BDF"
    )
    assert sol.success
    assert sol.y[3, -1] > 1.5 * y0[3]


def test_pth_scale_invariance():
    y1, y2 = steady_state(pth=20.0), steady_state(pth=60.0)
    np.testing.assert_allclose(y2[:4] / y1[:4], 3.0)
    np.testing.assert_allclose(y2[4:20], y1[4:20])


def test_unclamped_steady_state_is_stationary(y0):
    dydt = deriv(0.0, y0, y0[0] + y0[1], C_OPT, P_OPT, D_OPT, 1.0, False, y0[3], D_OPT)
    assert np.allclose(dydt, 0.0, atol=1e-10)


def test_unclamped_requires_references(y0):
    with pytest.raises(ValueError):
        deriv(0.0, y0, y0[0] + y0[1], C_OPT, P_OPT, D_OPT, calcium_clamp=False)


def test_calcitriol_loss_lowers_calcium_without_clamp(y0):
    sol = solve_ivp(
        deriv, (0, 48), y0,
        args=(y0[0] + y0[1], C_OPT, P_OPT, 0.5 * D_OPT, 1.0, False, y0[3], D_OPT),
        method="BDF",
    )
    assert sol.success
    assert sol.y[22, -1] < 1.0
    assert sol.y[3, -1] > y0[3]


@pytest.mark.parametrize("c, p", [(CA_MAX + 0.01, P_OPT), (C_OPT, P_MIN - 0.01)])
def test_out_of_range_inputs_rejected(y0, c, p):
    with pytest.raises(ValueError):
        steady_state(c, p)
    with pytest.raises(ValueError):
        deriv(0.0, y0, y0[0] + y0[1], c, p, D_OPT)


def test_boundary_steady_state_is_admissible():
    y = steady_state(CA_MAX, P_MIN)
    assert np.all(y[:4] > 0) and y[20] > 0
    assert np.all(y[[9, 11, 13]] > 0)


def test_boundary_stays_bounded_over_months(y0):
    sol = solve_ivp(
        deriv, (0, 24 * 180), y0, args=(y0[0] + y0[1], CA_MAX, P_MIN, D_OPT), method="BDF"
    )
    assert sol.success
    assert np.all(np.isfinite(sol.y))
    assert np.all(sol.y[[9, 11, 13]] < 10)
