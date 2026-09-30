import numpy as np
import pytest

from ptg_model.core_functions import (
    rate_adj,
    casr_occupancy,
    phosphate_factor,
    release_rate,
    gland_growth,
)
from ptg_model.parameters import (
    KC,
    C_REF,
    C_OPT,
    P_OPT,
    CA_MG_TO_MM,
    RELEASE_MAX,
    RELEASE_MIN,
    GROWTH_RATE,
    GROWTH_DECEL,
)


def test_rate_adj_scalar():
    r, a = 10, 0.2
    assert np.isclose(rate_adj(0.5, [r, a]), (r - a * r) * 0.5 + a * r)
    assert np.isclose(rate_adj(1.5, [r, a]), r)


def test_rate_adj_vectorized():
    r, a = 8, 0.1
    c = np.array([0.2, 0.8, 1.2])
    np.testing.assert_allclose(rate_adj(c, [r, a]), np.where(c < 1, (r - a * r) * c + a * r, r))


def test_casr_occupancy():
    c = np.linspace(0.5, 2.0, 20)
    np.testing.assert_allclose(casr_occupancy(c), c / (KC + c), rtol=1e-12)
    assert np.all(np.diff(casr_occupancy(c)) > 0)
    assert casr_occupancy(1e9) == pytest.approx(1.0, rel=1e-6)


def test_phosphate_factor():
    assert phosphate_factor(P_OPT) == pytest.approx(1.0)
    assert phosphate_factor(1.2 * P_OPT) > 1.0 > phosphate_factor(0.8 * P_OPT)


def test_release_rate_decreases_with_calcium():
    rates = release_rate(np.linspace(0.8, 1.6, 200))
    assert np.all(np.diff(rates) <= 1e-12)
    assert np.all((rates >= RELEASE_MIN) & (rates <= RELEASE_MAX))


def test_release_rate_half_maximal_at_reference():
    assert release_rate(C_REF) == pytest.approx((RELEASE_MAX + RELEASE_MIN) / 2, rel=1e-12)


def test_release_rate_scales_with_phosphate():
    assert release_rate(1.0, rp=0.5) < release_rate(1.0, rp=2.0)


def test_resting_gland_can_surge():
    rest = release_rate(C_OPT * CA_MG_TO_MM)
    assert C_REF < C_OPT * CA_MG_TO_MM
    assert RELEASE_MAX / rest > 4.0


def test_gland_growth():
    s0 = 1.0
    for fold in (0.5, 0.9, 1.0):
        assert gland_growth(fold * s0, s0) == 0.0
    for fold in (1.05, 1.5, 2.0, 5.0):
        want = GROWTH_RATE * (fold - 1.0) ** (2 / 3) / fold**GROWTH_DECEL
        assert gland_growth(fold * s0, s0) == pytest.approx(want, rel=1e-12)


def test_gland_growth_decelerates():
    rates = np.array([gland_growth(f, 1.0) for f in np.linspace(1.01, 20.0, 400)])
    assert rates[-1] < rates.max() / 100
