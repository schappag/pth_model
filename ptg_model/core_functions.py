"""Rate adjustment, CaSR occupancy, PTH release and gland growth."""

import numpy as np

from ptg_model.parameters import (
    KC,
    C_REF,
    RELEASE_HILL,
    RELEASE_MAX,
    RELEASE_MIN,
    P_OPT,
    P_MG_TO_MM,
    PHOS_HIGH,
    PHOS_LOW,
    PHOS_HILL,
    GROWTH_RATE,
    GROWTH_DECEL,
)


def rate_adj(c, parameterset):
    """Rate that falls linearly below c = 1 to a fraction of its reference value."""
    r, a = parameterset
    return np.where(c < 1, (r - a * r) * c + a * r, r)


def casr_occupancy(c):
    """CaSR occupancy at ionized calcium c (mmol/L)."""
    c = np.asarray(c, dtype=float)
    return c / (KC + c)


def phosphate_factor(p):
    """Scaling of maximal release by phosphate p (mg/dL), 1 at P_OPT."""
    kp = P_OPT * P_MG_TO_MM

    def f(x):
        x = x * P_MG_TO_MM
        return PHOS_HIGH + (PHOS_LOW - PHOS_HIGH) * x**PHOS_HILL / (x**PHOS_HILL + kp**PHOS_HILL)

    return f(P_OPT) / f(p)


def release_rate(c, rp=1.0):
    """PTH release rate at ionized calcium c (mmol/L) and phosphate factor rp."""
    ratio = casr_occupancy(c) / casr_occupancy(C_REF)
    a = RELEASE_MAX * rp
    return (a - RELEASE_MIN) / (1 + ratio**RELEASE_HILL) + RELEASE_MIN


def gland_growth(cell_mass, s0):
    """Growth of the gland carrying capacity, zero at or below s0."""
    fold = cell_mass / s0
    return GROWTH_RATE * max(0.0, fold - 1.0) ** (2 / 3) / max(fold, 1.0) ** GROWTH_DECEL
