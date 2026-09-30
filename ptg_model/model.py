"""System of ODEs describing PTG biology."""

import numpy as np

from ptg_model.parameters import (
    C_OPT,
    P_OPT,
    D_OPT,
    CA_MG_TO_MM,
    K1,
    K2,
    KA,
    DEGRAD,
    PROLIF,
    PROD,
    CLEAR,
    KCA,
    RCA,
    KD,
    RD,
    SENS_COUPLING,
    ADAPT_GAIN,
    MEM_CA_SENS,
    MEM_D_SENS,
    MEM_P_SENS,
    MEM_CA_DEGRAD,
    MEM_CA_PROD,
    MEM_CA_PROLIF,
    MEM_P_DEGRAD,
    MEM_P_PROD,
    MEM_P_PROLIF,
    FB_PTH_RATE,
    FB_PTH_GAIN,
    FB_D_RATE,
    FB_D_GAIN,
)
from ptg_model.utils import stim, sens, check_range
from ptg_model.core_functions import rate_adj, release_rate, phosphate_factor, gland_growth

N_STATES = 23

STATE_NAMES = [
    "resting cells",
    "active cells",
    "stored PTH",
    "plasma PTH",
    "calcium sensitivity",
    "calcitriol sensitivity",
    "calcium stimulus, sensitivity",
    "calcitriol stimulus, sensitivity",
    "phosphate stimulus, sensitivity",
    "degradation adaptation",
    "calcium stimulus, degradation",
    "production adaptation",
    "calcium stimulus, production",
    "proliferation adaptation",
    "calcium stimulus, proliferation",
    "sensed calcium",
    "sensed calcitriol",
    "phosphate stimulus, degradation",
    "phosphate stimulus, production",
    "phosphate stimulus, proliferation",
    "carrying capacity",
    "calcium factor, PTH feedback",
    "calcium factor, calcitriol feedback",
]


def _value(x, t):
    return x(t) if callable(x) else x


def _memory(s, y, rate):
    return (s * (1 - np.sign(s) * y) - y) * rate


def deriv(
    t, y, s0, calcium, phosphate, calcitriol, gfr=1.0,
    calcium_clamp=True, pth_ref=None, calcitriol_ref=None,
):
    """Right hand side for solve_ivp.

    calcium (mg/dL), phosphate (mg/dL), calcitriol (ng/L) and gfr (relative
    residual renal function) are constants or functions of t. s0 is the
    initial cell mass, y[0] + y[1]. Without calcium clamp, calcium is scaled
    by y[21] * y[22], which follow plasma PTH and calcitriol relative to
    pth_ref (model units, y0[3]) and calcitriol_ref (ng/L).
    """
    c = _value(calcium, t)
    p = _value(phosphate, t)
    d = _value(calcitriol, t)
    gfr = _value(gfr, t)
    if not calcium_clamp:
        if pth_ref is None or calcitriol_ref is None:
            raise ValueError("calcium_clamp=False requires pth_ref and calcitriol_ref")
        c = c * y[21] * y[22]
    check_range(c, p)

    sc = stim(c - C_OPT, "c")
    sp = stim(p - P_OPT, "p")
    sd = stim(d - D_OPT, "d")
    sc_sensed = stim(y[15] - C_OPT, "c")
    rel = release_rate(y[15] * CA_MG_TO_MM, phosphate_factor(p))

    dydt = np.zeros(N_STATES)
    dydt[0] = -K1 * y[0] + K2 * y[1]
    dydt[1] = (
        K1 * y[0]
        - K2 * y[1]
        - KA * y[1]
        + rate_adj(y[13], PROLIF) * y[1] * np.log(y[20] / (y[1] + y[0]))
    )
    dydt[2] = y[0] * rate_adj(y[11], PROD) - rel * y[2] - rate_adj(y[9], DEGRAD) * y[2]
    dydt[3] = rel * y[2] - y[3] * rate_adj(gfr, CLEAR)

    dydt[4] = KCA * ((y[6] - 2 * y[8]) * y[4] + SENS_COUPLING * (y[5] - 1)) + RCA * (1 - y[4])
    dydt[5] = KD * ((y[7] - 2 * y[8]) * y[5] + SENS_COUPLING * (y[4] - 1)) + RD * (1 - y[5])
    dydt[6] = _memory(sc, y[6], MEM_CA_SENS)
    dydt[7] = _memory(sd, y[7], MEM_D_SENS)
    dydt[8] = _memory(sp, y[8], MEM_P_SENS)

    dydt[9] = ADAPT_GAIN * KCA * (y[10] - y[17]) * y[9] + RCA * (1 - y[9])
    dydt[10] = _memory(sc_sensed, y[10], MEM_CA_DEGRAD)
    dydt[11] = ADAPT_GAIN * KCA * (y[12] - y[18]) * y[11] + RCA * (1 - y[11])
    dydt[12] = _memory(sc_sensed, y[12], MEM_CA_PROD)
    dydt[13] = ADAPT_GAIN * KCA * (y[14] - y[19]) * y[13] + RCA * (1 - y[13])
    dydt[14] = _memory(sc_sensed, y[14], MEM_CA_PROLIF)

    dydt[15] = sens(y[4], y[5]) * c - y[15]
    dydt[16] = sens(y[4], y[5]) * d - y[16]

    dydt[17] = _memory(sp, y[17], MEM_P_DEGRAD)
    dydt[18] = _memory(sp, y[18], MEM_P_PROD)
    dydt[19] = _memory(sp, y[19], MEM_P_PROLIF)

    dydt[20] = gland_growth(y[0] + y[1], s0)

    target_pth, target_d = 1.0, 1.0
    if not calcium_clamp:
        target_pth = 1 + np.tanh(FB_PTH_GAIN * (y[3] - pth_ref))
        target_d = 1 + np.tanh(FB_D_GAIN * (d - calcitriol_ref))
    dydt[21] = FB_PTH_RATE * (target_pth - y[21])
    dydt[22] = FB_D_RATE * (target_d - y[22])

    # zero derivatives below 1e-12 for numerical stability
    dydt[np.abs(dydt) < 1e-12] = 0
    return dydt
