"""Steady state of the PTG model for given patient levels."""

import numpy as np

from ptg_model.parameters import (
    C_OPT,
    P_OPT,
    D_OPT,
    CA_MG_TO_MM,
    PTH_PG_TO_MODEL,
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
)
from ptg_model.utils import stim, sens, check_range
from ptg_model.core_functions import rate_adj, release_rate, phosphate_factor


def _memory_steady(s):
    return s / (1 + s * np.sign(s))


def steady_state(calcium=C_OPT, phosphate=P_OPT, calcitriol=D_OPT, pth=40.0, gfr=1.0):
    """Steady state for calcium (mg/dL), phosphate (mg/dL), calcitriol (ng/L),
    plasma PTH (pg/mL) and relative residual renal function gfr."""
    check_range(calcium, phosphate)
    yc = _memory_steady(stim(calcium - C_OPT, "c"))
    yp = _memory_steady(stim(phosphate - P_OPT, "p"))
    yd = _memory_steady(stim(calcitriol - D_OPT, "d"))

    a = KCA * (yc - 2 * yp)
    b = KCA * SENS_COUPLING
    aq = KD * (yd - 2 * yp)
    bq = KD * SENS_COUPLING
    ysc = (b - b * (bq - RD) / (aq - RD) - RCA) / (a - b * bq / (aq - RD) - RCA)
    ysd = (-RD + bq * (1 - ysc)) / (aq - RD)

    csensed = sens(ysc, ysd) * calcium
    dsensed = sens(ysc, ysd) * calcitriol
    cstar = _memory_steady(stim(csensed - C_OPT, "c"))
    csstar = RCA / (RCA - ADAPT_GAIN * KCA * (cstar - yp))

    rel = release_rate(csensed * CA_MG_TO_MM, phosphate_factor(phosphate))
    s4 = pth * PTH_PG_TO_MODEL
    s3 = s4 * rate_adj(gfr, CLEAR) / rel
    s1 = (rel + rate_adj(csstar, DEGRAD)) * s3 / rate_adj(csstar, PROD)
    s2 = K1 * s1 / K2
    cap = np.exp(KA / rate_adj(csstar, PROLIF)) * (s1 + s2)

    return np.array(
        [
            s1, s2, s3, s4,
            ysc, ysd, yc, yd, yp,
            csstar, cstar, csstar, cstar, csstar, cstar,
            csensed, dsensed,
            yp, yp, yp,
            cap,
            1.0, 1.0,
        ],
        dtype=float,
    )
