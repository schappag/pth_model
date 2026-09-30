"""Smooth interpolation, stimulus and sensitivity functions."""

import numpy as np

from ptg_model.parameters import STIM, SENS_ENDPOINTS, CA_MAX, P_MIN


def check_range(calcium, phosphate):
    """Raise ValueError if calcium (mg/dL) or phosphate (mg/dL) is outside the model range."""
    if np.any(np.asarray(calcium) > CA_MAX):
        raise ValueError(f"ionized calcium {calcium} mg/dL above CA_MAX = {CA_MAX}")
    if np.any(np.asarray(phosphate) < P_MIN):
        raise ValueError(f"phosphate {phosphate} mg/dL below P_MIN = {P_MIN}")


def smooth_pw(x, endpoints, alpha=80):
    """Smooth piecewise linear interpolation through endpoints of shape (2, N)."""
    endpoints_x, endpoints_y = endpoints
    beta = endpoints_x[1:-1]
    jp = (endpoints_y[1:] - endpoints_y[:-1]) / (endpoints_x[1:] - endpoints_x[:-1])
    bp = (jp[-1] + jp[0]) / 2
    cp = (jp[1:] - jp[:-1]) / 2
    ap = endpoints_y[0] - np.sum(cp * np.abs(beta))
    a_p = ap - np.sum(cp * beta)
    b_p = bp + np.sum(cp)
    x = np.asarray(x, dtype=float)
    z = -alpha * (x[..., None] - beta)
    out = a_p + b_p * x + np.sum(cp * np.logaddexp(0, z), axis=-1) * 2 / alpha
    return out[()]


def stim(val, param):
    """Stimulus of a deviation from the reference level; param is 'c', 'p' or 'd'."""
    c1, c2, k, l = STIM[param]
    return l / (1 + np.exp(-k * (val - c1))) + l / (1 + np.exp(-k * (val - c2))) - l


def sens(c, d):
    """Sensitivity to calcium and calcitriol, normalised to 1 at c = d = 1."""
    avg = (c + d) / 2
    return smooth_pw(avg, SENS_ENDPOINTS) / smooth_pw(1, SENS_ENDPOINTS)
