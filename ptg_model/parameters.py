"""Model parameters. Time is in hours."""

import numpy as np

# healthy reference levels
C_OPT = 5.0  # ionized calcium, mg/dL
P_OPT = 3.6  # phosphate, mg/dL
D_OPT = 40.0  # calcitriol, ng/L

CA_MAX = 5.5  # ionized calcium, mg/dL
P_MIN = 3.0  # phosphate, mg/dL

# unit conversions
CA_MG_TO_MM = 1 / 4  # calcium, mg/dL to mmol/L
P_MG_TO_MM = 0.323  # phosphate, mg/dL to mmol/L
PTH_PG_TO_MODEL = 3 / 9.434  # PTH, pg/mL to model units

# cell cycle
K2 = 0.03 * 60  # active to resting
K1 = 4 * K2  # resting to active
KA = 0.001 * 60  # apoptosis

# PTG functions, [rate, fraction retained at zero input]
DEGRAD = [0.012 * 60, 0.2]
PROLIF = [0.03 * 60, 2]
PROD = [6.6 / 0.1 * 60, 2]
CLEAR = [0.632 * 60, 0.7]

# CaSR and release
KC = 2  
C_REF = 1.1881
RELEASE_HILL = 100
RELEASE_MAX = 0.14 * 60
RELEASE_MIN = 0.001 * 60

# phosphate effect on release
PHOS_HIGH = 0.3
PHOS_LOW = 0.15
PHOS_HILL = 4.5

# gland growth
GROWTH_RATE = 6* 3e-6
GROWTH_DECEL = 3.0

# sensitivity to calcium and calcitriol
KCA = 0.5
RCA = 0.5
KD = 0.001
RD = 0.001
SENS_COUPLING = 0.1
SENS_ENDPOINTS = np.array([[0, 0.5, 1, 2, 10], [0.8, 0.85, 1, 1.01, 1.05]])

# adaptation of degradation, production and proliferation
ADAPT_GAIN = 50

# stimulus functions, (c1, c2, k, l)
STIM = {
    "c": (-2.2, 2.2, 3, 1),
    "p": (-2.5, 2.5, 2.5, 1),
    "d": (-30, 30, 0.1, 1),
}

# calcium feedback, used only without calcium clamp
FB_PTH_RATE = 0.001 / 4 * (C_OPT / 4) * 0.4
FB_PTH_GAIN = 0.1
FB_D_RATE = 0.05
FB_D_GAIN = 0.1

# stimulus memory rates
MEM_CA_SENS = 0.15
MEM_D_SENS = 0.015
MEM_P_SENS = 0.05
MEM_CA_DEGRAD = 10
MEM_CA_PROD = 0.1
MEM_CA_PROLIF = 0.35
MEM_P_DEGRAD = 0.1
MEM_P_PROD = 0.03
MEM_P_PROLIF = 0.005
