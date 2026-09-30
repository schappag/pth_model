# Parathyroid Gland Systems Biology (Python Implementation)

## Overview
This repository provides a Python implementation of the quantitative systems physiology model of parathyroid gland (PTG) biology published in its basic form in [1].

The model describes the regulation of parathyroid hormone (PTH) secretion, intracellular degradation, synthesis, clearance, and proliferation, incorporating effects of extracellular ionized calcium, phosphate, and calcitriol. It is particularly focused on the altered PTG biology found in patients with chronic kidney disease (CKD) on hemodialysis and serves as a tool for studying secondary hyperparathyroidism.

## Key Features
- The calcium sensing receptor (CaSR) drives PTH release through its occupancy by ionized calcium.
- Timescales range from the release of stored PTH within minutes to cellular proliferation over months.
- Intracellular PTH degradation, PTH production and cell proliferation adapt to sustained changes in calcium and phosphate.
- The sensitivity of the gland to calcium and calcitriol adapts to their plasma levels.
- Model predictions have been validated against published data for acute hypocalcemia, hysteresis and the development of secondary hyperparathyroidism [1, 2].

## Repository Structure
- `ptg_model/`
  - `parameters.py`: all model parameters
  - `model.py`: system of ODEs
  - `steady_state.py`: steady state for given calcium, phosphate, calcitriol and PTH levels
  - `core_functions.py`: CaSR occupancy, PTH release, rate adjustment and gland growth
  - `utils.py`: smooth piecewise linear function, stimulus and sensitivity functions
- `example_notebook.ipynb`: hysteresis, acute hypocalcemia, and chronic changes in phosphate and calcitriol
- `tests/`: unit tests

## Usage
```python
from scipy.integrate import solve_ivp
from ptg_model.model import deriv
from ptg_model.steady_state import steady_state
from ptg_model.parameters import C_OPT, P_OPT, D_OPT

y0 = steady_state(pth=40.0)
s0 = y0[0] + y0[1]
sol = solve_ivp(deriv, (0, 2), y0, args=(s0, 0.9 * C_OPT, P_OPT, D_OPT), method="BDF")
```
Calcium, phosphate and calcitriol may be constants or functions of time (hours). Plasma PTH is `sol.y[3]`.
By default calcium is clamped (`calcium_clamp=True`): ionized calcium is an input and does not respond to PTH. This is the intended mode of the model. The calcium feedback (`calcium_clamp=False`, with `pth_ref` and `calcitriol_ref`) and residual renal function (`gfr`) are experimental. Use them only with parameters that are justified for the question at hand; the feedback parameters `FB_*` are not validated.

Ionized calcium must not exceed `CA_MAX` (5.5 mg/dL) and phosphate must not fall below `P_MIN` (3.0 mg/dL); `steady_state` and `deriv` raise a `ValueError` otherwise.

## Installation
```bash
git clone https://github.com/schappag/pth_model.git
cd pth_model
pip install -r requirements.txt
```

## Running Tests
```bash
pytest
```

## Contributing
Use Issues to report bugs or model inconsistencies and to suggest extensions.

## References
1. Schappacher-Tilp G, Cherif A, Fuertinger DH, Bushinsky D, Kotanko P. A mathematical model of parathyroid gland biology. Physiol Rep. 2019;7(7):e14045. doi: 10.14814/phy2.14045
2. Pirklbauer M, Bushinsky DA, Kotanko P, Schappacher-Tilp G. Personalized prediction of short- and long-term PTH changes in maintenance hemodialysis patients. Front Med (Lausanne). 2021;8:704970. doi: 10.3389/fmed.2021.704970
