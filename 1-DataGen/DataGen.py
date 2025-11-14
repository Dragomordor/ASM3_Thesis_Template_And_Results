# -----------------------------
## Import Libraries - Standard
# -----------------------------
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import configparser
from scipy.integrate import solve_ivp

# -----------------------------
# Paths
# -----------------------------
current_dir = Path.cwd()
top_dir = current_dir.parent if current_dir.name == '1-DataGen' else current_dir

libs_dir_path = top_dir / "0-Libs"
config_dir   = top_dir / "0-Config"
data_dir     = top_dir / "0-Data"

# Output dirs
super_dir        = data_dir / "00-SuperRes"
highres_dir      = data_dir / "0-HighRes"
routine_dir      = data_dir / "1-Routine"
active_dir       = data_dir / "2-Active"
long_active_dir  = data_dir / "3-LongActive"
long_routine_dir = data_dir / "4-LongRoutine"

for d in [super_dir, highres_dir, routine_dir, active_dir, long_active_dir, long_routine_dir]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Import libs
# -----------------------------
if str(libs_dir_path) not in sys.path:
    sys.path.append(str(libs_dir_path))

from asm3_model import ode_system_wrapper
from plant_config import get_reactor_initial_values

# -----------------------------
# Helpers
# -----------------------------
def _nonneg(x):
    """Clip scalars/arrays to be >= 0."""
    return np.clip(x, 0.0, None)


def ar1_deterministic_series(start_val, phi, n):
    """AR(1) ; X0 = start_val, Xt = phi*X_{t-1}."""
    x = np.zeros(n)
    x[0] = start_val
    for i in range(1, n):
        x[i] = phi * x[i-1]
        if x[i] < 0:
            x[i] = 0.0
    return x

def states_AR1_variation(Influent_SAV, t_eval, phi, seed=None):
    """
    Returns shape (n, 13): [time] + 12 variables.
    Influent_SAV[var] = (start_val, variance)
    """
    if seed is not None:
        np.random.seed(seed)
    n = len(t_eval)
    out = np.zeros((n, 13))           # time + 12 vars
    out[:, 0] = t_eval
    # init
    for j, var_name in enumerate(Influent_SAV.keys(), 1):
        out[0, j] = float(Influent_SAV[var_name][0])

    # AR(1) with innovations
    for i in range(1, n):
        for j, var_name in enumerate(Influent_SAV.keys(), 1):
            var = float(Influent_SAV[var_name][1])
            out[i, j] = phi * out[i-1, j] + np.random.normal(0.0, np.sqrt(var)) if var > 0 else phi * out[i-1, j]
            if out[i, j] < 0.0:
                out[i, j] = 0.0
    return out


def flow_AR1_Sinusoidal_variation(flow_influent_params, t_eval, phi, seed=None):
    """
    Returns shape (n, 2): [time, flowrate]
    flow_influent_params keys: start_val, ar_variance, sin_variance, amplitude, period_days, phase_pi
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(t_eval)
    flow_data = np.zeros((n, 2))
    flow_data[:, 0] = t_eval
    start_val  = float(flow_influent_params['start_val'])
    ar_var     = float(flow_influent_params.get('ar_variance', 0.0))
    sin_var    = float(flow_influent_params.get('sin_variance', 0.0))
    A          = float(flow_influent_params['amplitude'])
    period     = float(flow_influent_params['period_days'])  # PERIOD (days)
    phase_pi   = float(flow_influent_params.get('phase_pi', 0.0))

    # init
    flow_data[0, 1] = start_val
    ar = np.zeros(n)
    ar[0] = 0.0

    omega = 2.0 * np.pi / period
    phase = phase_pi * np.pi

    for i in range(1, n):
        # AR(1) drift
        ar[i] = phi * ar[i-1] + (np.random.normal(0.0, np.sqrt(ar_var)) if ar_var > 0 else 0.0)

        # Sinusoid with step noise on the sinusoid term
        sin_step = A * np.sin(omega * t_eval[i] + phase) + (np.random.normal(0.0, np.sqrt(sin_var)) if sin_var > 0 else 0.0)

        q = start_val + ar[i] + sin_step
        flow_data[i, 1] = q if q > 0.0 else 0.0

    return flow_data

def compounds_from_states(states_array):
    """
    Build compounds [Time, Flowrate, COD, NH4+NH3, NO3+NO2, TKN, Alkalinity, TSS]
    from states array with columns:
      0 T, 1 Q, 2 S_O2, 3 S_I, 4 S_S, 5 S_NH4, 6 S_N2, 7 S_NOX, 8 S_ALK,
      9 X_I, 10 X_S, 11 X_H, 12 X_STO, 13 X_A, 14 X_SS
    """
    # keep time & flow
    T = states_array[:, 0]
    Q = _nonneg(states_array[:, 1])

    # compounds (clip each to ≥ 0)
    COD = _nonneg(states_array[:, 3] + states_array[:, 4] + states_array[:, 9] +
                  states_array[:,10] + states_array[:,12] + states_array[:,11] +
                  states_array[:,13])
    NH4 = _nonneg(states_array[:, 5])
    NOX = _nonneg(states_array[:, 7])
    TKN = _nonneg(states_array[:, 5] + states_array[:, 6])
    ALK = _nonneg(states_array[:, 8])
    TSS = _nonneg(states_array[:,14])

    return np.column_stack([T, Q, COD, NH4, NOX, TKN, ALK, TSS])


def select_subset_from_super(super_array, super_ppd, *, points_per_day=None, points_per_week=None, t_final=None):
    """
    Exact subsample from super grid using integer strides.
    - super_ppd: super points per day (e.g., 20)
    - points_per_day or points_per_week: choose one
    - t_final: clip by time (days)
    """
    n = super_array.shape[0]
    if t_final is not None:
        # inclusive index up to t_final
        max_idx = int(round(t_final * super_ppd))
        max_idx = min(max_idx, n-1)
        arr = super_array[:max_idx+1]
    else:
        arr = super_array

    if points_per_day is not None:
        step_days = 1.0 / float(points_per_day)
    elif points_per_week is not None:
        step_days = 7.0 / float(points_per_week)
    else:
        raise ValueError("Specify points_per_day or points_per_week.")

    stride = int(round(step_days * super_ppd))
    stride = max(1, stride)

    idx = np.arange(0, arr.shape[0], stride, dtype=int)
    # Always include the last point of the window if exactly on grid
    if idx[-1] != arr.shape[0]-1:
        # Check if end aligns; if not, leave as-is (consistent stride)
        pass
    return arr[idx]

# -----------------------------
# Config
# -----------------------------
config = configparser.ConfigParser()
config.read(config_dir / "config.ini")

seed = int(config['OVERALL']['seed'])
rng = np.random.default_rng(seed)

# Time / sampling
tmax = int(config['DATA']['tmax'])  # days

# Super points/day (defaults to 20 if key missing)
try:
    super_ppd = int(config['DATA']['super_points_per_day'])
except Exception:
    super_ppd = 20  # requested default

# HighRes target = 4 pts/day (as requested)
highres_ppd = 4

# Other dataset horizons and sampling (from config)
t_final_routine       = float(config['DATA']['routine_days'])
t_final_long_routine  = float(config['DATA']['long_routine_days'])
t_final_active        = float(config['DATA']['active_days'])
t_final_long_active   = float(config['DATA']['long_active_days'])

routine_ppw           = float(config['DATA']['routine_points_per_week'])
long_routine_ppw      = float(config['DATA']['long_routine_points_per_week'])
active_ppd            = float(config['DATA']['active_points_per_day'])
long_active_ppd       = float(config['DATA']['long_active_points_per_day'])

if any(x > tmax for x in [t_final_routine, t_final_long_routine, t_final_active, t_final_long_active]):
    raise ValueError("Final time for routine, long routine, active or long active data cannot be more than tmax.")

# Process parameters
phi_AR1 = float(config['DATA']['phi_ar1'])

# Flow params (used for base dynamics)
influent_flow_start_val  = float(config['DATA']['influent_flow_start_val'])
influent_flow_ar_variance = float(config['DATA']['influent_flow_ar_variance'])
influent_flow_sin_variance = float(config['DATA']['influent_flow_sin_variance'])
influent_flow_period     = float(config['DATA']['influent_flow_period'])
influent_flow_amplitude  = float(config['DATA']['influent_flow_amplitude'])
influent_flow_phase      = float(config['DATA']['influent_flow_phase'])

# States starting values and noise std (post-process noise uses second value)
SO_2_constant   = float(config['DATA']['SO_2_constant'])
influent_S_I    = tuple(map(float, config['DATA']['influent_S_I'].split(',')))
influent_S_S    = tuple(map(float, config['DATA']['influent_S_S'].split(',')))
influent_S_NH4  = tuple(map(float, config['DATA']['influent_S_NH4'].split(',')))
influent_S_N2   = tuple(map(float, config['DATA']['influent_S_N2'].split(',')))
influent_S_NOX  = tuple(map(float, config['DATA']['influent_S_NOX'].split(',')))
influent_S_ALK  = tuple(map(float, config['DATA']['influent_S_ALK'].split(',')))
influent_X_I    = tuple(map(float, config['DATA']['influent_X_I'].split(',')))
influent_X_S    = tuple(map(float, config['DATA']['influent_X_S'].split(',')))
influent_X_H    = tuple(map(float, config['DATA']['influent_X_H'].split(',')))
influent_X_STO  = tuple(map(float, config['DATA']['influent_X_STO'].split(',')))
influent_X_A    = tuple(map(float, config['DATA']['influent_X_A'].split(',')))
influent_X_SS   = tuple(map(float, config['DATA']['influent_X_SS'].split(',')))

use_corrupted_data = config['DATA'].getboolean('use_corrupted_data')
effluent_S_I_corrupt_std   = float(config['DATA']['effluent_S_I_corrupt_std'])
effluent_S_S_corrupt_std   = float(config['DATA']['effluent_S_S_corrupt_std'])
effluent_S_NH4_corrupt_std = float(config['DATA']['effluent_S_NH4_corrupt_std'])
effluent_S_N2_corrupt_std  = float(config['DATA']['effluent_S_N2_corrupt_std'])
effluent_S_NOX_corrupt_std = float(config['DATA']['effluent_S_NOX_corrupt_std'])
effluent_S_ALK_corrupt_std = float(config['DATA']['effluent_S_ALK_corrupt_std'])
effluent_X_I_corrupt_std   = float(config['DATA']['effluent_X_I_corrupt_std'])
effluent_X_S_corrupt_std   = float(config['DATA']['effluent_X_S_corrupt_std'])
effluent_X_H_corrupt_std   = float(config['DATA']['effluent_X_H_corrupt_std'])
effluent_X_STO_corrupt_std = float(config['DATA']['effluent_X_STO_corrupt_std'])
effluent_X_A_corrupt_std   = float(config['DATA']['effluent_X_A_corrupt_std'])
effluent_X_SS_corrupt_std  = float(config['DATA']['effluent_X_SS_corrupt_std'])

# True params (unchanged)
true_param_k_H      = float(config['TRUEPARAMS']['true_param_k_H'])
true_param_K_X      = float(config['TRUEPARAMS']['true_param_K_X'])
true_param_k_STO    = float(config['TRUEPARAMS']['true_param_small_k_STO'])
true_param_eta_NOX  = float(config['TRUEPARAMS']['true_param_eta_NOX'])
true_param_K_O2     = float(config['TRUEPARAMS']['true_param_K_O2'])
true_param_K_NOX    = float(config['TRUEPARAMS']['true_param_K_NOX'])
true_param_K_S      = float(config['TRUEPARAMS']['true_param_K_S'])
true_param_K_STO    = float(config['TRUEPARAMS']['true_param_big_K_STO'])
true_param_mu_H     = float(config['TRUEPARAMS']['true_param_mu_H'])
true_param_K_NH4    = float(config['TRUEPARAMS']['true_param_K_NH4'])
true_param_K_ALK    = float(config['TRUEPARAMS']['true_param_K_ALK'])
true_param_b_H_O2   = float(config['TRUEPARAMS']['true_param_b_H_O2'])
true_param_b_H_NOX  = float(config['TRUEPARAMS']['true_param_b_H_NOX'])
true_param_b_STO_O2 = float(config['TRUEPARAMS']['true_param_b_STO_O2'])
true_param_b_STO_NOX= float(config['TRUEPARAMS']['true_param_b_STO_NOX'])
true_param_mu_A     = float(config['TRUEPARAMS']['true_param_mu_A'])
true_param_K_A_NH4  = float(config['TRUEPARAMS']['true_param_K_A_NH4'])
true_param_K_A_O2   = float(config['TRUEPARAMS']['true_param_K_A_O2'])
true_param_K_A_ALK  = float(config['TRUEPARAMS']['true_param_K_A_ALK'])
true_param_b_A_O2   = float(config['TRUEPARAMS']['true_param_b_A_O2'])
true_param_b_A_NOX  = float(config['TRUEPARAMS']['true_param_b_A_NOX'])
true_param_f_S_I    = float(config['TRUEPARAMS']['true_param_f_S_I'])
true_param_Y_STO_O2 = float(config['TRUEPARAMS']['true_param_Y_STO_O2'])
true_param_Y_STO_NOX= float(config['TRUEPARAMS']['true_param_Y_STO_NOX'])
true_param_Y_H_O2   = float(config['TRUEPARAMS']['true_param_Y_H_O2'])
true_param_Y_H_NOX  = float(config['TRUEPARAMS']['true_param_Y_H_NOX'])
true_param_Y_A      = float(config['TRUEPARAMS']['true_param_Y_A'])
true_param_f_X_I    = float(config['TRUEPARAMS']['true_param_f_X_I'])
true_param_i_N_S_I  = float(config['TRUEPARAMS']['true_param_i_N_S_I'])
true_param_i_N_S_S  = float(config['TRUEPARAMS']['true_param_i_N_S_S'])
true_param_i_N_X_I  = float(config['TRUEPARAMS']['true_param_i_N_X_I'])
true_param_i_N_X_S  = float(config['TRUEPARAMS']['true_param_i_N_X_S'])
true_param_i_N_BM   = float(config['TRUEPARAMS']['true_param_i_N_BM'])
true_param_i_SS_X_I = float(config['TRUEPARAMS']['true_param_i_SS_X_I'])
true_param_i_SS_X_S = float(config['TRUEPARAMS']['true_param_i_SS_X_S'])
true_param_i_SS_BM  = float(config['TRUEPARAMS']['true_param_i_SS_BM'])

r1_V = float(config['REACTOR']['r1_V'])

# -----------------------------
# SUPER GRID (20 pts/day default)
# -----------------------------
n_super = super_ppd * tmax + 1
t_eval_super = np.linspace(0.0, float(tmax), n_super)

# -----------------------------
# Build influent on SUPER grid (stochastic processes like before)
# -----------------------------
Influent_SAV = {
    "S_I":   influent_S_I, "S_S":   influent_S_S, "S_NH4": influent_S_NH4,
    "S_N2":  influent_S_N2,"S_NOX": influent_S_NOX, "S_ALK": influent_S_ALK,
    "X_I":   influent_X_I, "X_S":   influent_X_S, "X_H":   influent_X_H,
    "X_STO": influent_X_STO,"X_A":  influent_X_A, "X_SS":  influent_X_SS
}

# States with AR(1) innovations (as before)
states_super = states_AR1_variation(Influent_SAV, t_eval_super, phi=phi_AR1, seed=seed)

# Flow: PERIOD in days from config; PHASE in π-units; include per-step sinusoid noise
flow_params_super = {
    "start_val":   influent_flow_start_val,
    "ar_variance": influent_flow_ar_variance,   # 0 per your config; set >0 to enable AR innovations
    "sin_variance": influent_flow_sin_variance, # 40 per your config (per-step)
    "amplitude":   influent_flow_amplitude,
    "period_days": influent_flow_period,        # 2 days -> 0.5 cycles/day
    "phase_pi":    influent_flow_phase          # π-units (0..1 typical)
}
flow_super = flow_AR1_Sinusoidal_variation(flow_params_super, t_eval_super, phi=phi_AR1, seed=seed)

# Combine SUPER influent: [Time, Flow, S_O2, S_I, ... X_SS]
super_influent = np.zeros((n_super, 15))
super_influent[:, 0] = t_eval_super
super_influent[:, 1] = flow_super[:, 1]
super_influent[:, 2] = SO_2_constant
super_influent[:, 3:] = states_super[:, 1:]   # 12 vars

# -----------------------------
# Effluent simulation on SUPER grid
# -----------------------------
theta_true = {
    'k_H': true_param_k_H, 'K_X': true_param_K_X, 'k_STO': true_param_k_STO,
    'eta_NOX': true_param_eta_NOX, 'K_O2': true_param_K_O2, 'K_NOX': true_param_K_NOX,
    'K_S': true_param_K_S, 'K_STO': true_param_K_STO, 'mu_H': true_param_mu_H,
    'K_NH4': true_param_K_NH4, 'K_ALK': true_param_K_ALK,
    'b_H_O2': true_param_b_H_O2, 'b_H_NOX': true_param_b_H_NOX,
    'b_STO_O2': true_param_b_STO_O2, 'b_STO_NOX': true_param_b_STO_NOX,
    'mu_A': true_param_mu_A, 'K_A_NH4': true_param_K_A_NH4,
    'K_A_O2': true_param_K_A_O2, 'K_A_ALK': true_param_K_A_ALK,
    'b_A_O2': true_param_b_A_O2, 'b_A_NOX': true_param_b_A_NOX,
    'f_S_I': true_param_f_S_I, 'Y_STO_O2': true_param_Y_STO_O2, 'Y_STO_NOX': true_param_Y_STO_NOX,
    'Y_H_O2': true_param_Y_H_O2, 'Y_H_NOX': true_param_Y_H_NOX,
    'Y_A': true_param_Y_A, 'f_X_I': true_param_f_X_I,
    'i_N_S_I': true_param_i_N_S_I, 'i_N_S_S': true_param_i_N_S_S,
    'i_N_X_I': true_param_i_N_X_I, 'i_N_X_S': true_param_i_N_X_S,
    'i_N_BM': true_param_i_N_BM, 'i_SS_X_I': true_param_i_SS_X_I,
    'i_SS_X_S': true_param_i_SS_X_S, 'i_SS_BM': true_param_i_SS_BM
}
theta_true_array = np.array(list(theta_true.values()))

y0 = get_reactor_initial_values(top_dir)
reactor_volumes = [r1_V]

ode_system_true = lambda t, y: ode_system_wrapper(t, y, theta_true_array, super_influent, reactor_volumes)
sol_true = solve_ivp(fun=ode_system_true, t_span=(0.0, float(tmax)), y0=y0, t_eval=t_eval_super, method='RK45')

Qout = super_influent[:,1]
super_effluent = np.column_stack([sol_true.t, Qout, sol_true.y.T])

# Clip all state variables (cols 2..end) to be nonnegative
super_effluent[:, 1:] = _nonneg(super_effluent[:, 1:])

# Optional corruption (unchanged behavior, applied on SUPER then inherited by subsets)
if use_corrupted_data:
    super_effluent[:, 3]  += rng.normal(0, effluent_S_I_corrupt_std,   size=super_effluent.shape[0])
    super_effluent[:, 4]  += rng.normal(0, effluent_S_S_corrupt_std,   size=super_effluent.shape[0])
    super_effluent[:, 5]  += rng.normal(0, effluent_S_NH4_corrupt_std, size=super_effluent.shape[0])
    super_effluent[:, 6]  += rng.normal(0, effluent_S_N2_corrupt_std,  size=super_effluent.shape[0])
    super_effluent[:, 7]  += rng.normal(0, effluent_S_NOX_corrupt_std, size=super_effluent.shape[0])
    super_effluent[:, 8]  += rng.normal(0, effluent_S_ALK_corrupt_std, size=super_effluent.shape[0])
    super_effluent[:, 9]  += rng.normal(0, effluent_X_I_corrupt_std,   size=super_effluent.shape[0])
    super_effluent[:, 10] += rng.normal(0, effluent_X_S_corrupt_std,   size=super_effluent.shape[0])
    super_effluent[:, 11] += rng.normal(0, effluent_X_H_corrupt_std,   size=super_effluent.shape[0])
    super_effluent[:, 12] += rng.normal(0, effluent_X_STO_corrupt_std, size=super_effluent.shape[0])
    super_effluent[:, 13] += rng.normal(0, effluent_X_A_corrupt_std,   size=super_effluent.shape[0])
    super_effluent[:, 14] += rng.normal(0, effluent_X_SS_corrupt_std,  size=super_effluent.shape[0])

    super_effluent[:, 3:]  = _nonneg(super_effluent[:, 3:])
    print("Using corrupted effluent data")
else:
    print("Using true effluent data (no corruption)")

# -----------------------------
# Build compounds (SUPER)
# -----------------------------
super_influent_comp = compounds_from_states(super_influent)
super_effluent_comp = compounds_from_states(super_effluent)

# -----------------------------
# Save SUPER
# -----------------------------
pd.DataFrame(super_influent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(super_dir / 'Super_Influent_States.csv', index=False)
pd.DataFrame(super_effluent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(super_dir / 'Super_Effluent_States.csv', index=False)
pd.DataFrame(super_influent_comp, columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(super_dir / 'Super_Influent_Compounds.csv', index=False)
pd.DataFrame(super_effluent_comp, columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(super_dir / 'Super_Effluent_Compounds.csv', index=False)

# -----------------------------
# Exact subset: HIGHRES (4/day over full tmax)
# -----------------------------
highres_influent = select_subset_from_super(super_influent, super_ppd, points_per_day=highres_ppd, t_final=tmax)
highres_effluent = select_subset_from_super(super_effluent, super_ppd, points_per_day=highres_ppd, t_final=tmax)

pd.DataFrame(highres_influent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(highres_dir / 'HighRes_Influent_States.csv', index=False)
pd.DataFrame(highres_effluent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(highres_dir / 'HighRes_Effluent_States.csv', index=False)

pd.DataFrame(compounds_from_states(highres_influent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(highres_dir / 'HighRes_Influent_Compounds.csv', index=False)
pd.DataFrame(compounds_from_states(highres_effluent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(highres_dir / 'HighRes_Effluent_Compounds.csv', index=False)

# -----------------------------
# Exact subset: ACTIVE (points/day from config, first t_final_active days)
# -----------------------------
active_influent  = select_subset_from_super(super_influent, super_ppd, points_per_day=active_ppd, t_final=t_final_active)
active_effluent  = select_subset_from_super(super_effluent, super_ppd, points_per_day=active_ppd, t_final=t_final_active)

pd.DataFrame(active_influent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(active_dir / 'Active_Influent_States.csv', index=False)
pd.DataFrame(active_effluent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(active_dir / 'Active_Effluent_States.csv', index=False)

pd.DataFrame(compounds_from_states(active_influent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(active_dir / 'Active_Influent_Compounds.csv', index=False)
pd.DataFrame(compounds_from_states(active_effluent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(active_dir / 'Active_Effluent_Compounds.csv', index=False)

# -----------------------------
# Exact subset: LONG ACTIVE (points/day from config, first t_final_long_active days)
# -----------------------------
long_active_influent = select_subset_from_super(super_influent, super_ppd, points_per_day=long_active_ppd, t_final=t_final_long_active)
long_active_effluent = select_subset_from_super(super_effluent, super_ppd, points_per_day=long_active_ppd, t_final=t_final_long_active)

pd.DataFrame(long_active_influent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(long_active_dir / 'LongActive_Influent_States.csv', index=False)
pd.DataFrame(long_active_effluent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(long_active_dir / 'LongActive_Effluent_States.csv', index=False)

pd.DataFrame(compounds_from_states(long_active_influent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(long_active_dir / 'LongActive_Influent_Compounds.csv', index=False)
pd.DataFrame(compounds_from_states(long_active_effluent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(long_active_dir / 'LongActive_Effluent_Compounds.csv', index=False)

# -----------------------------
# Exact subset: ROUTINE (2 per week by default, first t_final_routine days)
# -----------------------------
routine_influent = select_subset_from_super(super_influent, super_ppd, points_per_week=routine_ppw, t_final=t_final_routine)
routine_effluent = select_subset_from_super(super_effluent, super_ppd, points_per_week=routine_ppw, t_final=t_final_routine)

pd.DataFrame(routine_influent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(routine_dir / 'Routine_Influent_States.csv', index=False)
pd.DataFrame(routine_effluent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(routine_dir / 'Routine_Effluent_States.csv', index=False)

pd.DataFrame(compounds_from_states(routine_influent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(routine_dir / 'Routine_Influent_Compounds.csv', index=False)
pd.DataFrame(compounds_from_states(routine_effluent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(routine_dir / 'Routine_Effluent_Compounds.csv', index=False)

# -----------------------------
# Exact subset: LONG ROUTINE (points/week from config across t_final_long_routine)
# -----------------------------
long_routine_influent = select_subset_from_super(super_influent, super_ppd, points_per_week=long_routine_ppw, t_final=t_final_long_routine)
long_routine_effluent = select_subset_from_super(super_effluent, super_ppd, points_per_week=long_routine_ppw, t_final=t_final_long_routine)

pd.DataFrame(long_routine_influent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(long_routine_dir / 'LongRoutine_Influent_States.csv', index=False)
pd.DataFrame(long_routine_effluent, columns=['Time','Flowrate','S_O2','S_I','S_S','S_NH4','S_N2','S_NOX','S_ALK','X_I','X_S','X_H','X_STO','X_A','X_SS']).to_csv(long_routine_dir / 'LongRoutine_Effluent_States.csv', index=False)

pd.DataFrame(compounds_from_states(long_routine_influent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(long_routine_dir / 'LongRoutine_Influent_Compounds.csv', index=False)
pd.DataFrame(compounds_from_states(long_routine_effluent), columns=['Time','Flowrate','COD','NH4+NH3','NO3+NO2','TKN','Alkalinity','TSS']).to_csv(long_routine_dir / 'LongRoutine_Effluent_Compounds.csv', index=False)

print("All data files generated successfully!")