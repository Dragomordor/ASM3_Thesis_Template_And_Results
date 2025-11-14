# --------------------------------------------------------------------
## Import Libraries
# --------------------------------------------------------------------

print(f'Start of profile likelihood script')

import numpy as np
from pathlib import Path
import sys
import configparser
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, least_squares
from scipy.stats import f as f_dist
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm import tqdm


print(f'Imported libraries')
# -----------------------------
## File Paths
# -----------------------------

# Current Directory
current_dir = Path.cwd()
# Top Directory (for .py it is same as current_dir, for .ipynb it is one level up)
    # Sometimes use .parent ( for ipynb and .py in oracle cloud), someimes use nothing (.py on windows)
# top_dir = current_dir
top_dir = current_dir.parent if current_dir.name == '3-Identifiability' else current_dir

# Libs Directory
libs_dir = str(top_dir / "0-Libs")
# Config
config_dir = top_dir / "0-Config"

# Data Directory
data_dir = top_dir / "0-Data"   
    # HighRes (0)
highres_dir = str(data_dir / "0-HighRes")
    # Routine (1)
routine_dir = str(data_dir / "1-Routine")
    # Active (2)
active_dir = str(data_dir / "2-Active")
    # LongActive (3)
long_active_dir = str(data_dir / "3-LongActive")
    # LongRoutine (4)
long_routine_dir = str(data_dir / "4-LongRoutine")

# Sensitivity Results Directory
sensitivity_results_dir = top_dir / "2-Sensitivity" / "Results"

# Results Directory
results_dir = (top_dir / "3-Identifiability" / "Results")

# -----------------------------
## Import Libraries - Custom
# -----------------------------
sys.path.append(libs_dir)

from plant_config import get_reactor_initial_values
from asm3_model import ode_system_wrapper

# -----------------------------
## Configs
# -----------------------------

# Config File
config = configparser.ConfigParser()
config.read(config_dir / "config.ini")
   # Seed for random number generator
seed = int(config['OVERALL']['seed'])        # Random seed
np.random.seed(seed)                         # Set random seed
data_to_use_for_identifiability = str(config['OVERALL']['data_to_use_for_run'])
    # True theta params
true_param_k_H = float(config['TRUEPARAMS']['true_param_k_H'])              # True k_H
true_param_K_X = float(config['TRUEPARAMS']['true_param_K_X'])              # True K_X
true_param_k_STO = float(config['TRUEPARAMS']['true_param_small_k_STO'])          # True k_STO
true_param_eta_NOX = float(config['TRUEPARAMS']['true_param_eta_NOX'])      # True eta_NOX
true_param_K_O2 = float(config['TRUEPARAMS']['true_param_K_O2'])            # True K_O2
true_param_K_NOX = float(config['TRUEPARAMS']['true_param_K_NOX'])          # True K_NOX
true_param_K_S = float(config['TRUEPARAMS']['true_param_K_S'])              # True K_S
true_param_K_STO = float(config['TRUEPARAMS']['true_param_big_K_STO'])          # True K_STO
true_param_mu_H = float(config['TRUEPARAMS']['true_param_mu_H'])            # True mu_H
true_param_K_NH4 = float(config['TRUEPARAMS']['true_param_K_NH4'])          # True K_NH4
true_param_K_ALK = float(config['TRUEPARAMS']['true_param_K_ALK'])          # True K_ALK
true_param_b_H_O2 = float(config['TRUEPARAMS']['true_param_b_H_O2'])        # True b_H_O2
true_param_b_H_NOX = float(config['TRUEPARAMS']['true_param_b_H_NOX'])      # True b_H_NOX
true_param_b_STO_O2 = float(config['TRUEPARAMS']['true_param_b_STO_O2'])    # True b_STO_O2
true_param_b_STO_NOX = float(config['TRUEPARAMS']['true_param_b_STO_NOX'])  # True b_STO_NOX
true_param_mu_A = float(config['TRUEPARAMS']['true_param_mu_A'])            # True mu_A
true_param_K_A_NH4 = float(config['TRUEPARAMS']['true_param_K_A_NH4'])      # True K_A_NH4
true_param_K_A_O2 = float(config['TRUEPARAMS']['true_param_K_A_O2'])        # True K_A_O2
true_param_K_A_ALK = float(config['TRUEPARAMS']['true_param_K_A_ALK'])      # True K_A_ALK
true_param_b_A_O2 = float(config['TRUEPARAMS']['true_param_b_A_O2'])        # True b_A_O2
true_param_b_A_NOX = float(config['TRUEPARAMS']['true_param_b_A_NOX'])      # True b_A_NOX
true_param_f_S_I = float(config['TRUEPARAMS']['true_param_f_S_I'])          # True f_S_I
true_param_Y_STO_O2 = float(config['TRUEPARAMS']['true_param_Y_STO_O2'])    # True Y_STO_O2
true_param_Y_STO_NOX = float(config['TRUEPARAMS']['true_param_Y_STO_NOX'])  # True Y_STO_NOX
true_param_Y_H_O2 = float(config['TRUEPARAMS']['true_param_Y_H_O2'])        # True Y_H_O2
true_param_Y_H_NOX = float(config['TRUEPARAMS']['true_param_Y_H_NOX'])      # True Y_H_NOX
true_param_Y_A = float(config['TRUEPARAMS']['true_param_Y_A'])              # True Y_A
true_param_f_X_I = float(config['TRUEPARAMS']['true_param_f_X_I'])          # True f_X_I
true_param_i_N_S_I = float(config['TRUEPARAMS']['true_param_i_N_S_I'])      # True i_N_S_I
true_param_i_N_S_S = float(config['TRUEPARAMS']['true_param_i_N_S_S'])      # True i_N_S_S
true_param_i_N_X_I = float(config['TRUEPARAMS']['true_param_i_N_X_I'])      # True i_N_X_I
true_param_i_N_X_S = float(config['TRUEPARAMS']['true_param_i_N_X_S'])      # True i_N_X_S
true_param_i_N_BM = float(config['TRUEPARAMS']['true_param_i_N_BM'])        # True i_N_BM
true_param_i_SS_X_I = float(config['TRUEPARAMS']['true_param_i_SS_X_I'])    # True i_SS_X_I
true_param_i_SS_X_S = float(config['TRUEPARAMS']['true_param_i_SS_X_S'])    # True i_SS_X_S
true_param_i_SS_BM = float(config['TRUEPARAMS']['true_param_i_SS_BM'])      # True i_SS_BM

theta_true = {
    'k_H': true_param_k_H,
    'K_X': true_param_K_X,
    'k_STO': true_param_k_STO,
    'eta_NOX': true_param_eta_NOX,
    'K_O2': true_param_K_O2,
    'K_NOX': true_param_K_NOX,
    'K_S': true_param_K_S,
    'K_STO': true_param_K_STO,
    'mu_H': true_param_mu_H,
    'K_NH4': true_param_K_NH4,
    'K_ALK': true_param_K_ALK,
    'b_H_O2': true_param_b_H_O2,
    'b_H_NOX': true_param_b_H_NOX,
    'b_STO_O2': true_param_b_STO_O2,
    'b_STO_NOX': true_param_b_STO_NOX,
    'mu_A': true_param_mu_A,
    'K_A_NH4': true_param_K_A_NH4,
    'K_A_O2': true_param_K_A_O2,
    'K_A_ALK': true_param_K_A_ALK,
    'b_A_O2': true_param_b_A_O2,
    'b_A_NOX': true_param_b_A_NOX,
    'f_S_I': true_param_f_S_I,
    'Y_STO_O2': true_param_Y_STO_O2,
    'Y_STO_NOX': true_param_Y_STO_NOX,
    'Y_H_O2': true_param_Y_H_O2,
    'Y_H_NOX': true_param_Y_H_NOX,
    'Y_A': true_param_Y_A,
    'f_X_I': true_param_f_X_I,
    'i_N_S_I': true_param_i_N_S_I,
    'i_N_S_S': true_param_i_N_S_S,
    'i_N_X_I': true_param_i_N_X_I,
    'i_N_X_S': true_param_i_N_X_S,
    'i_N_BM': true_param_i_N_BM,
    'i_SS_X_I': true_param_i_SS_X_I,
    'i_SS_X_S': true_param_i_SS_X_S,
    'i_SS_BM': true_param_i_SS_BM,
}

    # Reactor volumes
r1_V = float(config['REACTOR']['r1_V'])        # Volume of reactor 1
    # Parameter Ranges
range_param_k_H = tuple(map(float, config['PARAMRANGES']['range_param_k_H'].split(',')))              # Range of k_H
range_param_K_X = tuple(map(float, config['PARAMRANGES']['range_param_K_X'].split(',')))              # Range of K_X
range_param_k_STO = tuple(map(float, config['PARAMRANGES']['range_param_small_k_STO'].split(',')))          # Range of k_STO
range_param_eta_NOX = tuple(map(float, config['PARAMRANGES']['range_param_eta_NOX'].split(',')))      # Range of eta_NOX
range_param_K_O2 = tuple(map(float, config['PARAMRANGES']['range_param_K_O2'].split(',')))            # Range of K_O2
range_param_K_NOX = tuple(map(float, config['PARAMRANGES']['range_param_K_NOX'].split(',')))          # Range of K_NOX
range_param_K_S = tuple(map(float, config['PARAMRANGES']['range_param_K_S'].split(',')))              # Range of K_S
range_param_K_STO = tuple(map(float, config['PARAMRANGES']['range_param_big_K_STO'].split(',')))          # Range of K_STO
range_param_mu_H = tuple(map(float, config['PARAMRANGES']['range_param_mu_H'].split(',')))            # Range of mu_H
range_param_K_NH4 = tuple(map(float, config['PARAMRANGES']['range_param_K_NH4'].split(',')))          # Range of K_NH4
range_param_K_ALK = tuple(map(float, config['PARAMRANGES']['range_param_K_ALK'].split(',')))          # Range of K_ALK
range_param_b_H_O2 = tuple(map(float, config['PARAMRANGES']['range_param_b_H_O2'].split(',')))        # Range of b_H_O2
range_param_b_H_NOX = tuple(map(float, config['PARAMRANGES']['range_param_b_H_NOX'].split(',')))      # Range of b_H_NOX
range_param_b_STO_O2 = tuple(map(float, config['PARAMRANGES']['range_param_b_STO_O2'].split(',')))    # Range of b_STO_O2
range_param_b_STO_NOX = tuple(map(float, config['PARAMRANGES']['range_param_b_STO_NOX'].split(',')))  # Range of b_STO_NOX
range_param_mu_A = tuple(map(float, config['PARAMRANGES']['range_param_mu_A'].split(',')))            # Range of mu_A
range_param_K_A_NH4 = tuple(map(float, config['PARAMRANGES']['range_param_K_A_NH4'].split(',')))      # Range of K_A_NH4
range_param_K_A_O2 = tuple(map(float, config['PARAMRANGES']['range_param_K_A_O2'].split(',')))        # Range of K_A_O2
range_param_K_A_ALK = tuple(map(float, config['PARAMRANGES']['range_param_K_A_ALK'].split(',')))      # Range of K_A_ALK
range_param_b_A_O2 = tuple(map(float, config['PARAMRANGES']['range_param_b_A_O2'].split(',')))        # Range of b_A_O2
range_param_b_A_NOX = tuple(map(float, config['PARAMRANGES']['range_param_b_A_NOX'].split(',')))      # Range of b_A_NOX
range_param_f_S_I = tuple(map(float, config['PARAMRANGES']['range_param_f_S_I'].split(',')))          # Range of f_S_I
range_param_Y_STO_O2 = tuple(map(float, config['PARAMRANGES']['range_param_Y_STO_O2'].split(',')))    # Range of Y_STO_O2
range_param_Y_STO_NOX = tuple(map(float, config['PARAMRANGES']['range_param_Y_STO_NOX'].split(',')))  # Range of Y_STO_NOX
range_param_Y_H_O2 = tuple(map(float, config['PARAMRANGES']['range_param_Y_H_O2'].split(',')))        # Range of Y_H_O2
range_param_Y_H_NOX = tuple(map(float, config['PARAMRANGES']['range_param_Y_H_NOX'].split(',')))      # Range of Y_H_NOX
range_param_Y_A = tuple(map(float, config['PARAMRANGES']['range_param_Y_A'].split(',')))              # Range of Y_A
range_param_f_X_I = tuple(map(float, config['PARAMRANGES']['range_param_f_X_I'].split(',')))          # Range of f_X_I
range_param_i_N_S_I = tuple(map(float, config['PARAMRANGES']['range_param_i_N_S_I'].split(',')))      # Range of i_N_S_I
range_param_i_N_S_S = tuple(map(float, config['PARAMRANGES']['range_param_i_N_S_S'].split(',')))      # Range of i_N_S_S
range_param_i_N_X_I = tuple(map(float, config['PARAMRANGES']['range_param_i_N_X_I'].split(',')))      # Range of i_N_X_I
range_param_i_N_X_S = tuple(map(float, config['PARAMRANGES']['range_param_i_N_X_S'].split(',')))      # Range of i_N_X_S
range_param_i_N_BM = tuple(map(float, config['PARAMRANGES']['range_param_i_N_BM'].split(',')))        # Range of i_N_BM
range_param_i_SS_X_I = tuple(map(float, config['PARAMRANGES']['range_param_i_SS_X_I'].split(',')))    # Range of i_SS_X_I
range_param_i_SS_X_S = tuple(map(float, config['PARAMRANGES']['range_param_i_SS_X_S'].split(',')))    # Range of i_SS_X_S
range_param_i_SS_BM = tuple(map(float, config['PARAMRANGES']['range_param_i_SS_BM'].split(',')))      # Range of i_SS_BM

# For identifiability
num_passes_per_param = int(config['IDENTIFIABILITY']['num_passes_per_param'])
NAASI_threshold = float(config['IDENTIFIABILITY']['NAASI_threshold'])  # NAASI threshold for identifiability

# ----------------------------------------------------------
# Other Configs
# solver_method='BDF'
solver_method = str(config['IDENTIFIABILITY']['solver_method'])
# ----------------------------------------------------------
## Load Data from csv
# ----------------------------------------------------------

    # Highres (0)
data_highres_influent_states = pd.read_csv(highres_dir + "/HighRes_Influent_States.csv")
data_highres_effluent_states = pd.read_csv(highres_dir + "/HighRes_Effluent_States.csv")
    # Routine (1)
data_routine_influent_states = pd.read_csv(routine_dir + "/Routine_Influent_States.csv")
data_routine_effluent_states = pd.read_csv(routine_dir + "/Routine_Effluent_States.csv")
    # Active (2)
data_active_influent_states = pd.read_csv(active_dir + "/Active_Influent_States.csv")
data_active_effluent_states = pd.read_csv(active_dir + "/Active_Effluent_States.csv")
    # LongActive (3)
data_longactive_influent_states = pd.read_csv(long_active_dir + "/LongActive_Influent_States.csv")
data_longactive_effluent_states = pd.read_csv(long_active_dir + "/LongActive_Effluent_States.csv")
    # LongRoutine (4)
data_longroutine_influent_states = pd.read_csv(long_routine_dir + "/LongRoutine_Influent_States.csv")
data_longroutine_effluent_states = pd.read_csv(long_routine_dir + "/LongRoutine_Effluent_States.csv")

# ----------------------------------------------------------
## Data to use for identifiability
# ----------------------------------------------------------

data_mapping_influent = {
    'HighRes': data_highres_influent_states,
    'Routine': data_routine_influent_states,
    'Active': data_active_influent_states,
    'LongActive': data_longactive_influent_states,
    'LongRoutine': data_longroutine_influent_states,
}
data_mapping_effluent = {
    'HighRes': data_highres_effluent_states,
    'Routine': data_routine_effluent_states,
    'Active': data_active_effluent_states,
    'LongActive': data_longactive_effluent_states,
    'LongRoutine': data_longroutine_effluent_states,
}

try:
    Data_Influent = data_mapping_influent[data_to_use_for_identifiability]
    Data_Effluent = data_mapping_effluent[data_to_use_for_identifiability]
except KeyError:
    raise ValueError("Invalid data for identifiability")

print(f'Imported configs and data for identifiability: {data_to_use_for_identifiability}')
# ----------------------------------------------------------
## Parameter Ranges from Priors
# ----------------------------------------------------------

# Dictionary with minimum and maximum values for each parameter
theta_ranges = {
    'k_H': range_param_k_H,
    'K_X': range_param_K_X,
    'k_STO': range_param_k_STO,
    'eta_NOX': range_param_eta_NOX,
    'K_O2': range_param_K_O2,
    'K_NOX': range_param_K_NOX,
    'K_S': range_param_K_S,
    'K_STO': range_param_K_STO,
    'mu_H': range_param_mu_H,
    'K_NH4': range_param_K_NH4,
    'K_ALK': range_param_K_ALK,
    'b_H_O2': range_param_b_H_O2,
    'b_H_NOX': range_param_b_H_NOX,
    'b_STO_O2': range_param_b_STO_O2,
    'b_STO_NOX': range_param_b_STO_NOX,
    'mu_A': range_param_mu_A,
    'K_A_NH4': range_param_K_A_NH4,
    'K_A_O2': range_param_K_A_O2,
    'K_A_ALK': range_param_K_A_ALK,
    'b_A_O2': range_param_b_A_O2,
    'b_A_NOX': range_param_b_A_NOX,
    'f_S_I': range_param_f_S_I,
    'Y_STO_O2': range_param_Y_STO_O2,
    'Y_STO_NOX': range_param_Y_STO_NOX,
    'Y_H_O2': range_param_Y_H_O2,
    'Y_H_NOX': range_param_Y_H_NOX,
    'Y_A': range_param_Y_A,
    'f_X_I': range_param_f_X_I,
    'i_N_S_I': range_param_i_N_S_I,
    'i_N_S_S': range_param_i_N_S_S,
    'i_N_X_I': range_param_i_N_X_I,
    'i_N_X_S': range_param_i_N_X_S,
    'i_N_BM': range_param_i_N_BM,
    'i_SS_X_I': range_param_i_SS_X_I,
    'i_SS_X_S': range_param_i_SS_X_S,
    'i_SS_BM': range_param_i_SS_BM,
}

# Update individual ranges - adjust upper or lower bounds for some parameters according to results obtained already, if they did not produce a clear profile likelihood.
# Example: theta_ranges['K_X'] = (1*theta_ranges['K_X'][0], 3*theta_ranges['K_X'][1])
    # TODO - MANUAL
theta_ranges['k_H'] = (1*theta_ranges['k_H'][0], 3*theta_ranges['k_H'][1])
theta_ranges['K_X'] = (0.01, 3*theta_ranges['K_X'][1])
theta_ranges['k_STO'] = (1*theta_ranges['k_STO'][0], 8*theta_ranges['k_STO'][1])
theta_ranges['K_NOX'] = (1*theta_ranges['K_NOX'][0], 8*theta_ranges['K_NOX'][1])
theta_ranges['K_S'] = (0.01, 3*theta_ranges['K_S'][1])
theta_ranges['mu_H'] = (1.5*theta_ranges['mu_H'][0], 3*theta_ranges['mu_H'][1])
theta_ranges['b_H_O2'] = (0.01, 3*theta_ranges['b_H_O2'][1])
theta_ranges['b_STO_O2'] = (0.01, 3*theta_ranges['b_STO_O2'][1])
theta_ranges['mu_A'] = (1*theta_ranges['mu_A'][0], 4*theta_ranges['mu_A'][1])
theta_ranges['K_A_NH4'] = (0.2*theta_ranges['K_A_NH4'][0], 5*theta_ranges['K_A_NH4'][1])
theta_ranges['K_A_O2'] = (0.01, 5*theta_ranges['K_A_O2'][1])
theta_ranges['K_A_ALK'] = (0.01, 8*theta_ranges['K_A_ALK'][1])
theta_ranges['Y_STO_NOX'] = (0.01, 1*theta_ranges['Y_STO_NOX'][1])
theta_ranges['Y_H_O2'] = (0.01, 1*theta_ranges['Y_H_O2'][1])
theta_ranges['Y_H_NOX'] = (0.01, 1*theta_ranges['Y_H_NOX'][1])
theta_ranges['Y_A'] = (0.01, 1*theta_ranges['Y_A'][1])
theta_ranges['i_N_BM'] = (1*theta_ranges['i_N_BM'][0], 9*theta_ranges['i_N_BM'][1])
theta_ranges['Y_STO_O2'] = (0.01, 3*theta_ranges['Y_STO_O2'][1])

# Change all ranges to 1e-6 to 20 as example
# for key in theta_ranges:
#     theta_ranges[key] = (1e-6, 20)

# ----------------------------------------------------------
## ODE System
# ----------------------------------------------------------

# Time from data
t_eval = Data_Influent['Time'].values
# Initial values
y0 = get_reactor_initial_values(top_dir=top_dir)
# Ode system
ode_system = lambda t, y, theta: ode_system_wrapper(t=t, y=y, theta=theta, influentData=Data_Influent.to_numpy(), reactorVolumes=[r1_V])

# ----------------------------------------------------------
## Profile Likelihood Functions
# ----------------------------------------------------------

## Objective Function
def obj_fun_SSE(ode_function, theta, y0, true_output_data, solver_method=solver_method,
                always_fixed_param_idxs=None, always_fixed_param_vals=None, fixed_param_idx=None, fixed_param_val=None
                ):
    """
    Objective function to minimize the sum of squared errors (SSE) between model predictions and true output data.

    Args:
        ode_function: function, the ODE system to solve
        theta: array, parameters for the ODE system
        y0: array, initial values for the ODE system
        true_output_data: dataframe, true output data to compare against
        solver_method: str, method to use for solving the ODE system (default is 'BDF')
    Returns:
        obj_val: float, the objective function value (sum of squared errors)
    """
    t_eval = true_output_data['Time'].values
    tspan = (t_eval.min(), t_eval.max())

    # Fix always fixed parameters if provided
    if always_fixed_param_idxs is not None:
        for idx, val in zip(always_fixed_param_idxs, always_fixed_param_vals):
            theta[idx] = val
    # Fix a specific parameter if provided
    if fixed_param_idx is not None:
        theta[fixed_param_idx] = fixed_param_val

    # Solve the ODE system with the given parameters
    ode_system = lambda t, y: ode_function(t, y, theta)
    sol = solve_ivp(fun=ode_system, t_span=tspan, y0=y0, t_eval=t_eval, method=solver_method, rtol=1e-4, atol=1e-6)
    
    model_time = sol.t
    model_data = sol.y.T
        # Interpolate model values at true data time points - TODO: Redundancy but sometimes needed?
    model_interp_data = np.column_stack([
        np.interp(x=t_eval, xp=model_time, fp=model_data[:, i]) for i in range(model_data.shape[1])
    ])
    # Residuals (difference between true data and model predictions)
    residuals = (model_interp_data - true_output_data.iloc[:, 2:].values)**2  # Exclude 'Time' and 'Flowrate' 
    # Objective function value -- Sum of squared residuals
    obj_val = np.sum(residuals)
    return obj_val
# End of obj_fun() function

# Alternative: residuals for scipy least-squares
def obj_fun_lstsq(ode_function, theta, y0, true_output_data, solver_method=solver_method,
                always_fixed_param_idxs=None, always_fixed_param_vals=None, fixed_param_idx=None, fixed_param_val=None
                  ):
    """
    Objective function to compute residuals for least_squares optimization. Residuals are the differences between model predictions and true output data.

    Args:
        ode_function: function, the ODE system to solve
        theta: array, parameters for the ODE system
        y0: array, initial values for the ODE system
        true_output_data: dataframe, true output data to compare against
        fixed_param_idx: int, index of the parameter to be fixed (optional)
        fixed_param_val: float, value of the fixed parameter (optional)
        always_fixed_param_idxs: list, indices of parameters that are always fixed (optional)
        always_fixed_param_vals: list, values of the always fixed parameters (optional)
        solver_method: str, method to use for solving the ODE system (default is 'BDF')
    Returns:
        residuals: array, the residuals (differences between model predictions and true output data)

    """
    t_eval = None

    raw_time = true_output_data['Time'].values
    raw_output_data = true_output_data.drop(columns=['Time', 'Flowrate'])
    column_labels = raw_output_data.columns

    # Fix always fixed parameters if provided
    if always_fixed_param_idxs is not None:
        for idx, val in zip(always_fixed_param_idxs, always_fixed_param_vals):
            theta[idx] = val
    # Fix a specific parameter if provided
    if fixed_param_idx is not None:
        theta[fixed_param_idx] = fixed_param_val

    # Solve the ODE system with the given parameters
    ode_system = lambda t, y: ode_function(t, y, theta)
    t_span = (raw_time.min(), raw_time.max())
    sol = solve_ivp(fun=ode_system, t_span=t_span, y0=y0, t_eval=t_eval, method=solver_method, 
                # rtol=1e-6, atol=1e-6
                )
    model_time = sol.t
    interp_func = interp1d(model_time, sol.y, kind='linear', axis=1, fill_value='extrapolate')
    # model_data = sol.y.T
    model_data = interp_func(raw_time).T
    model_interp_data = pd.DataFrame(model_data, columns=column_labels, index=raw_time)

        # Interpolate model values at true data time points - TODO: Redundancy but sometimes needed?
    # model_interp_data = np.column_stack([
    #     np.interp(x=t_eval, xp=model_time, fp=model_data[:, i]) for i in range(model_data.shape[1])
    # ])
    # Residuals (difference between true data and model predictions)
    # residuals = (model_interp_data - true_output_data.iloc[:, 2:].values)  # Exclude 'Time' and 'Flowrate' 
    residuals = (raw_output_data - model_interp_data.values).values.flatten() # Flatten the residuals to a 1D array
    return residuals
# End of residuals_fun() function

# NLL Objective Function alternative
# def obj_fun_nll(ode_function, theta, y0, true_output_data, fixed_param_idx=None, fixed_param_val=None, always_fixed_param_idxs=None, always_fixed_param_vals=None, solver_method=solver_method):
#     """
#     Negative Log-Likelihood (NLL) Objective Function
#     Args:
#         ode_function: function, the ODE system to solve
#         theta: array, parameters for the ODE system
#         y0: array, initial values for the ODE system
#         true_output_data: dataframe, true output data to compare against
#         fixed_param_idx: int, index of the parameter to be fixed (optional)
#         fixed_param_val: float, value of the fixed parameter (optional)
#         always_fixed_param_idxs: list, indices of parameters that are always fixed (optional)
#         always_fixed_param_vals: list, values of the always fixed parameters (optional)
#         solver_method: str, method to use for solving the ODE system (default is 'BDF')
#     Returns:
#         NLL: float, the negative log-likelihood value
#     """
#     t_eval = true_output_data['Time'].values
#     tspan = (t_eval.min(), t_eval.max())

#     # # if always fixed params are provided, replace values in theta
#     if always_fixed_param_idxs is not None:
#         for idx, val in zip(always_fixed_param_idxs, always_fixed_param_vals):
#             theta[idx] = val

#     # if fixed param is provided, replace value in theta (skip that index if always fixed)
#     if fixed_param_idx is not None:
#         theta[fixed_param_idx] = fixed_param_val

#     # Solve the ODE system with the given parameters
#     ode_system = lambda t, y: ode_function(t, y, theta)
#     sol = solve_ivp(fun=ode_system, t_span=tspan, y0=y0, t_eval=t_eval, method=solver_method)

#     model_time = sol.t
#     model_data = sol.y.T
#        # Interpolate model values at true data time points - TODO: Redundancy but sometimes needed?
#     model_interp_data = np.column_stack([
#         np.interp(x=t_eval, xp=model_time, fp=model_data[:, i]) for i in range(model_data.shape[1])
#     ])
#     # Residuals (difference between true data and model predictions)
#     residuals = model_interp_data - true_output_data.iloc[:, 2:].values  # Exclude 'Time' and 'Flowrate' columns
    
#     # Estimate sigma^2 (variance of residuals)
#     sigma2 = np.var(residuals, ddof=1)  # Use sample variance (ddof=1)

#     # Compute Negative Log-Likelihood
#     n = residuals.size  # Total number of data points
#     NLL = (n / 2) * np.log(2 * np.pi * sigma2) + (1 / (2 * sigma2)) * np.sum(residuals**2)

# # Return Negative Log-Likelihood as objective function value
#     return NLL
# # End of obj_fun() function


## Optimize Params while fixing one
# def optimize_with_fixed_param(ode_function, fixed_param_idx, fixed_param_val, y0, true_output_data, theta_ranges, theta_init=None, always_fixed_param_idxs=None, always_fixed_param_vals=None, constraints=None, solver_method=solver_method):
#     """
#     Function to optimize parameters of the ODE system while keeping one parameter fixed. Uses scipy's minimize function.
#     Args:
#         ode_function: function, the ODE system to solve
#         fixed_param_idx: int, index of the parameter to be fixed
#         fixed_param_val: float, value of the fixed parameter
#         y0: array, initial values for the ODE system
#         true_output_data: dataframe, true output data to compare against
#         theta_ranges: dict, dictionary with ranges for each parameter
#         theta_init: list, initial guess for the parameters (optional)
#         always_fixed_param_idxs: list, indices of parameters that are always fixed (optional)
#         always_fixed_param_vals: list, values of the always fixed parameters (optional)
#         constraints: dict, constraints for the optimization (optional)
#         solver_method: str, method to use for solving the ODE system (default is 'BDF')
#     Returns:
#         result: OptimizeResult, the result of the optimization
#     """
#     # Wrapper objective function
#     def obj_function_wrapper(theta):
#         return obj_fun_SSE(
#             ode_function=ode_function,
#             theta=theta,
#             y0=y0,
#             true_output_data=true_output_data,
#             fixed_param_idx=fixed_param_idx,
#             fixed_param_val=fixed_param_val,
#             always_fixed_param_idxs=always_fixed_param_idxs,
#             always_fixed_param_vals=always_fixed_param_vals,
#             solver_method=solver_method
#         )

#     # Create Bounds from theta_ranges
#         # For minimize --- Sequence of (min, max) pairs for each element in x. None is used to specify no bound.
#     bounds = [theta_ranges[k] for k in theta_ranges.keys()]
#         # For least_squares --- Bounds for variables, specified as a tuple of two lists/arrays (lower_bounds, upper_bounds)
#     # bounds = ([theta_ranges[k][0] for k in theta_ranges.keys()], [theta_ranges[k][1] for k in theta_ranges.keys()])

#     # Initial guess for theta - If not provided, use the midpoint of the range
#     if theta_init is None:
#         theta_init = [(theta_ranges[k][0] + theta_ranges[k][1]) / 2 for k in theta_ranges.keys()]
    
#     # Always fixed params
#     if always_fixed_param_idxs is not None:
#         for idx, val in zip(always_fixed_param_idxs, always_fixed_param_vals):
#             theta_init[idx] = val

#     # Fix the parameter corresponding to the fixed_param_idx -- currently being optimized
#     theta_init[fixed_param_idx] = fixed_param_val

#     result = minimize(obj_function_wrapper, theta_init, bounds=bounds, constraints=constraints)
#     # Alternative using least_squares
#     # result = least_squares(obj_function_wrapper_res, theta_init, bounds=bounds)

#     return result
# # End of optimize_with_fixed_param() function
print(f'Functions setup')
# --------------------------------------------------------------------
## Profile Likelihood for each parameter - Setup
# --------------------------------------------------------------------

# Instead used dictionary with ranges -- bigger range
param_name_indices_bounds = {
    'k_H': (0, theta_ranges['k_H']),
    'K_X': (1, theta_ranges['K_X']),
    'k_STO': (2, theta_ranges['k_STO']),
    'eta_NOX': (3, theta_ranges['eta_NOX']),
    'K_O2': (4, theta_ranges['K_O2']),
    'K_NOX': (5, theta_ranges['K_NOX']),
    'K_S': (6, theta_ranges['K_S']),
    'K_STO': (7, theta_ranges['K_STO']),
    'mu_H': (8, theta_ranges['mu_H']),
    'K_NH4': (9, theta_ranges['K_NH4']),
    'K_ALK': (10, theta_ranges['K_ALK']),
    'b_H_O2': (11, theta_ranges['b_H_O2']),
    'b_H_NOX': (12, theta_ranges['b_H_NOX']),
    'b_STO_O2': (13, theta_ranges['b_STO_O2']),
    'b_STO_NOX': (14, theta_ranges['b_STO_NOX']),
    'mu_A': (15, theta_ranges['mu_A']),
    'K_A_NH4': (16, theta_ranges['K_A_NH4']),
    'K_A_O2': (17, theta_ranges['K_A_O2']),
    'K_A_ALK': (18, theta_ranges['K_A_ALK']),
    'b_A_O2': (19, theta_ranges['b_A_O2']),
    'b_A_NOX': (20, theta_ranges['b_A_NOX']),
    'f_S_I': (21, theta_ranges['f_S_I']),
    'Y_STO_O2': (22, theta_ranges['Y_STO_O2']),
    'Y_STO_NOX': (23, theta_ranges['Y_STO_NOX']),
    'Y_H_O2': (24, theta_ranges['Y_H_O2']),
    'Y_H_NOX': (25, theta_ranges['Y_H_NOX']),
    'Y_A': (26, theta_ranges['Y_A']),
    'f_X_I': (27, theta_ranges['f_X_I']),
    'i_N_S_I': (28, theta_ranges['i_N_S_I']),
    'i_N_S_S': (29, theta_ranges['i_N_S_S']),
    'i_N_X_I': (30, theta_ranges['i_N_X_I']),
    'i_N_X_S': (31, theta_ranges['i_N_X_S']),
    'i_N_BM': (32, theta_ranges['i_N_BM']),
    'i_SS_X_I': (33, theta_ranges['i_SS_X_I']),
    'i_SS_X_S': (34, theta_ranges['i_SS_X_S']),
    'i_SS_BM': (35, theta_ranges['i_SS_BM']),
}

num_params = len(param_name_indices_bounds)
num_passes_per_param = num_passes_per_param

# Initialize the resultsDataFrame
columns = [param_name+'_fix' for param_name in param_name_indices_bounds.keys()]
index = range(num_passes_per_param)
results_df = pd.DataFrame(index=index, columns=columns)

print(f'Setting up profile likelihood')

# --------------------------------------------------------------------
## Profile Likelihood for each parameter - Run
# --------------------------------------------------------------------

# Decide which params to always fix
    # Based on Sensitivity Analysis rseults + (i_SS_X_I, i_SS_X_S, i_SS_BM --- always fixed)

# From Top_dir/2-Sensitivity/Results/NAASI_states.csv, the parameters and NAASI values are retrieved
    # Example - The parameters with NAASI values less than 0.1 are always fixed in the profile likelihood analysis

# ----------------------------------------------------------

all_param_idx = {
    'k_H': 0,
    'K_X': 1,
    'k_STO': 2,
    'eta_NOX': 3,
    'K_O2': 4,
    'K_NOX': 5,
    'K_S': 6,
    'K_STO': 7,
    'mu_H': 8,
    'K_NH4': 9,
    'K_ALK': 10,
    'b_H_O2': 11,
    'b_H_NOX': 12,
    'b_STO_O2': 13,
    'b_STO_NOX': 14,
    'mu_A': 15,
    'K_A_NH4': 16,
    'K_A_O2': 17,
    'K_A_ALK': 18,
    'b_A_O2': 19,
    'b_A_NOX': 20,
    'f_S_I': 21,
    'Y_STO_O2': 22,
    'Y_STO_NOX': 23,
    'Y_H_O2': 24,
    'Y_H_NOX': 25,
    'Y_A': 26,
    'f_X_I': 27,
    'i_N_S_I': 28,
    'i_N_S_S': 29,
    'i_N_X_I': 30,
    'i_N_X_S': 31,
    'i_N_BM': 32,
    'i_SS_X_I': 33,
    'i_SS_X_S': 34,
    'i_SS_BM': 35
}

    # From sensitivity results directory, NAASI_combined.csv
NAASI_results = pd.read_csv(sensitivity_results_dir / 'NAASI_combined.csv')
NAASI_results = NAASI_results.set_index('Parameter')
# Get parameters with NAASI values below threshold
params_to_always_fix = {
    param: NAASI_results.loc[param, 'NAASI'] for param in NAASI_results.index if NAASI_results.loc[param, 'NAASI'] < NAASI_threshold
}
# Print the parameters to always fix based on NAASI results 
print(f'Parameters to always fix based on NAASI results: ')
for param, naasi in params_to_always_fix.items():
    print(f'  {param}: {naasi}')

state_names = {
    'State 1': 'S_O2',
    'State 2': 'S_I',
    'State 3': 'S_S',
    'State 4': 'S_NH4',
    'State 5': 'S_N2',
    'State 6': 'S_NOX',
    'State 7': 'S_ALK',
    'State 8': 'X_I',
    'State 9': 'X_S',
    'State 10': 'X_H',
    'State 11': 'X_STO',
    'State 12': 'X_A',
    'State 13': 'X_SS'
}

always_fixed_param_idxs = [all_param_idx[name] for name in params_to_always_fix.keys()]
# set always fixed parameter values based on true values for identifiability
always_fixed_param_vals = [theta_true[name] for name in params_to_always_fix.keys()]

# Create CI_threshold dictionary
CI_threshold = {}

# Initial guess for the first pass, default is the midpoint of the bounds. Set it to true value for easier convergence
# theta_init = [(theta_ranges[k][0] + theta_ranges[k][1]) / 2 for k in theta_ranges.keys()]
# or set to true value for easier convergence
theta_init = [theta_true[k] for k in theta_ranges.keys()]

print("------------------------------------")

print(f'Parameters that will be evaluated in the profile likelihood:')
for param_name, (param_idx, param_bounds) in param_name_indices_bounds.items():
    if param_name in params_to_always_fix:
        print(f'(Fixed)  {param_name} -- Fixed at {always_fixed_param_vals[always_fixed_param_idxs.index(param_idx)]}')
    else:
        print(f'(PLA)    {param_name} -- Range for PLA: {param_bounds}')
print("------------------------------------")

print(f'Starting Profile likelihood\n')

# Loop over all parames and get lower and upper bound
bounds = []
# Normal parameters - set lower bound of 0 and upper bound of infinity
for _, _ in enumerate(all_param_idx):
    bounds.append((0, np.inf))
# Change bounds for always fixed parameters to a small range around the fixed value
for idx, val in zip(always_fixed_param_idxs, always_fixed_param_vals):
    bounds[idx] = (val - 1e-6, val + 1e-6)  # Small range around the fixed value

# if param is fixed, set theta_init for that parameter
if always_fixed_param_idxs is not None and always_fixed_param_vals is not None:
    for idx, val in zip(always_fixed_param_idxs, always_fixed_param_vals):
        theta_init[idx] = val # Set initial guess for fixed parameter

# Loop over all parameters -- Skip always fixed parameters
for param_name, (param_idx, param_bounds) in param_name_indices_bounds.items():

    # if param is always fixed, skip
    if param_name in params_to_always_fix:
        print(f'{param_name} is always fixed, skipping...')
        continue

    # if params are fixed, update theta_init
    if always_fixed_param_idxs is not None and always_fixed_param_vals is not None:
        for idx, val in zip(always_fixed_param_idxs, always_fixed_param_vals):
            theta_init[idx] = val # Set initial guess for fixed parameter

    # Print current parameter
    print('----------------------------------')
    print(f'Profile Likelihood for {param_name}')
    print('----------------------------------')

    # Initialize arrays to store the results
    theta = np.zeros((num_passes_per_param, num_params))
    proflike_param_obj_val = np.zeros((num_params, num_passes_per_param))
    # sol_param_passes = np.zeros((num_params, num_passes_per_param, (num_params+1), len(t_eval)))
    # Get parameter bounds and range for current parameter
    param_range = np.linspace(param_bounds[0], param_bounds[1], num_passes_per_param)

    # --------------------------------------------------------
    # Helper function for a single fixed param point
    def run_profile_point(param_val, param_idx_here, param_name_here, theta_init_here):
        theta_init_here = theta_init_here.copy()
        theta_init_here[param_idx_here] = param_val
        # Set the parameter initial guess

        # SSE method - single objective function
        # res_fun_sse = lambda x: obj_fun_SSE(
        #     ode_function=ode_system,
        #     theta=x,
        #     y0=y0,
        #     true_output_data=Data_Effluent,
        #     solver_method=solver_method,
        #     always_fixed_param_idxs=always_fixed_param_idxs,
        #     always_fixed_param_vals=always_fixed_param_vals,
        #     fixed_param_idx=param_idx,
        #     fixed_param_val=param_val
        # )
        # result = minimize(
        #     fun=res_fun_sse,
        #     x0=theta_init,
        #     bounds=bounds,
        #     # method='SLSQP',  
        # )
        # obj_val = result.fun # Objective value (SSE)
        # return {'param_val': param_val, 'obj_val': obj_val, 'theta': result.x, 'success': result.success}

        # least squares method - vector of residuals
        res_fun_lstq = lambda x: obj_fun_lstsq(
            ode_function=ode_system,
            theta=x,
            y0=y0,
            true_output_data=Data_Effluent,
            solver_method=solver_method,
            always_fixed_param_idxs=always_fixed_param_idxs,
            always_fixed_param_vals=always_fixed_param_vals,
            fixed_param_idx=param_idx,
            fixed_param_val=param_val
        )
        bounds_lstsq = (np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
        result = least_squares(
            fun=res_fun_lstq,
            x0=theta_init_here,
            bounds=bounds_lstsq,
            # method='trf',
            # jac='3-point',
        )

        obj_val = np.sum(result.fun**2) # Objective value (least squares)
        return {'param_val': param_val, 'obj_val': obj_val, 'theta': result.x, 'success': result.success}
    # --------------------------------------------------------

    # Parallelize the profile likelihood runs for the current parameter
    n_jobs = 4  # Number of parallel jobs based on available CPU cores
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_profile_point)(param_val, param_idx, param_name, theta_init) for param_val in tqdm(param_range)
    )
    # Sequentially run the profile likelihood for the current parameter
    # results_list = [run_profile_point(param_val, param_idx, param_name, theta_init) for param_val in tqdm(param_range)]

    # store results
    for i, res in enumerate(results_list):
        proflike_param_obj_val[param_idx, i] = res['obj_val']
        results_df.at[i, param_name+'_fix'] = {'obj_val': res['obj_val'], 'theta': res['theta']}
        print(f'Pass {i+1}/{num_passes_per_param}: {param_name}={res["param_val"]},  obj={res["obj_val"]:.3e}, success={res["success"]}')
    
    # --------------------------------------------------------
    # Confidence Interval Calculation & plotting
    # For current parameter, find 95% CI  -- For SSE this uses F-test
        # CI_threshold = SSE_min * (1 + (1/(n-p))*F_(1,n-p,0.alpha))
        # SSE_min = minimum SSE value for the current parameter
        # n = number of data points
        # p = number of parameters
        # F_(1,n-p,0.alpha) = F-distribution value
            # Numerator degrees of freedom = 1 (since we are testing one parameter)
            # Denominator degrees of freedom = n - p (number of data points - number of parameters)
            # Significance level alpha = 0.05 (for 95% CI)

    CI_min_sse = np.min(proflike_param_obj_val[param_idx, :]) # Minimum SSE value for the current parameter
    CI_n = len(Data_Effluent['Time'])
    CI_p = num_params # Number of parameters
    CI_alpha = 0.05 # Significance level
    CI_f_value = f_dist.ppf(1-CI_alpha, 1, CI_n-CI_p) # F-distribution value
    CI_threshold[param_name] = CI_min_sse * (1 + (1/(CI_n-CI_p))*CI_f_value)

    # Plot the results for the current parameter
    fontsize = 24
    plt.figure(figsize=(24, 10))
    plt.title(f'Profile Likelihood for {param_name}', fontsize=fontsize)
    plt.plot(param_range, proflike_param_obj_val[param_idx], linestyle='-', color='b', label='Profile Likelihood') # fixed parameter values vs. objective function value
    plt.axhline(y=CI_threshold[param_name], color='r', linestyle='--', label='95% CI Threshold')
    plt.grid(False)
    plt.xlabel(f"Fixed value of {param_name}", fontsize=fontsize)
    plt.ylabel('Objective Function (SSE)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize=fontsize)
    # plt.savefig(f'{str(results_dir)}/ProfileLikelihood_{param_name}.png')
    plt.savefig(results_dir / f'ProfileLikelihood_{param_name}.png')

# End of loop

# --------------------------------------------------------
# Save the results DataFrame to a pickle file
# results_df.to_pickle('Results/ProfileLikelihoodResults.pkl')
results_df.to_pickle(results_dir / 'ProfileLikelihoodResults.pkl')


# --------------------------------------------------------------------
## After Loops, Print Results 

# print first few rows of the results dataframe
print("------------------------------------")
print(results_df.head())

# Confirm results saved
print("------------------------------------")
print(f'Profile Likelihood results saved"')
