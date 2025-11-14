# --------------------------------------------------------------------
## Import Libraries
# --------------------------------------------------------------------


import numpy as np
from pathlib import Path
import sys
import configparser
import pandas as pd
import pymc
import pytensor.tensor as pt
from scipy.stats import norm

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import sunode
import sunode.wrappers.as_pytensor as sunode_pytensor
from pymc.ode import DifferentialEquation




# -----------------------------
## File Paths
# -----------------------------

# Current Directory
current_dir = Path.cwd()
# Top Directory
top_dir = current_dir.parent if current_dir.name == '5-Sampling' else current_dir


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

# PCA Results File
pca_results_path = str(top_dir / "4-Correlations" / "Results" / "PCA_Results.pkl")

# Results Directory
results_dir = (top_dir / "5-Sampling" / "Results")

# -----------------------------
## Import Libraries - Custom
# -----------------------------
sys.path.append(libs_dir)

from plant_config import get_reactor_initial_values
from asm3_model import ode_system_wrapper
from run_sampling import run_model_NUTS, run_model_HamiltonianMC

# -----------------------------
## Configs
# -----------------------------

# Config File
config = configparser.ConfigParser()
config.read(config_dir / "config.ini")
   # Seed for random number generator
seed = int(config['OVERALL']['seed'])
np.random.seed(seed)                         # Set random seed
data_to_use_for_sampling = str(config['OVERALL']['data_to_use_for_run'])
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
    # True values
true_param_k_H = float(config['TRUEPARAMS']['true_param_k_H'])          
true_param_K_X = float(config['TRUEPARAMS']['true_param_K_X'])          
true_param_k_STO = float(config['TRUEPARAMS']['true_param_small_k_STO'])      
true_param_eta_NOX = float(config['TRUEPARAMS']['true_param_eta_NOX'])  
true_param_K_O2 = float(config['TRUEPARAMS']['true_param_K_O2'])
true_param_K_NOX = float(config['TRUEPARAMS']['true_param_K_NOX'])
true_param_K_S = float(config['TRUEPARAMS']['true_param_K_S'])
true_param_K_STO = float(config['TRUEPARAMS']['true_param_big_K_STO'])
true_param_mu_H = float(config['TRUEPARAMS']['true_param_mu_H'])
true_param_K_NH4 = float(config['TRUEPARAMS']['true_param_K_NH4'])
true_param_K_ALK = float(config['TRUEPARAMS']['true_param_K_ALK'])
true_param_b_H_O2 = float(config['TRUEPARAMS']['true_param_b_H_O2'])
true_param_b_H_NOX = float(config['TRUEPARAMS']['true_param_b_H_NOX'])
true_param_b_STO_O2 = float(config['TRUEPARAMS']['true_param_b_STO_O2'])
true_param_b_STO_NOX = float(config['TRUEPARAMS']['true_param_b_STO_NOX'])
true_param_mu_A = float(config['TRUEPARAMS']['true_param_mu_A'])
true_param_K_A_NH4 = float(config['TRUEPARAMS']['true_param_K_A_NH4'])
true_param_K_A_O2 = float(config['TRUEPARAMS']['true_param_K_A_O2'])
true_param_K_A_ALK = float(config['TRUEPARAMS']['true_param_K_A_ALK'])
true_param_b_A_O2 = float(config['TRUEPARAMS']['true_param_b_A_O2'])
true_param_b_A_NOX = float(config['TRUEPARAMS']['true_param_b_A_NOX'])
true_param_f_S_I = float(config['TRUEPARAMS']['true_param_f_S_I'])
true_param_Y_STO_O2 = float(config['TRUEPARAMS']['true_param_Y_STO_O2'])
true_param_Y_STO_NOX = float(config['TRUEPARAMS']['true_param_Y_STO_NOX'])
true_param_Y_H_O2 = float(config['TRUEPARAMS']['true_param_Y_H_O2'])
true_param_Y_H_NOX = float(config['TRUEPARAMS']['true_param_Y_H_NOX'])
true_param_Y_A = float(config['TRUEPARAMS']['true_param_Y_A'])
true_param_f_X_I = float(config['TRUEPARAMS']['true_param_f_X_I'])
true_param_i_N_S_I = float(config['TRUEPARAMS']['true_param_i_N_S_I'])
true_param_i_N_S_S = float(config['TRUEPARAMS']['true_param_i_N_S_S'])
true_param_i_N_X_I = float(config['TRUEPARAMS']['true_param_i_N_X_I'])
true_param_i_N_X_S = float(config['TRUEPARAMS']['true_param_i_N_X_S'])
true_param_i_N_BM = float(config['TRUEPARAMS']['true_param_i_N_BM'])
true_param_i_SS_X_I = float(config['TRUEPARAMS']['true_param_i_SS_X_I'])
true_param_i_SS_X_S = float(config['TRUEPARAMS']['true_param_i_SS_X_S'])
true_param_i_SS_BM = float(config['TRUEPARAMS']['true_param_i_SS_BM'])
    # From identifiability
NAASI_threshold = float(config['IDENTIFIABILITY']['NAASI_threshold'])  # NAASI threshold for identifiability
    # Sampling
solver_method = str(config['SAMPLING']['solver_method'])
config_tuning_samples = int(config['SAMPLING']['tuning_samples'])
config_draw_samples = int(config['SAMPLING']['draw_samples'])
config_sample_chains = int(config['SAMPLING']['run_chains'])
config_sample_cores = int(config['SAMPLING']['run_cores'])

# ----------------------------------------------------------
## Load Data from csv
# ----------------------------------------------------------

    # Highres (0)
data_highres_influent_states = pd.read_csv(highres_dir + "/HighRes_Influent_States.csv")
data_highres_effluent_states = pd.read_csv(highres_dir + "/HighRes_Effluent_States.csv")
data_highres_influent_compounds = pd.read_csv(highres_dir + "/HighRes_Influent_Compounds.csv")
data_highres_effluent_compounds = pd.read_csv(highres_dir + "/HighRes_Effluent_Compounds.csv")
    # Routine (1)
data_routine_influent_states = pd.read_csv(routine_dir + "/Routine_Influent_States.csv")
data_routine_effluent_states = pd.read_csv(routine_dir + "/Routine_Effluent_States.csv")
data_routine_influent_compounds = pd.read_csv(routine_dir + "/Routine_Influent_Compounds.csv")
data_routine_effluent_compounds = pd.read_csv(routine_dir + "/Routine_Effluent_Compounds.csv")
    # Active (2)
data_active_influent_states = pd.read_csv(active_dir + "/Active_Influent_States.csv")
data_active_effluent_states = pd.read_csv(active_dir + "/Active_Effluent_States.csv")
data_active_influent_compounds = pd.read_csv(active_dir + "/Active_Influent_Compounds.csv")
data_active_effluent_compounds = pd.read_csv(active_dir + "/Active_Effluent_Compounds.csv")
    # LongActive (3)
data_longactive_influent_states = pd.read_csv(long_active_dir + "/LongActive_Influent_States.csv")
data_longactive_effluent_states = pd.read_csv(long_active_dir + "/LongActive_Effluent_States.csv")
data_longactive_influent_compounds = pd.read_csv(long_active_dir + "/LongActive_Influent_Compounds.csv")
data_longactive_effluent_compounds = pd.read_csv(long_active_dir + "/LongActive_Effluent_Compounds.csv")
    # LongRoutine (4)
data_longroutine_influent_states = pd.read_csv(long_routine_dir + "/LongRoutine_Influent_States.csv")
data_longroutine_effluent_states = pd.read_csv(long_routine_dir + "/LongRoutine_Effluent_States.csv")
data_longroutine_influent_compounds = pd.read_csv(long_routine_dir + "/LongRoutine_Influent_Compounds.csv")
data_longroutine_effluent_compounds = pd.read_csv(long_routine_dir + "/LongRoutine_Effluent_Compounds.csv")

# ----------------------------------------------------------
## Data to use for identifiability
# ----------------------------------------------------------

data_mapping_influent_states = {
    'HighRes': data_highres_influent_states,
    'Routine': data_routine_influent_states,
    'Active': data_active_influent_states,
    'LongActive': data_longactive_influent_states,
    'LongRoutine': data_longroutine_influent_states,
}
data_mapping_effluent_states = {
    'HighRes': data_highres_effluent_states,
    'Routine': data_routine_effluent_states,
    'Active': data_active_effluent_states,
    'LongActive': data_longactive_effluent_states,
    'LongRoutine': data_longroutine_effluent_states,
}
data_mapping_influent_compounds = {
    'HighRes': data_highres_influent_compounds,
    'Routine': data_routine_influent_compounds,
    'Active': data_active_influent_compounds,
    'LongActive': data_longactive_influent_compounds,
    'LongRoutine': data_longroutine_influent_compounds,
}
data_mapping_effluent_compounds = {
    'HighRes': data_highres_effluent_compounds,
    'Routine': data_routine_effluent_compounds,
    'Active': data_active_effluent_compounds,
    'LongActive': data_longactive_effluent_compounds,
    'LongRoutine': data_longroutine_effluent_compounds,
}

try:
    Data_Influent_states = data_mapping_influent_states[data_to_use_for_sampling]
    Data_Effluent_states = data_mapping_effluent_states[data_to_use_for_sampling]
    Data_Influent_compounds = data_mapping_influent_compounds[data_to_use_for_sampling]
    Data_Effluent_compounds = data_mapping_effluent_compounds[data_to_use_for_sampling]
except KeyError:
    raise ValueError("Invalid data for sanpling. Choose from HighRes, Routine, LongRoutine, HalfActive (Deprecated), Active, or LongActive.")

# ----------------------------------------------------------
print("Config and Data loaded")

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

# True values dict -- Just for reference / Debugging
true_values = {
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

# ----------------------------------------------------------
## Gradient Model Definition
# ----------------------------------------------------------

# Define the model using PyMC
    # Priors over all theta values
    # Initial values for ODE system [For now, use the same as the true values with no distribution]
    # ODE system solution using those parameters (using sunODE or PyMC ode solver)
        # ODE system specified in file 'asm3_model.py'
    # noise variance
    # Likelihood function (Nomral, mean = ODE solution, variance = known noise variance, observed = data)

# Priors mean and standard deviation for each parameter
    # Set all std deviations to 0 here, change them after
    # Set means to true values
# ----------------------------------------------------------
# Priors
    # Normal distribution for all parameters
        # Mean = Midpoint between the range
        # Std = to give range limits as 95% CI -- therefore: Var = (High - Low) / (2 * Critical Value), where Critical Value = 1.96 for 95% CI

alpha = 0.05
crit_value = norm.ppf(1 - (alpha / 2)) # For 95% CI

priors = {
    # Priors for parameters -- mean and std
    'k_H':          [np.mean(range_param_k_H),          (range_param_k_H[1] - range_param_k_H[0]) / (2 * crit_value)],    # (g COD_X_S) / (g COD_X_H * d), Hydrolysis rate constant
    'K_X':          [np.mean(range_param_K_X),          (range_param_K_X[1] - range_param_K_X[0]) / (2 * crit_value)],    # (g COD_X_S) / (g COD_X_H), Hydrolysis saturation constant
    'k_STO':        [np.mean(range_param_k_STO),        (range_param_k_STO[1] - range_param_k_STO[0]) / (2 * crit_value)],    # (g COD_S_S) / (g COD_X_H * d), Storage rate constant
    'eta_NOX':      [np.mean(range_param_eta_NOX),      (range_param_eta_NOX[1] - range_param_eta_NOX[0]) / (2 * crit_value)],    # - , Anoxic reduction factor
    'K_O2':         [np.mean(range_param_K_O2),         (range_param_K_O2[1] - range_param_K_O2[0]) / (2 * crit_value)],    # (g O2 / m3), Saturation constant for S_NO2
    'K_NOX':        [np.mean(range_param_K_NOX),        (range_param_K_NOX[1] - range_param_K_NOX[0]) / (2 * crit_value)],    # (g (NO3-)-N / m3), Saturation constant for S_NOX
    'K_S':          [np.mean(range_param_K_S),          (range_param_K_S[1] - range_param_K_S[0]) / (2 * crit_value)],    # (g COD_S_S / m3), Saturation constant for Substrate S_S
    'K_STO':        [np.mean(range_param_K_STO),        (range_param_K_STO[1] - range_param_K_STO[0]) / (2 * crit_value)],    # (g COD_X_STO / g COD_X_H), Saturation constant for X_STO
    'mu_H':         [np.mean(range_param_mu_H),         (range_param_mu_H[1] - range_param_mu_H[0]) / (2 * crit_value)],    # (d^-1), Heterotrophic maximum specific growth rate of X_H
    'K_NH4':        [np.mean(range_param_K_NH4),        (range_param_K_NH4[1] - range_param_K_NH4[0]) / (2 * crit_value)],    # (g N / m3), Saturation constant for ammonium, S_NH4
    'K_ALK':        [np.mean(range_param_K_ALK),        (range_param_K_ALK[1] - range_param_K_ALK[0]) / (2 * crit_value)],    # (mole HCO3- / m3), Saturation constant for alkalinity for X_H
    'b_H_O2':       [np.mean(range_param_b_H_O2),       (range_param_b_H_O2[1] - range_param_b_H_O2[0]) / (2 * crit_value)],    # (d^-1), Aerobic endogenous respiration rate of X_H
    'b_H_NOX':      [np.mean(range_param_b_H_NOX),      (range_param_b_H_NOX[1] - range_param_b_H_NOX[0]) / (2 * crit_value)],    # (d^-1), Anoxic endogenous respiration rate of X_H
    'b_STO_O2':     [np.mean(range_param_b_STO_O2),     (range_param_b_STO_O2[1] - range_param_b_STO_O2[0]) / (2 * crit_value)],    # (d^-1), Aerobic endogenous respiration rate for X_STO
    'b_STO_NOX':    [np.mean(range_param_b_STO_NOX),    (range_param_b_STO_NOX[1] - range_param_b_STO_NOX[0]) / (2 * crit_value)],    # (d^-1), Anoxic endogenous respiration rate for X_STO
    'mu_A':         [np.mean(range_param_mu_A),         (range_param_mu_A[1] - range_param_mu_A[0]) / (2 * crit_value)],    # (d^-1), Autotrophic maximum specific growth rate of X_A
    'K_A_NH4':      [np.mean(range_param_K_A_NH4),      (range_param_K_A_NH4[1] - range_param_K_A_NH4[0]) / (2 * crit_value)],    # (g N / m3), Ammonium substrate saturation constant for X_A
    'K_A_O2':       [np.mean(range_param_K_A_O2),       (range_param_K_A_O2[1] - range_param_K_A_O2[0]) / (2 * crit_value)],    # (g O2 / m3), Oxygen saturation for nitrifiers
    'K_A_ALK':      [np.mean(range_param_K_A_ALK),      (range_param_K_A_ALK[1] - range_param_K_A_ALK[0]) / (2 * crit_value)],    # (mole HCO3- / m3), Bicarbonate saturation for nitrifiers
    'b_A_O2':       [np.mean(range_param_b_A_O2),       (range_param_b_A_O2[1] - range_param_b_A_O2[0]) / (2 * crit_value)],    # (d^-1), Aerobic endogenous respiration rate of X_A
    'b_A_NOX':      [np.mean(range_param_b_A_NOX),      (range_param_b_A_NOX[1] - range_param_b_A_NOX[0]) / (2 * crit_value)],    # (d^-1), Anoxic endogenous respiration rate of X_A
    'f_S_I':        [np.mean(range_param_f_S_I),        (range_param_f_S_I[1] - range_param_f_S_I[0]) / (2 * crit_value)],    # (g COD_S_I) / (g COD_X_s), Production of S_I in hydrolisis
    'Y_STO_O2':     [np.mean(range_param_Y_STO_O2),     (range_param_Y_STO_O2[1] - range_param_Y_STO_O2[0]) / (2 * crit_value)],    # (g COD_X_STO) / (g COD_S_S), Aerobic yield of stored product per S_S
    'Y_STO_NOX':    [np.mean(range_param_Y_STO_NOX),    (range_param_Y_STO_NOX[1] - range_param_Y_STO_NOX[0]) / (2 * crit_value)],    # (g COD_X_STO) / (g COD_S_S), Anoxic yield of stored product per S_S
    'Y_H_O2':       [np.mean(range_param_Y_H_O2),       (range_param_Y_H_O2[1] - range_param_Y_H_O2[0]) / (2 * crit_value)],    # (g COD_X_H) / (g COD_S_STO), Aerobic yield of heterotrophic biomass
    'Y_H_NOX':      [np.mean(range_param_Y_H_NOX),      (range_param_Y_H_NOX[1] - range_param_Y_H_NOX[0]) / (2 * crit_value)],    # (g COD_X_H) / (g COD_S_STO), Anoxic yield of heterotrophic biomass
    'Y_A':          [np.mean(range_param_Y_A),          (range_param_Y_A[1] - range_param_Y_A[0]) / (2 * crit_value)],    # (g COD_X_A) / (g N_S_NOX), Yield of autotrophic biomass per NO3-N
    'f_X_I':        [np.mean(range_param_f_X_I),        (range_param_f_X_I[1] - range_param_f_X_I[0]) / (2 * crit_value)],    # (g COD_X_I) / (g COD_X_BM), Production of X_I in endogenous repiration
    'i_N_S_I':      [np.mean(range_param_i_N_S_I),      (range_param_i_N_S_I[1] - range_param_i_N_S_I[0]) / (2 * crit_value)],    # (g N) / (g COD_S_I), N content of S_I
    'i_N_S_S':      [np.mean(range_param_i_N_S_S),      (range_param_i_N_S_S[1] - range_param_i_N_S_S[0]) / (2 * crit_value)],    # (g N) / (g COD_S_S), N content of S_S
    'i_N_X_I':      [np.mean(range_param_i_N_X_I),      (range_param_i_N_X_I[1] - range_param_i_N_X_I[0]) / (2 * crit_value)],    # (g N) / (g COD_X_I), N content of X_I
    'i_N_X_S':      [np.mean(range_param_i_N_X_S),      (range_param_i_N_X_S[1] - range_param_i_N_X_S[0]) / (2 * crit_value)],    # (g N) / (g COD_X_S), N content of X_S
    'i_N_BM':       [np.mean(range_param_i_N_BM),       (range_param_i_N_BM[1] - range_param_i_N_BM[0]) / (2 * crit_value)],    # (g N) / (g COD_X_BM), N content of biomass, X_H, X_A
    'i_SS_X_I':     [np.mean(range_param_i_SS_X_I),     (range_param_i_SS_X_I[1] - range_param_i_SS_X_I[0]) / (2 * crit_value)],    # (g SS) / (g COD_X_I), SS to COD ratio for X_I
    'i_SS_X_S':     [np.mean(range_param_i_SS_X_S),     (range_param_i_SS_X_S[1] - range_param_i_SS_X_S[0]) / (2 * crit_value)],    # (g SS) / (g COD_X_S), SS to COD ratio for X_S
    'i_SS_BM':      [np.mean(range_param_i_SS_BM),      (range_param_i_SS_BM[1] - range_param_i_SS_BM[0]) / (2 * crit_value)],    # (g SS) / (g COD_X_BM), SS to COD ratio for biomass, X_H, X_A
    # Priors for noise variance
    'sigma_COD':        [0, 1], # Noise variance
    'sigma_NH4':        [0, 1], # Noise variance
    'sigma_NOx':        [0, 1], # Noise variance
    'sigma_TKN':        [0, 1], # Noise variance
    'sigma_Alkalinity': [0, 1], # Noise variance
    'sigma_TSS':        [0, 1]  # Noise variance
}

# ----------------------------------------------------------
print("Priors Defined")

# -----------------------------------------------------------
## Model Ode System
# -----------------------------------------------------------

# model_t_eval = np.linspace(Data_Influent_states['Time'].min(), Data_Influent_states['Time'].max(), 1000)
model_t_eval = Data_Effluent_states['Time']
model_input_data = Data_Influent_states.to_numpy()
model_reactor_volumes = [
    r1_V,  # Volume of reactor 1
]
model_y0 = get_reactor_initial_values(top_dir=top_dir)

# interpolated influent data
interpolated_influent_data = Data_Influent_states.copy()
interpolated_influent_data = interpolated_influent_data.set_index('Time').reindex(model_t_eval).interpolate(method='linear').reset_index()
# Interpolated output data
interpolated_output_data_states = Data_Effluent_states.copy()
interpolated_output_data_states = interpolated_output_data_states.set_index('Time').reindex(model_t_eval).interpolate(method='linear').reset_index()
interpolated_output_data_compounds = Data_Effluent_compounds.copy()
interpolated_output_data_compounds = interpolated_output_data_compounds.set_index('Time').reindex(model_t_eval).interpolate(method='linear').reset_index()

model_ode_function = lambda y, t, theta: ode_system_wrapper(t, y, theta, interpolated_influent_data.to_numpy(), model_reactor_volumes, use_polynomial=True)
model_n_states = len(model_y0)  # Number of states in the ODE system
model_n_theta = len(true_values)  # Number of parameters in the ODE system

model_ode_system = DifferentialEquation(
    func=model_ode_function,
    times=model_t_eval,
    n_states=model_n_states,  
    n_theta=model_n_theta,
    t0=model_t_eval[0],
)

# -----------------------------------------------------------
## Sensitivity Analysis Results
# -----------------------------------------------------------

NAASI_results = pd.read_csv(sensitivity_results_dir / 'NAASI_combined.csv')
NAASI_results = NAASI_results.set_index('Parameter')
# Get parameters with NAASI values below threshold
params_to_always_fix = {
    param: priors[param][0] for param in NAASI_results.index if NAASI_results.loc[param, 'NAASI'] < NAASI_threshold
}
# Print the parameters to always fix based on NAASI results 
print(f'Parameters to always fix based on NAASI results (Param, value):')
for param, fixed_val in params_to_always_fix.items():
    print(f'    {param}:    {fixed_val}')


# -----------------
## PCA Results
# -----------------

# PCA Results from Correlation - PCA previous step
    # Import PCA results from pkl file
pca_results = pd.read_pickle(pca_results_path)
pca_eigenvectors = pca_results['eigenvectors']
pca_scaler_mean = pca_results['scaler_mean']
pca_scaler_scale = pca_results['scaler_scale']
pca_param_names = pca_results['param_names']
pca_n_params_pca = len(pca_param_names)
pca_n_components = pca_results['n_components']

# parameter names specify order in which the parameters are stored in the eigenvectors, therfore what order they will be transformed to
# Print order of parameters to check
print("PCA Parameter Names (Order):")
for i, param in enumerate(pca_param_names):
    print(f"[{i}] {param}")

    # Transpose eigenvectors to get (pca_n_params_pca, pca_n_components)
pca_eigenvectors_transposed = pt.constant(pca_eigenvectors.T)
pca_mean_pt = pt.constant(pca_scaler_mean)
pca_scale_pt = pt.constant(pca_scaler_scale)

# -----------------------------------------------------------
## PyMC Model
# -----------------------------------------------------------

# Observed data for the likelihood function
# observed_data_states = Data_Effluent_states.drop(columns=['Time', 'Flowrate']).to_numpy()
# observed_data_compounds = Data_Effluent_compounds.drop(columns=['Time', 'Flowrate']).to_numpy()

observed_data_states = interpolated_output_data_states.drop(columns=['Time', 'Flowrate']).to_numpy()
observed_data_compounds = interpolated_output_data_compounds.drop(columns=['Time', 'Flowrate']).to_numpy()

# FIXED - Specify which are fixed (get lilst of names from parameters to always fix)
model_fixed_params = list(params_to_always_fix.keys())
# PCA - Parameters (in order) for pca results
model_pca_params = pca_param_names.copy()
# SPECIAL case for 0 (1e-6) parameters -- TODO: MANUAL
model_zero_params = ['K_A_ALK']
# FREE - All other parameters are free to sample
model_free_params = [param for param in priors.keys() if param not in model_fixed_params and param not in model_pca_params and param not in model_zero_params]

# Print the parameters to be fixed, PCA, zero, and free
print(f'--------------------------------------------------------')
print(f'Parameters to be fixed (Param, value):')
print(f'--------------------------------------------------------')
for param in model_fixed_params:
    print(f'    {param}')
print(f'--------------------------------------------------------')
print(f'Parameters to be PCA transformed (Param):')
print(f'--------------------------------------------------------')
for param in model_pca_params:
    print(f'    {param}')
print(f'--------------------------------------------------------')
print(f'Parameters to be set to zero (Param):')
print(f'--------------------------------------------------------')
for param in model_zero_params:
    print(f'    {param}')
print(f'--------------------------------------------------------')
print(f'Parameters to be sampled (Param):')
print(f'--------------------------------------------------------')

for param in model_free_params:
    print(f'    {param}')
print(f'--------------------------------------------------------')
print(f'\nEnsure that the parameters to be fixed, PCA, zero, and free are correct before proceeding with the model')


# PyMC model - Prior and Likelihood
with pymc.Model() as model:
    ## Priors - theta - parameters of the ODE 

    # The following parameters are fixed to prior means:
        # i_SS_BM, i_SS_X_I, i_SS_X_S, f_S_I, i_N_S_I, K_NH4, 
        # i_N_S_S, i_N_X_I, K_STO, K_ALK, i_N_X_S, b_A_NOX, b_H_NOX,
        # b_STO_NOX, i_N_BM, f_X_I, K_O2, b_A_O2, b_STO_O2
    # The following parameters are identifiable from the PLA graphs:
        # k_H, K_X, K_S, b_H_O2, K_A_NH4, Y_STO_O2, Y_STO_NOX, Y_H_O2, Y_A
    # The others are non-identifiable: 
        # k_STO, eta_NOX, K_NOX, mu_H, mu_A, K_A_O2, {K_A_ALK - Special Case}, Y_H_NOX
    # MAYBE Special cases (for those going to 0):
        # K_A_ALK

    ## Fixed parameters  - overwrite the normal distribution with fixed values
        # TODO - Must be done manually!
    i_SS_BM =     priors['i_SS_BM'][0]
    i_SS_X_I =    priors['i_SS_X_I'][0]
    i_SS_X_S =    priors['i_SS_X_S'][0]
    f_S_I =       priors['f_S_I'][0]
    i_N_S_I =     priors['i_N_S_I'][0]
    K_NH4 =      priors['K_NH4'][0]
    i_N_S_S =     priors['i_N_S_S'][0]
    i_N_X_I =     priors['i_N_X_I'][0]
    K_STO =       priors['K_STO'][0]
    K_ALK =       priors['K_ALK'][0]
    i_N_X_S =     priors['i_N_X_S'][0]
    b_A_NOX =     priors['b_A_NOX'][0]
    b_H_NOX =     priors['b_H_NOX'][0]
    b_STO_NOX =   priors['b_STO_NOX'][0]
    i_N_BM =      priors['i_N_BM'][0]
    f_X_I =       priors['f_X_I'][0]
    K_O2 =        priors['K_O2'][0]
    b_A_O2 =      priors['b_A_O2'][0]
    b_STO_O2 =    priors['b_STO_O2'][0]

    # TODO - Must be done manually!
    ## Identifiable Parameters - included in likelihood function
    k_H = pymc.TruncatedNormal("k_H", mu=priors['k_H'][0], sigma=priors['k_H'][1], initval=priors['k_H'][0], lower=0)
    K_X = pymc.TruncatedNormal("K_X", mu=priors['K_X'][0], sigma=priors['K_X'][1], initval=priors['K_X'][0], lower=0)
    K_S = pymc.TruncatedNormal("K_S", mu=priors['K_S'][0], sigma=priors['K_S'][1], initval=priors['K_S'][0], lower=0)
    b_H_O2 = pymc.TruncatedNormal("b_H_O2", mu=priors['b_H_O2'][0], sigma=priors['b_H_O2'][1], initval=priors['b_H_O2'][0], lower=0)
    K_A_NH4 = pymc.TruncatedNormal("K_A_NH4", mu=priors['K_A_NH4'][0], sigma=priors['K_A_NH4'][1], initval=priors['K_A_NH4'][0], lower=0)
    Y_STO_O2 = pymc.TruncatedNormal("Y_STO_O2", mu=priors['Y_STO_O2'][0], sigma=priors['Y_STO_O2'][1], initval=priors['Y_STO_O2'][0], lower=0)
    Y_STO_NOX = pymc.TruncatedNormal("Y_STO_NOX", mu=priors['Y_STO_NOX'][0], sigma=priors['Y_STO_NOX'][1], initval=priors['Y_STO_NOX'][0], lower=0)
    Y_H_O2 = pymc.TruncatedNormal("Y_H_O2", mu=priors['Y_H_O2'][0], sigma=priors['Y_H_O2'][1], initval=priors['Y_H_O2'][0], lower=0)
    Y_A = pymc.TruncatedNormal("Y_A", mu=priors['Y_A'][0], sigma=priors['Y_A'][1], initval=priors['Y_A'][0], lower=0)

    ## Non-Identifiable parameters (From PCA)
        # 1) Latent variables in PCA space
    n_pca_components = pca_results['n_components']
    z = pymc.Normal("z", mu=0, sigma=1, shape=n_pca_components) # normal priors on latent PCA space
        # 2) Affine transofrm: Map from latent space back to standardized parameter space
    theta_std_pca = pt.dot(pca_eigenvectors_transposed, z) 
        # 3) Inverse standardization
    theta_pca = theta_std_pca * pca_scale_pt + pca_mean_pt
        # 4) Ensure theta_pca has all positive values
        # Potential adds constraint that adjusts probability density
    pymc.Potential("theta_pca_positive", pt.switch(
        pt.all(theta_pca > 0), # Ensure all parameters are positive
        0, # No penalty if all parameters are positive, log(1) = 0
        -np.inf # Negative infinity if any parameter is negative (log likelihood will be zero)
        ))  
        # Alt, scale using softplus to ensure positivity, but values close to zero are scaled more than larger values, therefore small values are penalized more
    # theta_pca = pt.softplus(theta_pca).copy()

        # 5) Define Non-identifiable parameters
    # TODO - Must be done manually!
    # Using order of pca_param_names to define the order of the parameters in theta_pca
    k_STO = pymc.Deterministic("k_STO", theta_pca[pca_param_names.index("k_STO")])
    eta_NOX = pymc.Deterministic("eta_NOX", theta_pca[pca_param_names.index("eta_NOX")])
    K_NOX = pymc.Deterministic("K_NOX", theta_pca[pca_param_names.index("K_NOX")])
    mu_H = pymc.Deterministic("mu_H", theta_pca[pca_param_names.index("mu_H")])
    mu_A = pymc.Deterministic("mu_A", theta_pca[pca_param_names.index("mu_A")])
    K_A_O2 = pymc.Deterministic("K_A_O2", theta_pca[pca_param_names.index("K_A_O2")])
    Y_H_NOX = pymc.Deterministic("Y_H_NOX", theta_pca[pca_param_names.index("Y_H_NOX")])

    # ## Special cases for 0 (1e-6) from PLA
    # TODO - Must be done manually!
    K_A_ALK = 1e-6

    # For scipy - put params in list, in order of the ode system
    """
    Reference order of parameters in the ode system:
        "k_H": theta[0],
        "K_X": theta[1],
        "k_STO": theta[2],
        "eta_NOX": theta[3],
        "K_O2": theta[4],
        "K_NOX": theta[5],
        "K_S": theta[6],
        "K_STO": theta[7],
        "mu_H": theta[8],
        "K_NH4": theta[9],
        "K_ALK": theta[10],
        "b_H_O2": theta[11],
        "b_H_NOX": theta[12],
        "b_STO_O2": theta[13],
        "b_STO_NOX": theta[14],
        "mu_A": theta[15],
        "K_A_NH4": theta[16],
        "K_A_O2": theta[17],
        "K_A_ALK": theta[18],
        "b_A_O2": theta[19],
        "b_A_NOX": theta[20],
        "f_S_I": theta[21],
        "Y_STO_O2": theta[22],
        "Y_STO_NOX": theta[23],
        "Y_H_O2": theta[24],
        "Y_H_NOX": theta[25],
        "Y_A": theta[26],
        "f_X_I": theta[27],
        "i_N_S_I": theta[28],
        "i_N_S_S": theta[29],
        "i_N_X_I": theta[30],
        "i_N_X_S": theta[31],
        "i_N_BM": theta[32],
        "i_SS_X_I": theta[33],
        "i_SS_X_S": theta[34],
        "i_SS_BM": theta[35]
    """
    model_theta = [
        k_H, K_X, k_STO, eta_NOX, K_O2, K_NOX, K_S, K_STO,
        mu_H, K_NH4, K_ALK, b_H_O2, b_H_NOX, b_STO_O2, b_STO_NOX, mu_A, 
        K_A_NH4, K_A_O2, K_A_ALK, b_A_O2, b_A_NOX, f_S_I, Y_STO_O2, Y_STO_NOX,
        Y_H_O2, Y_H_NOX, Y_A, f_X_I, i_N_S_I, i_N_S_S, i_N_X_I, i_N_X_S, 
        i_N_BM, i_SS_X_I, i_SS_X_S, i_SS_BM
    ]
    
    # ODE solution - pymc/scipy
    ode_solution = model_ode_system(
        y0=model_y0,
        theta=model_theta,
    )
    # Get the states from the ODE solution
    y_hat = {
        'r1_S_O2': ode_solution[:, 0],
        'r1_S_I': ode_solution[:, 1],
        'r1_S_S': ode_solution[:, 2],
        'r1_S_NH4': ode_solution[:, 3],
        'r1_S_N2': ode_solution[:, 4],
        'r1_S_NOX': ode_solution[:, 5],
        'r1_S_ALK': ode_solution[:, 6],
        'r1_X_I': ode_solution[:, 7],
        'r1_X_S': ode_solution[:, 8],
        'r1_X_H': ode_solution[:, 9],
        'r1_X_STO': ode_solution[:, 10],
        'r1_X_A': ode_solution[:, 11],
        'r1_X_SS': ode_solution[:, 12]
    }

    # Determine COD, NH4, NOx, Alkalinity, TSS from the states
    model_COD = y_hat['r1_S_I'] + y_hat['r1_S_S'] + y_hat['r1_X_I'] + y_hat['r1_X_S'] + y_hat['r1_X_H'] + y_hat['r1_X_A'] + y_hat['r1_X_STO']
    model_NH4 = y_hat['r1_S_NH4']
    model_NOx = y_hat['r1_S_NOX']
    model_TKN = y_hat['r1_S_N2'] + y_hat['r1_S_NH4'] 
    model_Alkalinity = y_hat['r1_S_ALK']
    model_TSS = y_hat['r1_X_SS']

    # Limit COD, NH4, NOx, Alkalinity, TSS to be non-negative
    model_COD = pymc.math.maximum(model_COD, 0)
    model_NH4 = pymc.math.maximum(model_NH4, 0)
    model_NOx = pymc.math.maximum(model_NOx, 0)
    model_TKN = pymc.math.maximum(model_TKN, 0)
    model_Alkalinity = pymc.math.maximum(model_Alkalinity, 0)
    model_TSS = pymc.math.maximum(model_TSS, 0)

    # Noise variance
    sigma_COD = pymc.HalfNormal("sigma_COD", sigma=priors['sigma_COD'][1])
    sigma_NH4 = pymc.HalfNormal("sigma_NH4", sigma=priors['sigma_NH4'][1])
    sigma_NOx = pymc.HalfNormal("sigma_NOx", sigma=priors['sigma_NOx'][1])
    sigma_TKN = pymc.HalfNormal("sigma_TKN", sigma=priors['sigma_TKN'][1])
    sigma_Alkalinity = pymc.HalfNormal("sigma_Alkalinity", sigma=priors['sigma_Alkalinity'][1])
    sigma_TSS = pymc.HalfNormal("sigma_TSS", sigma=priors['sigma_TSS'][1])

    # Likelihood - Conditioned on the observed (training) data
    pymc.Normal('COD', mu=model_COD, sigma=sigma_COD, observed=observed_data_compounds[:,0])
    pymc.Normal('NH4', mu=model_NH4, sigma=sigma_NH4, observed=observed_data_compounds[:,1])
    pymc.Normal('NOx', mu=model_NOx, sigma=sigma_NOx, observed=observed_data_compounds[:,2])
    pymc.Normal('TKN', mu=model_TKN, sigma=sigma_TKN, observed=observed_data_compounds[:,3])
    pymc.Normal('Alkalinity', mu=model_Alkalinity, sigma=sigma_Alkalinity, observed=observed_data_compounds[:,4])
    pymc.Normal('TSS', mu=model_TSS, sigma=sigma_TSS, observed=observed_data_compounds[:,5])
# End of prior and likelihood model definition
print("\nModel created successfully.")

# ----------------------------------------------------------
## NUTS sampler -- Sunode
# ----------------------------------------------------------
# on Windows, cores=1 is required otherwise the sampler will fail (windows 11)
# on Linux, cores=1+ can be used to use all available cpu cores (mathcing core and chain count means full parallelization)

if __name__ == '__main__':
    __spec__ = None
    # Using NUTS sampler
    trace = run_model_NUTS(
        # vars_list=vars_list,
        model=model, 
        tune=config_tuning_samples, draws=config_draw_samples, 
        chains=config_sample_chains, cores=config_sample_cores,
        progressbar='combined'
    )
    # Using HamiltonianMC sampler
    # trace = run_model_HamiltonianMC(
    # #    vars_list=vars_list,
    #     model=model, 
    #     tune=config_tuning_samples, draws=config_draw_samples, 
    #     chains=config_sample_chains, cores=config_sample_cores,
    #     progressbar='combined'
    # )
    
print("Model run successfully.")

# ----------------------------------------------------------
## Saving the trace to a file
# ----------------------------------------------------------

# Save the trace to a file
trace.to_netcdf(results_dir / "trace.nc")

print("Trace saved successfully.")