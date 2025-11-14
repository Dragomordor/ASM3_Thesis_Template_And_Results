# ----------------------------------------------------------
## Import Libraries
# ----------------------------------------------------------

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import configparser

from scipy.integrate import solve_ivp

# -----------------------------
## File Paths
# -----------------------------

# Current Directory
current_dir = Path.cwd()
# Top Directory (for py it is this directory, for jupyter, and py in oracle cloud, it is the parent directory)
top_dir = current_dir.parent if current_dir.name == '2-Sensitivity' else current_dir

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


# Results Directory
results_dir = str(top_dir / "2-Sensitivity" / "Results")



# -----------------------------
## Import Libraries - Custom
# -----------------------------
sys.path.append(libs_dir)

from plant_config import get_reactor_initial_values
from asm3_model import ode_system_wrapper


# ----------------------------------------------------------
## Load From Config File
# ----------------------------------------------------------
config = configparser.ConfigParser()
config.read(config_dir / "config.ini")
seed = int(config['OVERALL']['seed'])        # Random seed
np.random.seed(seed)
data_to_use_for_sensitivity = str(config['OVERALL']['data_to_use_for_run'])  # Data to use for sensitivity analysis

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
    # Reactor volumes
r1_V = float(config['REACTOR']['r1_V'])        # Volume of reactor 1

# Sensitivity analysis parameters
epsilon = float(config['SENSITIVITY']['perturbation_epsilon'])  # Perturbation value for sensitivity analysis


# ----------------------------------------------------------
## Load Data from csv
# ----------------------------------------------------------

    # Highres (0)
data_highres_influent_states = pd.read_csv(highres_dir + "/HighRes_Influent_States.csv")
# data_highres_effluent_states = pd.read_csv(highres_dir + "/HighRes_Effluent_States.csv")
    # Routine (1)
data_routine_influent_states = pd.read_csv(routine_dir + "/Routine_Influent_States.csv")
# data_routine_effluent_states = pd.read_csv(routine_dir + "/Routine_Effluent_States.csv")
    # Active (2)
data_active_influent_states = pd.read_csv(active_dir + "/Active_Influent_States.csv")
# data_active_effluent_states = pd.read_csv(active_dir + "/Active_Effluent_States.csv")
    # LongActive (3)
data_longactive_influent_states = pd.read_csv(long_active_dir + "/LongActive_Influent_States.csv")
# data_longactive_effluent_states = pd.read_csv(long_active_dir + "/LongActive_Effluent_States.csv")
    # LongRoutine (4)
data_longroutine_influent_states = pd.read_csv(long_routine_dir + "/LongRoutine_Influent_States.csv")
# data_longroutine_effluent_states = pd.read_csv(long_routine_dir + "/LongRoutine_Effluent_States.csv")


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

try:
    Data_Influent = data_mapping_influent[data_to_use_for_sensitivity]
    # Data_Effluent = data_mapping_effluent[data_to_use_for_sensitivity]
except KeyError:
    raise ValueError("Invalid data for sanpling. Choose from HighRes, Routine, LongRoutine, Active, or LongActive.")

print(f'Imported configs and data for sensitivity analysis : {data_to_use_for_sensitivity}')

# ----------------------------------------------------------
## ODE System
# ----------------------------------------------------------
# Time span from higres data
# tspan = (data_influent_states['Time'].values[0], data_influent_states['Time'].values[-1])
# t_eval = data_influent_states['Time'].values
tspan = (Data_Influent['Time'].values[0], Data_Influent['Time'].values[-1])
t_eval = Data_Influent['Time'].values

# Initial values
y0 = get_reactor_initial_values(top_dir=top_dir)
# Ode system
ode_system = lambda t, y, theta: ode_system_wrapper(t=t, y=y, theta=theta, influentData=Data_Influent.to_numpy(), reactorVolumes=[r1_V])

# True theta dataframe
theta_true = pd.DataFrame({
    'k_H': [true_param_k_H],
    'K_X': [true_param_K_X],
    'k_STO': [true_param_k_STO],
    'eta_NOX': [true_param_eta_NOX],
    'K_O2': [true_param_K_O2],
    'K_NOX': [true_param_K_NOX],
    'K_S': [true_param_K_S],
    'K_STO': [true_param_K_STO],
    'mu_H': [true_param_mu_H],
    'K_NH4': [true_param_K_NH4],
    'K_ALK': [true_param_K_ALK],
    'b_H_O2': [true_param_b_H_O2],
    'b_H_NOX': [true_param_b_H_NOX],
    'b_STO_O2': [true_param_b_STO_O2],
    'b_STO_NOX': [true_param_b_STO_NOX],
    'mu_A': [true_param_mu_A],
    'K_A_NH4': [true_param_K_A_NH4],
    'K_A_O2': [true_param_K_A_O2],
    'K_A_ALK': [true_param_K_A_ALK],
    'b_A_O2': [true_param_b_A_O2],
    'b_A_NOX': [true_param_b_A_NOX],
    'f_S_I': [true_param_f_S_I],
    'Y_STO_O2': [true_param_Y_STO_O2],
    'Y_STO_NOX': [true_param_Y_STO_NOX],
    'Y_H_O2': [true_param_Y_H_O2],
    'Y_H_NOX': [true_param_Y_H_NOX],
    'Y_A': [true_param_Y_A],
    'f_X_I': [true_param_f_X_I],
    'i_N_S_I': [true_param_i_N_S_I],
    'i_N_S_S': [true_param_i_N_S_S],
    'i_N_X_I': [true_param_i_N_X_I],
    'i_N_X_S': [true_param_i_N_X_S],
    'i_N_BM': [true_param_i_N_BM],
    'i_SS_X_I': [true_param_i_SS_X_I],
    'i_SS_X_S': [true_param_i_SS_X_S],
    'i_SS_BM': [true_param_i_SS_BM],
})

# ----------------------------------------------------------
## Functions
# ----------------------------------------------------------

### Function to simulate model given theta
def simulate_model(ode_system, t_eval, y0, theta_df, return_states=True):
    """
    Function to simulate the model given theta

    Args:
        ode_system: function, ode system
        t_eval: array, time points to evaluate function at
        y0: array, initial values
        theta_df: dataframe, theta values

    Returns:
        state_df: dataframe, state values
    """
    # Define state column names
    state_names = [
        'S_O2', 'S_I', 'S_S', 'S_NH4', 'S_N2', 
        'S_NOX', 'S_ALK', 'X_I', 'X_S', 'X_H', 
        'X_STO', 'X_A', 'X_SS'
    ]
    # Simulation time span
    tspan = (t_eval.min(), t_eval.max())
    
    # Create ODE system with current parameter set
    ode_sys = lambda t, y: ode_system(t, y, theta_df.values[0])

    # Solve the ODE system using solve_ivp
    sol = solve_ivp(fun=ode_sys, t_span=tspan, y0=y0, t_eval=t_eval, method='BDF')

    # ----------------------------------------------------------
    # Create a DataFrame from the solution with proper state names
    og_state_df = pd.DataFrame(data=sol.y.T, columns=state_names)
    og_state_df.index = t_eval

    # Calculate derived states (COD, NH4, NOx, TKN, Alkalinity, TSS)
    og_state_df['COD'] = (og_state_df['S_I'] + og_state_df['S_S'] + og_state_df['X_I'] + 
                        og_state_df['X_S'] + og_state_df['X_H'] + og_state_df['X_A'] + og_state_df['X_STO'])
    og_state_df['NH4'] = og_state_df['S_NH4']
    og_state_df['NOx'] = og_state_df['S_NOX']
    og_state_df['TKN'] = og_state_df['S_NH4'] + og_state_df['S_N2']
    og_state_df['Alkalinity'] = og_state_df['S_ALK']
    og_state_df['TSS'] = og_state_df['X_SS']

    # Limit COD, NH4, NOx, Alkalinity, TSS to be non-negative
    og_state_df['COD'] = og_state_df['COD'].clip(lower=0)
    og_state_df['NH4'] = og_state_df['NH4'].clip(lower=0)
    og_state_df['NOx'] = og_state_df['NOx'].clip(lower=0)
    og_state_df['TKN'] = og_state_df['TKN'].clip(lower=0)
    og_state_df['Alkalinity'] = og_state_df['Alkalinity'].clip(lower=0)
    og_state_df['TSS'] = og_state_df['TSS'].clip(lower=0)

    # Create output state dafraframe with only time, COD, NH4, NOx, Alkalinity, TSS
    compound_df = og_state_df.loc[:, ['COD', 'NH4', 'NOx', 'TKN', 'Alkalinity', 'TSS']]
    compound_df.index = t_eval

    # Return the state DataFrame (or compound df)
    if return_states:
        return og_state_df
    else:
        return compound_df
    
print("Defined functions and imported libraries for sensitivity analysis.")

# ----------------------------------------------------------
## Sensitivity Analysis -- (1) True Theta simulation
# ----------------------------------------------------------

# True theta values simulation

# Create dataframe for true theta values
    # Rows: theta values
    # Columns: state values
    # Values: state values
true_theta_df = pd.DataFrame(theta_true, index=[0])

# Simulate the model with true theta values, i.e. Y(theta)
true_state_df = simulate_model(
    ode_system=ode_system,
    t_eval=t_eval,
    y0=y0,
    theta_df=true_theta_df,
    return_states=True
)

true_compound_df = simulate_model(
    ode_system=ode_system,
    t_eval=t_eval,
    y0=y0,
    theta_df=true_theta_df,
    return_states=False
)

print("Simulated the model with true theta values.")
# ----------------------------------------------------------
## Sensitivity Analysis -- (2) Perturbed Theta simulation (States)
# ----------------------------------------------------------

# Create a 3d dataframe to store the sensitivity analysis results
    # where:
    # Rows are time points
    # Columns are parameters
    # Depth are states
    # values are sensitivity values

# States
output_states = ['S_O2', 'S_I', 'S_S', 'S_NH4', 'S_N2', 'S_NOX', 'S_ALK', 'X_I', 'X_S', 'X_H', 'X_STO', 'X_A', 'X_SS']
output_compounds = ['COD', 'NH4', 'NOx', 'TKN', 'Alkalinity', 'TSS']

multi_columns_states = pd.MultiIndex.from_product([theta_true.keys(), output_states], names=['Parameter', 'State'])
multi_columns_compounds = pd.MultiIndex.from_product([theta_true.keys(), output_compounds], names=['Parameter', 'State'])

# Create a dataframe to store the state values themselves for each perturbation, i.e. Y(theta+epsilon) with MultiIndex columns
big_state_df = pd.DataFrame(index=t_eval, columns=multi_columns_states)
big_compound_df = pd.DataFrame(index=t_eval, columns=multi_columns_compounds)

# Create a dataframe to store the sensitivity values (normalized) for each perturbation, i.e. (dY/dtheta)*(theta/Y(theta)) where dY/dtheta = ( Y(theta+epsilon) - Y(theta) ) / epsilon
big_state_sensitivity_df = pd.DataFrame(index=t_eval, columns=multi_columns_states)
big_compound_sensitivity_df = pd.DataFrame(index=t_eval, columns=multi_columns_compounds)

# Theta to skip for sensitivity analysis -- they are fixed for this study
theta_to_skip = ['i_SS_X_I', 'i_SS_X_S', 'i_SS_BM']

# Loop through each parameter
for theta_name in theta_true.keys():
    # Create a dataframe for the current theta values
    theta_df = true_theta_df.copy()

    # Perturb the current theta value (If the theta is not in the list of thetas to skip)
    if theta_name not in theta_to_skip:
        theta_df[theta_name] = theta_df[theta_name] + epsilon

    # Simulate the model with the perturbed theta values
    perturbed_state_df = simulate_model(ode_system, t_eval, y0, theta_df, return_states=True)
    perturbed_compound_df = simulate_model(ode_system, t_eval, y0, theta_df, return_states=False)

    # Store the perturbed state values for all key outputs
    for state in output_states:
        big_state_df[(theta_name, state)] = perturbed_state_df[state]
    for state in output_compounds:
        big_compound_df[(theta_name, state)] = perturbed_compound_df[state]

    # Calculate the normalized sensitivity values for all key outputs
    for state in output_states:
        big_state_sensitivity_df[(theta_name, state)] = ( (perturbed_state_df[state] - true_state_df[state]) / epsilon ) * (theta_df[theta_name].values / true_state_df[state])
    for state in output_compounds:
        big_compound_sensitivity_df[(theta_name, state)] = ( (perturbed_compound_df[state] - true_compound_df[state]) / epsilon ) * (theta_df[theta_name].values / true_compound_df[state])

print("Simulated the model with perturbed theta values")

# ----------------------------------------------------------
## Calculate the average absolute sensitivity index (AASI)
# ----------------------------------------------------------

# Get the average absolute sensitivity Index (AASI) for each parameter
    # 1) Sum the absolute sensitivity values for each state for each parameter across all time points
    # 2) Sum the absolute sensitivity values for each parameter across all states
    # (No division is required as they are normalized by the state value alreay in the sensitivity calculation)

# 1. Compute the absolute values of the sensitivity
abs_state_sensitivity_df = big_state_sensitivity_df.abs()
abs_compound_sensitivity_df = big_compound_sensitivity_df.abs()

# 2. Sum the absolute sensitivity values for each state for each parameter across all time points
sum_abs_statte_sensitivity_df = abs_state_sensitivity_df.sum(axis=0)
sum_abs_statte_sensitivity_df = abs_compound_sensitivity_df.sum(axis=0)


# 3. Sum the absolute sensitivity values for each parameter across all states
AASI_state_df = sum_abs_statte_sensitivity_df.groupby(level='Parameter').sum()
AASI_compound_df = sum_abs_statte_sensitivity_df.groupby(level='Parameter').sum()
    # Change AASI_df to have parameter name as first column and AASI as second column
AASI_state_df = AASI_state_df.reset_index().rename(columns={0: 'AASI'})
AASI_compound_df = AASI_compound_df.reset_index().rename(columns={0: 'AASI'})
    # Save the AASI to a csv file
AASI_state_df.to_csv(results_dir + '/AASI_states.csv', index=False)
AASI_compound_df.to_csv(results_dir + '/AASI_compounds.csv', index=False)


# Normalize AASI values
Normalized_AASI_state_df = AASI_state_df.copy()
Normalized_AASI_state_df['AASI'] = AASI_state_df['AASI'] / AASI_state_df['AASI'].max()
Normalized_AASI_compound_df = AASI_compound_df.copy()
Normalized_AASI_compound_df['AASI'] = AASI_compound_df['AASI'] / AASI_compound_df['AASI'].max()
    # Sort the dataframe by Normalized AASI values
Normalized_AASI_state_df = Normalized_AASI_state_df.sort_values(by='AASI', ascending=True)
Normalized_AASI_compound_df = Normalized_AASI_compound_df.sort_values(by='AASI', ascending=True)

    # Save the Normalized Ranked AASI to a csv file
Normalized_AASI_state_df.to_csv(results_dir + '/NAASI_states.csv', index=False)
Normalized_AASI_compound_df.to_csv(results_dir + '/NAASI_compounds.csv', index=False)

# ----------------------------------------------------------
## Combined AASI and NAASI for states and compounds
# ----------------------------------------------------------

# Combined AASI and NAASI for states and compounds
    # Take highest AASI value for each parameter from both states and compounds
combined_AASI_df = pd.DataFrame({
    'Parameter': AASI_state_df['Parameter'],
    'AASI': AASI_state_df['AASI'].combine(AASI_compound_df['AASI'], max)
})
combined_AASI_df = combined_AASI_df.sort_values(by='AASI', ascending=True) # sort
combined_AASI_df.to_csv(results_dir + '/AASI_combined.csv', index=False)

# Combined NAASI for states and compounds
combined_NAASI_df = pd.DataFrame({
    'Parameter': Normalized_AASI_state_df['Parameter'],
    'NAASI': Normalized_AASI_state_df['AASI'].combine(Normalized_AASI_compound_df['AASI'], max)
})
combined_NAASI_df = combined_NAASI_df.sort_values(by='NAASI', ascending=True) # sort
combined_NAASI_df.to_csv(results_dir + '/NAASI_combined.csv', index=False)

print("Calculated the average absolute sensitivity index (AASI) and saved to csv files.")