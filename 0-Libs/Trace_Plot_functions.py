# ---------------------------------------------------------
## Libraries
# ---------------------------------------------------------

# import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.integrate import solve_ivp
from asm3_model import ode_system_wrapper

# ---------------------------------------------------------
## Functions
# ---------------------------------------------------------

## Plot Data
def plot_data(data, data_to_plot, ax, plot_kwargs, ylim=None):
    # Have plot_kwargs be the following format:
        # plot_kwargs = {'lw': 2, 'marker': 'o', 'markersize': 10, 'label': 'data', 'color': 'blue', 'title': 'Data', 'alpha': 1}
    lw = plot_kwargs.get('lw', 0)
    marker = plot_kwargs.get('marker', 'o')
    markersize = plot_kwargs.get('markersize', 10)
    label = plot_kwargs.get('label', 'data')
    color = plot_kwargs.get('color', 'blue')
    title = plot_kwargs.get('title', 'Data')
    alpha = plot_kwargs.get('alpha', 1)
    
    # data_to_plot is a string of the column name to plot:
        # data_to_plot: 'Flowrate', 'COD', 'NH4', 'NOx', 'Alkalinity', 'TSS'
    # Units for data_to_plot:
    unit_choices = {
        'Flowrate': 'Flowrate (m3/day)',
        'COD': 'COD (g/m3)',
        'NH4': 'NH4 (g/m3)',
        'NOx': 'NOx (g/m3)',
        'Alkalinity': 'Alkalinity (g/m3)',
        'TSS': 'TSS (g/m3)',

        'S_O2': 'S_O2 (g/m3)',
        'S_I': 'S_I (g/m3)',
        'S_S': 'S_S (g/m3)',
        'S_NH4': 'S_NH4 (g/m3)',
        'S_N2': 'S_N2 (g/m3)',
        'S_NOX': 'S_NOX (g/m3)',
        'S_ALK': 'S_ALK (g/m3)',
        'X_I': 'X_I (g/m3)',
        'X_S': 'X_S (g/m3)',
        'X_H': 'X_H (g/m3)',
        'X_STO': 'X_STO (g/m3)',
        'X_A': 'X_A (g/m3)',
        'X_SS': 'X_SS (g/m3)'
    }

    unit_to_plot = unit_choices[data_to_plot]

    # Plot Data
    ax.plot(data.Time, data.loc[:, data_to_plot], lw=lw, marker=marker, markersize=markersize, label=label, color=color, alpha=alpha)
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 0.5))
    # Set xlim equal to time min and max
    ax.set_xlim(data.Time.min(), data.Time.max())
    # Set ylim equal to extend 5% beyond data limits (but not less than 0)
    if ylim is not None:
        ax.set_ylim(max(data[data_to_plot].min() - ylim*abs(data[data_to_plot].min()), 0), data[data_to_plot].max() + ylim*abs(data[data_to_plot].max()))
    # Set x and y labels
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel(unit_to_plot, fontsize=12)
    ax.set_title(title, fontsize=15)
    return ax
# End of plot_data function

def sim_all_states(
        ode_fun,
        t_eval,
        y0,
        param_samples_df
    ):
    # Define state column names and their meanings
    state_names = [
        'S_O2', 'S_I', 'S_S', 'S_NH4', 'S_N2', 
        'S_NOX', 'S_ALK', 'X_I', 'X_S', 'X_H', 
        'X_STO', 'X_A', 'X_SS'
    ]
    
    # Simulation time span
    tspan = (t_eval.min(), t_eval.max())
    
    # Initialize list to store all simulated states
    all_states = []
    
    # Iterate over each set of parameters (each row in the parameter dataframe)
    for i, param_set in param_samples_df.iterrows():
        # Create ODE system with current parameter set
        ode_system = lambda t, y: ode_fun(t, y, param_set.values)
        
        # Solve the ODE system using solve_ivp
        sol = solve_ivp(fun=ode_system, t_span=tspan, y0=y0, t_eval=t_eval, method='RK45')

        # Create a DataFrame from the solution with proper state names
        state_df = pd.DataFrame(data=sol.y.T, columns=state_names)
        
        # Calculate derived states (COD, NH4, NOx, Alkalinity, TSS)
        state_df['COD'] = (state_df['S_I'] + state_df['S_S'] + state_df['X_I'] + 
                           state_df['X_S'] + state_df['X_H'] + state_df['X_A'] + state_df['X_STO'])
        state_df['NH4'] = state_df['S_NH4']
        state_df['NOx'] = state_df['S_NOX']
        state_df['Alkalinity'] = state_df['S_ALK']
        state_df['TSS'] = state_df['X_SS']

        # Limit COD, NH4, NOx, Alkalinity, TSS to be non-negative
        state_df['COD'] = state_df['COD'].clip(lower=0)
        state_df['NH4'] = state_df['NH4'].clip(lower=0)
        state_df['NOx'] = state_df['NOx'].clip(lower=0)
        state_df['Alkalinity'] = state_df['Alkalinity'].clip(lower=0)
        state_df['TSS'] = state_df['TSS'].clip(lower=0)

        # Append the relevant state results (COD, NH4, NOx, Alkalinity, TSS) to all_states list
        all_states.append(state_df[['COD', 'NH4', 'NOx', 'Alkalinity', 'TSS']])

        print('Parameter set ', i, ' simulated successfully')
    
    return all_states
# End of sim_all_states function


def plot_inference_line_for_state(
        time,
        states,
        state_to_plot,
        ax,
        plot_kwargs
    ):
    # Have plot_kwargs be the following format:
        # plot_kwargs = {'lw': 1, 'marker': 'None', 'markersize': 0, 'label': 'data', 'color': 'blue', 'title': 'Data', 'alpha': 1}
    lw = plot_kwargs.get('lw', 2)
    marker = plot_kwargs.get('marker', 'None')
    markersize = plot_kwargs.get('markersize', 0)
    label = plot_kwargs.get('label', 'data')
    color = plot_kwargs.get('color', 'blue')
    title = plot_kwargs.get('title', 'Data')
    alpha = plot_kwargs.get('alpha', 1)

    # State to plot is a string of the column name to plot:
        # state_to_plot: 'COD', 'NH4', 'NOx', 'Alkalinity', 'TSS'
    
    # Units for state_to_plot:
    unit_choices = {
        'COD': 'COD (g/m3)',
        'NH4': 'NH4 (g/m3)',
        'NOx': 'NOx (g/m3)',
        'Alkalinity': 'Alkalinity (g/m3)',
        'TSS': 'TSS (g/m3)',
        'S_O2': 'S_O2 (g/m3)',
        'S_I': 'S_I (g/m3)',
        'S_S': 'S_S (g/m3)',
        'S_NH4': 'S_NH4 (g/m3)',
        'S_N2': 'S_N2 (g/m3)',
        'S_NOX': 'S_NOX (g/m3)',
        'S_ALK': 'S_ALK (g/m3)',
        'X_I': 'X_I (g/m3)',
        'X_S': 'X_S (g/m3)',
        'X_H': 'X_H (g/m3)',
        'X_STO': 'X_STO (g/m3)',
        'X_A': 'X_A (g/m3)',
        'X_SS': 'X_SS (g/m3)'
    }
    unit_to_plot = unit_choices[state_to_plot]

    ax.plot(time, states[state_to_plot], lw=lw, marker=marker, markersize=markersize, label=label, color=color, alpha=alpha)
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 0.5))

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel(unit_to_plot, fontsize=12)
    ax.set_title(title, fontsize=15)
    return ax
# End of plot_inference_line_for_state function

def plot_all_inference_lines(
        data,
        data_to_plot,
        all_states,
        t_eval,
        ax,
        plot_kwargs_data,
        plot_kwargs_inference,
        ylim = None
    ):
    """
    Args:
        data (DataFrame): DataFrame containing the data to plot
        data_to_plot (str): Column name of the data to plot
        all_states (list of DataFrames): List of DataFrames containing the simulated states
        t_eval (array): Time points at which the states were simulated
        ax (Axes): Matplotlib Axes object to plot on
        plot_kwargs_data (dict): Dictionary of keyword arguments to pass to the plot
        plot_kwargs_inference (dict): Dictionary of keyword arguments to pass to the plot
        ylim (float): Amount to extend the y-axis limits beyond the data limits

    Returns:
        None : Plots the data and all inference lines on the given ax object
    """

    time = t_eval
    # Plot raw data
    plot_data(data=data, data_to_plot=data_to_plot, ax=ax, plot_kwargs=plot_kwargs_data)

    # Plot all inference lines (plots each row of all_states)
    for states_idx in range(len(all_states)):
        # Extract row of states
        states = all_states[states_idx]
        # Plot the state
        plot_inference_line_for_state(time=time, states=states, state_to_plot=data_to_plot, ax=ax, plot_kwargs=plot_kwargs_inference)

    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:2], labels=labels[:2], fontsize=12, loc='upper left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Data and Inference Lines for ' + data_to_plot, fontsize=15)
    ax.set_xlim(time.min(), time.max())

    # Find max and min of data and all states for ylim
    data_max = data[data_to_plot].max()
    data_min = data[data_to_plot].min()
    states_max = np.max([states[data_to_plot].max() for states in all_states])
    states_min = np.min([states[data_to_plot].min() for states in all_states])
    max_val = max(data_max, states_max)
    min_val = min(data_min, states_min)

    # Set ylim equal to extend 10% beyond data limits (but not less than 0)
    if ylim is not None:
        ax.set_ylim(max(min_val - ylim*abs(min_val), -0.01), max_val + ylim*abs(max_val))

# End of plot_all_inference_lines function


    