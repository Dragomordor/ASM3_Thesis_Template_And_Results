# ----------------------------------------------------------
## Purpose
# ----------------------------------------------------------
"""
Python script that contains the ASM3 ode system for the simulation.
"""

# ----------------------------------------------------------
## Importing libraries
# ----------------------------------------------------------

import numpy as np

# ----------------------------------------------------------
## Function to define the simultaneous ODE system - Mass Balance
# ----------------------------------------------------------

# Simplified version of the ASM3 model - only using arrays for theta and reactorVolumes
def ode_system(y, t, theta, influentData, reactorVolumes, use_polynomial=False):
    """
    Function to define the ODE system for the ASM3 model.

    Args:
    y: array
        Array containing the state variables (concentrations of compounds) at time t (g Compound / m3)
    t: float
        Time (day) at time t 
    theta: Array
        Array containing the model parameters
    influentData: Array
        Array containing the influent data for the plant reactor model
            Each column of the array corresponds to the following data:
            [Time, Flowrate, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
            Each row of the array corresponds to the data at a specific time indicated by the first column
    reactorVolumes: Array
        Array containing the volumes of each reactor in the plant

    Returns:
    dYdt: array
        Array containing the derivatives of the state variables at time t
    """
    # Put theta into dictionary for easier access
    theta_dict = {
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
    }
    # Ensure y of all is positive
    y = np.abs(y)

    # Put states into dictionary for easier access
    y_dict = {
        "r1_S_O2": y[0],
        "r1_S_I": y[1],
        "r1_S_S": y[2],
        "r1_S_NH4": y[3],
        "r1_S_N2": y[4],
        "r1_S_NOX": y[5],
        "r1_S_ALK": y[6],
        "r1_X_I": y[7],
        "r1_X_S": y[8],
        "r1_X_H": y[9],
        "r1_X_STO": y[10],
        "r1_X_A": y[11],
        "r1_X_SS": y[12]
        # ... Add other reactors
    }

    NumReactors = len(reactorVolumes)

    # Reactor volume dictionary
    reactorVolumes_dict = {
        "V1": reactorVolumes[0],
        # ... Add other reactors
    }

    # ------------
    ### 1) Reaction rates for each state variable -- from asm3_reaction_rates
    # ------------
    ## Reactor 1
    # get r1_y (dict) from y_dict
    r1_y = {
        "S_O2": y_dict["r1_S_O2"],
        "S_I": y_dict["r1_S_I"],
        "S_S": y_dict["r1_S_S"],
        "S_NH4": y_dict["r1_S_NH4"],
        "S_N2": y_dict["r1_S_N2"],
        "S_NOX": y_dict["r1_S_NOX"],
        "S_ALK": y_dict["r1_S_ALK"],
        "X_I": y_dict["r1_X_I"],
        "X_S": y_dict["r1_X_S"],
        "X_H": y_dict["r1_X_H"],
        "X_STO": y_dict["r1_X_STO"],
        "X_A": y_dict["r1_X_A"],
        "X_SS": y_dict["r1_X_SS"]
    }
    r1_rates = asm3_reaction_rates(r1_y, theta_dict)

    ## Reactor 2
    # TODO: Add other reactors

    # ------------
    #### 2) Mass Balance
    # TODO: Do this in separate function
    # ------
    ## SETUP
    # ------
    # Get influent data at time t
    if use_polynomial: # For pymc implementation
        plant_inlet = get_influent_polyfit(influentData, t)
    else:
        plant_inlet = get_influent_at_time(influentData, t)

    # ------
    ## Reactor 1
    # ------
    ## Flowrate Balance
    # Reactor 1 Inlet = Plant Inlet
    r1_Inlets = plant_inlet       # TODO: Might update this for RAS
    r1_Outlets = {
        "Qout": r1_Inlets["Qin"],
        "S_O2": r1_y["S_O2"],
        "S_I": r1_y["S_I"],
        "S_S": r1_y["S_S"],
        "S_NH4": r1_y["S_NH4"],
        "S_N2": r1_y["S_N2"],
        "S_NOX": r1_y["S_NOX"],
        "S_ALK": r1_y["S_ALK"],
        "X_I": r1_y["X_I"],
        "X_S": r1_y["X_S"],
        "X_H": r1_y["X_H"],
        "X_STO": r1_y["X_STO"],
        "X_A": r1_y["X_A"],
        "X_SS": r1_y["X_SS"]
    }

    ## Component Balance -- ( (InFlow * InConc - OutFlow * OutConc) / Volume ) + Reaction Rate
    r1_dC_S_O2_dt = 0 # Constant for study
    r1_dC_S_I_dt = ((( r1_Inlets["Qin"] * r1_Inlets["S_I"] ) - ( r1_Outlets["Qout"] * r1_Outlets["S_I"] )) / reactorVolumes_dict["V1"]) + r1_rates["S_I"]
    r1_dC_S_S_dt = ((( r1_Inlets["Qin"] * r1_Inlets["S_S"] ) - ( r1_Outlets["Qout"] * r1_Outlets["S_S"] )) / reactorVolumes_dict["V1"]) + r1_rates["S_S"]
    r1_dC_S_NH4_dt = ((( r1_Inlets["Qin"] * r1_Inlets["S_NH4"] ) - ( r1_Outlets["Qout"] * r1_Outlets["S_NH4"] )) / reactorVolumes_dict["V1"]) + r1_rates["S_NH4"]
    r1_dC_S_N2_dt = ((( r1_Inlets["Qin"] * r1_Inlets["S_N2"] ) - ( r1_Outlets["Qout"] * r1_Outlets["S_N2"] )) / reactorVolumes_dict["V1"]) + r1_rates["S_N2"]
    r1_dC_S_NOX_dt = ((( r1_Inlets["Qin"] * r1_Inlets["S_NOX"] ) - ( r1_Outlets["Qout"] * r1_Outlets["S_NOX"] )) / reactorVolumes_dict["V1"]) + r1_rates["S_NOX"]
    r1_dC_S_ALK_dt = ((( r1_Inlets["Qin"] * r1_Inlets["S_ALK"] ) - ( r1_Outlets["Qout"] * r1_Outlets["S_ALK"] )) / reactorVolumes_dict["V1"]) + r1_rates["S_ALK"]
    r1_dC_X_I_dt = ((( r1_Inlets["Qin"] * r1_Inlets["X_I"] ) - ( r1_Outlets["Qout"] * r1_Outlets["X_I"] )) / reactorVolumes_dict["V1"]) + r1_rates["X_I"]
    r1_dC_X_S_dt = ((( r1_Inlets["Qin"] * r1_Inlets["X_S"] ) - ( r1_Outlets["Qout"] * r1_Outlets["X_S"] )) / reactorVolumes_dict["V1"]) + r1_rates["X_S"]
    r1_dC_X_H_dt = ((( r1_Inlets["Qin"] * r1_Inlets["X_H"] ) - ( r1_Outlets["Qout"] * r1_Outlets["X_H"] )) / reactorVolumes_dict["V1"]) + r1_rates["X_H"]
    r1_dC_X_STO_dt = ((( r1_Inlets["Qin"] * r1_Inlets["X_STO"] ) - ( r1_Outlets["Qout"] * r1_Outlets["X_STO"] )) / reactorVolumes_dict["V1"]) + r1_rates["X_STO"]
    r1_dC_X_A_dt = ((( r1_Inlets["Qin"] * r1_Inlets["X_A"] ) - ( r1_Outlets["Qout"] * r1_Outlets["X_A"] )) / reactorVolumes_dict["V1"]) + r1_rates["X_A"]
    r1_dC_X_SS_dt = ((( r1_Inlets["Qin"] * r1_Inlets["X_SS"] ) - ( r1_Outlets["Qout"] * r1_Outlets["X_SS"] )) / reactorVolumes_dict["V1"]) + r1_rates["X_SS"]

    dr1_dt = [
        r1_dC_S_O2_dt,
        r1_dC_S_I_dt,
        r1_dC_S_S_dt,
        r1_dC_S_NH4_dt,
        r1_dC_S_N2_dt,
        r1_dC_S_NOX_dt,
        r1_dC_S_ALK_dt,
        r1_dC_X_I_dt,
        r1_dC_X_S_dt,
        r1_dC_X_H_dt,
        r1_dC_X_STO_dt,
        r1_dC_X_A_dt,
        r1_dC_X_SS_dt
    ]

    # ------
    ## Reactor 2
    # ------
    # TODO: Add other reactors
    # TODO: Add recycle, bypass, oxygen pumping

    # ------
    ## State Variables Return
    # ------
    # Output array dYdt containing the derivatives of the state variables at time t
        # Add other reactors together    
    dYdt = dr1_dt
    # Return the array of differential equations
    return dYdt
# End of ode_system function


# ----------------------------------------------------------
## Function to get influent data at time t
# ----------------------------------------------------------

def get_influent_at_time(influentData, t):
    """
        Function to get the influent data for the plant at a specific time t

        Args:
        influentData: Array
            Array containing the influent data for the plant reactor model
            Each column of the array corresponds to the following data:
            [Time, Flowrate, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
            Each row of the array corresponds to the data at a specific time indicated by the first column
        t: float
            Time at which the influent data is required (days)

        Returns:
        Influent_data: Array
            Array containing the influent data at time t
            Form: [Qin, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
        """
    # Throw an error if the influent data is not provided or is empty
    if influentData is None or len(influentData) == 0:
        raise ValueError("Influent data is empty or not provided")
    
    # Get the influent data at time t
    plant_time = influentData[:, 0] # Get the time data, first column of the array plant_influent_data

    ## Perform linear interpolation for each variable using np.interp
    Qin = np.interp(t, plant_time, influentData[:, 1])
    S_O2 = np.interp(t, plant_time, influentData[:, 2])
    S_I = np.interp(t, plant_time, influentData[:, 3])
    S_S = np.interp(t, plant_time, influentData[:, 4])
    S_NH4 = np.interp(t, plant_time, influentData[:, 5])
    S_N2 = np.interp(t, plant_time, influentData[:, 6])
    S_NOX = np.interp(t, plant_time, influentData[:, 7])
    S_ALK = np.interp(t, plant_time, influentData[:, 8])
    X_I = np.interp(t, plant_time, influentData[:, 9])
    X_S = np.interp(t, plant_time, influentData[:, 10])
    X_H = np.interp(t, plant_time, influentData[:, 11])
    X_STO = np.interp(t, plant_time, influentData[:, 12])
    X_A = np.interp(t, plant_time, influentData[:, 13])
    X_SS = np.interp(t, plant_time, influentData[:, 14])

    # Return dictionary
    influent_dict = {
        "Qin": Qin,
        "S_O2": S_O2,
        "S_I": S_I,
        "S_S": S_S,
        "S_NH4": S_NH4,
        "S_N2": S_N2,
        "S_NOX": S_NOX,
        "S_ALK": S_ALK,
        "X_I": X_I,
        "X_S": X_S,
        "X_H": X_H,
        "X_STO": X_STO,
        "X_A": X_A,
        "X_SS": X_SS
    }
    return influent_dict
# End of get_influent_at_time function



def get_influent_polyfit(influentData, t):
    """
    Function to get the influent data for the plant at a specific time t

    Args:
    influentData: Array
        Array containing the influent data for the plant reactor model
        Each column of the array corresponds to the following data:
        [Time, Flowrate, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
        Each row of the array corresponds to the data at a specific time indicated by the first column
    t: sympy.Symbol or float
        Time at which the influent data is required (days)

    Returns:
    Influent_data: dataclass
        dataclass containing the influent data at time t (symbolic expressions)
        Form: [Qin, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
    """
    # Throw an error if the influent data is not provided or is empty
    if influentData is None or len(influentData) == 0:
        raise ValueError("Influent data is empty or not provided")
 
    time_data = influentData[:, 0]  # Get the time data, first column of the array plant_influent_data
    Qin_data= influentData[:, 1]
    S_O2_data = influentData[:, 2]
    S_I_data = influentData[:, 3]
    S_S_data = influentData[:, 4]
    S_NH4_data = influentData[:, 5]
    S_N2_data = influentData[:, 6]
    S_NOX_data = influentData[:, 7]
    S_ALK_data = influentData[:, 8]
    X_I_data = influentData[:, 9]
    X_S_data = influentData[:, 10]
    X_H_data = influentData[:, 11]
    X_STO_data = influentData[:, 12]
    X_A_data = influentData[:, 13]
    X_SS_data = influentData[:, 14]

    poly_degree = 12
    # Fit a polynomial to each variable which depends on time using np.Polynomial.fit. The degree should allow all data points to be fitted
    coeff_Qin = np.polynomial.Polynomial.fit(x=time_data, y=Qin_data, deg=poly_degree)
    coeff_S_O2 = np.polynomial.Polynomial.fit(x=time_data, y=S_O2_data, deg=poly_degree)
    coeff_S_I = np.polynomial.Polynomial.fit(x=time_data, y=S_I_data, deg=poly_degree)
    coeff_S_S = np.polynomial.Polynomial.fit(x=time_data, y=S_S_data, deg=poly_degree)
    coeff_S_NH4 = np.polynomial.Polynomial.fit(x=time_data, y=S_NH4_data, deg=poly_degree)
    coeff_S_N2 = np.polynomial.Polynomial.fit(x=time_data, y=S_N2_data, deg=poly_degree)
    coeff_S_NOX = np.polynomial.Polynomial.fit(x=time_data, y=S_NOX_data, deg=poly_degree)
    coeff_S_ALK = np.polynomial.Polynomial.fit(x=time_data, y=S_ALK_data, deg=poly_degree)
    coeff_X_I = np.polynomial.Polynomial.fit(x=time_data, y=X_I_data, deg=poly_degree)
    coeff_X_S = np.polynomial.Polynomial.fit(x=time_data, y=X_S_data, deg=poly_degree)
    coeff_X_H = np.polynomial.Polynomial.fit(x=time_data, y=X_H_data, deg=poly_degree)
    coeff_X_STO = np.polynomial.Polynomial.fit(x=time_data, y=X_STO_data, deg=poly_degree)
    coeff_X_A = np.polynomial.Polynomial.fit(x=time_data, y=X_A_data, deg=poly_degree)
    coeff_X_SS = np.polynomial.Polynomial.fit(x=time_data, y=X_SS_data, deg=poly_degree)

    # Get the polynomial expression for each variable at time t -- example: Val = a*t^2 + b*t + c -- should explicity use t
    Qin_poly = coeff_Qin(t)
    S_O2_poly = coeff_S_O2(t)
    S_I_poly = coeff_S_I(t)
    S_S_poly = coeff_S_S(t)
    S_NH4_poly = coeff_S_NH4(t)
    S_N2_poly = coeff_S_N2(t)
    S_NOX_poly = coeff_S_NOX(t)
    S_ALK_poly = coeff_S_ALK(t)
    X_I_poly = coeff_X_I(t)
    X_S_poly = coeff_X_S(t)
    X_H_poly = coeff_X_H(t)
    X_STO_poly = coeff_X_STO(t)
    X_A_poly = coeff_X_A(t)
    X_SS_poly = coeff_X_SS(t)

    influent_dict = {
        "Qin": Qin_poly,
        "S_O2": S_O2_poly,
        "S_I": S_I_poly,
        "S_S": S_S_poly,
        "S_NH4": S_NH4_poly,
        "S_N2": S_N2_poly,
        "S_NOX": S_NOX_poly,
        "S_ALK": S_ALK_poly,
        "X_I": X_I_poly,
        "X_S": X_S_poly,
        "X_H": X_H_poly,
        "X_STO": X_STO_poly,
        "X_A": X_A_poly,
        "X_SS": X_SS_poly
    }

    return influent_dict


# ----------------------------------------------------------
## ASM3 Simulation - ASM3 simulates reaction rates for Mass Balance
# ----------------------------------------------------------

def asm3_reaction_rates(yReactor_dict, theta_dict):
    """
    Function to calculate the reaction rates for the ASM3 model.

    Args:
    yReactor: dataclass
        dataclass containing the state variables for the simulation at time t
    theta: dataclass
        dataclass containing the model parameters
        See form of theta at the end of the script

    Returns:
    rates: dataclass
        dataclass containing the reaction terms for the ASM3 model
    """
    # Pack theta into dictionary for easier access

    # scp is a dictionary containing the stoichiometric and compositional parameters of ASM3
    scp = stoich_comp_params_asm3(theta_dict)

    # Kinetic rate equations for ASM3, Table 6.1, p.111 (IWA, 2000)
    p1 = theta_dict["k_H"] * ( ( yReactor_dict["X_S"] / yReactor_dict["X_H"] ) / ( theta_dict["K_X"] + ( yReactor_dict["X_S"] / yReactor_dict["X_H"] ) ) )* yReactor_dict["X_H"]
    p2 = theta_dict["k_STO"] * ( yReactor_dict["S_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * ( yReactor_dict["S_S"] / ( theta_dict["K_S"] + yReactor_dict["S_S"] ) ) * yReactor_dict["X_H"]
    p3 = theta_dict["k_STO"] * theta_dict["eta_NOX"] * ( theta_dict["K_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * ( yReactor_dict["S_NOX"] / ( theta_dict["K_NOX"] + yReactor_dict["S_NOX"] ) ) * ( yReactor_dict["S_S"] / ( theta_dict["K_S"] + yReactor_dict["S_S"] ) ) * yReactor_dict["X_H"]
    p4 = theta_dict["mu_H"] * ( yReactor_dict["S_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * ( yReactor_dict["S_NH4"] / ( theta_dict["K_NH4"] + yReactor_dict["S_NH4"] ) ) * ( yReactor_dict["S_ALK"] / ( theta_dict["K_ALK"] + yReactor_dict["S_ALK"] ) ) * ( ( yReactor_dict["X_STO"] / yReactor_dict["X_H"] ) / ( theta_dict["K_STO"] + ( yReactor_dict["X_STO"] / yReactor_dict["X_H"] ) ) ) * yReactor_dict["X_H"]
    p5 = theta_dict["mu_H"] * theta_dict["eta_NOX"] * ( theta_dict["K_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * ( yReactor_dict["S_NOX"] / ( theta_dict["K_NOX"] + yReactor_dict["S_NOX"] ) ) * ( yReactor_dict["S_NH4"] / ( theta_dict["K_NH4"] + yReactor_dict["S_NH4"] ) ) * ( yReactor_dict["S_ALK"] / ( theta_dict["K_ALK"] + yReactor_dict["S_ALK"] ) ) * ( ( yReactor_dict["X_STO"] / yReactor_dict["X_H"] ) / ( theta_dict["K_STO"] + ( yReactor_dict["X_STO"] / yReactor_dict["X_H"] ) ) ) * yReactor_dict["X_H"]
    p6 = theta_dict["b_H_O2"] * ( yReactor_dict["S_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * yReactor_dict["X_H"]
    p7 = theta_dict["b_H_NOX"] * ( theta_dict["K_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * ( yReactor_dict["S_NOX"] / ( theta_dict["K_NOX"] + yReactor_dict["S_NOX"] ) ) * yReactor_dict["X_H"]
    p8 = theta_dict["b_STO_O2"] * ( yReactor_dict["S_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * yReactor_dict["X_STO"]
    p9 = theta_dict["b_STO_NOX"] * ( theta_dict["K_O2"] / ( theta_dict["K_O2"] + yReactor_dict["S_O2"] ) ) * ( yReactor_dict["S_NOX"] / ( theta_dict["K_NOX"] + yReactor_dict["S_NOX"] ) ) * yReactor_dict["X_STO"]
    p10 = theta_dict["mu_A"] * ( yReactor_dict["S_O2"] / ( theta_dict["K_A_O2"] + yReactor_dict["S_O2"] ) ) * ( yReactor_dict["S_NH4"] / ( theta_dict["K_A_NH4"] + yReactor_dict["S_NH4"] ) ) * ( yReactor_dict["S_ALK"] / ( theta_dict["K_A_ALK"] + yReactor_dict["S_ALK"] ) ) * yReactor_dict["X_A"]
    p11 = theta_dict["b_A_O2"] * ( yReactor_dict["S_O2"] / ( theta_dict["K_A_O2"] + yReactor_dict["S_O2"] ) ) * yReactor_dict["X_A"]
    p12 = theta_dict["b_A_NOX"] * ( yReactor_dict["S_NOX"] / ( theta_dict["K_NOX"] + yReactor_dict["S_NOX"] ) ) * yReactor_dict["X_A"]

    # Reaction rates for each Compound -> Sum of column on stoichiometric matrix. AKA, sum of each process related to each compound
       # Soluble Compounds
    S_O2 = p2*scp["x2"] + p4*scp["x4"] + p6*scp["x6"] + p8*scp["x8"] + p10*scp["x10"] + p11*scp["x11"]
    S_I = p1*theta_dict["f_S_I"]
    S_S = p1*scp["x1"] + p2*(-1) + p3*(-1)
    S_NH4 = p1*scp["y1"] + p2*scp["y2"] + p3*scp["y3"] + p4*scp["y4"] + p5*scp["y5"] + p6*scp["y6"] + p7*scp["y7"] + p10*scp["y10"] + p11*scp["y11"] + p12*scp["y12"]
    S_N2 = p3*(-scp["x3"]) + p5*(-scp["x5"]) + p7*(-scp["x7"]) + p9*(-scp["x9"]) + p12*(-scp["x12"])
    S_NOX = p3*scp["x3"] + p5*scp["x5"] + p7*scp["x7"] + p9*scp["x9"] + p10*(1/theta_dict["Y_A"]) + p12*scp["x12"]
    S_ALK = p1*scp["z1"] + p2*scp["z2"] + p3*scp["z3"] + p4*scp["z4"] + p5*scp["z5"] + p6*scp["z6"] + p7*scp["z7"] + p9*scp["z9"] + p10*scp["z10"] + p11*scp["z11"] + p12*scp["z12"]
        # Particulate Compounds
    X_I = p6*theta_dict["f_X_I"] + p7*theta_dict["f_X_I"] + p11*theta_dict["f_X_I"] + p12*theta_dict["f_X_I"]
    X_S = p1*(-1)
    X_H = p4*1 + p5*1 + p6*(-1) + p7*(-1)
    X_STO = p2*theta_dict["Y_STO_O2"] + p3*theta_dict["Y_STO_NOX"] + p4*(-1/theta_dict["Y_H_O2"]) + p5*(-1/theta_dict["Y_H_NOX"]) + p8*(-1) + p9*(-1)
    X_A = p10*1 + p11*(-1) + p12*(-1)
    X_SS = p1*(-theta_dict["i_SS_X_S"]) + p2*scp["t2"] + p3*scp["t3"] + p4*scp["t4"] + p5*scp["t5"] + p6*scp["t6"] + p7*scp["t7"] + p8*scp["t8"] + p9*scp["t9"] + p10*scp["t10"] + p11*scp["t11"] + p12*scp["t12"]
    
    # Dictionary containing the reaction rates for each reaction in the simulation
    rates = {
        "S_O2": S_O2, "S_I": S_I, "S_S": S_S, "S_NH4": S_NH4, "S_N2": S_N2, "S_NOX": S_NOX, "S_ALK": S_ALK,
        "X_I": X_I, "X_S": X_S, "X_H": X_H, "X_STO": X_STO, "X_A": X_A, "X_SS": X_SS
        }

    # Return the dataclass of reaction rates
    return rates
# End of asm3_reaction_rates function

def stoich_comp_params_asm3(theta_dict):
    """
    Function to define the stoichiometric and compositional parameters of ASM3.
    Args:
    theta: dataclass
        Dataclass containing the model parameters
        See form of theta at the end of the script
    
    Returns:
    scp: list
        List containing the stoichiometric and compositional parameters of ASM3
    """
    # Stoichiometric Parameters (Compounds 1 through 12), using conservation Equation 5.1, p.109 (IWA, 2000)
    x1 = 1 - theta_dict["f_S_I"]
    y1 = theta_dict["i_N_X_S"] - theta_dict["f_S_I"]*theta_dict["i_N_S_I"] - theta_dict["i_N_S_S"]*(x1)
    z1 = (1/14)*y1

    x2 = theta_dict["Y_STO_O2"] - 1
    y2 = theta_dict["i_N_S_S"]
    z2 = (1/14)*y2

    x3 = (theta_dict["Y_STO_NOX"] - 1) / 2.86
    y3 = theta_dict["i_N_S_S"]
    z3 = (1 / 14) * (y3 - x3)

    x4 = (theta_dict["Y_H_O2"] - 1) / theta_dict["Y_H_O2"]
    y4 = -theta_dict["i_N_BM"]
    z4 = (1 / 14) * y4

    x5 = (theta_dict["Y_H_NOX"] - 1) / (2.86 * theta_dict["Y_H_NOX"])
    y5 = -theta_dict["i_N_BM"]
    z5 = (1 / 14) * (y5 - x5)

    x6 = theta_dict["f_X_I"] - 1
    y6 = theta_dict["i_N_BM"] - theta_dict["f_X_I"] * theta_dict["i_N_X_I"]
    z6 = (1 / 14) * y6

    x7 = (theta_dict["f_X_I"] - 1) / 2.86
    y7 = theta_dict["i_N_BM"] - theta_dict["f_X_I"] * theta_dict["i_N_X_I"]
    z7 = (1 / 14) * (y7 - x7)

    x8 = -1

    x9 = -1 / 2.86
    z9 = -(1 / 14) * x9

    x10 = (theta_dict["Y_A"] - 4.57) / theta_dict["Y_A"]
    y10 = -(theta_dict["i_N_BM"] + (1 / theta_dict["Y_A"]))
    z10 = (1 / 14) * (y10 - (1 / theta_dict["Y_A"]))

    x11 = theta_dict["f_X_I"] - 1
    y11 = theta_dict["i_N_BM"] - theta_dict["f_X_I"] * theta_dict["i_N_X_I"]
    z11 = (1 / 14) * y11

    x12 = (theta_dict["f_X_I"] - 1) / 2.86
    y12 = theta_dict["i_N_BM"] - theta_dict["f_X_I"] * theta_dict["i_N_X_I"]
    z12 = (1 / 14) * (y12 - x12)

    # Stoichiometric Parameters (Compound 13, X_SS), using composition Equation 5.2, p.109 (IWA, 2000)
    t2 = theta_dict["Y_STO_O2"] * 0.6
    t3 = theta_dict["Y_STO_NOX"] * 0.6
    t4 = theta_dict["i_SS_BM"] - (0.6 / theta_dict["Y_H_O2"])
    t5 = theta_dict["i_SS_BM"] - (0.6 / theta_dict["Y_H_NOX"])
    t6 = theta_dict["f_X_I"] * theta_dict["i_SS_X_I"] - theta_dict["i_SS_BM"]
    t7 = theta_dict["f_X_I"] * theta_dict["i_SS_X_I"] - theta_dict["i_SS_BM"]
    t8 = -0.6
    t9 = -0.6
    t10 = theta_dict["i_SS_BM"]
    t11 = theta_dict["f_X_I"] * theta_dict["i_SS_X_I"] - theta_dict["i_SS_BM"]
    t12 = theta_dict["f_X_I"] * theta_dict["i_SS_X_I"] - theta_dict["i_SS_BM"]

    # Dictionary scp containing the stoichiometric and compositional parameters
    scp = {
        "x1": x1, "y1": y1, "z1": z1,
        "x2": x2, "y2": y2, "z2": z2,
        "x3": x3, "y3": y3, "z3": z3,
        "x4": x4, "y4": y4, "z4": z4,
        "x5": x5, "y5": y5, "z5": z5,
        "x6": x6, "y6": y6, "z6": z6,
        "x7": x7, "y7": y7, "z7": z7,
        "x8": x8,
        "x9": x9, "z9": z9,
        "x10": x10, "y10": y10, "z10": z10,
        "x11": x11, "y11": y11, "z11": z11,
        "x12": x12, "y12": y12, "z12": z12,
        "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6, "t7": t7, "t8": t8, "t9": t9, "t10": t10, "t11": t11, "t12": t12
    }

    # Return the dataclass containing the stoichiometric and compositional parameters
    return scp
# End of stoich_comp_params_asm3 function


# ----------------------------------------------------------
## Wrapper function for ode_system function
# ----------------------------------------------------------

def ode_system_wrapper(t, y, theta, influentData, reactorVolumes, use_polyfit=False):
    """
    Wrapper function for the ode_system function to pass additional arguments.

    Args:
    t: float
        Time at which the simulation is run (days)
    y: Array
        Array containing the state variables for the simulation at time t
    theta: dict
        Array containing the model parameters
    influentData: Array
        Array containing the influent data for the plant reactor model.
        Each column of the array corresponds to the following data:
        [Time, Flowrate, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS].
        Each row of the array corresponds to the data at a specific time indicated by the first column.
    reactorVolumes: dict
        Dictionary containing the volume of each reactor in the plant

    Returns:
    dYdt: Array
        Array containing the derivatives of the state variables at time t
    """
    # Call the ode_system function with the additional arguments
    dYdt = ode_system(y, t, theta, influentData, reactorVolumes, use_polyfit)
    return dYdt

# ----------------------------------------------------------
## Notes
# ----------------------------------------------------------

"""
FORM OF THETA

theta has 36 parameters for the ASM3 model. 21 are Kinetic parameters, 15 are Stoichiometric parameters.
theta = [
    k_H,        # 0
    K_X,        # 1
    k_STO,      # 2
    eta_NOX,    # 3
    K_O2,       # 4
    K_NOX,      # 5
    K_S,        # 6
    K_STO,      # 7
    mu_H,       # 8
    K_NH4,      # 9
    K_ALK,      # 10
    b_H_O2,     # 11
    b_H_NOX,    # 12
    b_STO_O2,   # 13
    b_STO_NOX,  # 14
    mu_A,       # 15
    K_A_NH4,    # 16
    K_A_O2,     # 17
    K_A_ALK,    # 18
    b_A_O2,     # 19
    b_A_NOX,    # 20
    f_S_I,      # 21
    Y_STO_O2,   # 22
    Y_STO_NOX,  # 23
    Y_H_O2,     # 24
    Y_H_NOX,    # 25
    Y_A,        # 26
    f_X_I,      # 27
    i_N_S_I,    # 28
    i_N_S_S,    # 29
    i_N_X_I,    # 30
    i_N_X_S,    # 31
    i_N_BM,     # 32
    i_SS_X_I,   # 33
    i_SS_X_S,   # 34
    i_SS_BM     # 35
]

FORM OF SCP
scp = [
    x1,  # 0
    y1,  # 1
    z1,  # 2
    x2,  # 3
    y2,  # 4
    z2,  # 5
    x3,  # 6
    y3,  # 7
    z3,  # 8
    x4,  # 9
    y4,  # 10
    z4,  # 11
    x5,  # 12
    y5,  # 13
    z5,  # 14
    x6,  # 15
    y6,  # 16
    z6,  # 17
    x7,  # 18
    y7,  # 19
    z7,  # 20
    x8,  # 21
    x9,  # 22
    z9,  # 23
    x10, # 24
    y10, # 25
    z10, # 26
    x11, # 27
    y11, # 28
    z11, # 29
    x12, # 30
    y12, # 31
    z12, # 32
    t2,  # 33
    t3,  # 34
    t4,  # 35
    t5,  # 36
    t6,  # 37
    t7,  # 38
    t8,  # 39
    t9,  # 40
    t10, # 41
    t11, # 42
    t12  # 43
]
"""