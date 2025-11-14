# ----------------------------------------------------------
## Purpose
# ----------------------------------------------------------
"""
Python script that contains the ode system for the simulation.
"""

# ----------------------------------------------------------
## Importing libraries
# ----------------------------------------------------------

# Import standard libraries
import numpy as np
import sympy as sp
from dataclasses import dataclass
# Import custom python files

# ----------------------------------------------------------
## Function to define the simultaneous ODE system - Mass Balance
# ----------------------------------------------------------

# Simplified version of the ASM3 model - only using arrays for theta and reactorVolumes
def ode_system_sunode(t, y, theta, influentData, reactorVolumes, use_polynomial=False):
    """
    Function to define the ODE system for the ASM3 model.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.

    Args:
    y: dataclass
        dataclass containing the state variables (concentrations of compounds) at time t (g Compound / m3)
        Form:
        [r1_S_O2, r1_S_I, r1_S_S, r1_S_NH4, r1_S_N2, r1_S_NOX, r1_S_ALK, r1_X_I, r1_X_S, r1_X_H, r1_X_STO, r1_X_A, r1_X_SS,
        r2_S_O2, r2_S_I, r2_S_S...] - for each reactor

    t: float
        Time (day) at time t 
    theta: dataclass
        dataclass containing the model parameters
    influentData: Array
        Array containing the influent data for the plant reactor model
            Each column of the array corresponds to the following data:
            [Time, Flowrate, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
            Each row of the array corresponds to the data at a specific time indicated by the first column
    reactorVolumes: dict
        dictionary containing the volumes of each reactor in the plant
        Form: [V1, V2, V3, ...]

    Returns:
    dYdt: dict
        dictionary containing the derivatives of the state variables at time t
    """

    NumReactors = len(reactorVolumes)

    # ------------
    ### 1) Reaction rates for each state variable -- from asm3_reaction_rates
    # ------------
    @dataclass
    class yReactor:
        S_O2: float
        S_I: float
        S_S: float
        S_NH4: float
        S_N2: float
        S_NOX: float
        S_ALK: float
        X_I: float
        X_S: float
        X_H: float
        X_STO: float
        X_A: float
        X_SS: float

    ## Reactor 1
    # State variables for Reactor 1
    r1_y = yReactor(
        y.r1_S_O2,
        y.r1_S_I,
        y.r1_S_S,
        y.r1_S_NH4,
        y.r1_S_N2,
        y.r1_S_NOX,
        y.r1_S_ALK,
        y.r1_X_I,
        y.r1_X_S,
        y.r1_X_H,
        y.r1_X_STO,
        y.r1_X_A,
        y.r1_X_SS
    )
    # r1_rates = Dataclass containing the reaction rates for each compound in Reactor 1 
        # Form: [S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
    r1_rates = asm3_reaction_rates(r1_y, theta)

    ## Reactor 2
    # TODO: Add other reactors

    # ------------
    #### 2) Mass Balance
    # TODO: Do this in separate function
    # ------
    ## SETUP
    # ------
    @dataclass
    class FlowData:
        Qin: float
        S_O2: float
        S_I: float
        S_S: float
        S_NH4: float
        S_N2: float
        S_NOX: float
        S_ALK: float
        X_I: float
        X_S: float
        X_H: float
        X_STO: float
        X_A: float
        X_SS: float

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

    r1_Outlets = FlowData(
        r1_Inlets.Qin,
        r1_y.S_O2,
        r1_y.S_I,
        r1_y.S_S,
        r1_y.S_NH4,
        r1_y.S_N2,
        r1_y.S_NOX,
        r1_y.S_ALK,
        r1_y.X_I,
        r1_y.X_S,
        r1_y.X_H,
        r1_y.X_STO,
        r1_y.X_A,
        r1_y.X_SS
    )

    ## Component Balance
    #  r1_dC_S_O2_dt = ( ( (r1_Inlets.Qin * r1_Inlets.S_O2) - (r1_Outlets.Qin * r1_Outlets.S_O2) ) / reactorVolumes['V1'] ) + r1_rates.S_O2
    r1_dC_S_O2_dt = 0 # Constant for study
    r1_dC_S_I_dt = ( ( (r1_Inlets.Qin * r1_Inlets.S_I) - (r1_Outlets.Qin * r1_Outlets.S_I) ) / reactorVolumes['V1'] ) + r1_rates.S_I
    r1_dC_S_S_dt = ( ( (r1_Inlets.Qin * r1_Inlets.S_S) - (r1_Outlets.Qin * r1_Outlets.S_S) ) / reactorVolumes['V1'] ) + r1_rates.S_S
    r1_dC_S_NH4_dt = ( ( (r1_Inlets.Qin * r1_Inlets.S_NH4) - (r1_Outlets.Qin * r1_Outlets.S_NH4) ) / reactorVolumes['V1'] ) + r1_rates.S_NH4
    r1_dC_S_N2_dt = ( ( (r1_Inlets.Qin * r1_Inlets.S_N2) - (r1_Outlets.Qin * r1_Outlets.S_N2) ) / reactorVolumes['V1'] ) + r1_rates.S_N2
    r1_dC_S_NOX_dt = ( ( (r1_Inlets.Qin * r1_Inlets.S_NOX) - (r1_Outlets.Qin * r1_Outlets.S_NOX) ) / reactorVolumes['V1'] ) + r1_rates.S_NOX
    r1_dC_S_ALK_dt = ( ( (r1_Inlets.Qin * r1_Inlets.S_ALK) - (r1_Outlets.Qin * r1_Outlets.S_ALK) ) / reactorVolumes['V1'] ) + r1_rates.S_ALK
    r1_dC_X_I_dt = ( ( (r1_Inlets.Qin * r1_Inlets.X_I) - (r1_Outlets.Qin * r1_Outlets.X_I) ) / reactorVolumes['V1'] ) + r1_rates.X_I
    r1_dC_X_S_dt = ( ( (r1_Inlets.Qin * r1_Inlets.X_S) - (r1_Outlets.Qin * r1_Outlets.X_S) ) / reactorVolumes['V1'] ) + r1_rates.X_S
    r1_dC_X_H_dt = ( ( (r1_Inlets.Qin * r1_Inlets.X_H) - (r1_Outlets.Qin * r1_Outlets.X_H) ) / reactorVolumes['V1'] ) + r1_rates.X_H
    r1_dC_X_STO_dt = ( ( (r1_Inlets.Qin * r1_Inlets.X_STO) - (r1_Outlets.Qin * r1_Outlets.X_STO) ) / reactorVolumes['V1'] ) + r1_rates.X_STO
    r1_dC_X_A_dt = ( ( (r1_Inlets.Qin * r1_Inlets.X_A) - (r1_Outlets.Qin * r1_Outlets.X_A) ) / reactorVolumes['V1'] ) + r1_rates.X_A
    r1_dC_X_SS_dt = ( ( (r1_Inlets.Qin * r1_Inlets.X_SS) - (r1_Outlets.Qin * r1_Outlets.X_SS) ) / reactorVolumes['V1'] ) + r1_rates.X_SS


    # OTR
    # r1_OTR = get_reactor_OTR(y.r1_S_O2)     # Oxygen Transfer Rate for Reactor 1
    # r1_dC_S_O2_dt += r1_OTR                 # Add OTR term to S_O2

    # ------
    ## Reactor 2
    # ------
    # TODO: Add other reactors
    # TODO: Add recycle, bypass, oxygen pumping

    # ------
    ## State Variables Return
    # ------
    return {
        # Reactor 1
        'r1_S_O2': r1_dC_S_O2_dt,
        'r1_S_I': r1_dC_S_I_dt,
        'r1_S_S': r1_dC_S_S_dt,
        'r1_S_NH4': r1_dC_S_NH4_dt,
        'r1_S_N2': r1_dC_S_N2_dt,
        'r1_S_NOX': r1_dC_S_NOX_dt,
        'r1_S_ALK': r1_dC_S_ALK_dt,
        'r1_X_I': r1_dC_X_I_dt,
        'r1_X_S': r1_dC_X_S_dt,
        'r1_X_H': r1_dC_X_H_dt,
        'r1_X_STO': r1_dC_X_STO_dt,
        'r1_X_A': r1_dC_X_A_dt,
        'r1_X_SS': r1_dC_X_SS_dt
        # TODO: Add other reactors
    }
# End of ode_system function


def get_influent_at_time(influentData, t):
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
        dataclass containing the influent data at time t
        Form: [Qin, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS]
    """
    # Throw an error if the influent data is not provided or is empty
    if influentData is None or len(influentData) == 0:
        raise ValueError("Influent data is empty or not provided")
 
    time_data = influentData[:, 0]  # Get the time data, first column of the array plant_influent_data

    ## Perform linear interpolation for each variable using np.interp
    Qin = np.interp(t, time_data, influentData[:, 1])
    S_O2 = np.interp(t, time_data, influentData[:, 2])
    S_I = np.interp(t, time_data, influentData[:, 3])
    S_S = np.interp(t, time_data, influentData[:, 4])
    S_NH4 = np.interp(t, time_data, influentData[:, 5])
    S_N2 = np.interp(t, time_data, influentData[:, 6])
    S_NOX = np.interp(t, time_data, influentData[:, 7])
    S_ALK = np.interp(t, time_data, influentData[:, 8])
    X_I = np.interp(t, time_data, influentData[:, 9])
    X_S = np.interp(t, time_data, influentData[:, 10])
    X_H = np.interp(t, time_data, influentData[:, 11])
    X_STO = np.interp(t, time_data, influentData[:, 12])
    X_A = np.interp(t, time_data, influentData[:, 13])
    X_SS = np.interp(t, time_data, influentData[:, 14])

    # dataclass containing the influent data at time t
    @dataclass
    class FlowData:
        Qin: float
        S_O2: float
        S_I: float
        S_S: float
        S_NH4: float
        S_N2: float
        S_NOX: float
        S_ALK: float
        X_I: float
        X_S: float
        X_H: float
        X_STO: float
        X_A: float
        X_SS: float

    Influent_data = FlowData(Qin, S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK, X_I, X_S, X_H, X_STO, X_A, X_SS)
    return Influent_data

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

    # dataclass containing the influent data at time t
    @dataclass
    class FlowData:
        Qin: float
        S_O2: float
        S_I: float
        S_S: float
        S_NH4: float
        S_N2: float
        S_NOX: float
        S_ALK: float
        X_I: float
        X_S: float
        X_H: float
        X_STO: float
        X_A: float
        X_SS: float

    Influent_data = FlowData(Qin_poly, S_O2_poly, S_I_poly, S_S_poly, S_NH4_poly, S_N2_poly, S_NOX_poly, S_ALK_poly, X_I_poly, X_S_poly, X_H_poly, X_STO_poly, X_A_poly, X_SS_poly)
    return Influent_data

# ----------------------------------------------------------
## Function to get the reactor OTR
# ----------------------------------------------------------

def get_reactor_OTR(C_S_O2):
    """
    Function to get the oxygen transfer rate (OTR) for the reactor.

    Args:
    C_S_O2: float
        Concentration of dissolved oxygen in the reactor (g O2 / m3) at time t

    Returns:
    OTR: float
        Oxygen transfer rate for the reactor
    """

    # For oxygen (S_O2), a term for oxygen saturation should be added
    # From Benchmark Simulation Model No. 2 (BSM2) - IWA Task Group, the Equation for oxygen saturation is:
        # OTR = K_La * (C_O2_sat - C_O2)
        # Where:
            # K_La = Oxygen mass transfer coefficient, (1/day) - Function of temperature
            # C_O2_sat = Oxygen saturation concentration, (g O2 / m3) - Function of temperature
            # C_O2 = Oxygen concentration, (g O2 / m3)
        # From BSM2:
        # K_La - Oxygen mass transfer coefficient, (1/day):
            # ASCE (1993) presents the generally accepted dependency of the oxygen transfer coefficient KLa on temperature:
            # K_La = ( 1.024 ^(T - 15) ) * K_La_15degC with T in degrees Celsius, (1/day)
            # K_La_15degC = constant value at 15 degrees Celsius, (1/day)
        # C_O2_sat - Oxygen saturation concentration (g O2 / m3):
            # C_O2_sat_Tas = 0.9997743214 * (8/10.5) * 6791.5 * K_Tk
            # Where:
                # K_Tk = 56.12 * exp(A - (B/T*) + C * ln(T*)) valid in range 273.15 to 348.15 K (0 to 75 degrees Celsius)
                # T* = (Tk/100) in K
                # Tk = Tas(Celsius) + 273.15
                # A = -66.7354
                # B = 87.4755
                # C = 24.4526

    # Oxygen mass transfer coefficient (1/day)
        # TODO: Temperature Change here
    Temp = 20                                       # Temperature in degrees Celsius
    K_La_15degC_h = 11.4                            # Oxygen transfer coefficient at 15 degree Celsius and 1 hour residence time (1/h)
    K_La_15degC = K_La_15degC_h / 24                # 1/day
    K_La = (1.024 ** (Temp - 15)) * K_La_15degC     # 1/day

    K_La = 10 # TODO: For testing purposes

    # Oxygen saturation concentration (g O2 / m3)
    T_kelvin = Temp + 273.15 # At 20 degrees Celsius = 293.15 K - within range 273.15 to 348.15 K
    T_star = T_kelvin / 100
    A_const = -66.7354
    B_const = 87.4755
    C_const = 24.4526
    K_Tk = 56.12 * np.exp(A_const - (B_const / T_star) + C_const * np.log(T_star))
    C_O2_sat = 0.9997743214 * (8/10.5) * 6791.5 * K_Tk 

    C_O2_sat = 10 # TODO: For testing purposes

    # Oxygen Transfer Rate (OTR) for reactor
    OTR = K_La * (C_O2_sat - C_S_O2) # (g O2 / m3 / day)
    return OTR
# End of get_reactor_OTR function

# ----------------------------------------------------------
## ASM3 Simulation - ASM3 simulates reaction rates for Mass Balance
# ----------------------------------------------------------

def asm3_reaction_rates(yReactor, theta):
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
    # scp is a dataclass containing the stoichiometric and compositional parameters of ASM3
    scp = stoich_comp_params_asm3(theta)

    # Kinetic rate equations for ASM3, Table 6.1, p.111 (IWA, 2000)
    p1 = theta.k_H * ( ( yReactor.X_S / yReactor.X_H ) / ( theta.K_X + ( yReactor.X_S / yReactor.X_H ) ) )* yReactor.X_H
    p2 = theta.k_STO * ( yReactor.S_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * ( yReactor.S_S / ( theta.K_S + yReactor.S_S ) ) * yReactor.X_H
    p3 = theta.k_STO * theta.eta_NOX * ( theta.K_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * ( yReactor.S_NOX / ( theta.K_NOX + yReactor.S_NOX ) ) * ( yReactor.S_S / ( theta.K_S + yReactor.S_S ) ) * yReactor.X_H
    p4 = theta.mu_H * ( yReactor.S_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * ( yReactor.S_NH4 / ( theta.K_NH4 + yReactor.S_NH4 ) ) * ( yReactor.S_ALK / ( theta.K_ALK + yReactor.S_ALK ) ) * ( ( yReactor.X_STO / yReactor.X_H ) / ( theta.K_STO + ( yReactor.X_STO / yReactor.X_H ) ) ) * yReactor.X_H
    p5 = theta.mu_H * theta.eta_NOX * ( theta.K_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * ( yReactor.S_NOX / ( theta.K_NOX + yReactor.S_NOX ) ) * ( yReactor.S_NH4 / ( theta.K_NH4 + yReactor.S_NH4 ) ) * ( yReactor.S_ALK / ( theta.K_ALK + yReactor.S_ALK ) ) * ( ( yReactor.X_STO / yReactor.X_H ) / ( theta.K_STO + ( yReactor.X_STO / yReactor.X_H ) ) ) * yReactor.X_H
    p6 = theta.b_H_O2 * ( yReactor.S_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * yReactor.X_H
    p7 = theta.b_H_NOX * ( theta.K_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * ( yReactor.S_NOX / ( theta.K_NOX + yReactor.S_NOX ) ) * yReactor.X_H
    p8 = theta.b_STO_O2 * ( yReactor.S_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * yReactor.X_STO
    p9 = theta.b_STO_NOX * ( theta.K_O2 / ( theta.K_O2 + yReactor.S_O2 ) ) * ( yReactor.S_NOX / ( theta.K_NOX + yReactor.S_NOX ) ) * yReactor.X_STO
    p10 = theta.mu_A * ( yReactor.S_O2 / ( theta.K_A_O2 + yReactor.S_O2 ) ) * ( yReactor.S_NH4 / ( theta.K_A_NH4 + yReactor.S_NH4 ) ) * ( yReactor.S_ALK / ( theta.K_A_ALK + yReactor.S_ALK ) ) * yReactor.X_A
    p11 = theta.b_A_O2 * ( yReactor.S_O2 / ( theta.K_A_O2 + yReactor.S_O2 ) ) * yReactor.X_A
    p12 = theta.b_A_NOX * ( yReactor.S_NOX / ( theta.K_NOX + yReactor.S_NOX ) ) * yReactor.X_A

    # Reaction rates for each Compound -> Sum of column on stoichiometric matrix. AKA, sum of each process related to each compound
       # Soluble Compounds
    S_O2 = p2*scp.x2 + p4*scp.x4 + p6*scp.x6 + p8*scp.x8 + p10*scp.x10 + p11*scp.x11
    S_I = p1*theta.f_S_I
    S_S = p1*scp.x1 + p2*(-1) + p3*(-1)
    S_NH4 = p1*scp.y1 + p2*scp.y2 + p3*scp.y3 + p4*scp.y4 + p5*scp.y5 + p6*scp.y6 + p7*scp.y7 + p10*scp.y10 + p11*scp.y11 + p12*scp.y12
    S_N2 = p3*(-scp.x3) + p5*(-scp.x5) + p7*(-scp.x7) + p9*(-scp.x9) + p12*(-scp.x12)
    S_NOX = p3*scp.x3 + p5*scp.x5 + p7*scp.x7 + p9*scp.x9 + p10*(1/theta.Y_A) + p12*scp.x12
    S_ALK = p1*scp.z1 + p2*scp.z2 + p3*scp.z3 + p4*scp.z4 + p5*scp.z5 + p6*scp.z6 + p7*scp.z7 + p9*scp.z9 + p10*scp.z10 + p11*scp.z11 + p12*scp.z12
        # Particulate Compounds
    X_I = p6*theta.f_X_I + p7*theta.f_X_I + p11*theta.f_X_I + p12*theta.f_X_I
    X_S = p1*(-1)
    X_H = p4*1 + p5*1 + p6*(-1) + p7*(-1)
    X_STO = p2*theta.Y_STO_O2 + p3*theta.Y_STO_NOX + p4*(-1/theta.Y_H_O2) + p5*(-1/theta.Y_H_NOX) + p8*(-1) + p9*(-1)
    X_A = p10*1 + p11*(-1) + p12*(-1)
    X_SS = p1*(-theta.i_SS_X_S) + p2*scp.t2 + p3*scp.t3 + p4*scp.t4 + p5*scp.t5 + p6*scp.t6 + p7*scp.t7 + p8*scp.t8 + p9*scp.t9 + p10*scp.t10 + p11*scp.t11 + p12*scp.t12
    
    # Dataclass containing the reaction rates for each reaction in the simulation
    @dataclass
    class ReactionRates:
        S_O2: float
        S_I: float
        S_S: float
        S_NH4: float
        S_N2: float
        S_NOX: float
        S_ALK: float
        X_I: float
        X_S: float
        X_H: float
        X_STO: float
        X_A: float
        X_SS: float

    rates = ReactionRates(
        S_O2, S_I, S_S, S_NH4, S_N2, S_NOX, S_ALK,
        X_I, X_S, X_H, X_STO, X_A, X_SS
    )

    # Return the dataclass of reaction rates
    return rates
# End of asm3_reaction_rates function


# ----------------------------------------------------------
def stoich_comp_params_asm3(theta):
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
    x1 = 1 - theta.f_S_I
    y1 = theta.i_N_X_S - theta.f_S_I*theta.i_N_S_I - theta.i_N_S_S*(x1)
    z1 = (1/14)*y1

    x2 = theta.Y_STO_O2 - 1
    y2 = theta.i_N_S_S
    z2 = (1/14)*y2

    x3 = (theta.Y_STO_NOX - 1)/2.86
    y3 = theta.i_N_S_S
    z3 = (1/14)*(y3 - x3)

    x4 = (theta.Y_H_O2 - 1)/theta.Y_H_O2
    y4 = -theta.i_N_BM
    z4 = (1/14)*y4

    x5 = (theta.Y_H_NOX - 1)/(2.86*theta.Y_H_NOX)
    y5 = -theta.i_N_BM
    z5 = (1/14)*(y5 - x5)

    x6 = theta.f_X_I - 1
    y6 = theta.i_N_BM - theta.f_X_I*theta.i_N_X_I
    z6 = (1/14)*y6

    x7 = (theta.f_X_I - 1)/2.86
    y7 = theta.i_N_BM - theta.f_X_I*theta.i_N_X_I
    z7 = (1/14)*(y7 - x7)

    x8 = -1

    x9 = -1/2.86
    z9 = -(1/14)*x9

    x10 = (theta.Y_A - 4.57)/theta.Y_A
    y10 = -(theta.i_N_BM+(1/theta.Y_A))
    z10 = (1/14)*(y10 - (1/theta.Y_A))

    x11 = theta.f_X_I - 1
    y11 = theta.i_N_BM - theta.f_X_I*theta.i_N_X_I
    z11 = (1/14)*y11

    x12 = (theta.f_X_I - 1)/2.86
    y12 = theta.i_N_BM - theta.f_X_I*theta.i_N_X_I
    z12 = (1/14)*(y12 - x12)

    # Stoichiometric Parameters (Compound 13, X_SS), using composition Equation 5.2, p.109 (IWA, 2000)
    t2 = theta.Y_STO_O2*0.6
    t3 = theta.Y_STO_NOX*0.6
    t4 = theta.i_SS_BM - (0.6/theta.Y_H_O2)
    t5 = theta.i_SS_BM - (0.6/theta.Y_H_NOX)
    t6 = theta.f_X_I*theta.i_SS_X_I - theta.i_SS_BM
    t7 = theta.f_X_I*theta.i_SS_X_I - theta.i_SS_BM
    t8 = -0.6
    t9 = -0.6
    t10 = theta.i_SS_BM
    t11 = theta.f_X_I*theta.i_SS_X_I - theta.i_SS_BM
    t12 = theta.f_X_I*theta.i_SS_X_I - theta.i_SS_BM

    # dataclass containing the stoichiometric and compositional parameters
    @dataclass
    class StoichCompParams:
        x1: float
        y1: float
        z1: float
        x2: float
        y2: float
        z2: float
        x3: float
        y3: float
        z3: float
        x4: float
        y4: float
        z4: float
        x5: float
        y5: float
        z5: float
        x6: float
        y6: float
        z6: float
        x7: float
        y7: float
        z7: float
        x8: float
        x9: float
        z9: float
        x10: float
        y10: float
        z10: float
        x11: float
        y11: float
        z11: float
        x12: float
        y12: float
        z12: float
        t2: float
        t3: float
        t4: float
        t5: float
        t6: float
        t7: float
        t8: float
        t9: float
        t10: float
        t11: float
        t12: float

    # Dataclass containing the stoichiometric and compositional parameters
    scp = StoichCompParams(
        x1, y1, z1,
        x2, y2, z2,
        x3, y3, z3,
        x4, y4, z4,
        x5, y5, z5,
        x6, y6, z6,
        x7, y7, z7,
        x8, x9, z9,
        x10, y10, z10,
        x11, y11, z11,
        x12, y12, z12,
        t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12
    )

    # Return the dataclass containing the stoichiometric and compositional parameters
    return scp
# End of stoich_comp_params_asm3 function

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