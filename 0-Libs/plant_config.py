# ----------------------------------------------------------
## Purpose
# ----------------------------------------------------------
"""
Python script that contains the starting values for the state variables of the reactors in the plant
"""

# ----------------------------------------------------------
## Importing libraries
# ----------------------------------------------------------
import configparser

# ----------------------------------------------------------
## Reactor initial values
# ----------------------------------------------------------

def get_reactor_initial_values(top_dir):
    # TODO move to libs
    """
    Function to define the initial values for each reactor state variable for the simulation

    Args:
    top_dir: Path
        Path to the top directory of the project
    
    Returns:
    reactor_initial_values: Array
        Array containing the initial values for each reactor state variable
        = [S_I_reactor1, S_S_reactor1, S_NH4_reactor1, S_N2_reactor1, S_NOX_reactor1, S_ALK_reactor1, ...]
    """

    # Config
    config_dir = top_dir / "0-Config"
    # Config File
    config = configparser.ConfigParser()
    config.read(config_dir / "config.ini")

    # ----------------------------------------------------------
    ## Empty arrays to store the initial values for each reactor state variable
    # ----------------------------------------------------------
        # Dissolved Compounds
    S_O2 = []
    S_I = []
    S_S = []
    S_NH4 = []
    S_N2 = []
    S_NOX = []
    S_ALK = []
        # Particulate Compounds
    X_I = []
    X_S = []
    X_H = []
    X_STO = []
    X_A = []
    X_SS = []
    # ----------------------------------------------------------
    ## Initial values of the state variables -- Compound inputs, Typical Wastewater Composition, Table 8.3 (IWA, 2000), p.115 for default values
    # ----------------------------------------------------------

    # initial values for reactor 1 state variables
        # Dissolved Compounds
    r1_S_O2_i = float(config['REACTOR']['r1_S_O2_i'])           # (g O2 / m3), Dissolved oxygen
    r1_S_I_i = float(config['REACTOR']['r1_S_I_i'])           # (g COD / m3), Soluble inert organics
    r1_S_S_i = float(config['REACTOR']['r1_S_S_i'])           # (g COD / m3), Readily biodegradable substrates
    r1_S_NH4_i = float(config['REACTOR']['r1_S_NH4_i'])         # (g N / m3), Ammonium
    r1_S_N2_i = float(config['REACTOR']['r1_S_N2_i'])           # (g N / m3), Dinitrogen, released by denitrification
    r1_S_NOX_i = float(config['REACTOR']['r1_S_NOX_i'])          # (g N / m3), Nitrate plus nitrite
    r1_S_ALK_i = float(config['REACTOR']['r1_S_ALK_i'])        # (mole HCO3- / m3), Alkalinity, bicarbonate
        # Particulate Compounds
    r1_X_I_i = float(config['REACTOR']['r1_X_I_i'])           # (g COD / m3), Inert particulate organics
    r1_X_S_i = float(config['REACTOR']['r1_X_S_i'])          # (g COD / m3), Slowly biodegradable substrates
    r1_X_H_i = float(config['REACTOR']['r1_X_H_i'])           # (g COD / m3), Heterotrophic biomass
    r1_X_STO_i = float(config['REACTOR']['r1_X_STO_i'])          # (g COD / m3), Organics stored by heterotrophs
    r1_X_A_i = float(config['REACTOR']['r1_X_A_i'])          # (g COD / m3), Autotrophic nitrifying biomass (>0)
    r1_X_SS_i = float(config['REACTOR']['r1_X_SS_i'])         # (g SS / m3), Total suspended solids

    # Add reactor 1 initial values to the arrays
    S_O2.append(r1_S_O2_i)
    S_I.append(r1_S_I_i)
    S_S.append(r1_S_S_i)
    S_NH4.append(r1_S_NH4_i)
    S_N2.append(r1_S_N2_i)
    S_NOX.append(r1_S_NOX_i)
    S_ALK.append(r1_S_ALK_i)
    X_I.append(r1_X_I_i)
    X_S.append(r1_X_S_i)
    X_H.append(r1_X_H_i)
    X_STO.append(r1_X_STO_i)
    X_A.append(r1_X_A_i)
    X_SS.append(r1_X_SS_i)

    # More reactors here if needed

    # ----------------------------------------------------------
    ## Create array of initial values for each reactor state variable
    # ----------------------------------------------------------
    reactor_initial_values = []
    num_reactors = len(S_O2)

    for i in range(num_reactors):
        reactor_initial_values.extend([
            S_O2[i],
            S_I[i],
            S_S[i],
            S_NH4[i],
            S_N2[i],
            S_NOX[i],
            S_ALK[i],
            X_I[i],
            X_S[i],
            X_H[i],
            X_STO[i],
            X_A[i],
            X_SS[i]
        ])

    # Return the initial values for each reactor state variable
    return reactor_initial_values
# End of function reactor_initial_values()

# ----------------------------------------------------------

## Notes:

# Form of reactor_initial_values:
"""
reactor_initial_values = [
    S_O2_reactor1, S_I_reactor1, S_S_reactor1, S_NH4_reactor1, S_N2_reactor1, S_NOX_reactor1, S_ALK_reactor1,
    X_I_reactor1, X_S_reactor1, X_H_reactor1, X_STO_reactor1, X_A_reactor1, X_SS_reactor1,
    S_O2_reactor2, S_I_reactor2, S_S_reactor2, S_NH4_reactor2, S_N2_reactor2, S_NOX_reactor2, S_ALK_reactor2,
    X_I_reactor2, X_S_reactor2, X_H_reactor2, X_STO_reactor2, X_A_reactor2, X_SS_reactor2,
    S_O2_reactor3, S_I_reactor3, S_S_reactor3, S_NH4_reactor3, S_N2_reactor3, S_NOX_reactor3, S_ALK_reactor3,
    X_I_reactor3, X_S_reactor3, X_H_reactor3, X_STO_reactor3, X_A_reactor3, X_SS_reactor3
]
therefore, the length of the array is 13 * num_reactors
form:
reactor_initial_values = [
    Comp1_reactor1, (13 values)
    Comp2_reactor1, 
    Comp3_reactor1, 
    ...,
    CompN_reactor1,
]

"""