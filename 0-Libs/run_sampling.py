# ----------------------------------------------------------
## Purpose
# ----------------------------------------------------------
"""
Run the model samplers in PyMC for the ASM3 model

Different model samplers are used from the PyMC library
"""

# ----------------------------------------------------------
## Importing libraries
# ----------------------------------------------------------

import pymc


# ----------------------------------------------------------
## Non Gradient Sampling
# ----------------------------------------------------------

# Slice Sampling
def run_model_Slice(vars_list, model, tune, draws, chains=4, cores=4):
    with model:
        trace = pymc.sample(step=[pymc.Slice(vars_list)], tune=tune, draws=draws, chains=chains, cores=cores)
    return trace

# Metropolis Sampling
def run_model_Metropolis(vars_list, model, tune, draws, chains=4, cores=4):
    with model:
        trace = pymc.sample(step=[pymc.Metropolis(vars_list)], tune=tune, draws=draws, chains=chains, cores=cores)
    return trace

# DEMetropolis Sampling
def run_model_DEMetroplis(vars_list, model, tune, draws, chains=4, cores=4):
    with model:
        trace = pymc.sample(step=[pymc.DEMetropolis(vars_list)], tune=tune, draws=draws, chains=chains, cores=cores)
    return trace

# DEMetropolisZ Sampling
def run_model_DEMetroplisZ(vars_list, model, tune, draws, chains=4, cores=4):
    with model:
        trace = pymc.sample(step=[pymc.DEMetropolisZ(vars_list)], tune=tune, draws=draws, chains=chains, cores=cores)
    return trace

# SMC Sampling
def run_model_SMC(vars_list, model, tune, draws, chains=4, cores=4):
    with model:
        trace = pymc.sample(step=[pymc.SMC(vars_list)], tune=tune, draws=draws, chains=chains, cores=cores)
    return trace

# ----------------------------------------------------------
## Gradient Sampling
# ----------------------------------------------------------

# HamiltonianMC Sampling
# def run_model_HamiltonianMC(vars_list, model, tune, draws, chains=4, cores=4, progressbar=False):
#     with model:
#         trace = pymc.sample(step=[pymc.HamiltonianMC(vars_list)], tune=tune, draws=draws, chains=chains, cores=cores, progressbar=progressbar)
#     return trace
def run_model_HamiltonianMC(model, tune, draws, chains=4, cores=4, progressbar=False):
    with model:
        step = pymc.HamiltonianMC()
        trace = pymc.sample(step=step, tune=tune, draws=draws, chains=chains, cores=cores, progressbar=progressbar)
    return trace

# NUTS Sampling
# def run_model_NUTS(vars_list, model, tune, draws, chains=4, cores=4, progressbar=False):
#     with model:
#         trace = pymc.sample(step=[pymc.NUTS(vars_list)], tune=tune, draws=draws, chains=chains, cores=cores, progressbar=progressbar)
#     return trace
def run_model_NUTS(model, tune, draws, chains=4, cores=4, progressbar=False):
    with model:
        step = pymc.NUTS()
        trace = pymc.sample(step=step, tune=tune, draws=draws, chains=chains, cores=cores, progressbar=progressbar)
    return trace



# ----------------------------------------------------------
## Sunode Sampling -- Just NUTS sampling without vars_list provided
# ----------------------------------------------------------

def run_model_sunode(model, tune, draws, chains=4, cores=4, progressbar=False, init='jitter+adapt_diag_grad'):
    with model:
        trace = pymc.sample(tune=tune, draws=draws, chains=chains, cores=cores, progressbar=progressbar, init=init)
    return trace