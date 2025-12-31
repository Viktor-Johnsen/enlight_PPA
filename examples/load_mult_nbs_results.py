'''
This script demonstrates how to load and access results
from previously run multiple NBS models (NBSMultModel).
'''
import pickle
import pandas as pd

from enlight_PPA.config_path import SIMULATIONS_DIR

# --- Load results ---
file_name = "mult_nbs_results__BL_0_[0.  0.3 0.6].pkl"
with open(SIMULATIONS_DIR / "NBS_results" / file_name, "rb") as f:
    results = pickle.load(f)

betas = list(results["res_S"].keys())
beta_O = betas[1]
beta_D = betas[2]
if "BL" in file_name:
    S, M = results["res_S"][beta_O][beta_D], results["res_M"][beta_O][beta_D]
    print("Strike price S=%.2f €/MWh and PPA contracted capacity M=%.2f MW loaded from the NBS solution." % (S, M))
else:
    S, gamma = results["res_S"][beta_O][beta_D], results["res_gamma"][beta_O][beta_D]
    print("Strike price S=%.2f €/MWh and PPA contracted capacity share %.2f %% loaded from the NBS solution." % (S, gamma*100))