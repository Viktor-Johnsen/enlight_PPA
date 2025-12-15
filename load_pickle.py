import pickle
import pandas as pd

# --- Load results ---
with open("mult_nbs_results__PaP_0.0.pkl", "rb") as f:
    results = pickle.load(f)

betas = list(results["res_gamma"].keys())
beta_O = betas[1]
beta_D = betas[2]

S, gamma = results["res_S"][beta_O][beta_D], results["res_gamma"][beta_O][beta_D]
print("Strike price S=%.2f €/MWh and PPA contracted capacity share %.2f %% loaded from the NBS solution." % (S, gamma*100))