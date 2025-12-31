'''
This script shows how to use the NBSMultModel to run multiples NBS models with varying risk
parameters (beta_O and beta_D). It then visualizes the impact of these parameters on the outcomes
using a heatmap. This specific example script uses synthetic data.
'''
import time
import numpy as np
from enlight_PPA.utils.nbs_utils import load_plot_configs, generate_data, specify_battery_data, perc
from enlight_PPA.models.nbs_model import NBSMultModel


if __name__ == "__main__":
    load_plot_configs()
    t0 = time.time()
    # Fixed parameters:
    alpha = 0.75  # CVaR: tail of interest

    # Capture price VRE: (d.P_DA_w * d.lambda_DA_w).sum() / d.P_DA_w.sum() = 96.38 €/MWh
    # Capture price load: - (d.L_t * d.lambda_DA_w).sum() / (d.W * d.L_t.sum()) = -98.1 €/MWh
    S_LB, S_UB = 96.38 * 0.5, 98.1*1.5  # PPA strike price bounds
    M_LB, M_UB = 0.01, 0.99  # BL volume bounds.
    gamma_LB, gamma_UB = 0, 1  # PaP capacity share bounds.

    # Profile type
    PPA_profile = 'PaP'
    BL_compliance_perc = 0.1

    # Define ranges for betas
    beta_D_list = np.round(np.arange(0.0, 0.21, 0.1), 2)  # avoid floating point issues
    beta_O_list = np.round(np.arange(0.0, 0.21, 0.1), 2)  # avoid floating point issues

    P_fore_w, lambda_DA_w, L_t, WTP = generate_data()
    P_batt, batt_eta, batt_Crate = specify_battery_data()

    runner = NBSMultModel(
        PPA_profile=PPA_profile,  # Type of PPA profile ('PaF', 'PaP', or 'BL')
        BL_compliance_perc=BL_compliance_perc, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
        P_fore_w=P_fore_w,
        P_batt=P_batt,
        batt_eta=batt_eta,
        batt_Crate=batt_Crate,
        L_t=L_t,
        lambda_DA_w=lambda_DA_w,
        WTP=WTP,
        # add_batt=True,
        S_LB=S_LB,  # Minimum PPA strike price
        S_UB=S_UB,  # Maximum PPA strike price
        M_LB=M_LB,  # BL: Minimum baseload volume
        M_UB=M_UB,  # BL: Maximum baseload volume
        gamma_LB=gamma_LB, # PaP: Minimum PPA capacity share volume
        gamma_UB=gamma_UB, # PaP: Minimum PPA capacity share volume
        alpha=alpha,  # CVaR: Tail of interest for CVaR
    )
    #%%
    runner.run_multiple_NBS_models(beta_O_list=beta_O_list,
                                    beta_D_list=beta_D_list)

    runner.visualize_risk_impact_heatmap()

    beta_O_chosen=beta_O_list[1]
    beta_D_chosen=beta_D_list[-1]

    # For debugging
    d = runner.models[beta_O_chosen][beta_D_chosen]
    # end

    d.visualize_example_outcome()
    d.visualize_example_profit_dist()
    d.verify_behaviour()
    print(f"Total time elapsed: {time.time()-t0:.2f}")
