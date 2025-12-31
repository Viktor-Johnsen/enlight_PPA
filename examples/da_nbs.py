'''
Example script to run the NBS model for a PPA negotiation between a VRE producer
and a buyer, including risk aversion parameters, and visualize results
using heatmaps. This script uses input data (e.g. forecasts) and results (power prices)
from DA market simulations.
'''
import time
import numpy as np
import matplotlib.pyplot as plt

# Module imports
from enlight_PPA.runners import NBSRunner
from enlight_PPA.utils.nbs_utils import load_plot_configs, prettify_subplots


if __name__ == "__main__":
    t0 = time.time()
    # Setup hyperparameters
    PPA_profile = "PaP"
    BL_compliance_perc = 0
    PPA_zone = "DK2"

    # Producer VRE capacity, Buyer annual consumption, strike price upper bound
    P_vre = 1  # MW
    x_buyer = 0.3  # ratio of average buyer power consumption relative to capacity of Producer VRE
    y_batt = 0.25  # MW_batt / MW_VRE
    S_UB = 60  # €/MWh

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    BL_compliance_rate=BL_compliance_perc,
                    PPA_zone=PPA_zone,
                    P_vre=P_vre,
                    # x_tot_Z, x_pv, x_wind_on, x_wind_off
                    x_buyer=x_buyer,
                    y_batt=y_batt,
                    S_UB=S_UB,
    )
    # For good measure, check that the scenarios align. If not, uncomment and run the first two lines.
    for da_obj in nbs_runner.da_data_dict.values():
        print(da_obj.bidding_zones)
    # nbs_runner.single_nbs(scenario_name="scenario_2")
    # d=nbs_runner.nbs_model
    # print(f"S = {d.S.X:.2f} €/MWh, volume = {d.M.X if d.PPA_profile=="BL" else d.gamma.X:.2f} {"MW" if d.PPA_profile=="BL" else "%"}")

    # Define ranges for beta
    beta_D_list = np.round(np.arange(0.0,0.81, 0.3), 2)  # avoid floating point issues
    beta_O_list = np.round(np.arange(0.0,0.81, 0.3), 2)  # avoid floating point issues
    nbs_runner.mult_nbs(beta_O_list=beta_O_list,
                        beta_D_list=beta_D_list)

    # Save results in a pickle file
    nbs_runner.save_mult_nbs()

    # Verify combliance rate
    d=nbs_runner.mult_nbs_models.models[beta_O_list[1]][beta_D_list[0]]

    d.visualize_example_profit_dist(bars=True)
    d.visualize_example_outcome(show_all_scens=True)
    d.verify_behaviour(w_BESS=3)
    nbs_runner.ppa_calcs_dict["scenario_1"].visualize_inputs(plot_hours=(100*24, 100*24+168))


    # d=nbs_runner.nbs_model
    if d.BL_compliance_perc > 0:
        print(d.v_min.X.sum(axis=0) / (d.T * d.M.X))
    print(f"That took {time.time()-t0:.2f} s")

    '''
    Temporarily plotting input data down here
    '''
    hours = range(180*24,180*24+72)
    alphas = np.linspace(1.0, 0.25, d.P_fore_w.shape[1])
    load_plot_configs()
    fig, ax = plt.subplots(2, 1, figsize=(12,6))
    # --- First plot: Producer forecasts and Buyer consumption profile ---
    for i, a in enumerate(alphas):
        ax[0].plot(d.P_fore_w[hours, i], alpha=a, label=f"Producer: Scen. {i+1}", ls='--')

    ax[0].plot(d.L_t[hours], linewidth=2, label="Buyer")
    # plt.legend()
    # plt.show()
    ax[0].set_title("Forecasts [MW]", loc='left')

    # --- Second plot: DA prices (outcome of DA market model) ---
    for i, a in enumerate(alphas):
        ax[1].plot(d.lambda_DA_w[hours, i], alpha=a, label=f" Scen. {i+1}")
    ax[1].set_xlabel("Hour")
    ax[1].set_title("DA price [€/MWh]", loc='left')

    prettify_subplots(ax)
    plt.tight_layout()
    plt.show()

    # --- Third plot: Price-duration curves ---
    fig, ax = plt.subplots(figsize=(12,6))
    for i in range(d.lambda_DA_w.shape[1]):
        ldc = sorted(d.lambda_DA_w[:,i])[::-1]
        ax.plot(ldc, label=f"Scen. {i+1}")
    prettify_subplots(ax)
    ax.set_xlabel("Hours [h]")
    ax.set_title("Price-duration curve [€/MWh]", loc="left")
    plt.show()
