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
from enlight_PPA.utils.nbs_utils import load_plot_configs, prettify_subplots, make_hour_formatter


if __name__ == "__main__":
    t0 = time.time()
    # Setup hyperparameters
    # FIRST: 1) simulate a PPA negotiation between a Producer and a Buyer
    # Setup hyperparameters
    PPA_profile = "BL"
    BL_compliance_perc = 0  # between 0 and 1
    PPA_zone = "DELU"

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    BL_compliance_rate=BL_compliance_perc,
                    PPA_zone=PPA_zone,
                    # For a specific Producer/Buyer: P_vre, x_pv, x_wind_on, x_wind_off, y_batt, x_buyer
                    # On a zonal level: x_tot_Z...
                    x_tot_Z=0.9999,  # use 99.99% of the total zonal VRE (Hydro ror excl.) capacity
                    y_batt=0,  # If 0: use the zonal-level P_BESS/P_VRE ratio
                    S_UB=45,
    )

    # For good measure, check that the scenarios align. If not, uncomment and run the first two lines.
    for da_obj in nbs_runner.da_data_dict.values():
        print(da_obj.bidding_zones)
    # nbs_runner.single_nbs(scenario_name="scenario_2")
    # d=nbs_runner.nbs_model
    # print(f"S = {d.S.X:.2f} €/MWh, volume = {d.M.X if d.PPA_profile=="BL" else d.gamma.X:.2f} {"MW" if d.PPA_profile=="BL" else "%"}")

    # Define ranges for beta
    beta_B_list = np.round(np.arange(0.0, 1.01, 0.2), 2)  # avoid floating point issues
    beta_P_list = np.round(np.arange(0.0, 1.01, 0.2), 2)  # avoid floating point issues
    nbs_runner.mult_nbs(beta_B_list=beta_B_list,
                        beta_P_list=beta_P_list)

    # Save results in a pickle file
    nbs_runner.save_mult_nbs()

    # Verify combliance rate
    d=nbs_runner.mult_nbs_models.models[0.6][0.2]

    d.visualize_example_profit_dist(bars=True, presentation=True)
    
    d2=nbs_runner.mult_nbs_models.models[0.2][0.6]
    d2.visualize_example_profit_dist(bars=True, presentation=True)
    
    d.visualize_example_outcome(show_all_scens=True)
    # d.verify_behaviour(w_BESS=3)
    nbs_runner.ppa_calcs_dict["scenario_1"].visualize_inputs(plot_hours=(100*24, 100*24+168))


    # d=nbs_runner.nbs_model
    if d.BL_compliance_perc > 0:
        print(d.v_min.X.sum(axis=0) / (d.T * d.M.X))
    print(f"That took {time.time()-t0:.2f} s")

    '''
    Temporarily plotting input data down here
    '''
    year = nbs_runner.da_data_dict["scenario_1"].solar_weather_year
    hours = range(100*24, 100*24+168)
    alphas = np.linspace(1.0, 1.0, d.P_fore_w.shape[1])
    load_plot_configs()
    fig, ax = plt.subplots(2, 1, figsize=(12,6))
    # Producer forecasts and Buyer consumption profilec
    c_idx=0
    for i, a in enumerate(alphas):
        c_idx = i+1
        if i == 3:
            c_idx += 1  # skip color to avoid too light color
        ax[0].plot(nbs_runner.P_fore_w[hours, i], alpha=a, label=f"Producer: Scen. {i+1}", c=nbs_runner.palette[c_idx], ls='--')
        if d.PPA_profile == "PaP" or d.PPA_profile == "PaF":
            ax[0].plot(d.gamma.X * nbs_runner.P_fore_w[hours, i], alpha=a, label=fr"PaP vol. ($\gamma={d.gamma.X:.2f}$)", c=nbs_runner.palette[c_idx], ls='-')
    if d.PPA_profile == "BL":
        ax[0].axhline(d.M.X * nbs_runner.P_fore_w.max(), linewidth=2, label="Producer BL vol. M", c=nbs_runner.palette[0], ls='-.')
    ax[0].plot(nbs_runner.ppa_calcs_dict["scenario_1"].B_fore_arr[hours], linewidth=2, label="Buyer", c=nbs_runner.palette[c_idx+2])
    # plt.legend()
    # plt.show()
    ax[0].set_title(fr"{PPA_profile} PPA outcome in {PPA_zone} where $\beta_B=${d.beta_B} and $\beta_P=${d.beta_P}"+"\nPower forecasts [MW]", loc='left')

    # DA prices (outcome of DA market model)
    for i, a in enumerate(alphas):
        c_idx = i+1
        if i == 3:
            c_idx += 1  # skip color to avoid too light color
        ax[1].plot(d.lambda_DA_w[hours, i], alpha=a, c=nbs_runner.palette[c_idx], label=f" Scen. {i+1}")
    # ax[1].set_xlabel("Hour")
    ax[1].axhline(d.S.X, c=nbs_runner.palette[0], ls='-.', label=r"PPA price $S$")

    ax[1].set_title("Spot price [€/MWh]", loc='left')
    for a in ax:
        a.xaxis.set_major_formatter(make_hour_formatter(year=year))
    ax[0].tick_params(axis="x", labelbottom=False)
    ax[1].tick_params(axis="x", rotation=15)
    prettify_subplots(ax, bbox_list=[1.02, 1.02])
    plt.tight_layout()
    plt.show()


    # Price-duration curves
    fig, ax = plt.subplots(figsize=(12,6))
    for i in range(d.lambda_DA_w.shape[1]):
        ldc = sorted(d.lambda_DA_w[:,i])[::-1]
        ax.plot(ldc, label=f"Scen. {i+1}")
    prettify_subplots(ax, bbox_list=[1.15, 1])
    ax.set_xlabel("Hours [h]")
    ax.set_title(f"Price-duration curves in {PPA_zone}\nSpot price [€/MWh]", loc="left")
    plt.show()
