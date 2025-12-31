import matplotlib.pyplot as plt

# Module imports
from enlight_PPA.data_ops import DataLoader, PPAInputData, PPAInputCalcs, NBSSetup
from enlight_PPA.utils.nbs_utils import load_plot_configs, unify_palette_cyclers, prettify_subplots, perc, generate_scenarios
import enlight_PPA.utils as utils
from enlight_PPA.models import NBSModel


if __name__=="__main__":  
    load_plot_configs()  
    # Setup hyperparameters
    logger = utils.setup_logging(log_file="nbs.log")
    PPA_profile = "BL"
    BL_compliance_rate = 0.0
    PPA_zone = "DK2"
    scenario_name = "scenario_4"

    # Instantiate objects and load power price results from DA market model
    data = DataLoader(
        scenario_name=scenario_name,
        logger=logger,
    )

    ppa_data = PPAInputData(
        Z=PPA_zone,
        P_vre=5e3/20,  # MW
        x_tot_Z=0,
        x_pv=0.5,
        x_wind_on=0.1,
        x_wind_off=0.4,
        y_batt=0.02,  # p_batt/p_vre,
        batt_Crate=1,
        x_buyer=0.32,
        ppa_logger=logger,
    )

    # Prepare forecasts for NBS modeling
    ppa_calcs = PPAInputCalcs(
        scenario_name=scenario_name,
        da_data=data,
        ppa_data=ppa_data,        
        ppa_logger=logger,
    )
    ppa_calcs.visualize_inputs(plot_hours=(90*24, 90*24+168))

    nbs_setup = NBSSetup(
        S_LB=0,
        S_UB=80,
        M_LB=0,
        M_UB=ppa_data.P_vre,
        gamma_LB=0,
        gamma_UB=1,
        beta_D=0.5,
        beta_O=0.15,
        alpha=0.75,
        nbs_setup_logger=logger,
    )

    P_fore_w = generate_scenarios(yearly_param=ppa_calcs.P_fore, noise_lvl=.1)
    lambda_DA_w = generate_scenarios(yearly_param=ppa_calcs.lambda_DA, noise_lvl=0.15)
    P_fore_w[:, 1] *= 0.8  # Example of scenario modification

    nbs_model = NBSModel(
        PPA_profile=PPA_profile,
        BL_compliance_perc=BL_compliance_rate,
        P_fore_w=P_fore_w, #[:8736,:], # -> used to verify that FREQ_hours has been implemented correctly
        P_batt=ppa_calcs.P_batt,
        batt_eta=ppa_data.batt_eta,
        batt_Crate=ppa_data.batt_Crate,
        L_t=ppa_calcs.B_fore_arr, #[:8736],
        lambda_DA_w=lambda_DA_w, #[:8736, :],
        WTP=data.voll_classic,
        # add_batt,
        # hp=None,
        S_LB=nbs_setup.S_LB,
        S_UB=nbs_setup.S_UB,
        M_LB=nbs_setup.M_LB,
        M_UB=nbs_setup.M_UB,
        gamma_LB=nbs_setup.gamma_LB,
        gamma_UB=nbs_setup.gamma_UB,
        beta_D=nbs_setup.beta_D,
        beta_O=nbs_setup.beta_O,
        alpha=nbs_setup.alpha,
        nbs_model_logger=logger,
    )
    nbs_model.solve_model()
    
    if nbs_model.PPA_profile == 'BL':
        print(f"PPA price: {nbs_model.S.X:.2f} €/MWh, volume: {nbs_model.M.X:.2f} MW")
    else:
        print(f"PPA price: {nbs_model.S.X:.2f} €/MWh, volume: {nbs_model.gamma.X*100:.2f} %")

    # Visualize results (and inputs)
    load_plot_configs()
    fig, ax = plt.subplots(2, 1, figsize=(12,6))
    unify_palette_cyclers(ax)
    ppa_calcs.P_fore.plot(ax=ax[0], label="P_fore")
    ppa_calcs.B_fore.plot(ax=ax[0], label="B_fore")
    if PPA_profile == "BL":
        ax[0].axhline(nbs_model.M.X, xmin=0, xmax=8760, label="M: PPA volume", c='r')
    elif PPA_profile in ["PaP", "PaF"]:
        ax[0].plot(nbs_model.gamma.X * ppa_calcs.P_fore, label=r"$\gamma$: PPA volume", c='r') 
    ax[0].set_title(f"{scenario_name}")
    ppa_calcs.lambda_DA.plot(ax=ax[1], label="lambda_DA")
    ax[1].axhline(nbs_model.S.X, xmin=0, xmax=8760, label="S: PPA price", c='r')
    prettify_subplots(ax)

    print("Capture price load: ", -(ppa_calcs.B_fore * ppa_calcs.lambda_DA).sum() / ppa_calcs.B_fore.sum())
    print("Capture price producer: ", (ppa_calcs.P_fore * ppa_calcs.lambda_DA).sum() / ppa_calcs.P_fore.sum())