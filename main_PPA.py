"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
from enlight.runner import EnlightRunner  # Updated import path
from pathlib import Path
from nbs_runner import NBSRunner
from ppa_input import PaP2DA


if __name__ == "__main__":
    # FIRST: simulate a PPA negotiation between a Producer and a Buyer
   # Setup hyperparameters
    PPA_profile = "PaP"
    PPA_zone = "DK1"

    # Producer VRE capacity
    # Using physical capacity in DK2 to get 12.37 €/MWh, 86.41 %
    # E_buyer = 21862408 * 0.9
    # P_vre = E_buyer / 8760 * 3 # triple 90% of the mean classical power demand

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    PPA_zone=PPA_zone,
                    # P_vre=P_vre,
                    x_tot_Z=0.99,  # use 90% of the total zonal VRE (Hydro ror excl.) capacity
                    x_pv=0.1281, # 46% of power capacity is solar PV, 3484
                    x_wind_on=0.6984, # 8% is onshore wind, 584
                    x_wind_off=0.1735, # 46% is offshore wind, 3490
                    y_batt=0,
                    x_buyer=0.36,  # average buyer power cons. is 30% of the total Producer VRE capacity
                    S_UB=50,
    )# 0.128109, 0.698434, 0.173457
    # DK2: 0.6/0.3 -> (45, 5, 50), (0.67, 0.03, 0.3), (0.75, 0, 0.25)
    # DK1: 0.95/0.125 -> (0.33, 0.25, 0.42)
    # FI: 0.99/0.36 -> (0.127, 0.70, 0.173)
    beta_O_list = [0.0]
    beta_D_list = [0.75]
    beta_O_chosen = beta_O_list[0]
    beta_D_chosen = beta_D_list[0]
    nbs_runner.mult_nbs(beta_O_list, beta_D_list)
    ppa_model=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen]
    nbs_runner.ppa_calcs_dict["scenario_1"].visualize_inputs(plot_hours=(90*24,90*24+168))

    if ppa_model.PPA_profile == 'BL':
        print(f"PPA price: {ppa_model.S.X:.2f} €/MWh, volume: {ppa_model.M.X:.2f} MW")
    else:
        print(f"PPA price: {ppa_model.S.X:.2f} €/MWh, volume: {ppa_model.gamma.X*100:.2f} %")
    '''
    What information do I need to pass to the DA model?
    - Whether to include a PPA : bool
    - Strike price: nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_O_chosen].S.X
    - off_wind_el_cap: nbs_runner.ppa_calcs_dict["scenario_1"].off_wind_el_cap
    - on_wind_el_cap: nbs_runner.ppa_calcs_dict["scenario_1"].on_wind_el_cap
    - solar_pv_el_cap: nbs_runner.ppa_calcs_dict["scenario_1"].solar_pv_el_cap
    PaP:
    - gamma: nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_O_chosen].gamma.X

    BL:
    - BL volume for annual compliance rate
    - batt cap (P/E)
    - make v_min...
    DA model 
    '''
    # Create an instance of PaP2DA to prepare for usage in EnlightRunner
    pap2da = PaP2DA(
        z=PPA_zone,
        s=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].S.X,
        gamma=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].gamma.X,
        solar_pv_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].solar_pv_el_cap,
        on_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].on_wind_el_cap,
        off_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].off_wind_el_cap,
    )

    # Run DA market model with the PPA
    da_runner_ppa = EnlightRunner()
    scenario_name = "scenario_2"
    da_runner_ppa.prepare_data_single_scenario(scenario_name=scenario_name)
    da_runner_ppa.load_data_single_simulation(scenario_name=scenario_name)
    da_runner_ppa.run_single_simulation(scenario_name=scenario_name + "/test", PaP2DA=pap2da)
    d_p = da_runner_ppa.enlight_model

    # Verify social welfare calculations
    print(f"{d_p.results_econ['social welfare']/1e9:.6f} b.€")
    print(f"{d_p.results_econ['social welfare perceived']/1e9:.6f} b.€")
    print(f"{d_p.model.objective.value/1e9:.6f} b.€")

    d_p.results_dict['electricity_prices'].plot()
