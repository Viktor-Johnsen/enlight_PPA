"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
from enlight.runner import EnlightRunner  # Updated import path
from pathlib import Path
from nbs_runner import NBSRunner


if __name__ == "__main__":
    # FIRST: simulate a PPA negotiation between a Producer and a Buyer
   # Setup hyperparameters
    PPA_profile = "PaP"
    PPA_zone = "DK1"

    # Producer VRE capacity
    #P_vre = 0.9*(3484+584+3490)  # MW -- 90% of VRE capacity in DK2...
    E_buyer = 21862408 * 0.9
    P_vre = E_buyer / 8760 * 3 # triple 90% of the mean classical power demand
    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    PPA_zone=PPA_zone,
                    P_vre=P_vre,
                    x_pv=0.46, #3484
                    x_wind_on=0.08, #584
                    x_wind_off=0.46, # 3490
                    y_batt = 0,
                    E_buyer=E_buyer,
                    S_UB=50,
    )
    nbs_runner.mult_nbs(beta_O_list=[0.25], beta_D_list=[0.25])
    ppa_model=nbs_runner.mult_nbs_models.models[0.25][0.25]

    if ppa_model.PPA_profile == 'BL':
        print(f"PPA price: {ppa_model.S.X:.2f} €/MWh, volume: {ppa_model.M.X:.2f} MW")
    else:
        print(f"PPA price: {ppa_model.S.X:.2f} €/MWh, volume: {ppa_model.gamma.X*100:.2f} %")

    
    # # Create an instance of the EnlightRunner
    # runner = EnlightRunner()
    # scenario_name = "scenario_2"
    # h=133  # needed for .visualize_data() and .visualize_results()

    # '''Combination of methods to VISUALIZE INPUT data:'''
    # # Creates instance of the DataProcessor:
    # runner.prepare_data_single_scenario(scenario_name=scenario_name)
    # # Creates instance of the DataLoader:
    # runner.load_data_single_simulation(scenario_name=scenario_name)
    # # Creates instance of the DataVisualizer. Data has to be prepared when running this:
    # # issues due to short palette...
    # # runner.visualize_data(example_hour=h)

    # '''Combination of methods to RUN a SINGLE simulation
    # and SHOW RESULTS for that simulation:'''
    # # Creates instance of the EnlightModel
    # runner.run_single_simulation(scenario_name=scenario_name + "/test")
    # # Creates instance of the ResultsVisualizer.
    # runner.visualize_results(example_hour=h)

    # # Verify social welfare calculations
    # print(f"{runner.enlight_model.results_econ['social welfare']/1e9:.6f} b.€")
    # print(f"{runner.enlight_model.results_econ['social welfare perceived']/1e9:.6f} b.€")
    # print(f"{runner.enlight_model.model.objective.value/1e9:.6f} b.€")

    # # To run all scenarios at once instead: ;)
    # # from enlight.runner import EnlightRunner  # Updated import path
    # # from pathlib import Path
    # # runner = EnlightRunner()
    # # runner.prepare_load_run_all_sims()