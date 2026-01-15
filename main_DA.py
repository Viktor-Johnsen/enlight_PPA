"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
from enlight_PPA.utils.utils import get_capture_price_zonal, get_capture_price_units, get_capture_price_vre
from enlight_PPA.runners import EnlightRunner  # Updated import path
from pathlib import Path

if __name__ == "__main__":
    # d_dict = {}
    # for scen in ["scenario_1", "scenario_2", "scenario_3", "scenario_4"]:
    # Create an instance of the EnlightRunner
    runner = EnlightRunner()
    # scenario_name = scen
    scenario_name = "scenario_4"
    h=133  # needed for .visualize_data() and .visualize_results()

    '''Combination of methods to VISUALIZE INPUT data:'''
    # Creates instance of the DataProcessor:
    runner.prepare_data_single_scenario(scenario_name=scenario_name)
    # Creates instance of the DataLoader:
    runner.load_data_single_simulation(scenario_name=scenario_name)
    # Creates instance of the DataVisualizer. Data has to be prepared when running this:
    # issues due to short palette...
    runner.visualize_data(example_hour=h, chosen_zones=["DK1", "DELU", "FR"])

    '''Combination of methods to RUN a SINGLE simulation
    and SHOW RESULTS for that simulation:'''
    # Creates instance of the EnlightModel
    runner.run_single_simulation(scenario_name=scenario_name)# + "/test")
    # Creates instance of the ResultsVisualizer.
    # runner.visualize_results(example_hour=h)

    # Verify social welfare calculations
    print(f"{runner.enlight_model.results_econ['social welfare']/1e9:.6f} b.€")
    print(f"{runner.enlight_model.results_econ['social welfare perceived']/1e9:.6f} b.€")
    print(f"{runner.enlight_model.model.objective.value/1e9:.6f} b.€")

    # Inspect capture prices
    d=runner.enlight_model
    #d_dict[scen] = d

#for scen, d in d_dict.items():
    vres = ["solar_pv", "wind_onshore", "wind_offshore", "hydro_ror"]

#for scen, d in d_dict.items():
    zone = "DELU"
    print("################################")
    print(f"Capture prices for scenario {scenario_name}:")
    print(f"Capture price inflex load:{get_capture_price_zonal(d, "demand_inflexible_classic", zone=zone, generator=False):.2f}")
    vre_tot_fore = (d.data.solar_pv_production + d.data.wind_onshore_production + d.data.wind_offshore_production)[zone]
    vre_tot_disp = (d.results_dict['solar_pv_offer_sol']+d.results_dict['wind_onshore_offer_sol']+d.results_dict['wind_offshore_offer_sol'])[zone]
    print(f"Capture price producer: {(vre_tot_disp * d.results_dict['electricity_prices'][zone]).sum() / vre_tot_fore.sum():.2f}")
    print(f"Capture price FLEX (classic) load: { get_capture_price_zonal(d, "demand_flexible_classic", zone, generator=False):.2f}")
    print(f"Capture price FLEX (DH) load:  {get_capture_price_units(d, "dh_units", d.data.L_DH_Z_df, zone, generator=False):.2f}")
    print(f"Capture price FLEX (PtX) load:  {get_capture_price_units(d, "ptx_units", d.data.L_PtX_Z_df, zone, generator=False):.2f}")
    # print("done.")
    print("======== VREs ========")
    for vre in vres:
        print(f"Capture price {vre}:  {get_capture_price_zonal(object=d, var_name=vre, zone=zone, generator=True):.2f}")
        print(f"Capture price {vre}:  {get_capture_price_vre(object=d, var_name=vre, zone=zone):.2f}")
    
    print(f"Total inflex in: {d.data.demand_inflexible_classic.sum().sum()/1e6:.2f} TWh")
    #for scen, d in d_dict.items():
    print(f"Total VRE forecast in {scenario_name}: {(d.data.solar_pv_production + d.data.wind_onshore_production + d.data.wind_offshore_production + d.data.hydro_ror_production).sum().sum()/1e6:.2f} TWh")
    # To run all scenarios at once instead: ;)
    # from enlight_PPA.runners import EnlightRunner  # Updated import path
    # from pathlib import Path
    # runner = EnlightRunner()
    # runner.prepare_load_run_all_sims()
    # print("done.")

    import matplotlib.pyplot as plt
    import pandas as pd
    # --- SUPPLY SIDE ---

    gen_sources = {
        "Wind onshore": d.results_dict["wind_onshore_offer_sol"].sum().sum(),
        "Wind offshore": d.results_dict["wind_offshore_offer_sol"].sum().sum(),
        "Solar PV": d.results_dict["solar_pv_offer_sol"].sum().sum(),
        "Hydro RoR": d.results_dict["hydro_ror_offer_sol"].sum().sum(),
        "Conventional": d.results_dict["conventional_units_offer_sol"].sum().sum(),
        "Hydro reservoir": d.results_dict["hydro_res_units_offer_sol"].sum().sum(),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        gen_sources.values(),
        labels=gen_sources.keys(),
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title(f"Total annual electricity generation by technology ({scenario_name})")
    ax.axis("equal")

    plt.tight_layout()
    plt.show()

    # --- DEMAND SIDE ---

    demand_destinations = {
        "Inflexible demand": d.results_dict["demand_inflexible_classic_bid_sol"].sum().sum(),
        "Flexible consumption": (
            d.results_dict["demand_flexible_classic_bid_sol"].sum().sum()
            + d.results_dict["ptx_units_bid_sol"].sum().sum()
            + d.results_dict["dh_units_bid_sol"].sum().sum()
        ),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        demand_destinations.values(),
        labels=demand_destinations.keys(),
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Total annual electricity demand consumption type")
    ax.axis("equal")

    plt.tight_layout()
    plt.show()
