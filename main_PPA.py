"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import time

from enlight.runner import EnlightRunner  # Updated import path
from enlight.model.energy_model import ADJUST_FLEX, ADJUST_TRANS
from pathlib import Path
from nbs_runner import NBSRunner
from ppa_input import PaP2DA, load_plot_configs, prettify_subplots


if __name__ == "__main__":
    t0 = time.time()
    load_plot_configs()
    ### For aligning all sims with the current setup config ###
    # runner = EnlightRunner()
    # runner.prepare_load_run_all_sims()
    ############################################################
    scenario_name = "scenario_1"

    # FIRST: 1) Run DA market model without the PPA
    da_runner = EnlightRunner()
    da_runner.prepare_data_single_scenario(scenario_name=scenario_name)
    da_runner.load_data_single_simulation(scenario_name=scenario_name)
    da_runner.run_single_simulation(scenario_name=scenario_name)
    d = da_runner.enlight_model

    # Verify social welfare calculations
    print(f"{d.results_econ['social welfare']/1e9:.6f} b.€")
    print(f"{d.results_econ['social welfare perceived']/1e9:.6f} b.€")
    print(f"{d.model.objective.value/1e9:.6f} b.€")

    # FIRST: 2) simulate a PPA negotiation between a Producer and a Buyer
    # Setup hyperparameters
    PPA_profile = "PaP"
    PPA_zone = "DK2"

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    PPA_zone=PPA_zone,
                    # For a specific Producer/Buyer: P_vre, x_pv, x_wind_on, x_wind_off, x_buyer
                    # On a zonal level: x_tot_Z...
                    x_tot_Z=0.9999,  # use 90% of the total zonal VRE (Hydro ror excl.) capacity
                    y_batt=0,
                    S_UB=27,
    )

    # For good measure, check that the scenarios align. If not, uncomment and run the first two lines.
    for da_obj in nbs_runner.da_data_dict.values():
        print(da_obj.bidding_zones)

    #%%
    beta_O_list = [0.1]
    beta_D_list = [0.5]
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
    print(f"NBS took {time.time()-t0:.2f} s.")

    #%%
    # SECOND: 1) Run DA market model with the PPA
    da_runner_ppa = EnlightRunner()
    da_runner_ppa.prepare_data_single_scenario(scenario_name=scenario_name)
    da_runner_ppa.load_data_single_simulation(scenario_name=scenario_name)
    da_runner_ppa.run_single_simulation(scenario_name=scenario_name + "/PPA", PaP2DA=pap2da)
    d_p = da_runner_ppa.enlight_model

    # Verify social welfare calculations
    print(f"{d_p.results_econ['social welfare']/1e9:.6f} b.€")
    print(f"{d_p.results_econ['social welfare perceived']/1e9:.6f} b.€")
    print(f"{d_p.model.objective.value/1e9:.6f} b.€")

    print(f"RAN THE MODELS!: PPA profile and zone: {PPA_profile} in {PPA_zone}.")
    # SECOND: 2) COMPARE DA w/w.o. PaP
    # Visualize hours with price change
    mask_prices = ~np.isclose(d.results_dict['electricity_prices'][PPA_zone], d_p.results_dict['electricity_prices'][PPA_zone])
    fig, ax = plt.subplots(figsize=(12,6))
    x = d.results_dict['electricity_prices'][PPA_zone][mask_prices].index
    ax.scatter(x=x, y=d.results_dict['electricity_prices'][PPA_zone][mask_prices], label="DA")
    ax.scatter(x=x, y=d_p.results_dict['electricity_prices'][PPA_zone][mask_prices], label="w/ PaP")
    ax.legend()
    plt.show()
    # Print the number of changes for each decision variable
    for k in list(d.results_dict.keys()):
        num_cols = len(d.results_dict[k].columns)
        print(f"Variable {k} changed: {(~np.isclose(d.results_dict[k], d_p.results_dict[k])).sum()} times out of {num_cols*8760}...")


    # THIRD: Inspect results to compare DA v. DA+PAP
    # PLOT electricity prices
    fig, ax = plt.subplots(figsize=(16,8))
    d.results_dict['electricity_prices'][PPA_zone].plot(ax=ax, label="DA")
    d_p.results_dict['electricity_prices'][PPA_zone].plot(ax=ax, label="w/ PaP")
    prettify_subplots(ax)
    plt.show()

    # Base (non-PaP) keys only
    keys = sorted(k for k in set(d.results_econ['profits']) if not k.endswith('_PaP'))

    units_techs = ['conventional_units', 'dh_units', 'hydro_res_units', 'ptx_units']
    units_mapping = {'conventional_units': d.data.G_Z_df, 'dh_units': d.data.L_DH_Z_df, 'hydro_res_units':d.data.G_hydro_res_Z_df, 'ptx_units' : d.data.L_PtX_Z_df}

    vals_d = []
    vals_dp_base = []
    vals_dp_pap = []

    for k in keys:
        # Units-based profits should be handled differently
        if k in ['conventional_units', 'dh_units', 'hydro_res_units', 'ptx_units']:
            count=0
            vals_d.append(d.results_econ['profits'].get(k, np.nan).dot(units_mapping[k]).loc[PPA_zone])

            base = d_p.results_econ['profits'].get(k, 0.0).dot(units_mapping[k]).loc[PPA_zone]
            pap  = d_p.results_econ['profits'].get(f"{k}_PaP", 0.0)
            if type(pap) != float:
                pap = pap.loc[PPA_zone]

            vals_dp_base.append(base)
            vals_dp_pap.append(pap)   
        elif k == 'demand_inflexible_classic':
            vals_d.append(d.results_econ['profits'].get(k, np.nan).loc[PPA_zone] - d.results_dict['demand_inflexible_classic_bid_sol'].sum(axis=0).loc[PPA_zone] * d.data.voll_classic)
            base = (
                - d_p.results_dict['demand_inflexible_classic_bid_sol'] * d_p.results_dict['electricity_prices']
            ).sum(axis=0)[PPA_zone]
            vals_dp_base.append(base)
            pap = (
                (d_p.results_dict['solar_pv_PaP_offer_sol'] + d_p.results_dict['wind_onshore_PaP_offer_sol'] + d_p.results_dict['wind_offshore_PaP_offer_sol'])
                * (d_p.results_dict['electricity_prices'] - d_p.PaP2DA.s)
            ).sum(axis=0)[PPA_zone]
            vals_dp_pap.append(pap)
        else:
            vals_d.append(d.results_econ['profits'].get(k, np.nan).loc[PPA_zone])

            base = d_p.results_econ['profits'].get(k, 0.0).loc[PPA_zone]
            pap  = d_p.results_econ['profits'].get(f"{k}_PaP", 0.0)
            if type(pap) != float:
                pap = pap.loc[PPA_zone]

            vals_dp_base.append(base)
            vals_dp_pap.append(pap)

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(16,8))

    ax.bar(x - width/2, vals_d, width, label="DA")
    ax.bar(x + width/2, vals_dp_base, width, label="DA with PPA")
    ax.bar(x + width/2, vals_dp_pap, width, bottom=vals_dp_base, label="PPA part")

    ax.set_xticks(x, keys, rotation=90)
    ax.set_ylabel("profits_tot")
    prettify_subplots(ax)
    plt.show()
    #conv, dh, hres, ptx, 

    # Base (non-PaP) keys only
    keys = sorted(k for k in set(d.results_econ['profits_tot']) if not k.endswith('_PaP'))

    vals_d = []
    vals_dp_base = []
    vals_dp_pap = []

    for k in keys:
        if k == 'demand_inflexible_classic':
            vals_d.append(0)
            vals_dp_base.append(0)
            vals_dp_pap.append(0)
        else:
            vals_d.append(d.results_econ['profits_tot'].get(k, np.nan))

            base = d_p.results_econ['profits_tot'].get(k, 0.0)
            pap  = d_p.results_econ['profits_tot'].get(f"{k}_PaP", 0.0)

            vals_dp_base.append(base)
            vals_dp_pap.append(pap)

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(16,8))

    ax.bar(x - width/2, vals_d, width, label="DA")
    ax.bar(x + width/2, vals_dp_base, width, label="DA with PPA")
    ax.bar(x + width/2, vals_dp_pap, width, bottom=vals_dp_base, label="PPA part")

    ax.set_xticks(x, keys, rotation=90)
    ax.set_ylabel("profits_tot")
    prettify_subplots(ax)
    plt.show()



    # FOURTH: Inspect PaP results
    # Prepare data
    # Classical inflex
    inflex_cons = d_p.data.demand_inflexible_classic[PPA_zone]
    # flex: "classical" + PtX + DH
    flex_cons = d_p.data.flexible_demands_dfs['demand_flexible_classic']['capacity'][0] + d_p.data.agg_dh.capacity_el.dot(d_p.data.L_DH_Z_xr) + d_p.data.agg_ptx.capacity_el.dot(d_p.data.L_PtX_Z_xr)
    # Storage: PHS + BESS
    stor_cons = d_p.data.agg_phs.capacity_el + d_p.data.agg_bess.capacity_el
    # Transmission: from + to
    cable_from = d_p.data.lines_a_b_df[d_p.data.lines_a_b_df["from_zone"] == PPA_zone]["1"]
    cable_to = d_p.data.lines_a_b_df[d_p.data.lines_a_b_df["to_zone"] == PPA_zone]["1"]
    trans_cap = cable_from.sum() + cable_to.sum()

    # PPA zone as idx in bidding zones list
    PPA_zone_idx = np.arange(len(d_p.bidding_zones))[np.array(d_p.bidding_zones)==PPA_zone]

    # Plot for inspection
    fig, ax = plt.subplots(figsize=(16,8))
    # Plot total VRE (excl. hydro ror) forecast
    # "Free" VRE
    vre_prod = (d_p.data.wind_onshore_production + d_p.data.wind_offshore_production + d_p.data.solar_pv_production)[PPA_zone]# + d_p.data.hydro_ror_production)[PPA_zone]
    # PPA-bound VRE
    # Total VRE
    # vre_prod.plot(ax=ax, label="VREs")

    # Plot the PaP VRE offered at negative prices
    vre_ppa_prod = (d_p.solar_pv_PaP_fore + d_p.wind_onshore_PaP_fore + d_p.wind_offshore_PaP_fore)[PPA_zone]
    vre_ppa_prod.plot(ax=ax, label="VRE in PaP")#, alpha=1)

    # Plot all types of consumption
    inflex_cons.plot(ax=ax, label="inflex")#, alpha=0.9)
    if ADJUST_FLEX > 0:
        (ADJUST_FLEX*flex_cons[PPA_zone_idx] + inflex_cons).plot(ax=ax, label="incl. flex")#, alpha=0.8)
    if ADJUST_FLEX > 0:
        (ADJUST_FLEX*(stor_cons[PPA_zone] + flex_cons[PPA_zone_idx]) + inflex_cons).plot(ax=ax, label="incl. flex+stor")#, alpha=0.7)
    (ADJUST_TRANS * trans_cap + ADJUST_FLEX * (stor_cons[PPA_zone] + flex_cons[PPA_zone_idx]) + inflex_cons).plot(ax=ax, label=f"{"flex+stor" if ADJUST_FLEX>0 else "."}{"+ trans" if ADJUST_TRANS>0 else "."}")#, alpha=0.6)

    d_p.data.hydro_ror_production[PPA_zone].plot(ax=ax, label="hydro_ror")
    ax.set_xlim(200*24,200*24+1544)
    ax.legend()
    plt.show()

    print("The gamma corresponds to the ratio of prod. as part of PaP and the total VRE prod. (excl. hydro ror):\n", np.round((vre_ppa_prod/vre_prod).mean(),5), np.round(pap2da.gamma * nbs_runner.x_tot_Z,5))

    nbs_runner.x_pv, nbs_runner.x_wind_on, nbs_runner.x_wind_off, nbs_runner.ppa_data.x_pv, nbs_runner.ppa_data.x_wind_on, nbs_runner.ppa_data.x_wind_off

    print("The gamma corresponds to the ratio of prod. as part of PaP and the total VRE prod. (excl. hydro ror):\n", np.round((vre_ppa_prod/vre_prod).mean(),5), np.round(pap2da.gamma * nbs_runner.x_tot_Z,5))
    print("PPA coverage is very high:", np.minimum(inflex_cons,vre_ppa_prod).sum() / inflex_cons.sum(), inflex_cons.sum()/vre_ppa_prod.sum())

    nbs_runner.x_pv, nbs_runner.x_wind_on, nbs_runner.x_wind_off, nbs_runner.ppa_data.x_pv, nbs_runner.ppa_data.x_wind_on, nbs_runner.ppa_data.x_wind_off

    # Inspect the zonal VRE capacity (excl. hror) in the Producer's portfolio
    nbs_runner.ppa_calcs_dict["scenario_1"].P/(d_p.data.solar_pv_production.max() + d_p.data.wind_onshore_production.max() + d_p.data.wind_offshore_production.max())

    # Inspect the ratio of inflex mean power cons. to Producer VRE cap (= x_buyer)
    nbs_runner.ppa_calcs_dict["scenario_1"].E_buyer/8760 /nbs_runner.ppa_calcs_dict["scenario_1"].P

    # Calculate the ratio of x_buyer for each bidding zone included
    d_p.data.demand_inflexible_classic.sum().div(8760)/(d_p.data.solar_pv_production.max() + d_p.data.wind_onshore_production.max() + d_p.data.wind_offshore_production.max())

    # Inspect capture prices
    print("Capture price inflex load: ", -(d.data.demand_inflexible_classic[PPA_zone] * d.results_dict['electricity_prices'][PPA_zone]).sum() / d.data.demand_inflexible_classic[PPA_zone].sum())
    vre_tot = (d.data.solar_pv_production + d.data.wind_onshore_production + d.data.wind_offshore_production)[PPA_zone]
    print("Capture price producer: ", (vre_tot * d.results_dict['electricity_prices'][PPA_zone]).sum() / vre_tot.sum())
    print("Capture price FLEX (classic) load", -(d.results_dict['demand_flexible_classic_bid_sol'][PPA_zone] * d.results_dict['electricity_prices'][PPA_zone]).sum() / d.results_dict['demand_flexible_classic_bid_sol'][PPA_zone].sum())
    print("Capture price FLEX (DH) load", -(d.results_dict["dh_units_bid_sol"].dot(d.data.L_DH_Z_df)[PPA_zone] * d.results_dict['electricity_prices'][PPA_zone]).sum() / d.results_dict["dh_units_bid_sol"].dot(d.data.L_DH_Z_df)[PPA_zone].sum())
    print("Capture price FLEX (PtX) load", -(d.results_dict["ptx_units_bid_sol"].dot(d.data.L_PtX_Z_df)[PPA_zone] * d.results_dict['electricity_prices'][PPA_zone]).sum() / d.results_dict["ptx_units_bid_sol"].dot(d.data.L_PtX_Z_df)[PPA_zone].sum())
    # %%
    # Analyze new dispatch of a presumed marginal generatorDK2 Waste
    prices_p = d_p.results_dict['electricity_prices']
    prices_p_filtered = prices_p[prices_p.lt(0).any(axis=1)][PPA_zone]
    marg_gen = "DK2 Waste"
    plt.scatter(x=prices_p_filtered.index, y=d.results_dict['conventional_units_offer_sol'][marg_gen].loc[prices_p_filtered.index], label="DA")
    plt.scatter(x=prices_p_filtered.index, y=d_p.results_dict['conventional_units_offer_sol'][marg_gen].loc[prices_p_filtered.index], label="DA+PPA")
    plt.axhline(y=d.data.agg_g.loc["DK2 Waste"].capacity_el)
    plt.legend()
    plt.show()

    import pandas as pd
    tot_vre_ppa_disp_d = pd.concat([d.results_dict[tech][PPA_zone] for tech in ['solar_pv_offer_sol', 'wind_onshore_offer_sol','wind_offshore_offer_sol']], axis=1).sum(axis=1)
    tot_vre_ppa_disp_dp = pd.concat([d_p.results_dict[tech][PPA_zone] for tech in ['solar_pv_offer_sol', 'wind_onshore_offer_sol','wind_offshore_offer_sol', 'solar_pv_PaP_offer_sol', 'wind_onshore_PaP_offer_sol','wind_offshore_PaP_offer_sol']], axis=1).sum(axis=1)
    plt.scatter(x=prices_p_filtered.index, y=tot_vre_ppa_disp_d[prices_p_filtered.index], label="DA")
    plt.scatter(x=prices_p_filtered.index, y=tot_vre_ppa_disp_dp[prices_p_filtered.index], label="DA+PPA")
    plt.legend()
    plt.show()