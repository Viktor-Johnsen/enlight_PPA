"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
#%%
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Module imports
from enlight_PPA.data_ops import PaP2DA, BL2DA
from enlight_PPA.models.energy_model import ADJUST_FLEX, ADJUST_TRANS  # relevant for EnlightModel
from enlight_PPA.utils.nbs_utils import plot_market_clearing_outcome, load_plot_configs, prettify_subplots, unify_palette_cyclers
from enlight_PPA.runners import EnlightRunner, NBSRunner


if __name__ == "__main__":
    t0 = time.time()
    load_plot_configs()
    scenario_name = "scenario_1"

    # FIRST: 1) simulate a PPA negotiation between a Producer and a Buyer
    # Setup hyperparameters
    PPA_profile = "BL"
    PPA_zone = "DELU"

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    BL_compliance_rate=0.0,
                    PPA_zone=PPA_zone,
                    # For a specific Producer/Buyer: P_vre, x_pv, x_wind_on, x_wind_off, y_batt, x_buyer
                    # On a zonal level: x_tot_Z...
                    x_tot_Z=0.9999,  # use 99.99% of the total zonal VRE (Hydro ror excl.) capacity
                    S_UB=40,
    )

    # For good measure, check that the scenarios align. If not, uncomment and run the first two lines.
    for da_obj in nbs_runner.da_data_dict.values():
        print(da_obj.bidding_zones)

    beta_O_list = [0.1]
    beta_D_list = [0.5]
    beta_O_chosen = beta_O_list[0]
    beta_D_chosen = beta_D_list[0]
    nbs_runner.mult_nbs(beta_O_list, beta_D_list)
    ppa_model=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen]
    nbs_runner.ppa_calcs_dict["scenario_1"].visualize_inputs(plot_hours=(90*24,90*24+168))

    if ppa_model.PPA_profile == 'BL':
        print(f"PPA price: {ppa_model.S.X:.2f} €/MWh, volume: {ppa_model.M.X * nbs_runner.P_fore_w.max():.2f} MW ({ppa_model.M.X:.2f} of maximum forecast: {nbs_runner.P_fore_w.max():.2f} MW)")
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
    #%%
    # Create an instance of PaP2DA to prepare for usage in EnlightRunner
    if PPA_profile == "BL":
        bl2da = BL2DA(
            z=PPA_zone,
            compl_rate=0.6,
            s=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].S.X,
            m=nbs_runner.mult_nbs_models.results_volume[beta_O_chosen][beta_D_chosen],  # M.X is normalized!
            solar_pv_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].solar_pv_el_cap,
            on_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].on_wind_el_cap,
            off_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].off_wind_el_cap,
            P_batt=nbs_runner.ppa_calcs_dict["scenario_1"].P_batt,
            E_batt=nbs_runner.ppa_calcs_dict["scenario_1"].E_batt,
        )
    elif PPA_profile == "PaP":
        pap2da = PaP2DA(
            z=PPA_zone,
            s=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].S.X,
            gamma=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].gamma.X,
            solar_pv_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].solar_pv_el_cap,
            on_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].on_wind_el_cap,
            off_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].off_wind_el_cap,
        )
    else:
        raise Exception("KeyError: try either BL or PaP PPA profiles. A PaF won't cause any distortion and is meaningless to include in the DA market.")
    print(f"NBS took {time.time()-t0:.2f} s.")
    
    # SECOND: 1) Run DA market model with the PPA
    if PPA_profile == "PaP":
        da_runner_ppa = EnlightRunner()
        da_runner_ppa.prepare_data_single_scenario(scenario_name=scenario_name)
        da_runner_ppa.load_data_single_simulation(scenario_name=scenario_name)
        if PPA_profile == "BL":
            da_runner_ppa.run_single_simulation(scenario_name=scenario_name + "/PPA", PPA2DA=bl2da)
        elif PPA_profile == "PaP":
            da_runner_ppa.run_single_simulation(scenario_name=scenario_name + "/PPA", PPA2DA=pap2da)
        d_p = da_runner_ppa.enlight_model

        # Verify social welfare calculations
        print(f"{d_p.results_econ['social welfare']/1e9:.6f} b.€")
        print(f"{d_p.results_econ['social welfare perceived']/1e9:.6f} b.€")
        print(f"{d_p.model.objective.value/1e9:.6f} b.€")

    if PPA_profile == "BL":
        ##### RUNNING MULTIPLE DAs with different compliance rates for PPA BL #####
        bl2da_dict = {}
        d_p_dict = {}
        # crs = [0.50, 0.60, 0.70, 0.75]
        crs = [0.00, 0.9]#, 0.70, 0.85]#, 0.90]
        for cr in crs:
            print(cr)
            bl2da_dict[cr] = BL2DA(
                    z=PPA_zone,
                    compl_rate=cr,
                    s=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].S.X,
                    m=nbs_runner.mult_nbs_models.results_volume[beta_O_chosen][beta_D_chosen],  # M.X is normalized!
                    solar_pv_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].solar_pv_el_cap,
                    on_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].on_wind_el_cap,
                    off_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].off_wind_el_cap,
                    P_batt=nbs_runner.ppa_calcs_dict["scenario_1"].P_batt,
                    E_batt=nbs_runner.ppa_calcs_dict["scenario_1"].E_batt,
            )
            da_runner_ppa = EnlightRunner()
            da_runner_ppa.prepare_load_run_single_sim(scenario_name=scenario_name, results_path_optional=f"PPA_{PPA_profile}_{cr}", PPA2DA=bl2da_dict[cr])
            d_p_dict[cr] = da_runner_ppa.enlight_model
        print("Finished running multiple BL PPAs: ", crs)

        # inspect cr results
        hours = np.arange(120*24+1, 120*24+72, 1)
        # 65,   66,   94,   95,  104 in mask
        crs_opt = np.array([cr for cr in crs if d_p_dict[cr].model.status == "ok"])
        cr_ref = 0.0

        # vre_free_fore = {"solar_pv" : d_p_dict[0.00].data.solar_pv_production, "wind_on": d_p_dict[0.00].data.wind_onshore_production, "wind_off": d_p_dict[0.00].data.wind_offshore_production}
        # vre_ppa_fore = {"solar_pv": d_p_dict[0.00].solar_pv_PPA_fore, "wind_on": d_p_dict[0.00].wind_onshore_PPA_fore, "wind_off": d_p_dict[0.00].wind_offshore_PPA_fore}
        # vre_avail = {k: vre_free_fore[k] + vre_ppa_fore[k] for k in vre_free_fore.keys()}
        # vre_free_prod = {"solar_pv" : d_p_dict[0.00].solar_pv_offer.sol, "wind_on": d_p_dict[0.00].wind_onshore_offer.sol, "wind_off": d_p_dict[0.00].wind_offshore_offer.sol}
        # vre_bl_prod = {"solar_pv" : d_p_dict[0.00].solar_pv_BL_offer.sol, "wind_on": d_p_dict[0.00].wind_onshore_BL_offer.sol, "wind_off": d_p_dict[0.00].wind_offshore_BL_offer.sol}
        # vre_prod = {k: vre_free_fore[k] + vre_ppa_fore[k] for k in vre_free_fore.keys()}
        #vre_curt = vre_avail - vre_prod
        #vre_curt_norm = vre_curt/vre_avail.max()
        #vre_curt_norm
        fig1, ax1 = plt.subplots(
            len(crs_opt),
            figsize=(12, 6),
            constrained_layout=True,
        )
        fig2, ax2 = plt.subplots(
            len(crs_opt),
            figsize=(12, 6),
            constrained_layout=True,
        )
        ax1 = ax1.flatten()
        ax2 = ax2.flatten()
        ax2_2 = np.empty(len(ax2), dtype=object)
        unify_palette_cyclers(ax1)
        # unify_palette_cyclers(ax2)

        for i, cr in enumerate(crs_opt):
            # fig1:
            vres_BL_fore = (d_p_dict[cr].solar_pv_PPA_fore[PPA_zone] + d_p_dict[cr].wind_onshore_PPA_fore[PPA_zone] + d_p_dict[cr].wind_offshore_PPA_fore[PPA_zone]).loc[hours]
            vres_BL = (d_p_dict[cr].solar_pv_BL_offer.sol + d_p_dict[cr].wind_onshore_BL_offer.sol + d_p_dict[cr].wind_offshore_BL_offer.sol).sel(Z=PPA_zone).loc[hours].to_pandas()
            inflex = d_p_dict[cr].data.demand_inflexible_classic.loc[hours][PPA_zone]
            ppa_disp = d_p_dict[cr].total_BL_offer.sol.sel(Z=PPA_zone).loc[hours].to_pandas()
            bess_BL_ch = d_p_dict[cr].bess_units_BL_ch.sol.sel(Z=PPA_zone).loc[hours].to_pandas()
            bess_BL_dch = d_p_dict[cr].bess_units_BL_dch.sol.sel(Z=PPA_zone).loc[hours].to_pandas()
            direct_vres_BL = ppa_disp - bess_BL_dch

            
            ax1[i].fill_between(hours, 0, direct_vres_BL, label=f"PPA VRE: {cr}", alpha=.7)
            ax1[i].fill_between(hours, ppa_disp, ppa_disp + bess_BL_ch, label=f"PPA ch BESS: {cr}", alpha=.7)
            ax1[i].fill_between(hours, direct_vres_BL, ppa_disp, label=f"PPA BESS: {cr}", alpha=.7)
            vres_BL_fore.plot(ax=ax1[i], label="PPA VRE fores", lw=2)
            # vres_BL.plot(ax=ax1[i], label=f"PPA VREs", ls='-.')
            ppa_disp.plot(ax=ax1[i], label=f"PPA dispatch: {cr}", ls='-.', lw=2)
            inflex.plot(ax=ax1[i], label=f"Inflex", ls='-.')
            ax1[i].axhline(bl2da.m, xmin=0, xmax=1, label="M", alpha=.5, ls="--")

            # fig2:
            ax2_2[i] = ax2[i].twinx()
            el_export = d_p_dict[cr].electricity_export.sol.sel(Z=PPA_zone).loc[hours]
            Import = (-el_export.where(el_export < 0, 0.0)).to_pandas()
            Export = el_export.where(el_export > 0, 0.0).to_pandas()
            
            Import.plot(ax=ax2[i], label=f"Import {cr}")
            Export.plot(ax=ax2[i], label=f"Export {cr}")
            d_p_dict[cr].results_dict['electricity_prices'][PPA_zone].loc[hours].plot(ax=ax2_2[i], ls="--", label=f"DA prices -- {PPA_zone}")
            d_p_dict[cr].results_dict['electricity_prices']["DK1"].loc[hours].plot(ax=ax2_2[i], ls="--", label="DA prices -- DK1")
            # Add legend
            h1, l1 = ax2[i].get_legend_handles_labels()
            h2, l2 = ax2_2[i].get_legend_handles_labels()
            ax2[i].legend(
                h1 + h2,
                l1 + l2,
                loc="upper left",
                frameon=False,
                bbox_to_anchor=(1.05, 1.0)
            )
            #######
        

        prettify_subplots(ax1)  # [0, 2]
        prettify_subplots(ax2, legend=False)  # [0, 2]
        prettify_subplots(ax2_2, legend=False, grid=False)
        plt.show()

        # prettify_subplots(ax[np.arange(len(ax)) % 2 != 0], legend=False)#, grid=False) # [1, 3]
        # prettify_subplots(ax[:,1])
        # prettify_subplots(ax[:,2])
        # plt.tight_layout()
        # plt.show()
        # cr = 0.00
        # d_p_dict[cr].bess_units_BL_dch.sol.sel(Z=PPA_zone).loc[:168]
        
        fig, ax = plt.subplots(figsize=(12,6))
        param = 'electricity_prices' #'electricity_export_sol'
        for cr in crs_opt:
            y = d_p_dict[cr].results_dict[param][PPA_zone].sort_values()[::-1].values
            ax.plot(y, ls="-", label=f"{PPA_zone} {param} -- {cr}")
            for z in d_p_dict[0.00].bidding_zones:
                cp_inflex = (d_p_dict[cr].results_dict['electricity_prices'][z] * d_p_dict[cr].data.demand_inflexible_classic[z]).sum(axis=0) / d_p_dict[cr].data.demand_inflexible_classic[z].sum(axis=0)
                print(f"Inflex load capture price ({z}, {cr}): {cp_inflex:.2f} €/MWh")
        prettify_subplots(ax)
        plt.show()

        offers_cr = {}
        bids_cr = {}
        ##### PRODUCED STACKED LINE CHARTS
        for cr in crs_opt:
            offers_cr[cr], bids_cr[cr] = plot_market_clearing_outcome(dp=d_p_dict[cr], Z=PPA_zone, t=hours, bl2da=bl2da_dict[cr])

    if PPA_profile == "PaP":
        # SECOND: 2) Run DA market model without the PaP
        da_runner = EnlightRunner()
        da_runner.prepare_data_single_scenario(scenario_name=scenario_name)
        da_runner.load_data_single_simulation(scenario_name=scenario_name)
        da_runner.run_single_simulation(scenario_name=scenario_name)
        d = da_runner.enlight_model

        # Verify social welfare calculations
        print(f"{d.results_econ['social welfare']/1e9:.6f} b.€")
        print(f"{d.results_econ['social welfare perceived']/1e9:.6f} b.€")
        print(f"{d.model.objective.value/1e9:.6f} b.€")

        print(f"RAN THE MODELS!: PPA profile and zone: {PPA_profile} in {PPA_zone}.")

        # SECOND: 2) COMPARE DA w/w.o. PaP
        # Visualize hours with price change
        mask_prices = ~np.isclose(d.results_dict['electricity_prices'][PPA_zone], d_p.results_dict['electricity_prices'][PPA_zone])
        fig, ax = plt.subplots(figsize=(12,6))
        x = d.results_dict['electricity_prices'][PPA_zone][mask_prices].index
        ax.scatter(x=x, y=d.results_dict['electricity_prices'][PPA_zone][mask_prices], label="DA")
        ax.scatter(x=x, y=d_p.results_dict['electricity_prices'][PPA_zone][mask_prices], label="w/ PaP", alpha=0.5)
        ax.legend()
        ax.set_title(f"{PPA_zone}: Power prices before and after", loc='left')
        prettify_subplots(ax)
        ax.set_ylabel("DA price [€/MWh]")
        ax.set_xlabel("Hour of year [h]")
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
        ax.set_title(f"{PPA_zone}: Power prices before and after", loc='left')
        ax.set_ylabel("DA price [€/MWh]")
        ax.set_xlabel("Hour of year [h]")
        prettify_subplots(ax)
        plt.show()

        # PLOT electricity prices II: price-duration curves
        fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(d.results_dict['electricity_prices'][PPA_zone].sort_values()[::-1].values, label="DA")
        ax.plot(d_p.results_dict['electricity_prices'][PPA_zone].sort_values()[::-1].values, label="w/ PaP")
        ax.set_title(f"{PPA_zone}: Power prices before and after", loc='left')
        ax.set_ylabel("DA price [€/MWh]")
        ax.set_xlabel("Hour of year [h]")
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
                    * (d_p.results_dict['electricity_prices'] - d_p.PPA2DA.s)
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
        ax.set_ylabel("Profits [b.€]")
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
        ax.set_ylabel("Profits [b.€]")
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
        # e.g. for DELU: 'BE-DELU', 'BE-NL', 'DELU-NL', 'DK1-DELU', 'DK1-DK2', 'DK1-NL' 'DK2-DELU' -->  0: lower, 2: upper, 3: lower, 6: lower
        lines = d_p.lineflow.coords["L"].values
        lines_to = [str(l) for l in lines if l.endswith(PPA_zone)]
        lines_from = [str(l) for l in lines if l.startswith(PPA_zone)]
        trans_to = -d_p.lineflow.lower.sel(L=lines_to).min(dim="T").sum().item()
        trans_from = d_p.lineflow.upper.sel(L=lines_from).min(dim="T").item()
        trans_cap = trans_to + trans_from

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
        vre_ppa_fore = (d_p.solar_pv_PPA_fore + d_p.wind_onshore_PPA_fore + d_p.wind_offshore_PPA_fore)[PPA_zone]
        vre_ppa_fore.plot(ax=ax, label="VRE in PaP")#, alpha=1)

        # Plot all types of consumption
        inflex_cons.plot(ax=ax, label="inflex")#, alpha=0.9)
        if ADJUST_FLEX > 0:
            (ADJUST_FLEX*flex_cons[PPA_zone_idx] + inflex_cons).plot(ax=ax, label="incl. flex")#, alpha=0.8)
        if ADJUST_FLEX > 0:
            (ADJUST_FLEX*(stor_cons[PPA_zone] + flex_cons[PPA_zone_idx]) + inflex_cons).plot(ax=ax, label="incl. flex+stor")#, alpha=0.7)
        (ADJUST_TRANS * trans_cap + ADJUST_FLEX * (stor_cons[PPA_zone] + flex_cons[PPA_zone_idx]) + inflex_cons).plot(ax=ax, label=f"{"flex+stor" if ADJUST_FLEX>0 else "."}{"+ trans" if ADJUST_TRANS>0 else "."}")#, alpha=0.6)

        d_p.data.hydro_ror_production[PPA_zone].plot(ax=ax, label="hydro_ror")
        ax.set_xlim(200*24,200*24+1544)
        prettify_subplots(ax)
        ax.set_title(f"Power in {PPA_zone} production under PPA compared to CONS, FLEX, and TRANS capacities", loc='left')
        ax.set_ylabel("Power [MW]")
        plt.show()

        nbs_runner.x_pv, nbs_runner.x_wind_on, nbs_runner.x_wind_off, nbs_runner.ppa_data.x_pv, nbs_runner.ppa_data.x_wind_on, nbs_runner.ppa_data.x_wind_off

        print("The gamma corresponds to the ratio of prod. as part of PaP and the total VRE prod. (excl. hydro ror):\n", np.round((vre_ppa_fore/vre_prod).mean(),5), np.round(pap2da.gamma * nbs_runner.x_tot_Z,5))
        print(f"PPA coverage is very high {np.minimum(inflex_cons,vre_ppa_fore).sum() / inflex_cons.sum()*100:.2f}")
        print(f"Similar to the PPA coverage, the % of available Producer VRE consumed by the Buyer is {np.minimum(inflex_cons,vre_ppa_fore).sum() / vre_ppa_fore.sum()*100:.2f}")
        print(f"The ratio of total Buyer cons. and Producer gen. is: {inflex_cons.sum()/vre_ppa_fore.sum()*100:.2f}%")

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

        # Analyze new dispatch of a presumed marginal generatorDK2 Waste
        prices_p = d_p.results_dict['electricity_prices']
        prices_p_filtered = prices_p[prices_p.lt(0).any(axis=1)][PPA_zone]
        marg_gen = "DK2 Waste"
        fig, ax = plt.subplots(figsize=(12,6))
        unify_palette_cyclers(ax)
        ax.scatter(x=prices_p_filtered.index, y=d.results_dict['conventional_units_offer_sol'][marg_gen].loc[prices_p_filtered.index], label="DA")
        ax.scatter(x=prices_p_filtered.index, y=d_p.results_dict['conventional_units_offer_sol'][marg_gen].loc[prices_p_filtered.index], label="DA+PPA")
        ax.axhline(y=d.data.agg_g.loc["DK2 Waste"].capacity_el, label=f"{marg_gen} max. capacity")
        prettify_subplots(ax)
        ax.set_title('Behaviour of a "new marginal generator" after the PPA implementation', loc='left')
        ax.set_ylabel("Power generation [MW]")
        ax.set_xlabel("Hour of year [h]")
        plt.show()

        fig, ax = plt.subplots(figsize=(12,6))
        tot_vre_ppa_disp_d = pd.concat([d.results_dict[tech][PPA_zone] for tech in ['solar_pv_offer_sol', 'wind_onshore_offer_sol','wind_offshore_offer_sol']], axis=1).sum(axis=1)
        tot_vre_ppa_disp_dp = pd.concat([d_p.results_dict[tech][PPA_zone] for tech in ['solar_pv_offer_sol', 'wind_onshore_offer_sol','wind_offshore_offer_sol', 'solar_pv_PaP_offer_sol', 'wind_onshore_PaP_offer_sol','wind_offshore_PaP_offer_sol']], axis=1).sum(axis=1)
        ax.scatter(x=prices_p_filtered.index, y=tot_vre_ppa_disp_d[prices_p_filtered.index], label="DA")
        ax.scatter(x=prices_p_filtered.index, y=tot_vre_ppa_disp_dp[prices_p_filtered.index], label="DA+PPA")
        prettify_subplots(ax)
        ax.set_title("Behaviour of the Producer's VRE capacity after the PPA implementation", loc='left')
        ax.set_ylabel("Power generation [MW]")
        ax.set_xlabel("Hour of year [h]")
        plt.show()

        # Inspect specific hours to understand negative prices even though total FLEX+STOR+TRANS cap is not exceeded
        # When lambda_DA = -S
        hour_range_s = range(2910, 2919)  # For PPA_zone = "DELU"
        print("Power prices:\n", d_p.results_dict['electricity_prices'].loc[hour_range_s][PPA_zone])
        print("BESS:\n", (d_p.bess_units_SOC.sol/d_p.bess_units_SOC.upper).sel(T=hour_range_s, Z=PPA_zone).values)
        print("PHS:\n", (d_p.hydro_ps_units_SOC.sol/d_p.hydro_ps_units_SOC.upper).sel(T=hour_range_s, Z=PPA_zone).values)
        print("VREs:\n", (d_p.solar_pv_PaP_offer.sol/d_p.solar_pv_PaP_offer.upper).sel(T=hour_range_s, Z=PPA_zone).values, (d_p.wind_onshore_PaP_offer.sol/d_p.wind_onshore_PaP_offer.upper).sel(T=hour_range_s, Z=PPA_zone).values, (d_p.wind_offshore_PaP_offer.sol/d_p.wind_offshore_PaP_offer.upper).sel(T=hour_range_s, Z=PPA_zone).values)
        print("Trans:\n", (d_p.lineflow.sol).sel(T=hour_range_s).dot(d_p.data.L_Z_xr.sel(Z=PPA_zone)).values/trans_cap)
        # Find hours of interest:
        # mask1 = (d_p.results_dict['electricity_prices'][PPA_zone] < 0)
        # mask2 = (d_p.results_dict['electricity_prices'][PPA_zone] > -19)
        # range(1,8761)[mask1*mask2] --> e.g. [327,  328,  329,  330,  331,  332, 333,  343,  344,  345]
        # offers_df, bids_df = plot_market_clearing_outcome(dp=d_p, Z=PPA_zone, t=hour_range_s, bl2da=None)
        # DELU Waste at 1676.6 MW fully dispatched. BESS is marginal
        # When lambda_DA < -S
            # lambda_DA = -1.629937 <-- agg_bess: offer_price - (bid_price - (-S) )/eff_rt
            # -> d_p.data.agg_bess.loc[PPA_zone].offer_price_weighted - (d_p.data.agg_bess.loc[PPA_zone].bid_price_weighted - -pap2da.s)/d_p.data.bess_charging_efficiency**2
            # Negative DA prices:
                # [-19.29153215, -19.29153215, -17.8805855 , -17.8720855 ,
                # -3.37166374,  -3.37166374,  -2.44337717,  -2.11683271,
                # -1.62993724,  -0.30196102,  -0.06720315]
                    # marginal supply, T=[326:334] <-- offers_df & power prices
                # -19.29153215, -19.29153215: PPA VRE
                    # marginal demand, T=[668:679] <-- bids_df & power prices
                # -17.8805855, -17.8720855: BESS, discharges @ 0.03, 0.04 (when solar_pv or hydro_ror is marginal supply and price-setter)
                #  -3.37166374,  -3.37166374,...
                    # marginal demand, T=[5360:5368] <-- bids_df & power prices
                # -2.44337717: PHS or BESS, discharges @ 19.607 (PHS <- when xx is marginal yyy) or 18.19 (BESS <- when DK2 Electric Boiler is marginal demand).
                    # marginal demand, T=[3898:3904] <-- bids_df & power prices
                # -2.11683271: BESS, discharges @ 18.57559 (when "DK1 Eletric Boiler" is marginal demand and price-setter) 
                    # marginal supply, T=(326:334) <-- offers_df & power prices: offer_price - (bid_price - (-S) )/eff_rt
                # -1.62993724: BESS, charged @ -S!
                #  -0.30196102
                # -0.06720315
            #  caused by marginal costs:
                # [-1.62993725, -1.62993725,  0.03      ,  0.04      , 17.09931971,
                # 17.09931971, 18.19142156, 18.57559151, 19.14840971, 20.71073468,
                # 20.98692041]
                # -1.62993725, -1.62993725: PPA VRE.
                # 0.03      ,  0.04: free solar_pv and hydro ror.
                # 18.19142156, 18.57559151: DK2/1 Electric Boilers.
                
                # Looking at negative prices:
                # mask_u_0 = (d_p.results_dict['electricity_prices'][PPA_zone] < 0)
                # mask_u_neg2 = (d_p.results_dict['electricity_prices'][PPA_zone] < -2)
                # mask_l_neg19 = (d_p.results_dict['electricity_prices'][PPA_zone] > -19)
                # d_p.results_dict['electricity_prices'][PPA_zone][mask_u_0 * mask_u_neg2 * mask_l_neg19]
