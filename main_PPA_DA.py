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
from enlight_PPA.models.energy_model import ADJUST_FLEX, ADJUST_TRANS, ADJUST_STOR  # relevant for EnlightModel
from enlight_PPA.utils.nbs_utils import plot_market_clearing_outcome, load_plot_configs, prettify_subplots, unify_palette_cyclers, make_hour_formatter
from enlight_PPA.runners import EnlightRunner, NBSRunner

FAKE_PURE_ONWIND = False  # only for testing purposes!

if __name__ == "__main__":
    print(ADJUST_TRANS)
    t0 = time.time()
    load_plot_configs()
    scenario_name = "scenario_4"

    # FIRST: 1) simulate a PPA negotiation between a Producer and a Buyer
    # Setup hyperparameters
    PPA_profile = "BL"
    PPA_zone = "DELU"
    BL_compliance_rate = 0.0

    # Instantiate objects and load power price results from DA market model
    if FAKE_PURE_ONWIND:
        nbs_runner = NBSRunner(
            PPA_profile=PPA_profile,
            BL_compliance_rate=BL_compliance_rate,
            PPA_zone=PPA_zone,
            # For a specific Producer/Buyer: P_vre, x_pv, x_wind_on, x_wind_off, y_batt, x_buyer
            P_vre=89154,
            x_wind_on=1-2*1e-6,
            x_wind_off=1e-6,  # 0 causes an error for some reason
            x_pv=1e-6,
            # On a zonal level: x_tot_Z...
            x_buyer=0.5,
            y_batt=0.349,  # If 0: use the zonal-level P_BESS/P_VRE ratio
            batt_Crate=0.62,
            S_UB=45,
        )
    else:
        nbs_runner = NBSRunner(
            PPA_profile=PPA_profile,
            BL_compliance_rate=BL_compliance_rate,
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
    print("######\nTOTAL PtX capacity:" , nbs_runner.da_data_dict["scenario_4"].agg_ptx.groupby("zone_el").sum().capacity_el.sum(), "\n#######")

    beta_O_list = [0.4]
    beta_D_list = [0.4]
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
            compl_rate=BL_compliance_rate,
            s=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].S.X,
            m=nbs_runner.mult_nbs_models.results_volume[beta_O_chosen][beta_D_chosen],  # M.X is normalized!
            solar_pv_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].solar_pv_el_cap,  # MW...
            on_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].on_wind_el_cap,
            off_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].off_wind_el_cap, # MW...
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
    #%% 
    da_runner_ppa = EnlightRunner()
    # da_runner_ppa.prepare_data_single_scenario(scenario_name=scenario_name)
    da_runner_ppa.load_data_single_simulation(scenario_name=scenario_name)
    if PPA_profile == "BL":
        da_runner_ppa.run_single_simulation(scenario_name=scenario_name + "/PPA", PPA2DA=bl2da)
    elif PPA_profile == "PaP":
        da_runner_ppa.run_single_simulation(scenario_name=scenario_name + "/PPA", PPA2DA=pap2da)
    d_p = da_runner_ppa.enlight_model

    # Verify social welfare calculations
    print(f"{d_p.results_econ['social welfare']/1e9:.6f} b.€")
    print(f"{d_p.results_econ['social welfare perceived']/1e9:.6f} b.€ (if it differs from below, check slacks in obj!)")
    print(f"{d_p.model.objective.value/1e9:.6f} b.€")


    #%%

    if PPA_profile == "BL":
        ##### RUNNING MULTIPLE DAs with different compliance rates for PPA BL #####
        bl2da_dict = {}
        d_p_dict = {}
        # crs = [0.50, 0.60, 0.70, 0.75]
        crs = [0.816] # DELU, full VRE: 0.816. DELU, onwind: 0.724, full VRE 2040PtX: 0.930
        for cr in crs:
            print(cr)
            bl2da_dict[cr] = BL2DA(
                    z=PPA_zone,
                    compl_rate=cr,
                    s=nbs_runner.mult_nbs_models.models[beta_O_chosen][beta_D_chosen].S.X,
                    m=nbs_runner.mult_nbs_models.results_volume[beta_O_chosen][beta_D_chosen],  # M.X is normalized!
                    solar_pv_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].solar_pv_el_cap,  # MW...
                    on_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].on_wind_el_cap,
                    off_wind_el_cap=nbs_runner.ppa_calcs_dict["scenario_1"].off_wind_el_cap, # MW...
                    P_batt=nbs_runner.ppa_calcs_dict["scenario_1"].P_batt,
                    E_batt=nbs_runner.ppa_calcs_dict["scenario_1"].E_batt,
                )
            da_runner_ppa = EnlightRunner()
            da_runner_ppa.prepare_load_run_single_sim(scenario_name=scenario_name, results_path_optional=f"PPA_{PPA_profile}_{cr}", PPA2DA=bl2da_dict[cr])
            d_p_dict[cr] = da_runner_ppa.enlight_model
        print("Finished running multiple BL PPAs: ", crs)

        # also include the base case with 0% compliance rate (the original run above)
        if len(crs) == 1:
            crs.insert(0, BL_compliance_rate)
            bl2da_dict[BL_compliance_rate] = bl2da
            d_p_dict[BL_compliance_rate] = d_p

        # inspect cr results
        year = d_p_dict[crs[0]].data.solar_weather_year  # for axis formatter

        hours = np.arange(120*24+1, 120*24+168, 1)
        hours = np.arange(500, 668, 1)  # for quick testing of pure ONWIND use of BESS
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
            sharex=True,
        )
        fig2, ax2 = plt.subplots(
            len(crs_opt),
            figsize=(12, 6),
            constrained_layout=True,
            sharex=True,
        )
        ax1 = ax1.flatten()
        ax2 = ax2.flatten()
        ax2_2 = np.empty(len(ax2), dtype=object)
        unify_palette_cyclers(ax1)
        # unify_palette_cyclers(ax2)

        observed_PPA_complr = {}
        for i, cr in enumerate(crs_opt):
            # fig1:
            observed_PPA_complr[cr] = np.minimum(d_p_dict[cr].total_BL_offer.sol, d_p_dict[cr].PPA2DA.m).sum().item()/(d_p_dict[cr].PPA2DA.m * d_p_dict[cr].T)
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
            d_p_dict[cr].results_dict['electricity_prices'][PPA_zone].loc[hours].plot(ax=ax2_2[i], ls="--", label=f"{PPA_zone} spot price")
            # d_p_dict[cr].results_dict['electricity_prices']["DK1"].loc[hours].plot(ax=ax2_2[i], ls="--", label="DA prices -- DK1")
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
        ax1[0].set_title(f"{PPA_zone}: PPA VRE production and BESS operation under different ENFORCED compliance rates.\nCompl. rates are {crs_opt[0]:.3f}|{observed_PPA_complr[crs_opt[0]]:.3f} (enforced|observed)\nPower [MW]", loc='left')
        ax1[1].set_title(f"Compl. rates are {crs_opt[-1]:.3f}|{observed_PPA_complr[crs_opt[-1]]:.3f} (enforced|observed)\nPower [MW]", loc='left')
        ax1[1].xaxis.set_major_formatter(make_hour_formatter(year=year))
        ax1[1].tick_params(axis='x', rotation=15)
        ax1[1].set_xlabel("")  # remove automatic "Time"
        prettify_subplots(ax2, legend=False)  # [0, 2]
        ax2[0].set_title(f"{PPA_zone}: Import/Export under different ENFORCED compliance rates.\nCompl. rates are {crs_opt[0]:.3f}|{observed_PPA_complr[crs_opt[0]]:.3f} (enforced|observed)\nPower [MW]", loc='left')
        ax2[1].set_title(f"Compl. rates are {crs_opt[-1]:.3f}|{observed_PPA_complr[crs_opt[-1]]:.3f} (enforced|observed)\nPower [MW]", loc='left')
        ax2_2[0].set_title("Spot price [€/MWh] ", loc='right')
        ax2_2[1].set_title("Spot price [€/MWh] ", loc='right')
        ax2[1].xaxis.set_major_formatter(make_hour_formatter(year=year))
        ax2[1].tick_params(axis='x', rotation=15)
        ax2[1].set_xlabel("")  # remove automatic "Time"
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
        pdc_labels=[f"DA ({cr_ref})", f"w/ BL ({crs_opt[-1]:.3f})"]
        pdc = {}
        for i, cr in enumerate(crs_opt):
            pdc[cr] = d_p_dict[cr].results_dict[param][PPA_zone].sort_values()[::-1]
            y = pdc[cr].values
            ax.plot(y, ls="-", label=pdc_labels[i])
        prettify_subplots(ax)
        ax.set_xlabel("Hours [h]")
        ax.set_title(f"{PPA_zone}: Price-duration curves before and after {PPA_profile+(" (single-VRE)" if FAKE_PURE_ONWIND else "")}.\nSpot price [€/MWh]", loc='left')
        plt.show()

        for z in d_p_dict[0.00].bidding_zones:
            cp_inflex = {}
            for cr in crs_opt:
                cp_inflex[cr] = (d_p_dict[cr].results_dict['electricity_prices'][z] * d_p_dict[cr].data.demand_inflexible_classic[z]).sum(axis=0) / d_p_dict[cr].data.demand_inflexible_classic[z].sum(axis=0)
            print(f"Inflex load CP in {z}:({crs_opt[0]}) {cp_inflex[crs_opt[0]]:.2f} -- ({crs_opt[-1]}) {cp_inflex[crs_opt[-1]]:.2f} €/MWh")

        # Using PDC order to produce LDCs for the flexible bids
        units_mapping = {
                    'conventional_units': d_p_dict[cr_ref].data.G_Z_df,
                    'dh_units': d_p_dict[cr_ref].data.L_DH_Z_df,
                    'hydro_res_units': d_p_dict[cr_ref].data.G_hydro_res_Z_df,
                    'ptx_units': d_p_dict[cr_ref].data.L_PtX_Z_df
        }
        bid_types = [
            # 'demand_inflexible_classic_bid',  # not interesting in this context... it's always active.
            'demand_flexible_classic',  # flex_dem
            'ptx_units',  # flex_dem
            'dh_units',  # flex_dem
            'bess_units',  # stor_dem
            'hydro_ps_units',  # stor_dem
        ]
        bid_stors = [  # stors are already aggregated on zonal level
            'bess_units', 
            'hydro_ps_units',
        ]
        # BESS and PtX present in many zones, but check for DH and PHS:
            # Check if any DH units exist in the zone...
        if PPA_zone not in np.unique(d_p.data.dh_units_df.zone_el):
            bid_types.remove('dh_units')
        if PPA_zone not in np.unique(d_p.data.hydro_ps_units.zone_el):
            bid_types.remove('hydro_ps_units')
        
        fig, axs = plt.subplots(len(bid_types), figsize=(14, 10), sharex=True)
        for cr in crs_opt:
            # cr = 0.0
            for i, bid in enumerate(bid_types):
                x = np.arange(len(pdc[cr].index))
                if ('units' in bid and not bid in bid_stors):
                    y_units = d_p_dict[cr].results_dict[bid+"_bid_sol"].loc[pdc[cr].index]  # by units
                    y_z = y_units.dot(units_mapping[bid])[PPA_zone].values  # aggregate to zonal level
                else:  # flex classic
                    y_z = d_p_dict[cr].results_dict[bid+"_bid_sol"][PPA_zone].loc[pdc[cr].index].values
                # plot the (aggregated) zonal consumption in the order of the PDC
                axs[i].scatter(x=x, y=y_z, label=f"{bid} cr={cr}", alpha=0.1)
            for ax in axs:
                prettify_subplots(ax)
            axs[-1].set_xlabel("Hours [h]")
            axs[0].set_title(f"{PPA_zone}: Flexible load-duration curves based on zonal price-duration curve\nPower consumption[MW]", loc='left')
            plt.show()

        ##### PRODUCED STACKED LINE CHARTS
        offers_cr = {}
        bids_cr = {}
        for cr in crs_opt:
            offers_cr[cr], bids_cr[cr] = plot_market_clearing_outcome(dp=d_p_dict[cr], Z=PPA_zone, t=hours, bl2da=bl2da_dict[cr])

        fores = ["solar_pv_PPA_fore", "wind_onshore_PPA_fore", "wind_offshore_PPA_fore", "solar_pv_production", "wind_onshore_production", "wind_offshore_production"]
        vars = ["solar_pv_BL_offer", "wind_onshore_BL_offer", "wind_offshore_BL_offer", "solar_pv_offer", "wind_onshore_offer", "wind_offshore_offer"]

        curt_z = {}  # only for the chosen PPA zone
        curt_zonal = {}  # by zone
        curt_tot = {}  # total across zones
        for cr in crs_opt:
            curt_z[cr] = {}
            curt_zonal[cr] = {}
            curt_tot[cr] = {}
            for f, v in zip(fores, vars):
                k = v.removesuffix("_offer")
                curt_zonal[cr][k] = (getattr(d_p_dict[cr], f) - getattr(d_p_dict[cr], v).sol.to_pandas()).sum(axis=0)  # forecast - dispatch
                curt_tot[cr][k] = curt_zonal[cr][k].sum()
                curt_z[cr][k] = curt_zonal[cr][k].loc[PPA_zone]
            print(f"VRE (excl. hydro ror) for cr={cr}: {sum(curt_tot[cr].values())/1e6:.2f} TWh")
            print(f"In {PPA_zone}: {sum(curt_z[cr].values())/1e6:.2f} TWh")

        # Base (non-PaP) keys only
        keys = sorted(k for k in set(d_p_dict[cr_ref].results_econ['profits']) if not k.endswith('_PaP'))

        vals_d = []
        vals_dp_base = []
        vals_dp_pap = []

        for k in keys:
            if k in units_mapping:
                vals_d.append(
                    d_p_dict[cr_ref].results_econ['profits'].get(k, np.nan)
                    .dot(units_mapping[k]).loc[PPA_zone] / 1e9
                )

                base = (
                    d_p_dict[crs_opt[-1]].results_econ['profits']
                    .get(k, 0.0).dot(units_mapping[k]).loc[PPA_zone]
                )
                pap = d_p_dict[crs_opt[-1]].results_econ['profits'].get(f"{k}_PaP", 0.0)
                if not isinstance(pap, float):
                    pap = pap.loc[PPA_zone]

                vals_dp_base.append(base / 1e9)
                vals_dp_pap.append(pap / 1e9)

            elif k == 'demand_inflexible_classic':
                vals_d.append(
                        - (d_p_dict[cr_ref].results_dict['demand_inflexible_classic_bid_sol']
                           * d_p_dict[cr_ref].results_dict['electricity_prices']
                           ).sum(axis=0).loc[PPA_zone] / 1e9
                )

                base = (
                    - d_p_dict[crs_opt[-1]].results_dict['demand_inflexible_classic_bid_sol']
                    * d_p_dict[crs_opt[-1]].results_dict['electricity_prices']
                ).sum(axis=0)[PPA_zone]

                pap = (
                    d_p_dict[crs_opt[-1]].PPA2DA.m
                    * (d_p_dict[crs_opt[-1]].results_dict['electricity_prices']
                    - d_p_dict[crs_opt[-1]].PPA2DA.s)
                ).sum(axis=0)[PPA_zone]

                vals_dp_base.append(base / 1e9)
                vals_dp_pap.append(pap / 1e9)

            elif k == 'lineflow (CR)':
                df_CR = d_p_dict[cr_ref].results_econ['profits']['lineflow (CR)']
                vals_d.append(
                    sum(df_CR.loc[line] for line in df_CR.index if PPA_zone in line.split("-")) / 1e9
                )

                df_CR_p = d_p_dict[crs_opt[-1]].results_econ['profits']['lineflow (CR)']
                base = sum(df_CR_p.loc[line] for line in df_CR_p.index if PPA_zone in line.split("-"))

                vals_dp_base.append(base / 1e9)
                vals_dp_pap.append(0.0)

            elif k == "total_BL":
                vals_d.append(
                    d_p_dict[cr_ref].results_econ['profits_sw'][k + "_offer_sol"].loc[PPA_zone] / 1e9
                )

                base = (
                    d_p_dict[crs_opt[-1]].results_econ['profits_sw'][k + "_offer_sol"].loc[PPA_zone]
                )
                pap = d_p_dict[crs_opt[-1]].BL_net_settlement

                vals_dp_base.append(base / 1e9)
                vals_dp_pap.append(pap / 1e9)

            else:
                vals_d.append(
                    d_p_dict[cr_ref].results_econ['profits'][k].loc[PPA_zone] / 1e9
                )

                base = d_p_dict[crs_opt[-1]].results_econ['profits'].get(k, 0.0).loc[PPA_zone]
                pap = d_p_dict[crs_opt[-1]].results_econ['profits'].get(f"{k}_PaP", 0.0)
                if not isinstance(pap, float):
                    pap = pap.loc[PPA_zone]

                vals_dp_base.append(base / 1e9)
                vals_dp_pap.append(pap / 1e9)

        # set inflex demand to 0 to get a better look at the other bars
        dem_infl_idx = np.arange(len(keys))[np.array(keys)=='demand_inflexible_classic'][0]
        vals_d3 = vals_d.copy()
        vals_dp_base3 = vals_dp_base.copy()
        vals_dp_pap3 = vals_dp_pap.copy()
        vals_d3[dem_infl_idx] = 0
        vals_dp_base3[dem_infl_idx] = 0
        vals_dp_pap3[dem_infl_idx] = 0

        x = np.arange(len(keys))
        width = 0.35

        # Horizontal bar plots
        order = np.argsort(vals_d)

        keys_s      = np.array(keys)[order]
        vals_d_s    = np.array(vals_d)[order]
        vals_dp_b_s = np.array(vals_dp_base)[order]
        vals_dp_p_s = np.array(vals_dp_pap)[order]

        y = np.arange(len(keys_s))
        h = 0.25

        fig, ax = plt.subplots(1, 3, figsize=(14, 8), sharey=True, width_ratios=[1, 0.1, 1])

        # Plot 1
        ax[0].barh(y - h, vals_d_s, height=h, label="DA")
        ax[0].barh(y,     vals_dp_b_s, height=h, label="DA with PPA")
        ax[0].barh(y,     vals_dp_p_s, height=h, left=vals_dp_b_s, alpha=0.5, label="PPA part")

        # Plot 2 (inflex = 0, same order)
        ax[2].barh(y - h, np.array(vals_d3)[order], height=h)
        ax[2].barh(y,     np.array(vals_dp_base3)[order], height=h)
        ax[2].barh(y,     np.array(vals_dp_pap3)[order],
                height=h, alpha=0.5, left=np.array(vals_dp_base3)[order])

        # Prettifying + legend
        ax[0].set_yticks(y, keys_s)
        ax[0].invert_yaxis()
        prettify_subplots(ax[0], legend=False)
        prettify_subplots(ax[2], legend=False)
        handles, labels = ax[0].get_legend_handles_labels()
        ax[1].legend(handles, labels, loc="upper center", frameon=False)
        ax[1].axis("off")
        ax[0].set_xlabel("Payoff [b.€]")
        ax[2].set_xlabel("Payoff [b.€]")
        ax[0].set_title(fr"{PPA_zone}: Individual PS and CS by type -- {PPA_profile} ($M=${d_p.PPA2DA.m/1e3:.1f} GW, $S=${d_p.PPA2DA.s:.2f} €/MWh)", loc="left")
        ax[2].set_title(f"REPEATED without {keys_s[0]}", loc="left")

        plt.show()

        def get_cr_diff(var_name, crs, *, t=[0, 0.5], z="DELU"):
            var_diff = (d_p_dict[crs[1]].results_dict[var_name+"_sol"]-d_p_dict[crs[0]].results_dict[var_name+"_sol"])[PPA_zone].loc[hours]
            return var_diff
        var_names_diff = ['bess_units_bid', 'electricity_export', 'total_BL_offer']

        fig, ax = plt.subplots(2, 1, figsize=(14, 10))
        for var_name in var_names_diff:
            var_diff = get_cr_diff(var_name, crs_opt, t=hours[:-1], z=PPA_zone)
            ax[0].plot(var_diff, label=var_name)
        for cr in crs_opt:
            soc_frac = (d_p_dict[cr].bess_units_SOC.sol/d_p_dict[cr].bess_units_SOC.upper).sel(T=hours[1:], Z=PPA_zone)
            ax[1].plot(soc_frac, label=f"soc_frac: cr={cr:.2f}")
        prettify_subplots(ax)
        plt.show()

        print("Amount of times that the system BESS reaches maximum SOC")
        for cr in crs_opt:
            print(cr, np.isclose(d_p_dict[cr].bess_units_SOC.sol.sel(Z=PPA_zone).values, d_p_dict[cr].bess_units_SOC.upper.sel(Z=PPA_zone).max().item()).sum())
        for cr in crs_opt:
            print(cr, np.isclose(d_p_dict[cr].bess_units_SOC.sol.sel(Z=PPA_zone).values, d_p_dict[0.0].bess_units_SOC.upper.sel(Z=PPA_zone).max().item()).sum())

        print("Amount of charging cycles that the system BESS performs")
        for cr in crs_opt:
            print(cr, d_p_dict[cr].bess_units_bid.sol.sel(Z=PPA_zone).sum().item() * d_p_dict[0.0].data.bess_charging_efficiency / d_p_dict[0.0].bess_units_SOC.upper.sel(Z=PPA_zone).max().item())

        for cr in crs_opt:
            print(f"{d_p_dict[cr].results_econ['social welfare']/1e9:.6f} b.€")
            print(f"{d_p_dict[cr].results_econ['social welfare perceived']/1e9:.6f} b.€ (if it differs from below, check slacks in obj!)")
            print(f"{d_p_dict[cr].model.objective.value/1e9:.6f} b.€")

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
        year = d_p.data.solar_weather_year  # for axis formatter

        mask_prices = ~np.isclose(d.results_dict['electricity_prices'][PPA_zone], d_p.results_dict['electricity_prices'][PPA_zone])
        fig, ax = plt.subplots(figsize=(12,6))
        x = d.results_dict['electricity_prices'][PPA_zone][mask_prices].index
        ax.scatter(x=x, y=d.results_dict['electricity_prices'][PPA_zone][mask_prices], label="DA")
        ax.scatter(x=x, y=d_p.results_dict['electricity_prices'][PPA_zone][mask_prices], label="w/ PaP", alpha=0.5)
        ax.legend()
        ax.set_title(f"{PPA_zone}: Power prices before and after {PPA_profile}\nSpot price [€/MWh]", loc='left')
        prettify_subplots(ax)
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(make_hour_formatter(year=year))
        ax.tick_params(axis='x', rotation=15)
        ax.set_xlabel("")
        plt.show()

        # Print the number of changes for each decision variable
        for k in list(d.results_dict.keys()):
            num_cols = len(d.results_dict[k].columns)
            print(f"Variable {k} changed: {(~np.isclose(d.results_dict[k], d_p.results_dict[k])).sum()} times out of {num_cols*8760}...")


        # THIRD: Inspect results to compare DA v. DA+PAP
        # PLOT electricity prices
        fig, ax = plt.subplots(figsize=(16,8))
        d.results_dict['electricity_prices'][PPA_zone].plot(ax=ax, label="DA")
        d_p.results_dict['electricity_prices'][PPA_zone].plot(ax=ax, label="w/ PaP", ls='--')
        ax.set_title(f"{PPA_zone}: Power prices before and after {PPA_profile}\nSpot price [€/MWh]", loc='left')
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(make_hour_formatter(year=year))
        ax.tick_params(axis='x', rotation=15)
        ax.set_xlabel("")
        prettify_subplots(ax)
        plt.show()

        # PLOT electricity prices II: price-duration curves
        fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(d.results_dict['electricity_prices'][PPA_zone].sort_values()[::-1].values, label="DA")
        ax.plot(d_p.results_dict['electricity_prices'][PPA_zone].sort_values()[::-1].values, label="w/ PaP", ls='--')
        ax.set_title(f"{PPA_zone}: Price-duration curves before and after {PPA_profile}\nSpot price [€/MWh]", loc='left')
        ax.set_ylabel("")
        ax.set_xlabel("Hours [h]")
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
                vals_d.append(d.results_econ['profits'].get(k, np.nan).dot(units_mapping[k]).loc[PPA_zone] / 1e9)

                base = d_p.results_econ['profits'].get(k, 0.0).dot(units_mapping[k]).loc[PPA_zone]
                pap  = d_p.results_econ['profits'].get(f"{k}_PaP", 0.0)
                if type(pap) != float:
                    pap = pap.loc[PPA_zone]

                vals_dp_base.append(base/1e9)
                vals_dp_pap.append(pap/1e9)   
            elif k == 'demand_inflexible_classic':
                vals_d.append((d.results_econ['profits'].get(k, np.nan).loc[PPA_zone] - d.results_dict['demand_inflexible_classic_bid_sol'].sum(axis=0).loc[PPA_zone] * d.data.voll_classic)/1e9)
                base = (
                    - d_p.results_dict['demand_inflexible_classic_bid_sol'] * d_p.results_dict['electricity_prices']
                ).sum(axis=0)[PPA_zone]
                vals_dp_base.append(base/1e9)
                pap = (
                    (d_p.results_dict['solar_pv_PaP_offer_sol'] + d_p.results_dict['wind_onshore_PaP_offer_sol'] + d_p.results_dict['wind_offshore_PaP_offer_sol'])
                    * (d_p.results_dict['electricity_prices'] - d_p.PPA2DA.s)
                ).sum(axis=0)[PPA_zone]
                vals_dp_pap.append(pap/1e9)
            elif k == 'lineflow (CR)':
                df_CR = d.results_econ['profits']['lineflow (CR)']
                vals_d.append(sum(df_CR.loc[line] for line in df_CR.index if PPA_zone in line.split("-"))/1e9)

                df_CR_p = d_p.results_econ['profits']['lineflow (CR)']
                base = sum(df_CR_p.loc[line] for line in df_CR_p.index if PPA_zone in line.split("-"))
                pap = 0.0  # No PaP part for lines...
                vals_dp_base.append(base/1e9)
                vals_dp_pap.append(pap/1e9)
            else:
                vals_d.append(d.results_econ['profits'].get(k, np.nan).loc[PPA_zone]/1e9)

                # default base only counts "free" VREs
                base = d_p.results_econ['profits'].get(k, 0.0).loc[PPA_zone]
                # default pap only counts the PaP revenues -- not the missed DA revenues!
                pap  = d_p.results_econ['profits'].get(f"{k}_PaP", 0.0)
                if type(pap) != float:
                    # for solar PV, onshore wind, offshore wind:
                    pap = pap.loc[PPA_zone]
                    # print(f"PaP pap REVENUES for {k} in {PPA_zone}: {pap/1e9:.4f} b.€")
                    base += (d_p.results_dict[k+"_PaP_offer_sol"][PPA_zone]
                             * (d_p.results_dict['electricity_prices'][PPA_zone] - getattr(d_p.data, k+"_bid_price"))
                             ).sum(axis=0)
                    pap = (d_p.results_dict[k+"_PaP_offer_sol"][PPA_zone]
                             * (d_p.PPA2DA.s - d_p.results_dict['electricity_prices'][PPA_zone])
                             ).sum(axis=0)
                    # print(f"PaP base profits for {k} in {PPA_zone}: {base/1e9:.4f} b.€")
                    # print(f"PaP pap profits for {k} in {PPA_zone}: {pap/1e9:.4f} b.€")

                vals_dp_base.append(base/1e9)
                vals_dp_pap.append(pap/1e9)

        x = np.arange(len(keys))
        width = 0.35
        fig, ax = plt.subplots(figsize=(16,8))

        ax.bar(x - width/2, vals_d, width, label="DA")
        ax.bar(x + width/2, vals_dp_base, width, label="DA with PPA")
        ax.bar(x + width/2, vals_dp_pap, width, bottom=vals_dp_base, alpha=0.5, label="PPA part")

        ax.set_xticks(x, keys, rotation=90)
        ax.set_ylabel("")
        ax.set_title(fr"{PPA_zone}: Individual producer and consumer surpluses by technology/type -- before and after {PPA_profile} ($\gamma=${d_p.PPA2DA.gamma:.2f})"+"\nPayoff [b.€]", loc="left")
        prettify_subplots(ax)
        plt.show()
        #conv, dh, hres, ptx, 

        ''' Bar plots for the entire system (all zones)... not really that interesting
        # Base (non-PaP) keys only
        keys = sorted(k for k in set(d.results_econ['profits_tot']) if not k.endswith('_PaP'))

        vals_d2 = []
        vals_dp_base2 = []
        vals_dp_pap2 = []

        for k in keys:
            if k == 'demand_inflexible_classic':
                vals_d2.append(0)
                vals_dp_base2.append(0)
                vals_dp_pap2.append(0)
            else:
                vals_d2.append(d.results_econ['profits_tot'].get(k, np.nan))

                base2 = d_p.results_econ['profits_tot'].get(k, 0.0)
                pap2  = d_p.results_econ['profits_tot'].get(f"{k}_PaP", 0.0)

                vals_dp_base2.append(base2)
                vals_dp_pap2.append(pap2)

        x = np.arange(len(keys))
        width = 0.35
        fig, ax = plt.subplots(figsize=(16,8))

        ax.bar(x - width/2, vals_d2, width, label="DA")
        ax.bar(x + width/2, vals_dp_base2, width, label="DA with PPA")
        ax.bar(x + width/2, vals_dp_pap2, width, bottom=vals_dp_base2, label="PPA part")

        ax.set_xticks(x, keys, rotation=90)
        ax.set_ylabel("Profits [b.€]")
        prettify_subplots(ax)
        plt.show()
        '''
        # set inflex demand to 0 to get a better look at the other bars
        dem_infl_idx = np.arange(len(keys))[np.array(keys)=='demand_inflexible_classic'][0]
        vals_d3 = vals_d.copy()
        vals_dp_base3 = vals_dp_base.copy()
        vals_dp_pap3 = vals_dp_pap.copy()
        vals_d3[dem_infl_idx] = 0
        vals_dp_base3[dem_infl_idx] = 0
        vals_dp_pap3[dem_infl_idx] = 0

        fig, ax = plt.subplots(figsize=(16,8))
        ax.bar(x - width/2, vals_d3, width, label="DA")
        ax.bar(x + width/2, vals_dp_base3, width, label="DA with PPA")
        ax.bar(x + width/2, vals_dp_pap3, width, bottom=vals_dp_base3, alpha=0.5, label="PPA part")

        ax.set_xticks(x, keys, rotation=90)
        ax.set_ylabel("")
        ax.set_title(fr"{PPA_zone}: Individual producer and consumer surpluses by technology/type excl. inflex. demand -- before and after {PPA_profile} ($\gamma=${d_p.PPA2DA.gamma:.2f})"+"\nProfits [b.€]", loc="left")
        prettify_subplots(ax)
        plt.show()

        # Horizontal bar plots
        order = np.argsort(vals_d)

        keys_s      = np.array(keys)[order]
        vals_d_s    = np.array(vals_d)[order]
        vals_dp_b_s = np.array(vals_dp_base)[order]
        vals_dp_p_s = np.array(vals_dp_pap)[order]

        y = np.arange(len(keys_s))
        h = 0.25

        fig, ax = plt.subplots(1, 3, figsize=(14, 8), sharey=True, width_ratios=[1, 0.1, 1])

        # Plot 1
        ax[0].barh(y - h, vals_d_s, height=h, label="DA")
        ax[0].barh(y,     vals_dp_b_s, height=h, label="DA with PPA")
        ax[0].barh(y,     vals_dp_p_s, height=h, left=vals_dp_b_s, alpha=0.5, label="PPA part")

        # Plot 2 (inflex = 0, same order)
        ax[2].barh(y - h, np.array(vals_d3)[order], height=h)
        ax[2].barh(y,     np.array(vals_dp_base3)[order], height=h)
        ax[2].barh(y,     np.array(vals_dp_pap3)[order],
                height=h, alpha=0.5, left=np.array(vals_dp_base3)[order])

        # Prettifying + legend
        ax[0].set_yticks(y, keys_s)
        ax[0].invert_yaxis()
        prettify_subplots(ax[0], legend=False)
        prettify_subplots(ax[2], legend=False)
        handles, labels = ax[0].get_legend_handles_labels()
        ax[1].legend(handles, labels, loc="upper center", frameon=False)
        ax[1].axis("off")
        ax[0].set_xlabel("Payoff [b.€]")
        ax[2].set_xlabel("Payoff [b.€]")
        ax[0].set_title(fr"{PPA_zone}: Individual PS and CS by type -- {PPA_profile} ($S=${d_p.PPA2DA.s:.2f} €/MWh, $\gamma=${d_p.PPA2DA.gamma:.2f})", loc="left")
        ax[2].set_title(f"REPEATED without {keys_s[0]}", loc="left")

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
        trans_from = d_p.lineflow.upper.sel(L=lines_from).max(dim="T").sum().item()
        trans_cap = trans_to + trans_from

        # PPA zone as idx in bidding zones list
        PPA_zone_idx = np.arange(len(d_p.bidding_zones))[np.array(d_p.bidding_zones)==PPA_zone]

        # Plot for inspection
        fig, ax = plt.subplots(figsize=(16,8))
        vre_prod = (d_p.data.wind_onshore_production + d_p.data.wind_offshore_production + d_p.data.solar_pv_production)[PPA_zone]# + d_p.data.hydro_ror_production)[PPA_zone]

        # Plot the PaP VRE offered at negative prices
        vre_ppa_fore = (d_p.solar_pv_PPA_fore + d_p.wind_onshore_PPA_fore + d_p.wind_offshore_PPA_fore)[PPA_zone]
        vre_ppa_fore.plot(ax=ax, label="VRE in PaP")#, alpha=1)

        # Plot all types of consumption
        inflex_cons.plot(ax=ax, label="demand inflex. classic")#, alpha=0.9)
        if ADJUST_FLEX > 0:
            (ADJUST_FLEX*flex_cons[PPA_zone_idx] + inflex_cons).plot(ax=ax, label="+flex. demand")#, alpha=0.8)
        if ADJUST_STOR > 0:
            (ADJUST_STOR * stor_cons[PPA_zone] + ADJUST_FLEX * flex_cons[PPA_zone_idx] + inflex_cons).plot(ax=ax, label="+flex. demand & stor")#, alpha=0.7)
        (ADJUST_TRANS * trans_cap + ADJUST_STOR * stor_cons[PPA_zone] + ADJUST_FLEX * flex_cons[PPA_zone_idx] + inflex_cons).plot(ax=ax, label=f"{"+flex. demand" if ADJUST_FLEX>0 else "."}{" & stor" if ADJUST_STOR>0 else "."}{" & trans" if ADJUST_TRANS>0 else "."}")#, alpha=0.6)

        d_p.data.hydro_ror_production[PPA_zone].plot(ax=ax, label="hydro_ror")
        ax.set_xlim(200*24,200*24+1544)
        ax.set_title(fr"{PPA_zone}: Power production under {PPA_profile} (with $\gamma=${d_p.PPA2DA.gamma:.2f}) compared to bids (inflex., flex., and stor.) and transmission capacities"+"\nPower [MW]", loc='left')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(make_hour_formatter(year=year))
        ax.tick_params(axis='x', rotation=15)
        prettify_subplots(ax, bbox_list=[1.02, 0.8])
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
        if PPA_zone == "DK2":
            marg_gen = "DK2 Waste"
            fig, ax = plt.subplots(figsize=(12,6))
            unify_palette_cyclers(ax)
            ax.scatter(x=prices_p_filtered.index, y=d.results_dict['conventional_units_offer_sol'][marg_gen].loc[prices_p_filtered.index], label="DA")
            ax.scatter(x=prices_p_filtered.index, y=d_p.results_dict['conventional_units_offer_sol'][marg_gen].loc[prices_p_filtered.index], label="DA+PPA")
            ax.axhline(y=d.data.agg_g.loc["DK2 Waste"].capacity_el, label=f"{marg_gen} max. capacity")
            prettify_subplots(ax)
            ax.set_title('Behaviour of a "new marginal generator" after the PPA implementation', loc='left')
            ax.set_ylabel("Power generation [MW]")
            ax.set_xlabel("Hours [h]")
            plt.show()

        fig, ax = plt.subplots(figsize=(12,6))
        tot_vre_ppa_disp_d = pd.concat([d.results_dict[tech][PPA_zone] for tech in ['solar_pv_offer_sol', 'wind_onshore_offer_sol','wind_offshore_offer_sol']], axis=1).sum(axis=1)
        tot_vre_ppa_disp_dp = pd.concat([d_p.results_dict[tech][PPA_zone] for tech in ['solar_pv_offer_sol', 'wind_onshore_offer_sol','wind_offshore_offer_sol', 'solar_pv_PaP_offer_sol', 'wind_onshore_PaP_offer_sol','wind_offshore_PaP_offer_sol']], axis=1).sum(axis=1)
        ax.scatter(x=prices_p_filtered.index, y=tot_vre_ppa_disp_d[prices_p_filtered.index], label="DA")
        ax.scatter(x=prices_p_filtered.index, y=tot_vre_ppa_disp_dp[prices_p_filtered.index], label="DA+PPA")
        prettify_subplots(ax)
        ax.set_title("Behaviour of the Producer's VRE capacity after the PPA implementation", loc='left')
        ax.set_ylabel("Power generation [MW]")
        ax.set_xlabel("Hours [h]")
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
        # neg_vals, neg_counts = np.unique(d_p.results_dict['electricity_prices'][PPA_zone][mask1], return_counts=True)
        # (d_p.data.agg_bess.offer_price_weighted[PPA_zone] - (d_p.data.agg_bess.bid_price_weighted[PPA_zone] - (-neg_vals))/d_p.data.bess_charging_efficiency**2)
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

        # Plot cross-zonal VRE correlations
        for vre_var in ["solar_pv", "wind_onshore", "wind_offshore"]:
            corr_tab = getattr(d_p.data, vre_var+"_production").corr()
            cols = corr_tab.columns

            fig, ax = plt.subplots(figsize=(14, 10))
            corr_ax = ax.matshow(corr_tab, cmap="cividis")

            ax.set_xticks(np.arange(len(cols)))
            ax.set_yticks(np.arange(len(cols)))
            ax.set_xticklabels(cols)
            ax.set_yticklabels(cols)

            fig.colorbar(corr_ax)

            prettify_subplots(ax, legend=False, grid=False)
            plt.title(f"Cross-zonal correlation of {vre_var.upper()} ({scenario_name})")
            plt.show()

        print(pap2da.gamma)
        for i, k in enumerate(keys):
            print(f"{k:<25}: {vals_d[i]/1e9:>8.3f}, {vals_dp_base[i]/1e9:>8.3f}, {vals_dp_pap[i]/1e9:>8.3f}, base_ppa/da: {(vals_dp_base[i]/vals_d[i] if vals_d[i]>0 else 0):>8.3f}")

        fores = ["solar_pv_PPA_fore", "wind_onshore_PPA_fore", "wind_offshore_PPA_fore", "solar_pv_production", "wind_onshore_production", "wind_offshore_production"]
        vars = ["solar_pv_BL_offer", "wind_onshore_BL_offer", "wind_offshore_BL_offer", "solar_pv_offer", "wind_onshore_offer", "wind_offshore_offer"]

        curt_z = {}  # only for the chosen PPA zone
        curt_zonal = {}  # by zone
        curt_tot = {}  # total across zones
        for cr in crs_opt:
            curt_z[cr] = {}
            curt_zonal[cr] = {}
            curt_tot[cr] = {}
            for f, v in zip(fores, vars):
                k = v.removesuffix("_offer")
                curt_zonal[cr][k] = (getattr(d_p_dict[cr], f) - getattr(d_p_dict[cr], v).sol.to_pandas()).sum(axis=0)  # forecast - dispatch
                curt_tot[cr][k] = curt_zonal[cr][k].sum()
                curt_z[cr][k] = curt_zonal[cr][k].loc[PPA_zone]
            print(f"VRE (excl. hydro ror) for cr={cr}: {sum(curt_tot[cr].values())/1e6:.2f} TWh")
            print(f"In {PPA_zone}: {sum(curt_z[cr].values())/1e6:.2f} TWh")

        for mod in [d, d_p]:
            other_zones = mod.data.bidding_zones.copy()
            other_zones.remove(PPA_zone)
            inflex_sol = mod.results_dict['demand_inflexible_classic_bid_sol'] 
            prices = mod.results_dict['electricity_prices']
            cp_z = (inflex_sol[PPA_zone] * prices[PPA_zone]).sum(axis=0)/inflex_sol[PPA_zone].sum(axis=0)
            cp_sys = (inflex_sol * prices).sum(axis=0)[other_zones].sum()/inflex_sol.sum(axis=0)[other_zones].sum()
            print(f"Capture price in {PPA_zone} is {cp_z:.2f} €/MWh")
            print(f"Capture. price in the rest of the system is: {cp_sys:.2f} €/MWh")

# %%
