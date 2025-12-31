import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from enlight.model import EnlightModel
from ppa2da import BL2DA




def plot_market_clearing_outcome(
        dp : EnlightModel,
        Z : str,
        t : list[int],
        bl2da : BL2DA,
        ):
    # cr = crs_opt[0]          # or loop over cr
    # select a subset of sols (and fores) -- (t, z)
    export = dp.electricity_export.sol.sel(Z=Z).loc[t]

    offers = {}

    offers["Wind onshore"] = dp.wind_onshore_offer.sol.sel(Z=Z).loc[t]
    offers["Wind offshore"] = dp.wind_offshore_offer.sol.sel(Z=Z).loc[t]
    offers["Solar PV"] = dp.solar_pv_offer.sol.sel(Z=Z).loc[t]

    if dp.PaP:
        offers["Wind onshore PaP"] = dp.wind_onshore_PaP_offer.sol.sel(Z=Z).loc[t]
        offers["Wind offshore PaP"] = dp.wind_offshore_PaP_offer.sol.sel(Z=Z).loc[t]
        offers["Solar PV PaP"] = dp.solar_pv_PaP_offer.sol.sel(Z=Z).loc[t]

    if dp.BL:
        offers["BL"] = dp.total_BL_offer.sol.sel(Z=Z).loc[t]

    offers["Hydro RoR"] = dp.hydro_ror_offer.sol.sel(Z=Z).loc[t]
    offers["Conventional"] = dp.conventional_units_offer.sol.dot(
        dp.data.G_Z_xr
    ).sel(Z=Z).loc[t]

    offers["Hydro res"] = dp.hydro_res_units_offer.sol.dot(
        dp.data.G_hydro_res_Z_xr
    ).sel(Z=Z).loc[t]

    offers["Hydro PS"] = dp.hydro_ps_units_offer.sol.sel(Z=Z).loc[t]
    offers["BESS"] = dp.bess_units_offer.sol.sel(Z=Z).loc[t]

    offers["Import"] = (-export.where(export < 0, 0.0))

    offers_df = pd.concat(
        {k: v.to_pandas() for k, v in offers.items()},
        axis=1
    )
    bids = {}

    bids["Inflex demand"] = dp.demand_inflexible_classic_bid.sol.sel(Z=Z).loc[t]
    bids["Flex demand"] = dp.demand_flexible_classic_bid.sol.sel(Z=Z).loc[t]

    bids["Hydro PS bid"] = dp.hydro_ps_units_bid.sol.sel(Z=Z).loc[t]
    bids["BESS bid"] = dp.bess_units_bid.sol.sel(Z=Z).loc[t]

    bids["PtX"] = dp.ptx_units_bid.sol.dot(
        dp.data.L_PtX_Z_xr
    ).sel(Z=Z).loc[t]

    bids["DH"] = dp.dh_units_bid.sol.dot(
        dp.data.L_DH_Z_xr
    ).sel(Z=Z).loc[t]

    bids["Export"] = export.where(export > 0, 0.0)

    bids_df = pd.concat(
        {k: v.to_pandas() for k, v in bids.items()},
        axis=1
    )
    fig, ax = plt.subplots(figsize=(14,6))

    # stacked offers (outcome of LHS)
    offers_df.plot.area(
        ax=ax,
        stacked=True,
        lw=0,
        alpha=0.4
    )

    # bids as lines (RHS)
    for col in bids_df.columns:
        bids_df[col].plot(
            ax=ax,
            lw=2.3,
            linestyle="--",
            label=col
        )
    bids_tot = bids_df.sum(axis=1)
    bids_tot.plot(ax=ax, label="Total consumption", c='k', ls='--')

    ax.set_ylabel("Power [MW]")
    ax.set_title(f"Market clearing constraint outcome — CR = {cr}, Z = {Z}")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False
    )
    if dp.BL:
        ax.axhline(bl2da.m, color="k", lw=0.8, label="M (BL)")
    # for val in [16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 32,  64,  65,  67]:
    #     ax.axvline(val, color="k", alpha=0.25)
    plt.show()

    return offers_df, bids_df