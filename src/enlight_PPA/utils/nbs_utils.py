import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp

# Module imports
from enlight_PPA.data_ops import BL2DA


def load_plot_configs(only_get_palette : bool = False) -> None:
    palette1 = ["#7782f0", "#70e0aa", "#f8dd7a", "#fc9a67", "#f9ccc3", "#ef7a81", "#4ab877", "#a566b5"]
    nature_pastel1 = [
        "#595959",  # soft black / gray
        "#F3C76B",  # pastel orange
        "#89CAF0",  # pastel sky blue
        "#44C1A1",  # pastel bluish green
        "#F6EB88",  # pastel yellow
        "#4C97C9",  # pastel blue
        "#E98A4C",  # pastel vermillion
        "#DA95BF",  # pastel reddish purple
    ]
    nature_pastel2 = [
    "#B3B3B3",  # light gray
    "#F9DB9C",  # light orange
    "#A7D8F4",  # light sky blue
    "#7FD3BB",  # light bluish green
    "#FAF2B8",  # light yellow
    "#7FB6DA",  # light blue
    "#F2A97E",  # light vermillion
    "#E4B5D0",  # light reddish purple
    ]

    # Set default color palette, font sizes, and font family.
    # chose palette1 or nature_pastel1
    chosen_palette = nature_pastel1

    # Don't overwrite the plot configs every time loading the palette
    if only_get_palette:
        return chosen_palette

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=chosen_palette)  # matplotlib.pyplot
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 13
    })

    return chosen_palette
    
def unify_palette_cyclers(axs):  # run BEFORE plotting
    if not type(axs) == np.ndarray:
        ax = axs
        # Shape of subplots of (1,1
        ax._get_patches_for_fill = ax._get_lines
        return ax
    else:
        # For multiple axes
        for ax in axs: # Unify palette cyclers for different plot types
            ax._get_patches_for_fill = ax._get_lines
        return axs
    
def prettify_subplots(axs, legend=True, grid=True, bbox_list=[1, 1.02]):  # run AFTER plotting
    if not type(axs) == np.ndarray:
        # Shape of subplots of (1,1)
        ax = axs  # to symbolize that there is only one axis
        ax.spines[['right', 'top']].set_visible(False)  # Remove spines
        if grid:
            ax.grid(alpha=.25)  # Add opaque gridlines
        else:
            ax.grid(False)
        ax.margins(0.005)  # Remove whitespace inside each plot
        ax.spines[['bottom','left']].set_alpha(0.5)  # Introduce opacity to the x- and y-axes spines
        if legend:
            ax.legend(bbox_to_anchor=bbox_list, frameon=False)
        return ax
    else:
        for ax in axs:
            ax.spines[['right', 'top']].set_visible(False)  # Remove spines
            if grid:
                ax.grid(alpha=.25)  # Add opaque gridlines
            else:
                ax.grid(False)
            ax.margins(0.005)  # Remove whitespace inside each plot
            ax.spines[['bottom','left']].set_alpha(0.5)  # Introduce opacity to the x- and y-axes spines
            if legend:
                ax.legend(bbox_to_anchor=[1, 1.02], frameon=False)
        return axs

def make_negative_DA_price_mask(times, lambda_DA):
    # Initialize arrays
    mask = lambda_DA.ravel() < 0

    # Detect whether the price changes from negative to positive or positive to negative
    transitions_from = np.where(np.diff(mask.astype(int)) != 0)[0]
    transitions_to = np.where(np.diff(mask.astype(int)) != 0)[0] + 1

    # We want to insert points only "after" a negative hour
    insert_points_idx = [t for t in transitions_to if mask[t-1]]

    # Build new extended lists
    mask_ext = mask.astype(int).tolist()
    times_ext = times.tolist()

    for t in reversed(insert_points_idx):  # we reverse to avoid that the first insert's affect later indices
        mask_ext.insert(t, True)
        times_ext.insert(t, times[t])

    mask_ext = np.append(mask_ext, mask_ext[-1])
    times_ext = np.append(times_ext, times_ext[-1])

    return mask_ext, times_ext

def remove_mult_suffix(string: str, suffixes: list):
    '''
    Removes multiple types of suffixes to allow for simple naming
    conventions of PPA types while maintaining a versatile class.
    '''
    for s in suffixes:
        string = string.removesuffix(s)
    return string

def generate_data():  # <-- used in NBSModel to generate simple synthetic data
    np.random.seed(42)
    num_weeks = 52 # 1/7
    num_days = int(7 * num_weeks)
    T = 24 * num_days  # num hours
    W = 5  # num scenarios

    PROB_w = np.full(shape=W, fill_value=1/W)  # all scenarios are equiprobable

    # Generate forecasts
    dist = 0.5*np.random.weibull(1.5, size=(T, W))
    P_fore_w = dist/np.max(dist)  # normalized forecast between 0 and 1

    # Generate DA prices
    # lambda_DA_day = np.array([
    #     89.33, 89.14, 87.95, 86.89, 88.69, 98.73,
    #     113.97, 117.38, 108.84, 100.01, 72.64, 64.23,
    #     40.25, -23.12, 39.33, 71.01, 83.13, 110.93,
    #     125.91, 220.25, -195.33, 119.71, 108.31, 97.7
    #     ]).reshape(24, 1)  # September 1st prices
    lambda_DA_day = np.array([
        89.33, 89.14, 87.95, 86.89, 88.69, 98.73,
        113.97, 117.38, 108.84, 100.01, 72.64, 64.23,
        40.25, 23.12, 39.33, 71.01, 83.13, 110.93,
        125.91, 220.25, 195.33, 119.71, 108.31, 97.7
        ]).reshape(24, 1)  # September 1st prices

    lambda_DA = np.vstack([lambda_DA_day for _ in range(num_days)])
    lambda_DA = lambda_DA.ravel().reshape(T, 1)
    lambda_DA_coeffs = np.random.uniform(low=0.5, high=1.5, size=(T, W))
    lambda_DA_w = lambda_DA * lambda_DA_coeffs  # scenario-based prices

    # Use given day as example for load profile
    P_L = np.array([
        0.000112081 ,0.000107526 ,0.000102827 ,9.89E-05 ,9.63E-05 ,9.62E-05,
        9.70E-05 ,9.58E-05 ,9.76E-05 ,0.000100145 ,0.00010334 ,0.000106918
        ,0.000108268 ,0.000108537 ,0.000105907 ,0.000106967 ,0.000108994 ,0.000116045
        ,0.000117293 ,0.000115371 ,0.00011297 ,0.000110143 ,0.000113326 ,0.00011483
    ])
    P_L = P_L / max(P_L)  # normalize to [0, 1]
    # testing smth
    P_L = np.vstack([P_L.reshape(24, 1) for _ in range(num_days)])
    L_t = P_L.reshape(T, 1)  # reshape for broadcasting

    # To make PaP results interesting
    L_t = L_t * 0.3

    # Assume a marginal utility for the consumer
    WTP = 300  # €/MWh

    return P_fore_w, lambda_DA_w, L_t, WTP

def specify_battery_data():
    batt_power = 0.25  # MW
    batt_eta = float(np.sqrt(0.9))  # round-trip efficiency
    batt_Crate = 1  # C-rate (1/C-rate = hours of storage)
    return batt_power, batt_eta, batt_Crate

def var_to_pandas(var, name=None):
    """
    Convert a Gurobi variable (Var or MVar) to pandas object.
    """
    if var is None:
        return None

    # MVar (vector or matrix)
    if isinstance(var, gp.MVar):
        arr = var.X
        if arr.ndim == 1:
            return pd.Series(arr, name=name)
        elif arr.ndim == 2:
            return pd.DataFrame(arr)
        else:
            raise ValueError(f"Unsupported MVar dimension: {arr.ndim}")

    # Scalar Var
    if isinstance(var, gp.Var):
        return pd.Series({name: var.X})

    raise TypeError(f"Unsupported type: {type(var)}")

def perc(array, perc):
            return np.percentile(array, perc, axis=1)

def initialize_data():  # <-- used in hybrid_vre_in_da to generate simple synthetic data
    np.random.seed(42)  # for reproducibility
    num_days = 1
    T=num_days * 24
    W=10
    dist = 0.5*np.random.weibull(1.5, size=(T, W))
    P_fore_w = dist/np.max(dist)  # normalized forecast between 0 and 1

    # Generate DA prices
    lambda_DA_day = np.array([
        89.33, 89.14, 87.95, 86.89, 88.69, 98.73,
        113.97, 117.38, 108.84, 100.01, 72.64, 64.23,
        40.25, -23.12, 39.33, 71.01, 83.13, 110.93,
        125.91, 220.25, -195.33, 119.71, 108.31, 97.7
        ])  # September 1st prices

    lambda_DA = np.tile(lambda_DA_day, num_days)
    lambda_DA = lambda_DA.reshape(T, 1)

    lambda_DA_coeffs = np.random.uniform(low=0.5, high=1.5, size=(T, W))
    lambda_DA_w = lambda_DA * lambda_DA_coeffs  # scenario-based prices

    return T, W, lambda_DA_w, P_fore_w

def normalize_forecast_power(df: pd.DataFrame, Z: str):
    return df[Z] / df.max()[Z]

def generate_scenarios(yearly_param, noise_lvl=0.05):
    '''
    Until we have run the DA model for multiple weather years, I need to make some synthetic data.
    '''
    np.random.seed(42)
    num_hours = yearly_param.shape[0]
    num_scens = 4

    # Initialize new matrix with 10 year-scenarios
    mult_year_param = np.zeros((num_hours, num_scens))
    mult_year_param[:, 0] = yearly_param.values

    # Generate noise
    for s in range(1, num_scens):
        noise = np.random.normal(loc=0, scale=noise_lvl, size=num_hours)
        mult_year_param[:, s] = mult_year_param[:, 0] * (1 + noise)
    mult_year_param = np.maximum(mult_year_param, 0)
    # E.g. P_fore_w
    return mult_year_param

def plot_market_clearing_outcome(
        dp : object,  # type: EnlightModel
        Z : str,
        t : list[int],
        bl2da : BL2DA,
        *,
        year=2020,
        show=True,
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

    ax.set_ylabel("")
    ax.set_title(f"{Z}: Power balance from DA market clearing — compl. rate = {dp.PPA2DA.compl_rate}\nPower [MW]", loc="left")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False
    )
    if dp.BL:
        ax.axhline(bl2da.m, color="k", lw=0.8, label="M (BL)")
    # for val in [16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 32,  64,  65,  67]:
    #     ax.axvline(val, color="k", alpha=0.25)
    ax.xaxis.set_major_formatter(make_hour_formatter(year=year))
    ax.tick_params(axis='x', rotation=15)
    ax.set_xlabel("")
    prettify_subplots(ax)
    if show:
        plt.show()

    return offers_df, bids_df

def make_hour_formatter(year: int):
        '''
        Used to change x-axis from hour number to date-time format.
        '''
        start = pd.Timestamp(f"{year}-01-01")
        def _formatter(x, pos):
            return (start + pd.Timedelta(hours=int(x) - 1)).strftime("%Y-%m-%d %H:%M")
        return _formatter
