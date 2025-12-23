from pathlib import Path
from logging import Logger
from dataclasses import dataclass
from enlight.data_ops import DataLoader
import enlight.utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TypeAlias

# from pyclustering.cluster.kmedoids import kmedoids
# from pyclustering.utils import calculate_distance_matrix
# from pyclustering.utils import distance_metric
# from pyclustering.utils import type_metric
import numpy as np
import matplotlib.pyplot as plt

from PPA_modeling import load_plot_configs, unify_palette_cyclers, prettify_subplots
from nbs_modeling import NBSModel, generate_data

def normalize_forecast_power(df: pd.DataFrame, Z: str):
    return df[Z] / df.max()[Z]

@dataclass
class PaP2DA:
    '''
    Purely inputs.

    Pre-selects the data/results needed as inputs to the DA model
    in order to include PPAs in the model.

    To include a PaP all we need to pass are:
    - Bidding zone: Z
    - PPA price: S
    - PPA volume share: gamma
    - capacities of the techs in the Producer portfolio
        - off_wind_el_cap
        - on_wind_el_cap
        - solar_pv_el_cap
    '''
    # Bidding zone of producer and buyer
    z : str = "DK1"

    # Producer specs
    s : float = 5.0  # €/MWh -- PPA price
    gamma : float = 0.5  # p.u. of Producer total VRE capacity
    solar_pv_el_cap : float = 0
    on_wind_el_cap : float = 0
    off_wind_el_cap : float = 0
    ppa2da_logger : Logger | None = None

    def __post_init__(self):
        self.ppa2da_logger = self.ppa2da_logger or utils.setup_logging(log_file="nbs.log")
        self.ppa2da_logger.info("PaP2DA: Create instance to effectively transfer PPA outcome to DA market.")

@dataclass
class BL2DA:
    '''
    Purely inputs.

    Pre-selects the data/results needed as inputs to the DA model
    in order to include PPAs in the model.

    To include a PaP all we need to pass are:
    - Bidding zone: Z
    - PPA price: S
    - PPA volume share: gamma
    - capacities of the techs in the Producer portfolio
        - off_wind_el_cap
        - on_wind_el_cap
        - solar_pv_el_cap
    '''
    # Bidding zone of producer and buyer
    z : str = "DK1"

    # Producer specs
    s : float = 5.0  # €/MWh -- PPA price
    m : float = 0.5  # MW -- BL volume
    solar_pv_el_cap : float = 1
    on_wind_el_cap : float = 1
    off_wind_el_cap : float = 1
    ppa2da_logger : Logger | None = None

    def __post_init__(self):
        self.ppa2da_logger = self.ppa2da_logger or utils.setup_logging(log_file="nbs.log")
        self.ppa2da_logger.info("PaP2DA: Create instance to effectively transfer PPA outcome to DA market.")


# def week_reduction(fore_power : np.ndarray, lambda_DA : np.ndarray, scen0 : int, n_clusters : int = 4, plot : bool = False):
#     '''
#     Reduces the total hours needed to still capture the yearly profits properly.
#     Chooses representative weeks instead of single hours to preserve daily chonology.
#     '''
#     # Set seed for reproducibility
#     np.random.seed(42)

#     # Select the scenario-specific data and reshape so a single data point corresponds to a week in that scenario-year. A week-point consists of 168*2 = 336 coordinates.
#     P_fore_week_idx = fore_power[:168*52, scen0].reshape(52, 168)
#     lambda_DA_week_idx = lambda_DA[:168*52, scen0].reshape(52, 168)

#     data = np.hstack([P_fore_week_idx, lambda_DA_week_idx])

#     # Choose a metric. Many are available, also e.g. MANHATTAN
#     metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)

#     # Initialize medoids, build kmedoids model and run it
#     initial_medoids = np.arange(n_clusters).tolist()

#     k = kmedoids(data=data, initial_index_medoids=initial_medoids, metric=metric, tolerance=1e-3)#, data_type='points')

#     k.process()

#     if plot:
#         plt.scatter(x=data[:, :168],y=data[:, 168:336], alpha=.2, label="Data - x: fore power, y: price")
#         plt.scatter(x=data[k.get_medoids()][:, :168],y=data[k.get_medoids()][:, 168:336], alpha=.4, label="Representative points")
#         plt.xlabel("Power forecast [MW]")
#         plt.ylabel("DA price [€/MWh]")
#         plt.grid()
#         plt.legend()
#         plt.show()

#     # Calculate the "probability" of each week. More fitting terminology is probably the "weight" of each week.
#     cluster_size = np.array(list(map(lambda x: len(x), k.get_clusters())))
#     PROB_week = cluster_size / 52
#     PROB = PROB_week / 168  # probability should be divided unto all of the hours in that week

#     # For plotting examples of selected representative weeks
#     least_prob = np.argmin(PROB)
#     highest_prob = np.argmax(PROB)

#     if plot:
#         plt.plot(P_fore_week_idx[np.array(k.get_medoids())[[least_prob, highest_prob]]][:, :72].T)
#         plt.show()

#         plt.plot(fore_power[np.array(k.get_medoids())[least_prob]*168:np.array(k.get_medoids())[least_prob]*168+72, scen0])
#         plt.plot(fore_power[np.array(k.get_medoids())[highest_prob]*168:np.array(k.get_medoids())[highest_prob]*168+72, scen0])
#         plt.show()

#     P_fore_reduced = P_fore_week_idx[k.get_medoids()].ravel()
#     lambda_DA_reduced = lambda_DA_week_idx[k.get_medoids()].ravel()
#     PROB_hours = np.repeat(PROB, 168)  # same shape as P_fore_reduced, and lambda_DA_reduced

#     return P_fore_reduced, lambda_DA_reduced, PROB_hours, k.get_medoids(), k.get_clusters()

# def week_reduction_by_scenario(fore_power_w : np.ndarray, lambda_DA_w : np.ndarray, num_clusters : int = 4):
#     '''
#     Uses the function "week_reduction" to reduce the total hours needed to still capture the yearly profits properly
#     for all of the scenarios used.
#     '''
#     num_scens = fore_power_w.shape[1]

#     P_fore_red = np.empty(shape=(num_clusters * 168, num_scens), dtype=np.float64)
#     lambda_DA_red = np.empty(shape=(num_clusters * 168, num_scens), dtype=np.float64)
#     PROB_hours = np.empty(shape=(num_clusters * 168, num_scens), dtype=np.float64)
#     weeks_list = [[]]

#     # Iterate through the scenario-years to reduce the weeks used in each scenario
#     for w in range(num_scens):
#         P_fore_red[:, w], lambda_DA_red[:, w], PROB_hours[:, w], weeks, _ = week_reduction(fore_power_w, lambda_DA_w, scen0=w, n_clusters=num_clusters)
#         weeks_list.append(weeks)

#     return P_fore_red, lambda_DA_red, PROB_hours, weeks_list

    # Placeholder function until we can load the data
    #   (VRE forecasts) from multiple weather years and
    #   the resulting power prices from the DA market model

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
    
@dataclass
class PPAInputData:
    '''
    Purely inputs.

    Pre-configures the parameters needed as inputs to create the data
    required on the producer side.

    This function has two options. One intended for analyses from an
    individual stakeholder perspective and one from system perspective:
    - Individual: Use the following parameters if you want to give a physical capacity for the Producer portfolio.
        - P_vre
        - x_pv, x_wind_on, x_wind_off
        - x_buyer
    - System: If you want to assign a share of the total zonal capacity to the Producer portfolio.
        - x_tot_Z -- this parameter scales the Producer VRE capacity based on the total zonal capacity.
                  -- further, it also scales the buyer to maintain the ratio of:
                  -- mean inflex zonal cons. to total VRE capacity.
    '''
    # Bidding zone of producer and buyer
    Z : str = "DK1"

    # Producer specs
    P_vre : float = 1  # MW, total VRE capacity of producer
    x_tot_Z : float = 0  # used to override P_vre.
    x_pv : float = 1  # -, total solar pv share of VRE capacity
    x_wind_on : float = 0  # -, share of onshore wind
    x_wind_off : float = 0  # -, share of offshore wind
    y_batt : float = 0.25 # -, battery power ratio to VRE: P_batt/P_VRE
    batt_Crate : float = 1  # -, battery C-rate to determine energy capacity
    batt_eta : float = float(np.sqrt(0.9))

    # Buyer specs
    # E_buyer is calculated from P_vre (or x_tot_Z and P_vre_Z_tot)
    x_buyer : float = 0.5  # The average capacity of the buyer relative to the total Producer VRE capacity
    # x_buyer is equivalent to E_buyer/(P_vre * 8760)

    ppa_logger : Logger | None = None

    def __post_init__(self):
        self.ppa_logger = self.ppa_logger or utils.setup_logging(log_file="nbs.log")
        self.ppa_logger.info("PPAInputData: LOAD technical and economic specifications for producer and buyer.")

        self.validate_x()

    def validate_x(self):
        '''
        Validate the capacity shares provided by the user.
        '''
        if not (1>=
                np.array([self.x_pv, self.x_wind_on, self.x_wind_off]).all()
                >=0):
            raise ValueError(f"The VRE capacity shares must be between 0 and 1. Currently they are: {self.x_pv, self.x_wind_on, self.x_wind_off}")
        
        if self.x_pv + self.x_wind_on + self.x_wind_off != 1:
            raise ValueError(f"The total share of VREs must be 1. It is currently: {self.x_pv + self.x_wind_on + self.x_wind_off:.2f}")


class PPAInputCalcs:
    """
    Computation engine.

    Selects some of the energy system input data from a data loader instance and the power prices saved from a previous enlight DA run.
    
    Example usage:
    open interactive window in VSCode,
    >>> cd ../../
    run the script data_loader.py in the interactive window,
    >>> data = DataLoader(week=1, scenario_name="scenario_1")
    """
    def __init__(self,
                 scenario_name : str = "scenario_1",
                 da_data : DataLoader | None = None,
                 ppa_data : PPAInputData | None = None,
                 ppa_logger: Logger | None = None,
        ) -> None:
        self.scenario_name = scenario_name
        self.da_data = da_data
        self.ppa_data = ppa_data

        self.ppa_logger = ppa_logger or utils.setup_logging(name=__file__, log_file="nbs.log")
        self.ppa_logger.info("PPAInputCalcs: CALCULATE power forecast, batt specs, buyer profile, and load power prices.")

        # Do we have DataLoader objects already?
        if self.da_data is None:
            self.da_data = DataLoader(
                scenario_name=self.scenario_name,
                logger=self.ppa_logger
                )

        # If no PPA configuration instance is input to the object, just try to use the default.
        if self.ppa_data is None:
            # The default only works if scenario_1 has been run.
            self.ppa_data = PPAInputData()

        # Retrieve the parameters needed for NBSModel:
        #   - P_fore, B_fore, WTP, lambda_DA
        self.calculate_normalized_forecasts()
        self.calculate_forecasts()  # P_fore, B_fore

        # Load power prices from DA market model
        self.load_power_prices()

        self.calculate_batt_power()
        self.verify_batt_capacy_and_buyer_load()
        # self.hour_reduction(num_clusters=self.num_clusters)

    def calculate_normalized_forecasts(self):
        self.fore_solar_pv_pu = normalize_forecast_power(df=self.da_data.solar_pv_production, Z=self.ppa_data.Z)
        self.fore_on_wind_pu = normalize_forecast_power(df=self.da_data.wind_onshore_production, Z=self.ppa_data.Z)
        self.fore_off_wind_pu = normalize_forecast_power(df=self.da_data.wind_offshore_production, Z=self.ppa_data.Z)

        # Buyer's consumption forecast. Normalize by energy consumption
        self.fore_inflex_classic_pu = self.da_data.demand_inflexible_classic[self.ppa_data.Z] / self.da_data.demand_inflexible_classic.sum(axis=0)[self.ppa_data.Z]

    def calculate_forecasts(self):
        # Get the technology-specific installed capacity
        # Existing attributes for dynamic calling
        attrs_ppa_cfg = ["x_pv", "x_wind_on", "x_wind_off"]
        attrs_fore = ["solar_pv_production", "wind_onshore_production", "wind_offshore_production"]
        attrs_fore_pu = ["fore_solar_pv_pu", "fore_on_wind_pu", "fore_off_wind_pu"]
        # New attributes for dynamic assignment
        new_attrs_capacity = ["solar_pv_el_cap", "on_wind_el_cap", "off_wind_el_cap"]
        new_attrs_fore = ["P_fore_solar_pv", "P_fore_on_wind", "P_fore_off_wind"]

        # Relevant only if using x_tot_Z: P_vre_Z_tot = P_pv + P_onwind + P_offwind in zone Z
        self.P_vre_Z_tot = sum(getattr(self.da_data, fore_attr).max()[self.ppa_data.Z] for fore_attr in attrs_fore)
        # If e.g. x_tot_Z = 0.9, then 90% of the zonal VRE capacity is included in the Producer's portfolio.
        self.P_vre_Z = self.ppa_data.x_tot_Z * self.P_vre_Z_tot

        for tech in range(len(attrs_ppa_cfg)):
            if self.ppa_data.x_tot_Z > 0: # capacity share at bidding zone level
                # If using zonal capacity, we also use the zonal capacity shares. It's easier that way.
                # Overwrite, x_pv, x_wind_on and x_wind_off
                vre_max = getattr(self.da_data, attrs_fore[tech]).max()[self.ppa_data.Z]
                value = vre_max / self.P_vre_Z_tot  # e.g. solar_pv_production.max() / P_vre_Z_tot -- in the PPA zone
                setattr(self.ppa_data, attrs_ppa_cfg[tech], value)
                # e.g. for PV: value = x_pv * x_tot_Z * (P_pv[Z] + P_onwind[Z] + P_offwind[Z])
                value = getattr(self.ppa_data, attrs_ppa_cfg[tech]) * self.P_vre_Z
            else:
                # The default actual physical capacities are given:
                # e.g. solar_pv_el_cap = x_pv * P_vre
                value = getattr(self.ppa_data, attrs_ppa_cfg[tech]) * self.ppa_data.P_vre
            setattr(self, new_attrs_capacity[tech], value)

            if getattr(self, new_attrs_capacity[tech]) <= getattr(self.da_data, attrs_fore[tech]).max()[self.ppa_data.Z]:
                value = getattr(self, new_attrs_capacity[tech]) * getattr(self, attrs_fore_pu[tech])
                # e.g.: P_fore_solar_pv = solar_pv_el_cap * fore_solar_pv_pu
                setattr(self, new_attrs_fore[tech], value)
            else:
                raise Exception(f"ValueError: The capacity ({new_attrs_capacity[tech]}) of the PPA producer {getattr(self, new_attrs_capacity[tech]):.2f} is higher than the total capacity in the bidding zone {self.da_data.__getattribute__(attrs_fore[tech]).max()[self.ppa_data.Z]:.2f}")
        
        self.P = self.solar_pv_el_cap + self.on_wind_el_cap + self.off_wind_el_cap  # either corresponds to P_vre or P_vre_Z
        self.P_fore = self.P_fore_solar_pv + self.P_fore_on_wind + self.P_fore_off_wind

    def load_power_prices(self):
        prices_file = Path("simulations") / f"{self.scenario_name}/results/electricity_prices.csv"
        if prices_file.exists():
            df_prices = pd.read_csv(prices_file, index_col=0)
            self.lambda_DA = df_prices[self.ppa_data.Z]
        else:
            raise Exception(f"FileNotFoundError: Please provide an existing scenario name. No power prices are given under (full file path shown) {prices_file}.")

    def calculate_batt_power(self):
        self.P_batt = self.P * self.ppa_data.y_batt  # MW
        self.E_batt = self.P_batt / self.ppa_data.batt_Crate  # MWh

    def verify_batt_capacy_and_buyer_load(self):
        '''
        Check that the VRE, batt and buyer consumption levels are indeed below the zonal maximum.
        '''
        # Calculate the ratio of x_buyer for each bidding zone included
        if self.ppa_data.x_tot_Z > 0: # capacity share at bidding zone level
            self.x_buyer_Z = (self.da_data.demand_inflexible_classic.mean()[self.ppa_data.Z]
                        / self.P_vre_Z_tot)
            self.E_buyer = 8760 * self.x_buyer_Z * self.P # MWh/year
        else:
            self.E_buyer = 8760 * self.ppa_data.x_buyer * self.P  # MWh/year

        if self.E_buyer <= self.da_data.demand_inflexible_classic.sum(axis=0)[self.ppa_data.Z]:
            B_fore = self.E_buyer * self.fore_inflex_classic_pu
            self.B_fore = B_fore  # pd.Series
            self.B_fore_arr = B_fore.values.reshape(len(B_fore), 1)
        else:
            raise Exception(f"ValueError: The annual buyer consumption {self.E_buyer:.2f} exceeds the zonal total {self.da_data.demand_inflexible_classic.sum(axis=0)[self.ppa_data.Z]:.2f}")
        
        if not self.P_batt <= self.da_data.agg_bess.capacity_el[self.ppa_data.Z]:
            raise Exception(f"ValueError: The producer battery power capacity {self.P_batt:.2f} MW exceeds the zonal total {self.da_data.agg_bess.capacity_el[self.ppa_data.Z]:.2f} MW")
        elif not self.E_batt <= self.da_data.agg_bess.capacity_stor[self.ppa_data.Z]:
            raise Exception(f"ValueError: The producer battery energy capacity {self.E_batt:.2f} MWh exceeds the zonal total {self.da_data.agg_bess.capacity_stor[self.ppa_data.Z]:.2f} MWh")

    def visualize_inputs(self, plot_hours=(0, 8760)):
        # Visualize the capacities as a bar plot
        tech = ["Offshore Wind", "Onshore Wind", "Solar PV", "BESS", "VRE total"]
        cap = [self.off_wind_el_cap, self.on_wind_el_cap, self.solar_pv_el_cap, self.P_batt]
        tech_cap_pairs = list(zip(tech, cap))
        # Sort techs by capacity: low to high
        sorted_pairs = sorted(tech_cap_pairs, key=lambda x: x[1])
        # Separate into two lists
        tech_sorted, cap_sorted = zip(*sorted_pairs)
        tech = list(tech_sorted) + ["VRE total"]
        cap = list(cap_sorted) + [self.P]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        unify_palette_cyclers(ax)
        sns.barplot(ax=ax,
                    x=cap,
                    y=tech,
                    label='Producer',
                    orient='h',
        )
        sns.barplot(ax=ax,
                    x=[self.B_fore.mean()],
                    y=["Average hourly load"],
                    label='Buyer',
                    orient='h',
        )
        prettify_subplots(ax)
        ax.set_xlabel('Power [MW]')
        ax.set_title(f'PPA (in {self.ppa_data.Z}) producer capacities and average buyer consumption', loc='left')
        fig.tight_layout()
        plt.show()

        # Visualize the individual forecasts as a stacked line chart
        h0, hf = plot_hours

        fig, ax = plt.subplots(figsize=(12,6))
        unify_palette_cyclers(ax)
        ax.fill_between(self.P_fore.index[h0:hf], 0, self.P_fore_off_wind[h0:hf], label="OFFshore Wind")
        ax.fill_between(self.P_fore.index[h0:hf], self.P_fore_off_wind[h0:hf], (self.P_fore_off_wind+self.P_fore_on_wind)[h0:hf], label="ONshore Wind")
        ax.fill_between(self.P_fore.index[h0:hf], (self.P_fore - self.P_fore_solar_pv)[h0:hf], self.P_fore[h0:hf], label="Solar PV")
        sns.lineplot(ax=ax, data=self.B_fore[h0:hf], label="Buyer")
        prettify_subplots(ax)
        ax.set_ylabel('Power [MW]')
        ax.set_title(f'PPA (in {self.ppa_data.Z}) producer generation and buyer consumption profiles', loc='left')
        fig.tight_layout()
        plt.show()

        # Visualize power prices as a simple line plot
        fig, ax = plt.subplots(figsize=(12,6))
        unify_palette_cyclers(ax)
        sns.lineplot(ax=ax, data=self.lambda_DA, label=r"$\lambda^{DA}_t$")
        prettify_subplots(ax)
        ax.set_ylabel(f"Power price in {self.ppa_data.Z} [€/MWh]")
        ax.legend().remove()
        plt.show()

    # def hour_reduction(self, num_clusters : int = 6):
    #     (self.P_fore_w_red,
    #      self.lambda_DA_w_red,
    #      self.PROB_w_red,
    #      self.weeks_l
    #      ) = week_reduction_by_scenario(
    #          fore_power_w=self.P_fore_w,
    #          lambda_DA_w=self.lambda_DA_w,
    #          num_clusters=num_clusters
    #     )
    #     B_fore_arr_red = self.B_fore_arr[:168*52].reshape(52, 168)
    #     B_fore_arr_red = B_fore_arr_red[self.weeks_l[1]].ravel()
    #     self.B_fore_arr_red = B_fore_arr_red.reshape(len(B_fore_arr_red), 1)

@dataclass
class NBSSetup:
    '''
    Purely inputs.

    Pre-configures the bounds on PPA strike price and volume and values
    of CVaR parameters needed as inputs to create the NBSModel.
    '''
    # Strike price and volume bounds
    S_LB : float = 0  # Minimum PPA strike price
    S_UB : float = 1000  # Maximum PPA strike price
    M_LB : float = 0  # BL: Minimum baseload volume
    M_UB : float = 1  # BL: Maximum baseload volume
    gamma_LB : float = 0 # PaP: Minimum PPA capacity share volume
    gamma_UB : float = 1 # PaP: Minimum PPA capacity share volume
    
    # CVaR parameters
    beta_D : float = 0.5  # CVaR: Risk-aversion level of developer
    beta_O : float = 0.5  # CVaR: Risk-aversion level of off-taker
    alpha : float = 0.75

    nbs_setup_logger : Logger | None = None

    def __post_init__(self):
        self.nbs_setup_logger = self.nbs_setup_logger or utils.setup_logging(log_file="nbs.log")
        self.nbs_setup_logger.info("NBSSetup: SETUP NBS bounds and CVaR params.")


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