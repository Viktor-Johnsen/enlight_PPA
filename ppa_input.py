from pathlib import Path
from logging import Logger
from dataclasses import dataclass
from enlight.data_ops import DataLoader
import enlight.utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
from pyclustering.utils import distance_metric
from pyclustering.utils import type_metric
import numpy as np
import matplotlib.pyplot as plt

from PPA_modeling import load_plot_configs, unify_palette_cyclers, prettify_subplots
from nbs_modeling import NBSModel

def normalize_forecast_power(df: pd.DataFrame, Z: str):
    return df[Z] / df.max()[Z]

def week_reduction(fore_power : np.ndarray, lambda_DA : np.ndarray, scen0 : int, n_clusters : int = 4, plot : bool = False):
    '''
    Reduces the total hours needed to still capture the yearly profits properly.
    Chooses representative weeks instead of single hours to preserve daily chonology.
    '''
    # Set seed for reproducibility
    np.random.seed(42)

    # Select the scenario-specific data and reshape so a single data point corresponds to a week in that scenario-year. A week-point consists of 168*2 = 336 coordinates.
    P_fore_week_idx = fore_power[:168*52, scen0].reshape(52, 168)
    lambda_DA_week_idx = lambda_DA[:168*52, scen0].reshape(52, 168)

    data = np.hstack([P_fore_week_idx, lambda_DA_week_idx])

    # Choose a metric. Many are available, also e.g. MANHATTAN
    metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)

    # Initialize medoids, build kmedoids model and run it
    initial_medoids = np.arange(n_clusters).tolist()

    k = kmedoids(data=data, initial_index_medoids=initial_medoids, metric=metric, tolerance=1e-3)#, data_type='points')

    k.process()

    if plot:
        plt.scatter(x=data[:, :168],y=data[:, 168:336], alpha=.2, label="Data - x: fore power, y: price")
        plt.scatter(x=data[k.get_medoids()][:, :168],y=data[k.get_medoids()][:, 168:336], alpha=.4, label="Representative points")
        plt.xlabel("Power forecast [MW]")
        plt.ylabel("DA price [€/MWh]")
        plt.grid()
        plt.legend()
        plt.show()

    # Calculate the "probability" of each week. More fitting terminology is probably the "weight" of each week.
    cluster_size = np.array(list(map(lambda x: len(x), k.get_clusters())))
    PROB = cluster_size / 52

    # For plotting examples of selected representative weeks
    least_prob = np.argmin(PROB)
    highest_prob = np.argmax(PROB)

    if plot:
        plt.plot(P_fore_week_idx[np.array(k.get_medoids())[[least_prob, highest_prob]]][:, :72].T)
        plt.show()

        plt.plot(fore_power[np.array(k.get_medoids())[least_prob]*168:np.array(k.get_medoids())[least_prob]*168+72, scen0])
        plt.plot(fore_power[np.array(k.get_medoids())[highest_prob]*168:np.array(k.get_medoids())[highest_prob]*168+72, scen0])
        plt.show()

    P_fore_reduced = P_fore_week_idx[k.get_medoids()].ravel()
    lambda_DA_reduced = lambda_DA_week_idx[k.get_medoids()].ravel()
    PROB_hours = np.repeat(PROB, 168)  # same shape as P_fore_reduced, and lambda_DA_reduced

    return P_fore_reduced, lambda_DA_reduced, PROB_hours, k.get_medoids(), k.get_clusters()

def week_reduction_by_scenario(fore_power_w : np.ndarray, lambda_DA_w : np.ndarray, num_clusters : int = 4):
    '''
    Uses the function "week_reduction" to reduce the total hours needed to still capture the yearly profits properly
    for all of the scenarios used.
    '''
    num_scens = fore_power_w.shape[1]

    P_fore_red = np.empty(shape=(num_clusters * 168, num_scens), dtype=np.float64)
    lambda_DA_red = np.empty(shape=(num_clusters * 168, num_scens), dtype=np.float64)
    PROB_hours = np.empty(shape=(num_clusters * 168, num_scens), dtype=np.float64)
    weeks_list = [[]]

    # Iterate through the scenario-years to reduce the weeks used in each scenario
    for w in range(num_scens):
        P_fore_red[:, w], lambda_DA_red[:, w], PROB_hours[:, w], weeks, _ = week_reduction(fore_power_w, lambda_DA_w, scen0=w, n_clusters=num_clusters)
        weeks_list.append(weeks)

    return P_fore_red, lambda_DA_red, PROB_hours, weeks_list


@dataclass
class PPAInputData:
    '''
    Purely inputs.

    Pre-configures the parameters needed as inputs to create the data
    required on the producer side.
    '''
    # Bidding zone of producer and buyer
    Z : str = "DK1"

    # Producer specs
    P_vre : float = 1  # MW, total VRE capacity of producer
    x_pv : float = 1  # -, total solar pv share of VRE capacity
    x_wind_on : float = 0  # -, share of onshore wind
    x_wind_off : float = 0  # -, share of offshore wind
    y_batt : float = 0.25 # -, battery power ratio to VRE: P_batt/P_VRE
    batt_Crate : float = 1  # -, battery C-rate to determine energy capacity

    # Buyer specs
    E_buyer : float = 2000  # MWh, buyer total annual electricity consumption
    WTP : float = 5000  # €/MWh, default is the typical VOLL of an inflexible load

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
        
        if not (1>=self.x_pv + self.x_wind_on + self.x_wind_off >= 0):
            raise ValueError(f"The total share of VREs must be between 0 and 1. It is currently: {self.x_pv + self.x_wind_on + self.x_wind_off:.2f}")


class PPAInputCalcs:
    """
    Computation engine.

    Selects some of the energy system input data from a data loader instance and the power prices saved from a previous enlight DA run.
    
    Example usage:
    open interactive window in VSCode,
    >>> cd ../../
    run the script data_loader.py in the interactive window,
    >>> data = DataLoader(week=1, simulation_path='simulations/scenario_1/data')
    """
    def __init__(self,
                 data : DataLoader | None = None,
                 ppa_data : PPAInputData | None = None,
                 simulation_path: Path | None = None,
                 scenario_name : str = "scenario_1",
                 ppa_logger: Logger | None = None,
        ) -> None:
        self.data = data
        self.ppa_data = ppa_data
        self.simulation_path = simulation_path
        self.scenario_name = scenario_name

        self.ppa_logger = ppa_logger or utils.setup_logging(name=__file__, log_file="nbs.log")
        self.ppa_logger.info("PPAInputCalcs: CALCULATE power forecast, batt specs, buyer profile, and load power prices.")

        # Do we have a DataLoader object already?
        if self.data is None:
            # Can we use the input path to create a DataLoader instance, then?
            if self.simulation_path is not None:
                self.data = DataLoader(
                    input_path=Path(self.simulation_path / "data"),
                    logger=self.ppa_logger
                )
            # Raise exception and ask for new inputs.
            else:
                raise Exception(f"AttributeError: please insert an actual path to the data folder as NO DataLoader instance already exists.")
        
        # If no PPA configuration instance is input to the object, just try to use the default.
        if self.ppa_data is None:
            # The default only works if scenario_1 has been run.
            self.ppa_data = PPAInputData()

        # Retrieve the parameters needed for NBSModel:
        #   - P_fore, B_fore, WTP, lambda_DA
        self.calculate_normalized_forecasts()
        self.calculate_forecasts()  # P_fore, B_fore
        
        # Load power prices from DA market model
        if self.simulation_path.exists():
            df_prices = pd.read_csv(self.simulation_path / "results/electricity_prices.csv", index_col=0)
            self.lambda_DA = df_prices[self.ppa_data.Z]
        else:
            raise Exception(f"FileNotFoundError: Please provide an existing simulation path. No power prices are given under {self.simulation_path}.")

        self.WTP = self.ppa_data.WTP

        self.calculate_batt_power()
        self.verify_batt_capacy_and_buyer_load()

        self.generate_scenarios(self.P_fore, noise_lvl=.1, attr_name="P_fore_w")
        self.generate_scenarios(self.lambda_DA, noise_lvl=0.15, attr_name="lambda_DA_w")
        # self.shorten_years()  # <- function required to actually run model. Too big if using 8760 hours...
        
        # Finally, if we choose to only use representative weeks/hours:
        # if self.reduce_number_of_weeks:
        #     '''
        #     Reduce the number of weeks in each scenario-year to num_clusters,
        #       so e.g. if there are W scenarios and num_cluster=6:
        #     shape = (8760, W) -> (6 * 168, W).

        #     self.FREQ_hours should reflect the number of hours throughout the year
        #     that each hour represents.

        #     After consulting the calculations below,
        #     note that (self.PROB_hours * 52).sum(axis=0) = 8736.
        #         - The last (365th) day is omitted.
        #     '''
        #     num_clusters=45
        #     (self.P_fore_w,
        #     self.lambda_DA_w,
        #     self.PROB_hours,
        #     weeks
        #     ) = week_reduction_by_scenario(
        #         fore_power_w=self.P_fore_w,
        #         lambda_DA_w=self.lambda_DA_w,
        #         num_clusters=num_clusters
        #     )
        #     self.FREQ_hours = self.PROB_hours * 52  # shape=(num_clusters * 168, W)

        #     # Reduce L_t from 8760 to 168*num_clusters.
        #     self.B_fore_arr_red = self.B_fore_arr[:168*52].reshape(52, 168)
        #     self.B_fore_arr_red = self.B_fore_arr_red[weeks].ravel()
        #     self.B_fore_arr = self.B_fore_arr_red.reshape(num_clusters*168, 1)
        # else:
        #     self.FREQ_hours = None

    def verify_batt_capacy_and_buyer_load(self):
        '''
        Check that the VRE, batt and buyer consumption levels are indeed below the zonal maximum.
        '''
        if self.ppa_data.E_buyer <= self.data.demand_inflexible_classic.sum(axis=0)[self.ppa_data.Z]:
            B_fore = self.ppa_data.E_buyer * self.fore_inflex_classic_pu
            self.B_fore = B_fore  # pd.Series
            self.B_fore_arr = B_fore.values.reshape(len(B_fore), 1)
        else:
            raise Exception(f"ValueError: The annual buyer consumption {self.ppa_data.E_buyer:.2f} exceeds the zonal total {self.data.demand_inflexible_classic.sum(axis=0)[ppa_data.Z]:.2f}")
        
        if not self.P_batt <= self.data.agg_bess.capacity_el[self.ppa_data.Z]:
            raise Exception(f"ValueError: The producer battery power capacity {self.P_batt:.2f} MW exceeds the zonal total {self.data.agg_bess.capacity_el[self.ppa_data.Z]:.2f} MW")
        elif not self.E_batt <= self.data.agg_bess.capacity_stor[self.ppa_data.Z]:
            raise Exception(f"ValueError: The producer battery energy capacity {self.E_batt:.2f} MWh exceeds the zonal total {self.data.agg_bess.capacity_stor[self.ppa_data.Z]:.2f} MWh")

    def calculate_normalized_forecasts(self):
        self.fore_solar_pv_pu = normalize_forecast_power(df=self.data.solar_pv_production, Z=self.ppa_data.Z)
        self.fore_on_wind_pu = normalize_forecast_power(df=self.data.wind_onshore_production, Z=self.ppa_data.Z)
        self.fore_off_wind_pu = normalize_forecast_power(df=self.data.wind_offshore_production, Z=self.ppa_data.Z)

        # Buyer's consumption forecast. Normalize by energy consumption
        self.fore_inflex_classic_pu = self.data.demand_inflexible_classic[self.ppa_data.Z] / self.data.demand_inflexible_classic.sum(axis=0)[self.ppa_data.Z]

    def calculate_forecasts(self):
        # Get the technology-specific installed capacity
        ppa_cfg_attrs = ["x_pv", "x_wind_on", "x_wind_off"]
        fore_attrs = ["solar_pv_production", "wind_onshore_production", "wind_offshore_production"]
        fore_pu_attrs = ["fore_solar_pv_pu", "fore_on_wind_pu", "fore_off_wind_pu"]
        new_capacity_attrs = ["capacity_solar_pv", "capacity_on_wind", "capacity_off_wind"]
        new_fore_attrs = ["P_fore_solar_pv", "P_fore_on_wind", "P_fore_off_wind"]

        for tech in range(len(ppa_cfg_attrs)):
            value = getattr(self.ppa_data, ppa_cfg_attrs[tech]) * self.ppa_data.P_vre
            # e.g.: capacity_solar_pv = x_pv * P_vre
            setattr(self, new_capacity_attrs[tech], value)

            if getattr(self, new_capacity_attrs[tech]) <= getattr(self.data, fore_attrs[tech]).max()[self.ppa_data.Z]:
                value = getattr(self, new_capacity_attrs[tech]) * getattr(self, fore_pu_attrs[tech])
                # e.g.: P_fore_solar_pv = capacity_solar_pv * fore_solar_pv_pu
                setattr(self, new_fore_attrs[tech], value)
            else:
                raise Exception(f"ValueError: The capacity ({new_capacity_attrs[tech]}) of the PPA producer {getattr(self, new_capacity_attrs[tech]):.2f} is higher than the total capacity in the bidding zone {self.data.__getattribute__(fore_attrs[tech]).max()[self.ppa_data.Z]:.2f}")

        # # Scale the normalized hourly forecasts to the installed capacity of the producer
        # if self.capacity_solar_pv <= self.data.solar_pv_production.max()[self.ppa_data.Z]:
        #     self.P_fore_solar_pv = self.capacity_solar_pv * self.fore_solar_pv_pu
        # else:
        #     raise Exception(f"ValueError: The solar capacity of the PPA producer {self.capacity_solar_pv:.2f} is higher than the total solar capacity in the bidding zone {self.data.solar_pv_production.max()[self.ppa_data.Z]:.2f}")
        
        # self.P_fore_on_wind = self.capacity_on_wind * self.fore_on_wind_pu
        # self.P_fore_off_wind = self.capacity_off_wind * self.fore_off_wind_pu

        # Total VRE forecast of producer
        self.P_fore = self.P_fore_solar_pv + self.P_fore_on_wind + self.P_fore_off_wind

    def calculate_batt_power(self):
        self.P_batt = self.ppa_data.P_vre * self.ppa_data.y_batt  # MW
        self.E_batt = self.P_batt / self.ppa_data.batt_Crate  # MWh

    def visualize_inputs(self, plot_hours=(0, 8760)):
        # Visualize the capacities as a bar plot
        tech = ["Offshore Wind", "Onshore Wind", "Solar PV", "BESS", "VRE total"]
        cap = [self.capacity_off_wind, self.capacity_on_wind, self.capacity_solar_pv, self.P_batt]
        tech_cap_pairs = list(zip(tech, cap))
        # Sort techs by capacity: low to high
        sorted_pairs = sorted(tech_cap_pairs, key=lambda x: x[1])
        # Separate into two lists
        tech_sorted, cap_sorted = zip(*sorted_pairs)
        tech = list(tech_sorted) + ["VRE total"]
        cap = list(cap_sorted) + [self.ppa_data.P_vre]
        
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

    # Placeholder function until we can load the data
    #   (VRE forecasts) from multiple weather years and
    #   the resulting power prices from the DA market model
    def generate_scenarios(self, yearly_param, noise_lvl=0.05, attr_name="attr_name"):
        '''
        Until we have run the DA model for multiple weather years, I need to make some synthetic data.
        '''
        np.random.seed(42)
        num_hours = yearly_param.shape[0]
        num_scens = 5

        # Initialize new matrix with 10 year-scenarios
        mult_year_param = np.zeros((num_hours, num_scens))
        mult_year_param[:, 0] = yearly_param.values

        # Generate noise
        for s in range(1, num_scens):
            noise = np.random.normal(loc=0, scale=noise_lvl, size=num_hours)
            mult_year_param[:, s] = mult_year_param[:, 0] * (1 + noise)
        mult_year_param = np.maximum(mult_year_param, 0)
        # E.g. P_fore_w
        setattr(self, attr_name, mult_year_param)


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
    alpha : float = 0.8

    nbs_setup_logger : Logger | None = None

    def __post_init__(self):
        self.nbs_setup_logger = self.nbs_setup_logger or utils.setup_logging(log_file="nbs.log")
        self.nbs_setup_logger.info("NBSSetup: SETUP NBS bounds and CVaR params.")


if __name__=="__main__":  
    load_plot_configs()  
    # Setup hyperparameters
    logger = utils.setup_logging(log_file="nbs.log")
    PPA_profile = "PaF"
    BL_compliance_rate = 0.0
    PPA_zone = "DELU"
    simulation_path = Path(f'simulations/scenario_1')

    # Instantiate objects and load power price results from DA market model
    data = DataLoader(
        input_path=Path(simulation_path / "data"),
        logger=logger,
    )

    ppa_data = PPAInputData(
        Z=PPA_zone,
        P_vre=1,  # MW
        x_pv=0.5,
        x_wind_on=0.3,
        x_wind_off=0.2,
        y_batt=0.25,  # p_batt/p_vre,
        batt_Crate=1,
        E_buyer=0.4*8760,  # MWh
        WTP=data.voll_classic,  # €/MWh
        ppa_logger=logger,
    )

    # Prepare forecasts for NBS modeling
    ppa_calcs = PPAInputCalcs(
        data=data,
        ppa_data=ppa_data,
        simulation_path=simulation_path,
        scenario_name="scenario_1",
        ppa_logger=logger,
    )
    ppa_calcs.visualize_inputs(plot_hours=(90*24, 90*24+168))

    nbs_setup = NBSSetup(
        S_LB=0,
        S_UB=80,
        M_LB=0,
        M_UB=1,
        gamma_LB=0,
        gamma_UB=1,
        beta_D=0.3,
        beta_O=0.5,
        alpha=0.9,
        nbs_setup_logger=logger,
    )

    nbs_model = NBSModel(
        PPA_profile=PPA_profile,
        BL_compliance_perc=BL_compliance_rate,
        P_fore_w=ppa_calcs.P_fore_w, #[:8736,:], # -> used to verify that FREQ_hours has been implemented correctly
        L_t=ppa_calcs.B_fore_arr, #[:8736],
        lambda_DA_w=ppa_calcs.lambda_DA_w, #[:8736, :],
        WTP=ppa_data.WTP,
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
