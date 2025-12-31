from logging import Logger
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Module imports
from enlight_PPA.config_path import SIMULATIONS_DIR
from enlight_PPA.data_ops import DataLoader
import enlight_PPA.utils as utils  # <- for logging setup
from enlight_PPA.utils.nbs_utils import unify_palette_cyclers, prettify_subplots, normalize_forecast_power
 
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
        self.ppa_logger.info(f"PPAInputCalcs: CALCULATE power forecast, batt specs, buyer profile, and load power prices for {self.scenario_name}.")

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
        prices_file = SIMULATIONS_DIR / f"{self.scenario_name}/results/electricity_prices.csv"
        if prices_file.exists():
            df_prices = pd.read_csv(prices_file, index_col=0)
            self.lambda_DA = df_prices[self.ppa_data.Z]
        else:
            raise Exception(f"FileNotFoundError: Please provide an existing scenario name. No power prices are given under (full file path shown) {prices_file}.")

    def calculate_batt_power(self):
        if self.ppa_data.x_tot_Z > 0:
            # capacity share at bidding zone level
            # If using zonal capacity, we also use the zonal capacity shares. It's easier that way.
            # Overwrite, y_batt and batt_Crate
            self.P_bess_Z = self.da_data.bess_units_el_cap[0,:][np.array(self.da_data.bidding_zones) == self.ppa_data.Z][0]  # [0] turns array([float]) -> float
            self.E_bess_Z = self.da_data.bess_units_storage_cap[0,:][np.array(self.da_data.bidding_zones) == self.ppa_data.Z][0]

            self.P_batt = self.P_bess_Z * self.ppa_data.x_tot_Z
            self.E_batt = self.E_bess_Z * self.ppa_data.x_tot_Z
            # Overwrite any user-provided y_batt value
            self.y_batt = self.P_batt / self.P
        else:
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
