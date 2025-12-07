from pathlib import Path
from logging import Logger
from dataclasses import dataclass
from enlight.data_ops import DataLoader
import enlight.utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PPA_modeling import load_plot_configs, unify_palette_cyclers, prettify_subplots
from NBS_modeling import NBSModel

def normalize_forecast_power(df: pd.DataFrame, Z: str):
    return df[Z] / df.max()[Z]

@dataclass
class PPAConfig:
    '''
    Purely inputs.

    Pre-configures the parameters needed as inputs to create the data
    required on the producer side.
    '''
    # PPA profile and bidding zone
    PPA_profile : str = 'BL'
    Z : str = "DK1"

    # Power prices from DA model
    simulation_path : Path = Path("simulations/scenario_1")
    
    # Producer inputs
    P_vre : float = 1  # MW, total VRE capacity of producer
    x_pv : float = 1  # -, total solar pv share of VRE capacity
    x_wind_on : float = 0  # -, share of onshore wind
    x_wind_off : float = 0  # -, share of offshore wind
    y_batt : float = 0.25 # -, battery power ratio to VRE: P_batt/P_VRE
    batt_Crate : float = 1  # -, battery C-rate to determine energy capacity

    # Buyer inputs
    E_buyer : float = 2000  # MWh, buyer total annual electricity consumption
    WTP : float = 5000  # €/MWh, default is the typical VOLL of an inflexible load

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
    alpha : float = 0.9
    ppa_logger : Logger | None = None

    def __post_init__(self):
        self.ppa_logger = self.ppa_logger or utils.setup_logging(log_file="nbs.log")
        self.ppa_logger.info("SETUP PPA configuration")
        # If no power prices are provided as input, try to access the power prices of the first zone in scenario 1 as default.
        if self.simulation_path.exists():
            df_prices = pd.read_csv(self.simulation_path / "results/electricity_prices.csv", index_col=0)
            self.lambda_DA = df_prices[self.Z]
        else:
            raise Exception(f"FileNotFoundError: Please provide an existing simulation path. No power prices are given under {self.simulation_path}.")
        
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


@dataclass
class PrepareNBS:
    """
    Computation engine.

    Selects some of the energy system input data from a data loader instance and the power prices saved from a previous enlight DA run.
    
    Example usage:
    open interactive window in VSCode,
    >>> cd ../../
    run the script data_loader.py in the interactive window,
    >>> data = DataLoader(week=1, simulation_path='simulations/scenario_1/data')
    """
    data : DataLoader | None = None
    ppa_config : PPAConfig | None = None
    simulation_path: Path | None = None
    scenario_name : str = "scenario_1"
    nbs_data_logger: Logger | None = None

    def __post_init__(self):
        self.nbs_data_logger = self.nbs_data_logger or utils.setup_logging(name=__file__, log_file="nbs.log")
        self.nbs_data_logger.info("PREPARING NBS data")
        # Do we have a DataLoader object already?
        if self.data is None:
            # Can we use the input path to create a DataLoader instance, then?
            if self.simulation_path is not None:
                self.data = DataLoader(
                    input_path=Path(self.simulation_path / "data"),
                    logger=self.nbs_data_logger
                )
            # Raise exception and ask for new inputs.
            else:
                raise Exception(f"AttributeError: please insert an actual path to the data folder as NO DataLoader instance already exists.")
        
        # If no PPA configuration instance is input to the object, just try to use the default.
        if self.ppa_config is None:
            # The default only works if scenario_1 has been run.
            self.ppa_config = PPAConfig()

        
        # Retrieve the parameters needed for NBSModel:
        #   - P_fore, B_fore, WTP, lambda_DA
        self.calculate_normalized_forecasts()
        self.calculate_forecasts()  # P_fore, B_fore
        
        self.lambda_DA = self.ppa_config.lambda_DA
        self.WTP = self.ppa_config.WTP

        self.calculate_batt_power()
        self.generate_scenarios(self.P_fore, noise_lvl=1.0, attr_name="P_fore_w")
        self.generate_scenarios(self.lambda_DA, noise_lvl=0.5, attr_name="lambda_DA_w")
        # self.shorten_years()  # <- function required to actually run model. Too big if using 8760 hours...

    def verify_capacities(self):
        '''
        Check that the VRE, batt and buyer consumption levels are indeed below the zonal maximum.
        '''
        # Check solar
        # check on wind
        # check off wind
        # check cons
        # check batt

    def calculate_normalized_forecasts(self):
        self.fore_solar_pv_pu = normalize_forecast_power(df=self.data.solar_pv_production, Z=self.ppa_config.Z)
        self.fore_on_wind_pu = normalize_forecast_power(df=self.data.wind_onshore_production, Z=self.ppa_config.Z)
        self.fore_off_wind_pu = normalize_forecast_power(df=self.data.wind_offshore_production, Z=self.ppa_config.Z)

        # Buyer's consumption forecast. Normalize by energy consumption
        self.fore_inflex_classic_pu = self.data.demand_inflexible_classic[self.ppa_config.Z] / self.data.demand_inflexible_classic.sum(axis=0)[self.ppa_config.Z]

    def calculate_forecasts(self):
        # Get the technology-specific installed capacity
        ppa_cfg_attrs = ["x_pv", "x_wind_on", "x_wind_off"]
        fore_attrs = ["solar_pv_production", "wind_onshore_production", "wind_offshore_production"]
        fore_pu_attrs = ["fore_solar_pv_pu", "fore_on_wind_pu", "fore_off_wind_pu"]
        new_capacity_attrs = ["capacity_solar_pv", "capacity_on_wind", "capacity_off_wind"]
        new_fore_attrs = ["P_fore_solar_pv", "P_fore_on_wind", "P_fore_off_wind"]

        for tech in range(len(ppa_cfg_attrs)):
            value = getattr(self.ppa_config, ppa_cfg_attrs[tech]) * self.ppa_config.P_vre
            # e.g.: capacity_solar_pv = x_pv * P_vre
            setattr(self, new_capacity_attrs[tech], value)

            if getattr(self, new_capacity_attrs[tech]) <= getattr(self.data, fore_attrs[tech]).max()[self.ppa_config.Z]:
                value = getattr(self, new_capacity_attrs[tech]) * getattr(self, fore_pu_attrs[tech])
                # e.g.: P_fore_solar_pv = capacity_solar_pv * fore_solar_pv_pu
                setattr(self, new_fore_attrs[tech], value)
            else:
                raise Exception(f"ValueError: The capacity ({new_capacity_attrs[tech]}) of the PPA producer {getattr(self, new_capacity_attrs[tech]):.2f} is higher than the total solar capacity in the bidding zone {self.data.__getattribute__(fore_attrs[tech]).max()[self.ppa_config.Z]:.2f}")

        # # Scale the normalized hourly forecasts to the installed capacity of the producer
        # if self.capacity_solar_pv <= self.data.solar_pv_production.max()[self.ppa_config.Z]:
        #     self.P_fore_solar_pv = self.capacity_solar_pv * self.fore_solar_pv_pu
        # else:
        #     raise Exception(f"ValueError: The solar capacity of the PPA producer {self.capacity_solar_pv:.2f} is higher than the total solar capacity in the bidding zone {self.data.solar_pv_production.max()[self.ppa_config.Z]:.2f}")
        
        # self.P_fore_on_wind = self.capacity_on_wind * self.fore_on_wind_pu
        # self.P_fore_off_wind = self.capacity_off_wind * self.fore_off_wind_pu

        # Total VRE forecast of producer
        self.P_fore = self.P_fore_solar_pv + self.P_fore_on_wind + self.P_fore_off_wind

        # To avoid method overloading the buyer's consumption is scaled here as well
        if self.ppa_config.E_buyer <= self.data.demand_inflexible_classic.sum(axis=0)[self.ppa_config.Z]:
            B_fore = self.ppa_config.E_buyer * self.fore_inflex_classic_pu
            self.B_fore = B_fore  # pd.Series
            self.B_fore_arr = B_fore.values.reshape(len(B_fore), 1)
        else:
            raise Exception(f"ValueError: The annual buyer consumption {self.ppa_config.E_buyer:.2f} exceeds the zonal total {self.data.demand_inflexible_classic.sum(axis=0)[ppa_config.Z]:.2f}")

    def calculate_batt_power(self):
        self.P_batt = self.ppa_config.P_vre * self.ppa_config.y_batt  # MW
        self.E_batt = self.P_batt / self.ppa_config.batt_Crate  # MWh

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
        cap = list(cap_sorted) + [self.ppa_config.P_vre]
        
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
        ax.set_title(f'PPA (in {self.ppa_config.Z}) producer capacities and average buyer consumption', loc='left')
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
        ax.set_title(f'PPA (in {self.ppa_config.Z}) producer generation and buyer consumption profiles', loc='left')
        fig.tight_layout()
        plt.show()

        # Visualize power prices as a simple line plot
        fig, ax = plt.subplots(figsize=(12,6))
        unify_palette_cyclers(ax)
        sns.lineplot(ax=ax, data=self.lambda_DA, label=r"$\lambda^{DA}_t$")
        prettify_subplots(ax)
        ax.set_ylabel(f"Power price in {self.ppa_config.Z} [€/MWh]")
        ax.legend().remove()
        plt.show()

    # Placeholder function until we can load the data
    #   (VRE forecasts) from multiple weather years and
    #   the resulting power prices from the DA market model
    def generate_scenarios(self, yearly_param, noise_lvl=0.05, attr_name="attr_name"):
        '''
        Until we have run the DA model for multiple weather years, I need to make some synthetic data.
        '''
        num_hours = yearly_param.shape[0]
        num_scens = 10

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


if __name__=="__main__":  
    load_plot_configs()  
    # Setup hyperparameters
    logger = utils.setup_logging(log_file="nbs.log")
    scenario_name = 'scenario_1'
    PPA_zone = "DELU"

    # Setup file paths
    simulation_path = Path(f'simulations/{scenario_name}')
    data_path = simulation_path / "data"

    # Instantiate objects and load power price results from DA market model
    data = DataLoader(
        input_path=Path(data_path),
        logger=logger,
    )
    
    ppa_config = PPAConfig(
        PPA_profile='BL',
        Z=PPA_zone,
        simulation_path=simulation_path,
        P_vre=1,  # MW
        x_pv=0.5,
        x_wind_on=0.3,
        x_wind_off=0.2,
        y_batt=0.25,  # p_batt/p_vre,
        batt_Crate=1,
        E_buyer=0.4*8760,  # MWh
        WTP=data.voll_classic,  # €/MWh
        S_UB=100,  # €/MWh
        M_UB=1.0,  # MW
    )

    # Prepare NBS data for NBSModel
    # # Possible use
    # pnbs2 = PrepareNBS(
    #     ppa_config=ppa_config,
    #     simulation_path=simulation_path,
    #     scenario_name=scenario_name,
    #     logger=logger,
    # )
    # Intended use
    pnbs = PrepareNBS(
        data=data,
        ppa_config=ppa_config,
        scenario_name=scenario_name,
        logger=logger,
    )

    pnbs.visualize_inputs(plot_hours=(90*24, 90*24+168))

# from data_loader
    # .wind_offshore_production
    # .wind_onshore_production
    # .solar_pv_production
    # .hydro_ror_production
    # .demand_inflexible_classic
    # potentially:
    #   .flexible_demands_dfs['demand_flexible_classic']
# results from DA market model
    # lambda_zt = pd.read_csv("simulations/scenario_1/results/electricity_prices.csv", index_col="T")

# verify that zones in data and DA results are the same.
    # (runner.data.bidding_zones == dft.columns).all()

