from pathlib import Path
from dataclasses import dataclass
import numpy as np
import time

from enlight.data_ops import DataLoader
import enlight.utils as utils
from PPA_modeling import load_plot_configs, unify_palette_cyclers, prettify_subplots
from ppa_input import PPAInputData, PPAInputCalcs, NBSSetup
from nbs_modeling import NBSModel, NBSMultModel, generate_data

# Kmedoids conda env requires:
# conda create -n kmedoids python=3.10 numpy=1.26 scikit-learn=1.3 scikit-learn-extra
# pip install pyyaml pandas xarray gurobipy matplotlib seaborn tqdm

class NBSRunner:
    """
    Uses classes:
        - PPAConfig (ppa_config)
        - DataLoader (data)
        - PrepareNBS (nbs_data)
        - NBSModel (nbs_model)
    To effectively load the DA input data relevant for PPA negotiations
    as well as the results (power prices) from the DA market model.
    Finally, the NBS model is built and solved using Gurobipy.

    The NBSRunner handles configuration loading, data preparation, and model execution
    for NBS PPA modeling scenarios.
    """
    def __init__(self,
                 scenario_name : str = "scenario_1",
                 PPA_profile = "BL",
                 reduce_number_of_weeks : bool = False,
                 BL_compliance_rate = 0.0,
                 PPA_zone = "DK1",
                 simulation_path = Path("simulations/scenario_1"),
                 P_vre : float = 1,
                 E_buyer : float = 0.4*8760,
                 y_batt : float = 0.25,
                 S_UB : float = 250,
                 ) -> None:
        """Initialize the NBSRunner."""
        self.nbs_runner_logger = utils.setup_logging(log_file="nbs.log")
        self.nbs_runner_logger.info("NBSRunner: INITIALIZATION")        

        self.scenario_name = scenario_name
        self.PPA_profile = PPA_profile
        self.reduce_number_of_weeks = reduce_number_of_weeks
        self.BL_compliance_rate = BL_compliance_rate
        self.PPA_zone = PPA_zone
        self.simulation_path = simulation_path
        self.P_vre = P_vre
        self.E_buyer = E_buyer
        self.y_batt = y_batt
        self.S_UB = S_UB

        load_plot_configs()  # conform plotting palette and more
        self.setup_config_data()

    def setup_config_data(self) -> None:
        # 1) Instantiate a DataLoader object
        self.data = DataLoader(
            input_path=Path(self.simulation_path / "data"),
            logger=self.nbs_runner_logger,
        )
        # 2) Instantiate a PPA data instance
        self.ppa_data = PPAInputData(
            Z=self.PPA_zone,
            P_vre=self.P_vre,  # MW
            x_pv=0.4,
            x_wind_on=0.3,
            x_wind_off=0.3,
            y_batt=self.y_batt,  # p_batt/p_vre,
            batt_Crate=1,
            E_buyer=self.E_buyer,  # MWh
            WTP=self.data.voll_classic,  # €/MWh
            ppa_logger=self.nbs_runner_logger,
        )
        # 3) Instantiate a PPA calculation instance with user inputs and the PPAConfig
        self.ppa_calcs = PPAInputCalcs(
            data=self.data,
            ppa_data=self.ppa_data,
            simulation_path=self.simulation_path,
            scenario_name=self.scenario_name,
            ppa_logger=self.nbs_runner_logger,
        )
        if self.reduce_number_of_weeks:
            self.ppa_calcs.hour_reduction(num_clusters=5)

        # 4) Instantiate a NBS setup instance
        self.nbs_setup = NBSSetup(
            S_LB=0,
            S_UB=self.S_UB,
            M_LB=0,
            M_UB=self.P_vre,
            gamma_LB=0,
            gamma_UB=1,
            beta_D=0.5,
            beta_O=0.1,
            alpha=0.8,
            nbs_setup_logger=self.nbs_runner_logger,
        )

    def single_nbs(self) -> None:
        # 4) Instantiate and run an NBSModel
        self.nbs_model = NBSModel(
            PPA_profile=self.PPA_profile,  # BL or PaP
            BL_compliance_perc=self.BL_compliance_rate,
            P_fore_w=(self.ppa_calcs.P_fore_w_red if self.reduce_number_of_weeks
                      else self.ppa_calcs.P_fore_w),
            L_t=(self.ppa_calcs.B_fore_arr_red if self.reduce_number_of_weeks
                 else self.ppa_calcs.B_fore_arr),
            lambda_DA_w=(self.ppa_calcs.lambda_DA_w_red if self.reduce_number_of_weeks
                         else self.ppa_calcs.lambda_DA_w),
            WTP=self.ppa_data.WTP,
            S_UB=self.nbs_setup.S_UB,  # PPA price
            M_UB=self.nbs_setup.M_UB,  # BL volume
            gamma_UB=self.nbs_setup.gamma_UB,
            beta_D=self.nbs_setup.beta_D,
            beta_O=self.nbs_setup.beta_O,
            alpha=self.nbs_setup.alpha,
            nbs_model_logger=self.nbs_runner_logger,
            # LBs, hp, and add_batt have been left as default values.
        )

        self.nbs_model.solve_model()

        self.ppa_calcs.visualize_inputs(plot_hours=(90*24, 90*24+168))
        self.nbs_model.visualize_example_profit_hists()
        self.nbs_model.visualize_example_outcome()
        self.nbs_model.verify_behaviour()

        self.nbs_runner_logger.info("RAN a single NBS in NBSRunner with betas: (O:{self.ppa_config.beta_O}, D:{self.ppa_config.beta_D}) -- (profile: {self.PPA_profile})")        

    def mult_nbs(self, beta_O_list, beta_D_list) -> None:
        # only for testing the classes before using real DA input and results
        P_fore_w, lambda_DA_w, L_t, WTP = generate_data()
        up_to = 8760
        self.mult_nbs_models = NBSMultModel(
            PPA_profile=self.PPA_profile,  # Type of PPA profile ('PaF', 'PaP', or 'BL')
            BL_compliance_perc=self.BL_compliance_rate, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
            P_fore_w=self.ppa_calcs.P_fore_w, #[:up_to, :],
            L_t=self.ppa_calcs.B_fore_arr, #[:up_to],
            lambda_DA_w=self.ppa_calcs.lambda_DA_w, #[:up_to, :],
            WTP=self.ppa_data.WTP,
            # add_batt=None,
            S_LB=self.nbs_setup.S_LB,  # Minimum PPA strike price
            S_UB=self.nbs_setup.S_UB,  # Maximum PPA strike price
            M_LB=self.nbs_setup.M_LB,  # BL: Minimum baseload volume
            M_UB=self.nbs_setup.M_UB,  # BL: Maximum baseload volume
            gamma_LB=self.nbs_setup.gamma_LB, # PaP: Minimum PPA capacity share volume
            gamma_UB=self.nbs_setup.gamma_UB, # PaP: Minimum PPA capacity share volume
            alpha=self.nbs_setup.alpha,  # CVaR: Tail of interest for CVaR
            nbs_mult_logger=self.nbs_runner_logger,
        )

        self.mult_nbs_models.run_multiple_NBS_models(beta_O_list=beta_O_list, beta_D_list=beta_D_list)

        self.nbs_runner_logger.info("RAN multiple NBS in NBSRunner")        

        self.mult_nbs_models.visualize_risk_impact_heatmap()
   
if __name__=="__main__":
    t0 = time.time()
    # Setup hyperparameters
    PPA_profile = "BL"
    BL_compliance_perc = 0.1
    PPA_zone = "DELU"
    
    # Setup file paths
    simulation_path = Path(f'simulations/scenario_1')

    # Producer VRE capacity, Buyer annual consumption, strike price upper bound
    P_vre = 1  # MW
    E_buyer = 0.4*8760  # MWh
    y_batt = 0.25  # MW_batt / MW_VRE
    S_UB = 150  # €/MWh

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    reduce_number_of_weeks=True,
                    BL_compliance_rate=BL_compliance_perc,
                    PPA_zone=PPA_zone,
                    simulation_path=simulation_path,
                    P_vre=P_vre,
                    E_buyer=E_buyer,
                    y_batt=y_batt,
                    S_UB=S_UB,
    )

    # nbs_runner.single_nbs()

    # Define ranges for betas
    beta_D_list = np.round(np.arange(0.05, 1.01, 0.2), 2)  # avoid floating point issues
    beta_O_list = np.round(np.arange(0.05, 1.01, 0.2), 2)  # avoid floating point issues
    nbs_runner.mult_nbs(beta_O_list=beta_O_list,
                        beta_D_list=beta_D_list)

    # Verify combliance rate
    d=nbs_runner.mult_nbs_models.models[0.4][0.4]
    if d.BL_compliance_perc > 0:
        print(d.v_min.X.sum(axis=0) / (d.T * d.M.X))
    print(f"That took {time.time()-t0:.2f} s")