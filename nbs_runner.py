from pathlib import Path
import numpy as np
from enlight.data_ops import DataLoader
import enlight.utils as utils
from PPA_modeling import load_plot_configs, unify_palette_cyclers, prettify_subplots
from nbs_data_loader import PrepareNBS, PPAConfig
from NBS_modeling import NBSModel, NBSMultModel
from NBS_modeling import generate_data
import time


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
                 BL_compliance_rate = 0.0,
                 PPA_zone = "DK1",
                 simulation_path = Path("simulations/scenario_1"),
                 P_vre : float = 1,
                 E_buyer : float = 0.4*8760,
                 ) -> None:
        """Initialize the NBSRunner."""
        self.nbs_runner_logger = utils.setup_logging(log_file="nbs.log")
        self.nbs_runner_logger.info("INITIALIZE NBSRunner")        

        self.scenario_name = scenario_name
        self.PPA_profile = PPA_profile
        self.BL_compliance_rate = BL_compliance_rate
        self.PPA_zone = PPA_zone
        self.simulation_path = simulation_path
        self.P_vre = P_vre
        self.E_buyer = E_buyer

        load_plot_configs()  # conform plotting palette and more
        self.setup_config_data()

    def setup_config_data(self) -> None:
        # 1) Instantiate a DataLoader object
        self.data = DataLoader(
            input_path=Path(self.simulation_path / "data"),
            logger=self.nbs_runner_logger,
        )
        # 2) Instantiate a PPAConfig from a few user inputs
        self.ppa_config = PPAConfig(
            PPA_profile=self.PPA_profile,
            Z=self.PPA_zone,
            simulation_path=self.simulation_path,
            P_vre=self.P_vre,  # MW
            x_pv=0.4,
            x_wind_on=0.3,
            x_wind_off=0.3,
            y_batt=0.25,  # p_batt/p_vre,
            batt_Crate=1,
            E_buyer=self.E_buyer,  # MWh
            WTP=self.data.voll_classic,  # €/MWh
            S_LB=48,
            S_UB=148,  # €/MWh
            M_UB=self.P_vre,  # MW
            beta_D=0.2,
            beta_O=0.4
        )
        # 3) Instantiate a PrepareNBS with user inputs and the PPAConfig
        self.nbs_data = PrepareNBS(
            data=self.data,
            ppa_config=self.ppa_config,
            scenario_name=self.scenario_name,
            nbs_data_logger=self.nbs_runner_logger
        )

    def single_nbs(self) -> None:
        # 4) Instantiate and run an NBSModel
        self.nbs_model = NBSModel(
            PPA_profile=self.ppa_config.PPA_profile,  # BL or PaP
            BL_compliance_perc=self.BL_compliance_rate,
            P_fore_w=self.nbs_data.P_fore_w,
            L_t=self.nbs_data.B_fore_arr,
            lambda_DA_w=self.nbs_data.lambda_DA_w,
            WTP=self.ppa_config.WTP,
            S_UB=self.ppa_config.S_UB,  # PPA price
            M_UB=self.ppa_config.M_UB,  # BL volume
            gamma_UB=self.ppa_config.gamma_UB,
            beta_D=self.ppa_config.beta_D,
            beta_O=self.ppa_config.beta_O,
            alpha=self.ppa_config.alpha,
            nbs_model_logger=self.nbs_runner_logger,
            # LBs, hp, and add_batt have been left as default values.
        )

        self.nbs_model.solve_model()

        self.nbs_data.visualize_inputs(plot_hours=(90*24, 90*24+168))
        self.nbs_model.visualize_example_profit_hists()
        self.nbs_model.visualize_example_outcome()
        self.nbs_model.verify_behaviour()

        self.nbs_runner_logger.info("RAN a single NBS in NBSRunner with betas: (O:{self.ppa_config.beta_O}, D:{self.ppa_config.beta_D}) -- (profile: {self.PPA_profile})")        

    def mult_nbs(self, beta_O_list, beta_D_list) -> None:
        # self.mult_nbs_models = NBSMultModel(
        #     PPA_profile=self.ppa_config.PPA_profile,  # Type of PPA profile ('PaF', 'PaP', or 'BL')
        #     BL_compliance_perc=0.75, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
        #     P_fore_w=self.nbs_data.P_fore_w,
        #     L_t=self.nbs_data.B_fore_arr,
        #     lambda_DA_w=self.nbs_data.lambda_DA_w,
        #     WTP=self.ppa_config.WTP,
        #     # add_batt=True,
        #     S_LB=self.ppa_config.S_LB,  # Minimum PPA strike price
        #     S_UB=self.ppa_config.S_UB,  # Maximum PPA strike price
        #     M_LB=self.ppa_config.M_LB,  # BL: Minimum baseload volume
        #     M_UB=self.ppa_config.M_UB,  # BL: Maximum baseload volume
        #     gamma_LB=self.ppa_config.gamma_LB, # PaP: Minimum PPA capacity share volume
        #     gamma_UB=self.ppa_config.gamma_UB, # PaP: Minimum PPA capacity share volume
        #     alpha=self.ppa_config.alpha,  # CVaR: Tail of interest for CVaR
        # )
        # trying to improve stability and runtime
        P_fore_w, lambda_DA_w, L_t, WTP = generate_data()
        
        self.mult_nbs_models = NBSMultModel(
            PPA_profile=self.PPA_profile,  # Type of PPA profile ('PaF', 'PaP', or 'BL')
            BL_compliance_perc=self.BL_compliance_rate, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
            P_fore_w=P_fore_w,
            L_t=L_t,
            lambda_DA_w=lambda_DA_w,
            WTP=WTP,
            # add_batt=True,
            S_LB=96.38*0.5,  # Minimum PPA strike price
            S_UB=98.1*1.5,  # Maximum PPA strike price
            M_LB=0,  # BL: Minimum baseload volume
            M_UB=0.99,  # BL: Maximum baseload volume
            gamma_LB=0, # PaP: Minimum PPA capacity share volume
            gamma_UB=1, # PaP: Minimum PPA capacity share volume
            alpha=0.9,  # CVaR: Tail of interest for CVaR
            nbs_mult_logger=self.nbs_runner_logger,
        )

        self.mult_nbs_models.run_multiple_NBS_models(beta_O_list=beta_O_list, beta_D_list=beta_D_list)

        self.nbs_runner_logger.info("RAN multiple NBS in NBSRunner")        

        self.mult_nbs_models.visualize_risk_impact_heatmap()
   
if __name__=="__main__":
    # Setup hyperparameters
    PPA_profile = "BL"
    PPA_zone = "DELU"
    BL_compliance_perc = 0.9

    # Setup file paths
    simulation_path = Path(f'simulations/scenario_1')

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    BL_compliance_rate=BL_compliance_perc,
                    PPA_zone=PPA_zone,
                    simulation_path=simulation_path,
                    P_vre=1,
                    E_buyer=0.1*8760,
    )

    # nbs_runner.single_nbs()

    # Define ranges for betas
    beta_D_list = np.round(np.arange(0.8, 1.01, 0.1), 2)  # avoid floating point issues
    beta_O_list = np.round(np.arange(0.8, 1.01, 0.1), 2)  # avoid floating point issues
    nbs_runner.mult_nbs(beta_O_list=beta_O_list,
                        beta_D_list=beta_D_list)
