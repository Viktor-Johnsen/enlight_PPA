from pathlib import Path
from dataclasses import dataclass
import numpy as np
import time
import yaml
import pickle
from gurobipy import GRB

from enlight.data_ops import DataLoader
import enlight.utils as utils
from PPA_modeling import load_plot_configs, unify_palette_cyclers, prettify_subplots
from ppa_input import PPAInputData, PPAInputCalcs, NBSSetup#, generate_scenarios
from nbs_modeling import NBSModel, NBSMultModel, var_to_pandas #, generate_data

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
                 PPA_profile = "BL",
                 BL_compliance_rate = 0.0,
                 PPA_zone = "DK1",
                 P_vre : float = 1,
                 x_tot_Z : float = 0,
                 x_pv : float = 0.4,
                 x_wind_on :float = 0.3,
                 x_wind_off : float = 0.3,
                 x_buyer : float = 0.4,
                 y_batt : float = 0.25,
                 S_UB : float = 250,
                 ) -> None:
        """Initialize the NBSRunner."""
        # Not an input:
        self.config_path: Path = Path("config") / "config.yaml"

        # Inputs
        self.nbs_runner_logger = utils.setup_logging(log_file="nbs.log")
        self.nbs_runner_logger.info("NBSRunner: INITIALIZATION")        

        self.PPA_profile = PPA_profile
        self.BL_compliance_rate = BL_compliance_rate
        self.PPA_zone = PPA_zone
        self.P_vre = P_vre
        self.x_tot_Z = x_tot_Z
        self.x_pv = x_pv
        self.x_wind_on = x_wind_on
        self.x_wind_off = x_wind_off
        self.x_buyer = x_buyer
        self.y_batt = y_batt
        self.S_UB = S_UB

        load_plot_configs()  # conform plotting palette and more
        
        # Load scenario list
        self._load_config()

        # Setup general NBS (incl. CVaR) parameters.
        self._setup_PPA__NBS_data()
    
        # Load DA forecasts and capacities as well as results (only power prices)
        #   and combine these for all of the scenarios.
        self._combine_scenarios_to_single_arrs()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Load the configuration from the YAML file
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config_yaml = yaml.safe_load(file)

        # Extract configuration values
        self.scenario_list: list[str] = self.config_yaml.get("scenario_list", [])
        self.W = len(self.scenario_list)

        # Register into the logger
        self.nbs_runner_logger.info("Loaded scenario list of %d scenarios from config.", len(self.scenario_list))

    def _setup_PPA__NBS_data(self) -> None:
        '''
        This method instantiates classes that are general across ALL scenarios.
        '''
        # 1) Instantiate a PPA data instance
        self.ppa_data = PPAInputData(
            Z=self.PPA_zone,
            P_vre=self.P_vre,  # MW
            x_tot_Z=self.x_tot_Z,  # p.u.
            x_pv=self.x_pv,
            x_wind_on=self.x_wind_on,
            x_wind_off=self.x_wind_off,
            y_batt=self.y_batt,  # p_batt/p_vre,
            batt_eta=float(np.sqrt(0.9)),
            batt_Crate=1,
            x_buyer=self.x_buyer,  # P_buyer.mean()/P_vre
            ppa_logger=self.nbs_runner_logger,
        )
        # 2) Instantiate a NBS setup instance
        self.nbs_setup = NBSSetup(
            S_LB=0,
            S_UB=self.S_UB,
            M_LB=0,
            M_UB=self.P_vre,
            gamma_LB=0,
            gamma_UB=1,
            beta_D=0.5,
            beta_O=0.1,
            alpha=0.75,
            nbs_setup_logger=self.nbs_runner_logger,
        )

    def _combine_scenarios_to_single_arrs(self) -> None:
        '''
        This method creates DataLoader and PPAInputCalcs instances for each
        scenario included in the yaml config file.

        The forecasts from the DataLoader instances are used for calculations
        in the PPAInputCalcs instance primarily so as to tailor the forecasts to the specific
        producer and buyer. Secondarily it also calculates the battery power and
        verifies the inputs are valid in the given bidding zone.
        '''
        # Instantiate DataLoader objects for each scenario
        da_data_dict = {}
        # Instantiate PPA calculation instances for each scenario
        ppa_calcs_dict = {}
        # Create dicts for the forecasts and prices (not to be attributes)
        P_fore_dict = {}
        lambda_DA_dict = {}

        # Create the instances and saves dicts as attributes:
        for w in self.scenario_list:
            # 3) Instantiate the DA data instance
            da_data_dict[w] = DataLoader(
                scenario_name=w,
                logger=self.nbs_runner_logger,
            )
            # 4) Instantiate the NBS data calculations instance
            ppa_calcs_dict[w] = PPAInputCalcs(
                scenario_name=w,
                da_data=da_data_dict[w],
                ppa_data=self.ppa_data,
                ppa_logger=self.nbs_runner_logger,
            )
            P_fore_dict[w] = ppa_calcs_dict[w].P_fore
            lambda_DA_dict[w] = ppa_calcs_dict[w].lambda_DA

        self.da_data_dict = da_data_dict
        self.ppa_calcs_dict = ppa_calcs_dict

        # self.T = len(P_fore_dict[self.scenario_list[0]])  # = 8760
        self.P_fore_w = np.column_stack(
            [P_fore_dict[w] for w in self.scenario_list]
        )
        self.lambda_DA_w = np.column_stack(
            [lambda_DA_dict[w] for w in self.scenario_list]
        )
        # self.P_fore_ws = np.vstack(list(P_fore_dict.values())).reshape(self.T, self.W)
        # self.lambda_DA_ws = np.vstack(list(lambda_DA_dict.values())).reshape(self.T, self.W)

# # single_nbs does NOT currently work as intended...
#     def single_nbs(self, scenario_name : str = "scenario_1") -> None:
#         # Arbitrary scenario used to retrieve data that is
#         # NOT scenario-specific from DataLoader of PPAInputCalcs objects:
#         w0 = self.scenario_list[0]

#         # 5) Instantiate and run an NBSModel
#         self.nbs_model = NBSModel(
#             PPA_profile=self.PPA_profile,  # BL or PaP
#             BL_compliance_perc=self.BL_compliance_rate,
#             P_fore_w=self.P_fore_w,
#             P_batt=self.ppa_calcs_dict[w0].P_batt,  # use any scenario
#             batt_eta=self.ppa_data.batt_eta,
#             batt_Crate=self.ppa_data.batt_Crate,
#             L_t=self.ppa_calcs_dict[w0].B_fore_arr,  # use any scenario
#             lambda_DA_w=self.lambda_DA_w,
#             WTP=self.da_data_dict[w0].voll_classic,  # use any scenario
#             S_UB=self.nbs_setup.S_UB,  # PPA price
#             M_UB=self.nbs_setup.M_UB,  # BL volume
#             gamma_UB=self.nbs_setup.gamma_UB,
#             beta_D=self.nbs_setup.beta_D,
#             beta_O=self.nbs_setup.beta_O,
#             alpha=self.nbs_setup.alpha,
#             nbs_model_logger=self.nbs_runner_logger,
#             # LBs, hp, and add_batt have been left as default values.
#         )

#         self.nbs_model.solve_model()
#         self.ppa_calcs_dict[w0].visualize_inputs(plot_hours=(100*24, 100*24+168))
#         self.nbs_model.visualize_example_profit_dist(bars=True)
#         self.nbs_model.visualize_example_outcome(show_all_scens=True)
#         self.nbs_model.verify_behaviour(w_BESS=0)
#         self.nbs_runner_logger.info("RAN a single NBS in NBSRunner with betas: (O:{self.ppa_config.beta_O}, D:{self.ppa_config.beta_D}) -- (profile: {self.PPA_profile})")        

    def mult_nbs(self, beta_O_list, beta_D_list) -> None:
        self.mult_nbs_has_run = True
        self.beta_O_list = beta_O_list
        self.beta_D_list = beta_D_list
        # Arbitrary scenario used to retrieve data that is
        # NOT scenario-specific from DataLoader of PPAInputCalcs objects:
        w0 = self.scenario_list[0]

        # 5) Instantiate and run multiple NBSModels in an NBSMultModel
        self.mult_nbs_models = NBSMultModel(
            PPA_profile=self.PPA_profile,  # Type of PPA profile ('PaF', 'PaP', or 'BL')
            BL_compliance_perc=self.BL_compliance_rate, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
            P_fore_w=self.P_fore_w,  # combined scens
            P_batt=self.ppa_calcs_dict[w0].P_batt,  # use any scenario
            batt_eta=self.ppa_data.batt_eta,
            batt_Crate=self.ppa_data.batt_Crate,
            L_t=self.ppa_calcs_dict[w0].B_fore_arr,  # use any scenario
            lambda_DA_w=self.lambda_DA_w,  # combined scens
            WTP=self.da_data_dict[w0].voll_classic,  # use any scenario
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
   
    def save_mult_nbs(self) -> dict:
        if hasattr(self, "mult_nbs_has_run"):
            if self.mult_nbs_has_run:
                # Initialize a results dict
                res = {}

                # Save the PPA strike price and volume dicts:
                res["res_S"] = self.mult_nbs_models.results_S
                if self.mult_nbs_models.PPA_profile == "BL":
                    res["res_M"] = self.mult_nbs_models.results_volume
                else:
                    res["res_gamma"] = self.mult_nbs_models.results_volume

                # Save the results of each individual NBS model to allow for later inspection
                for beta_O in self.beta_O_list:
                    res[beta_O] = {}
                    for beta_D in self.beta_D_list:
                        d = self.mult_nbs_models.models[beta_O][beta_D]
                        # Do not load the "results" of any model that did not successfully run.
                        if d.model.Status == GRB.OPTIMAL:
                            res[beta_O][beta_D] = d.get_results()

                with open(f"mult_nbs_results__{self.PPA_profile}_{self.BL_compliance_rate}_{self.beta_O_list}.pkl", "wb") as f:
                    pickle.dump(res, f)


if __name__=="__main__":
    t0 = time.time()
    # Setup hyperparameters
    PPA_profile = "PaF"
    BL_compliance_perc = 0
    PPA_zone = "DK2"

    # Producer VRE capacity, Buyer annual consumption, strike price upper bound
    P_vre = 1  # MW
    x_buyer = 0.3  # ratio of average buyer power consumption relative to capacity of Producer VRE
    y_batt = 0.25  # MW_batt / MW_VRE
    S_UB = 50  # €/MWh

    # Instantiate objects and load power price results from DA market model
    nbs_runner = NBSRunner(
                    PPA_profile=PPA_profile,
                    BL_compliance_rate=BL_compliance_perc,
                    PPA_zone=PPA_zone,
                    P_vre=P_vre,
                    # x_tot_Z, x_pv, x_wind_on, x_wind_off
                    x_buyer=x_buyer,
                    y_batt=y_batt,
                    S_UB=S_UB,
    )

    # nbs_runner.single_nbs(scenario_name="scenario_2")
    # d=nbs_runner.nbs_model
    # print(f"S = {d.S.X:.2f} €/MWh, volume = {d.M.X if d.PPA_profile=="BL" else d.gamma.X:.2f} {"MW" if d.PPA_profile=="BL" else "%"}")

    # Define ranges for beta
    beta_D_list = np.round(np.arange(0.0,0.41, 0.2), 2)  # avoid floating point issues
    beta_O_list = np.round(np.arange(0.0,0.41, 0.2), 2)  # avoid floating point issues
    nbs_runner.mult_nbs(beta_O_list=beta_O_list,
                        beta_D_list=beta_D_list)

    # Save results in a pickle file
    nbs_runner.save_mult_nbs()

    # Verify combliance rate
    d=nbs_runner.mult_nbs_models.models[beta_O_list[1]][beta_D_list[0]]

    d.visualize_example_profit_dist(bars=True)
    d.visualize_example_outcome(show_all_scens=True)
    d.verify_behaviour(w_BESS=3)
    nbs_runner.ppa_calcs_dict["scenario_1"].visualize_inputs(plot_hours=(100*24, 100*24+168))
    

    # d=nbs_runner.nbs_model
    if d.BL_compliance_perc > 0:
        print(d.v_min.X.sum(axis=0) / (d.T * d.M.X))
    print(f"That took {time.time()-t0:.2f} s")

    '''
    Temporarily plotting input data down here
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    data = nbs_runner.mult_nbs_models.models[0.1][0.1]
    hours = range(180*24,180*24+72)
    alphas = np.linspace(1.0, 0.25, data.P_fore_w.shape[1])
    load_plot_configs()
    fig, ax = plt.subplots(2, 1, figsize=(12,6))
    # --- First plot ---
    for i, a in enumerate(alphas):
        ax[0].plot(data.P_fore_w[hours, i], alpha=a, label=f"Producer: Scen. {i+1}", ls='--')

    ax[0].plot(data.L_t[hours], linewidth=2, label="Buyer")
    # plt.legend()
    # plt.show()
    ax[0].set_title("Forecasts [MW]", loc='left')

    # --- Second plot ---
    for i, a in enumerate(alphas):
        ax[1].plot(data.lambda_DA_w[hours, i], alpha=a, label=f" Scen. {i+1}")
    ax[1].set_xlabel("Hour")
    ax[1].set_title("DA price [€/MWh]", loc='left')
    
    prettify_subplots(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    for i in range(d.lambda_DA_w.shape[1]):
        ldc = sorted(d.lambda_DA_w[:,i])[::-1]
        ax.plot(ldc, label=f"Scen. {i+1}")
    prettify_subplots(ax)
    ax.set_xlabel("Hours [h]")
    ax.set_title("Price-duration curve [€/MWh]", loc="left")
    plt.show()