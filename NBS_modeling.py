import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import enlight.utils as utils
import hybrid_vre_in_da as hv
from PPA_modeling import load_plot_configs, unify_palette_cyclers, prettify_subplots

import pandas as pd

###
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
###

def generate_data():
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

def perc(array, perc):
            return np.percentile(array, perc, axis=1)


class NBSModel:
    '''
    Nash Bargaining Solution (NBS) model for bilateral contracts
    between a renewable energy developer (D) and an off-taker (O).
    The model determines the optimal strike price (S) and
    baseload volume (M) of a power purchase agreement (PPA). There are a plethora
    of auxiliary variables necessary to:
        1) formulate the CVaR constraints,
        2) express the NBS objective function in a legible manner,
        3) partially linearize the objective function by using logarithms.

    This class is ready to use the data exported from the Enlight DA market model. It requires:
    - P_fore_w,t : Input to Enlight
    - L_w,t : Input to Enlight
    - lambda_DA_t : OUTput of Enlight    
    '''
    def __init__(self,
        PPA_profile: str = 'BL',  # Type of PPA profile ('PaF', 'PaP' or 'BL' )
        BL_compliance_perc : float = 0, # indicates the enforced compliance of the producer: meaning the % of BL PPA volume where the producer has to match the BL volume on an hourly basis
        P_fore_w : np.ndarray = None,
        P_batt : float = 1.0,
        batt_eta : float = float(np.sqrt(0.9)),
        batt_Crate : float = 1.0,
        L_t : np.ndarray = None,
        lambda_DA_w : np.ndarray = None,
        WTP : float = 0,
        add_batt: bool = None,
        hp : gp.Model = None,  # the optimal solution of the hybrid plant in DA math. model
        S_LB : float = 0,  # Minimum PPA strike price
        S_UB : float = 1000,  # Maximum PPA strike price
        M_LB : float = 0,  # BL: Minimum baseload volume
        M_UB : float = 1,  # BL: Maximum baseload volume
        gamma_LB : float = 0, # PaP: Minimum PPA capacity share volume
        gamma_UB : float = 1, # PaP: Minimum PPA capacity share volume
        beta_D : float = 0.5,  # CVaR: Risk-aversion level of developer
        beta_O : float = 0.5,  # CVaR: Risk-aversion level of off-taker
        alpha : float = 0.75,  # CVaR: Tail of interest for CVaR
        nbs_model_logger : logging.Logger | None = None,
    ) -> None:
    
        '''
        Initialize the NBS model with necessary parameters, variables, and constraints.
        '''
        self.nbs_model_logger = nbs_model_logger or utils.setup_logging(log_file="nbs.log")
        self.nbs_model_logger.info(f"NBSModel: INITIALIZING model with betas=(O:{beta_O}, D:{beta_D})")
        
        self.PPA_profile = PPA_profile

        self.BL = False
        if self.PPA_profile == 'BL':
            self.BL=True

        if self.BL:
            # If the PPA is PaF or PaF, then the compliance percentage is irrelevant anyways.
            if not (1 >= BL_compliance_perc >= 0):
                raise Exception(f"The chosen level of PPA BL compliance is not valid: {BL_compliance_perc}. It should be between [0,1].")
            self.BL_compliance_perc = BL_compliance_perc
        else:
            self.BL_compliance_perc = 0.0  # default to 0. Irrelevant, so it doesn't matter if it's the value input to the object instance.
        
        self.P_fore_w = P_fore_w
        self.P_batt = P_batt
        self.batt_eta = batt_eta
        self.batt_Crate = batt_Crate
        self.L_t = L_t
        self.lambda_DA_w = lambda_DA_w
        self.WTP = WTP
        
        self.add_batt = add_batt

        if self.add_batt is None:
            if PPA_profile == 'BL':
                self.add_batt = True
            else:
                self.add_batt = False

        self.hp = hp  # default is 'None'

        self.S_LB = S_LB
        self.S_UB = S_UB
        self.M_LB = M_LB
        self.M_UB = M_UB
        self.gamma_LB = gamma_LB
        self.gamma_UB = gamma_UB
        self.beta_D = beta_D
        self.beta_O = beta_O
        self.alpha = alpha

        self.palette = load_plot_configs(only_get_palette=True)

        self.T, self.W = self.P_fore_w.shape
        
        self.calc_aux_data()
        self.compute_disagreement_points()
        self.build_model()
    
    def calc_aux_data(self) -> None:
        # PARAMS needed in -||-
        '''
        Initialize hard-coded model data, parameters, and variables.
        Hard-coded data only used for testing.

        Parameters of which some are inputs to the class*:
        - P_fore_w[w, t] : power forecast in scenario w at time t
        - lambda_DA_w[w, t] : day-ahead market prices in hour t in scenario w
        - PROB_w[w] : probability of scenario w
        - L_w[w, t] : off-taker consumption in scenario w at time t
        - WTP : off-taker utility in €/MWh
        - *S_LB : minimum PPA strike price
        - *S_UB : maximum PPA strike price
        - *M_LB : minimum BL volume
        - *M_UB : maximum BL volume
        CVaR
            - *beta_D : risk-averseness level of developer
            - *beta_O : risk-averseness level of off-taker
            - *alpha : tail of interest
        '''
        self.PROB_w = np.full(shape=self.W, fill_value=1/self.W)  # all scenarios are equiprobable
        '''
        The DA offer depends on the settlement scheme, however,
        in both of the current structures (PaP current, BL current),
        the developer would offer P_fore at a price of 0. So, she would
        not be in the market if the DA price is less than 0.
        P_DA_w -> parameter
        p_DA_w -> variable in the case of future settlements: "PaP coupled" and "BL compliance"
        '''
        self.P_DA_w = self.P_fore_w * (self.lambda_DA_w >= 0)

        self.CP_D = (self.lambda_DA_w * self.P_DA_w).sum()/self.P_DA_w.sum()  # €/MWh - capture price of VRE
        self.eps = 1e-6  # small number to avoid log(0) in the model
        # self.eps_fake_cycling = 1e-5

    def compute_disagreement_points(self) -> None:
        # Default functions : disagreement points
        # Expected profit when purely offering in DA market
        PI_D_w = (self.lambda_DA_w * self.P_DA_w).sum(axis=0)  # expected profit in scenario w
        VAR_idx = round(self.W*(1-self.alpha)-1)  # "-1" because of 0-indexing
        VAR_D = np.sort(PI_D_w)[VAR_idx]
        ETA_D_w = VAR_D - PI_D_w  # Difference between VaR and expected profit in each scenario. We want to minimize this when we are risk-averse
        self.d_D = (
            (1 - self.beta_D) * sum([PI_D_w[w] * self.PROB_w[w] for w in range(self.W)])
            + self.beta_D * (VAR_D - 1/(1-self.alpha) * sum([ETA_D_w[w] * self.PROB_w[w] for w in range(self.W) if ETA_D_w[w] >= 0]))
        )
        self.PI_D_w, self.VAR_D, self.ETA_D_w = PI_D_w, VAR_D, ETA_D_w

        # Expected utility for off-taker when purely bidding in DA market
        PI_O_w = ( (self.WTP - self.lambda_DA_w) * self.L_t).sum(axis=0)  # expected power costs in scenario w
        VAR_O = np.sort(PI_O_w)[VAR_idx]  # we need the maximum loss
        ETA_O_w = VAR_O - PI_O_w  # Difference between VaR and expected utility in each scenario. We want to minimize this when we are risk-averse
        self.d_O = (
            (1 - self.beta_O) * sum([PI_O_w[w] * self.PROB_w[w] for w in range(self.W)])
            + self.beta_O * (VAR_O - 1/(1-self.alpha) * sum([ETA_O_w[w] * self.PROB_w[w] for w in range(self.W) if ETA_O_w[w] >= 0]))
        )
        self.PI_O_w, self.VAR_O, self.ETA_O_w = PI_O_w, VAR_O, ETA_O_w
        '''
        profit in low-profits scenarios = CVaR = (VaR_O - 1/(1-alpha) * sum([eta_O_w_[w] * PROB_w[w] for w in range(W) if eta_O_w_[w] >= 0]))
        The term has to be '-'. NOT '+'. Verify by inspecting the CVaR constraint on eta_w in the model formulation...
        '''

        if self.BL:  # -> overwrite d_D but keep d_O
            if self.hp is None:
                print("RUNNING HYBRID VRE")
                hp = hv.HybridVRE(
                    P_fore_w = self.P_fore_w,
                    lambda_DA_w = self.lambda_DA_w,
                    # model = None,
                    add_batt = self.add_batt,
                    batt_power = self.P_batt,
                    batt_eta = self.batt_eta,
                    batt_Crate = self.batt_Crate,
                )
                hp.build_model()
                hp.run_model()
                hp.get_results()
                print("FINISHED running HybridVRE")
                self.hp = hp
            else:
                print("USING PROVIDED HYBRID VRE MODEL")

            PI_D_w = (self.hp.p_DA.X * self.hp.lambda_DA_w).sum(axis=0)  # expected power revenues in each scenario w
            VAR_idx = round(self.W*(1-self.alpha)-1)  # "-1" because of 0-indexing
            VAR_D = np.sort(PI_D_w)[VAR_idx]
            ETA_D_w = VAR_D - PI_D_w
            # print("Expected Power Incomes per scenario:\n", PI_D_w)
            # print(f"Value at Risk at {alpha*100}% confidence level:\n", VAR_D)
            # print("Excess over VaR per scenario:\n", ETA_D_w)
            self.d_D = (
                (1 - self.beta_D) * self.hp.model.ObjVal  # Expected profit
                + self.beta_D  # CVaR term in objective
                    * (VAR_D
                       - 1/(1-self.alpha)
                         # probability-weighted profits BELOW CVaR. 0 if above:
                       * sum([ETA_D_w[w] * self.hp.PROB_w[w] for w in range(self.W) if ETA_D_w[w] >= 0])
                       )
            )# trying smth

            self.PI_D_w, self.VAR_D, self.ETA_D_w = PI_D_w, VAR_D, ETA_D_w

    def build_vars(self) -> None:
        '''
        # VARS needed in simple BL NBS from Anders' thesis
        # Volume:
            # M: fixed BL volume
            # or
            # gamma: PPA capacity share volume
        # S: PPA strike price (aka. simply "PPA price")
        # CVaR
            # zeta_D = Value-at-Risk for developer
            # zeta_O = Value-at-Risk for off-taker
            # eta_D_w = developer profit in scenario w
            # eta_O_w = off-taker profit in scenario w
        '''
        
        
        self.S = self.model.addVar(lb=self.S_LB, ub=self.S_UB, vtype=GRB.CONTINUOUS, name="S")        
        #CVaR vars:
        self.zeta_D = self.model.addVar(vtype=GRB.CONTINUOUS, name="zeta_D")
        self.zeta_O = self.model.addVar(vtype=GRB.CONTINUOUS, name="zeta_O")
        self.eta_D_w = self.model.addMVar(self.W, lb=0, vtype=GRB.CONTINUOUS, name="eta_D_w")
        self.eta_O_w = self.model.addMVar(self.W, lb=0, vtype=GRB.CONTINUOUS, name="eta_O_w")

        # Aux vars
        # Utility/profit variables
        self.u_D = self.model.addVar(vtype=GRB.CONTINUOUS, name="u_D")  # developer profit
        self.u_O = self.model.addVar(vtype=GRB.CONTINUOUS, name="u_O")  # off-taker net utility
        # Difference variables
        self.x_D = self.model.addVar(lb=self.eps, name="x_D")  # u_D - d_D
        self.x_O = self.model.addVar(lb=self.eps, name="x_O")  # u_O - d_O
        # Logarithm variables -- e.g. log(x_D) where x_D = u_D-d_D
        self.log_uD_dD = self.model.addVar(vtype=GRB.CONTINUOUS, name="log_uD_dD")  # logarithm of profit gain of developer from PPA
        self.log_uO_dO = self.model.addVar(vtype=GRB.CONTINUOUS, name="log_uO_dO")

    def BL_add_battery_vars_constrs(self) -> None:
        '''
        Initiate an instance of the HybridVRE class and add the vars and constrs
        on top of the CVaR and AUX vars defined above.
        '''
        # create HybridVRE instance
        hp_BL = hv.HybridVRE(
            P_fore_w = self.P_fore_w,
            lambda_DA_w = self.lambda_DA_w,
            model = self.model,  # 
            add_batt = self.add_batt,
            batt_power = self.P_batt,
            batt_eta = self.batt_eta,
            batt_Crate = self.batt_Crate,
        )

        # add the vars and cons for batt to NBS BL model formulation
        # and save the new variables for easy access and manipulation
        self.model, self.p_DA, self.SOC, self.y_dch, self.y_ch = hp_BL.build_and_extract_model_no_obj()
#HERE
    def build_profile_vars_constrs(self) -> None:
        '''
        This method builds the variables and constraints that are profile-specific.
        Meaning that they differ between a BL and a PaP electricity profile.
        All the other methods are general for a NBS mathematical model.
        '''
        # Utility/profit variables that EXCLUDE the CVaR term
        self.y_D = self.model.addMVar(self.W, vtype=GRB.CONTINUOUS, name="y_D")  # developer profit excluding CVaR
        self.y_O = self.model.addMVar(self.W, vtype=GRB.CONTINUOUS, name="y_O")  # off-taker net utility excluding CVaR

        # Profile-specific variables
        if self.PPA_profile in ['PaF', 'PaP']:
            self.gamma = self.model.addVar(lb=self.gamma_LB, ub=self.gamma_UB, vtype=GRB.CONTINUOUS, name="gamma")

        elif self.BL:
            self.M = self.model.addVar(lb=self.M_LB, ub=self.M_UB, vtype=GRB.CONTINUOUS, name="M")
            self.v_min = self.model.addMVar(shape=(self.T, self.W), lb=0, vtype=GRB.CONTINUOUS, name="v_min")

        else:
            raise Exception(f"{self.PPA_profile} is not a valid PPA electricity profile. Please try baseload (BL) or Pay-as-Produced (PaP)")

        # Link actual scenario profit/utility to CVaR-excluded variables
        if self.PPA_profile == 'PaF':
            # Decoupled structure where the developer behaves "normally" and offers at 0 price.
            # Thus, she is remunerated in the PPA for the power that she produces in hours with non-negative prices.
            # If allowing negative prices, use P_fore instead of P_DA.
            self.model.addConstrs((self.y_D[w] ==
                                gp.quicksum(
                                    (1 - self.gamma) * self.P_DA_w[t, w] * self.lambda_DA_w[t, w]  # DA revenues (don't sell if DA price < 0)
                                        + self.gamma * self.P_fore_w[t, w] * self.S # PaP PPA revenues (sell even if DA price < 0)
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yD_link_PaF')
            
            self.model.addConstrs((self.y_O[w] ==
                                gp.quicksum(
                                    self.gamma * self.P_fore_w[t, w] * (self.WTP - self.S)  # The PPA volume is paid at the strike price
                                    +
                                    (self.L_t.ravel()[t] - self.gamma * self.P_fore_w[t, w])
                                            * (self.WTP - self.lambda_DA_w[t, w])  # The rest is procured (or sold) in the DA market
                                        for t in range(self.T)
                                )
                                for w in range(self.W)), name='c_yO_link_PaF')
            
        elif self.PPA_profile == 'PaP':
            # Restrict PPA-remunerated volume to contracted volume.
            # Three options for PaP:
            # 1. Assume that the price difference is always settled to PPA price (offer at minimum allowed price) -> identical 
            # 2. Assume that at maximum PPA price is paid to the producer (offer at -PPA price)
            # 3. No PPA payment at negative prices (offer at 0 price)
            # including a 'p_PPA' variable would allow to always easily adapt the model to capture this behaviour. BUT it is unable to run efficiently for longer periods than 10 weeks...
            # We do not lose interesting insight by just modelling PaP option 3. PaP option 1. is included in PaF and can be used to analyze the effect of PaP 2. anyways.

            # Write objective function of the profit-maximizing developer
            #   and utility-maximizing off-taker.
            self.model.addConstrs((self.y_D[w] ==
                                gp.quicksum(
                                    (1 - self.gamma) * self.P_DA_w[t, w] * self.lambda_DA_w[t, w]  # DA revenues
                                    + self.gamma * self.P_DA_w[t, w] * self.S # PaP PPA revenues
                                    #+ self.gamma * self.P_fore_w[t, w] * self.S # PaP PPA revenues
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yD_link_PaP')
            self.model.addConstrs((self.y_O[w] ==
                                gp.quicksum(
                                    # They only pay for what the developer actually produces: p_DA
                                    #   not just for the forecast: P_fore_w.
                                    # Note that we use P_fore_w instead of P_DA_w, because the offers
                                    #   are now optimized as well, as for P_DA_w the assumption of MC=0 is made.
                                    #self.gamma * self.P_fore_w[t, w] * (self.WTP - self.S)  # The PPA volume is paid at the strike price
                                    self.gamma * self.P_DA_w[t, w] * (self.WTP - self.S)  # The PPA volume is paid at the strike price
                                    +
                                    (self.L_t.ravel()[t] - self.gamma * self.P_fore_w[t, w])
                                            * (self.WTP - self.lambda_DA_w[t, w])  # The rest is procured (or sold) in the DA market
                                        for t in range(self.T)
                                )
                                for w in range(self.W)), name='c_yO_link_PaP')

        elif self.BL:
            # Create battery variables and constraints in the model
            #   this makes it so that the DA offer is no longer trivially
            #   P_DA_w, but is instead optimized as p_DA!
            self.BL_add_battery_vars_constrs()

            # Define the profit excluding CVaR term for the developer
            '''
            BEFORE
            self.model.addConstrs((self.y_D[w] ==
                                gp.quicksum(
                                    # OLD, plain P_DA_w (= forecast in non-negative price hours):
                                    # self.lambda_DA_w[t,w] * self.P_DA_w[t,w]  # DA revenues
                                    # NEW, p_DA optimized with BESS
                                    (self.lambda_DA_w[t,w] * self.p_DA[t,w]  # DA revenues
                                    + (self.S - self.lambda_DA_w[t,w]) * self.M)  # BL PPA revenues
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yD_link_BL')
            self.model.addConstrs((self.y_O[w] ==
                                gp.quicksum(
                                    (self.L_t.ravel()[t] * (self.WTP - self.lambda_DA_w[t,w])  # Costs in DA
                                    - self.M * (self.S - self.lambda_DA_w[t,w]))  # Costs in BL PPA
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yO_link_BL')
            # further, add compliance:
            if self.BL_compliance_perc > 0:
                # v_min is the BL volume matched by the producer on hourly basis
                self.model.addConstrs((self.v_min[t, w] <= self.p_DA[t, w]
                                        for t in range(self.T) for w in range(self.W)),
                                        name="c_v_min__p_DA")
                # the producer cannot compensate for hours of low-compliance by exceeding the BL volume in other hours.
                self.model.addConstrs((self.v_min[t, w] <= self.M
                                        for t in range(self.T) for w in range(self.W)),
                                        name="c_v_min__M")
                # the annual compliance-volume
                self.model.addConstrs(
                    (gp.quicksum(self.v_min[t, w] for t in range(self.T))
                    >= self.BL_compliance_perc * self.T * self.M
                    for w in range(self.W)),
                    name="c_BL_compliance")
            '''
            # for w in range(self.W):
            #     self.model.addQConstr(
            #         self.y_D[w] ==
            #         (self.lambda_DA_w[:, w] * self.p_DA[:, w]).sum()
            #         + ((self.S - self.lambda_DA_w[:, w]) * self.M).sum(),
            #         name=f"c_yD_link_BL[{w}]"
            #     )
            # self.model.addQConstr(self.y_D ==
            #                         # OLD, plain P_DA_w (= forecast in non-negative price hours):
            #                         # self.lambda_DA_w[t,w] * self.P_DA_w[t,w]  # DA revenues
            #                         # NEW, p_DA optimized with BESS
            #                         ((self.lambda_DA_w * self.p_DA).sum(axis=0)  # DA revenues
            #                         + ((self.S - self.lambda_DA_w) * self.M).sum(axis=0)),  # BL PPA revenues
            #                         name='c_yD_link_BL')
            # self.model.addQConstr(self.y_O ==
            #                         (self.L_t * (self.WTP - self.lambda_DA_w)).sum(axis=0)  # Costs in DA
            #                         - (self.M * (self.S - self.lambda_DA_w)).sum(axis=0),  # Costs in BL PPA
            #                         name='c_yO_link_BL')
            self.Z = self.model.addVar(vtype=GRB.CONTINUOUS, name="Z")
            self.model.addQConstr(self.Z == self.S * self.M, name="aux_PPA_product")
            '''
            self.M_aux = self.model.addMVar(1, vtype=GRB.CONTINUOUS, name="M_auxMvar")
            self.model.addConstr(self.M == self.M_aux, name="c_aux_M")
            self.WTP_aux = self.model.addMVar(1, vtype=GRB.CONTINUOUS, name="WTP_auxMvar")
            self.model.addConstr(self.WTP_aux == self.WTP, name="c_aux_WTP")
            '''
            self.model.addConstr(self.y_D
                                 ==
                                 (self.lambda_DA_w * self.p_DA).sum(axis=0)  # DA revenues
                                 + self.Z * self.T - (self.M * self.lambda_DA_w).sum(axis=0),  # BL PPA revenues
                                 name='c_yD_link_BL')
            # OLD formulation due to bugs :(
            self.model.addConstrs((self.y_O[w] ==
                                gp.quicksum(
                                    (self.L_t.ravel()[t] * (self.WTP - self.lambda_DA_w[t,w])  # Costs in DA
                                    - self.M * (self.S - self.lambda_DA_w[t,w]))  # Costs in BL PPA
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yO_link_BL'
            )
            # Add auxiliary variable for the DA costs of the buyer so that
            # we can use vectorized formuation of the total expected net utility of the buyer:
            # "(self.L_t * (self.WTP - self.lambda_DA_w)).sum(axis=0)" is not allowed in y_O constraint.
            # self.model.addConstrs((self.y_O[w] ==
            #                     gp.quicksum(
            #                         (self.L_t.ravel()[t] * (self.WTP - self.lambda_DA_w[t,w])  # Costs in DA
            #                         - self.M * (self.S - self.lambda_DA_w[t,w]))  # Costs in BL PPA
            #                         for t in range(self.T)
            #                     ) for w in range(self.W)), name='c_yO_link_BL')
            # self.model.addConstr(self.y_O
            #                      ==
            #                      (self.L_t * (self.WTP_aux - self.lambda_DA_w)).sum(axis=0)
            #                      - self.Z * self.T - self.M_aux * self.lambda_DA_w.sum(axis=0))
            # self.buyer_DA_costs = self.model.addMVar(self.W, lb=0,vtype=GRB.CONTINUOUS, name="buyer_DA_costs")
            # self.WTP_fake = self.model.addMVar(1, lb=0, vtype=GRB.CONTINUOUS, name="WTP_fake")
            # self.model.addConstr(self.buyer_DA_costs == (self.L_t * (self.WTP_fake - self.lambda_DA_w)).sum(axis=0))
            # self.model.addConstr(self.y_O
            #                      ==
            #                      self.buyer_DA_costs  # Costs in DA by scenario
            #                      - self.Z * self.T + (self.M * self.lambda_DA_w).sum(axis=0),  # Costs in BL PPA
            #                      ,name='c_yO_link_BL')
            # further, add compliance:
            if self.BL_compliance_perc > 0:
                # v_min is the BL volume matched by the producer on hourly basis
                self.model.addConstr(self.v_min <= self.p_DA,
                                     name="c_v_min__p_DA")
                # the producer cannot compensate for hours of low-compliance by exceeding the BL volume in other hours.
                self.model.addConstr(self.v_min <= self.M,
                                     name="c_v_min__M")
                # the annual compliance-volume
                self.model.addConstr(
                    self.v_min.sum(axis=0)
                    >=
                    self.BL_compliance_perc * self.T * self.M,
                    name="c_BL_compliance")

        else:
            raise Exception(f"{self.PPA_profile} is not a valid profile type. Please choose 'PaF', 'PaP', 'BL', or 'BL-COMPLIANCE'." )
#HERE
    def build_aux_constrs(self) -> None:
        # Linear linking constraints
        self.model.addConstr(self.x_D == self.u_D - self.d_D, name="c_diff_D")
        self.model.addConstr(self.x_O == self.u_O - self.d_O, name="c_diff_O")

        self.model.addGenConstrLog(self.x_D, self.log_uD_dD, name="c_log_uD_dD")
        self.model.addGenConstrLog(self.x_O, self.log_uO_dO, name="c_log_uO_dO")

        # self.model.setObjective(self.log_uD_dD + self.log_uO_dO, sense=GRB.MAXIMIZE)

        # Add aux constraints to make obj readable
        self.model.addConstr(self.u_D
                        ==
                        (1 - self.beta_D)
                        * (self.PROB_w * self.y_D).sum()  # Expected profit
                        + self.beta_D
                        * (self.zeta_D  # The CVaR term
                           - 1/(1-self.alpha)
                           * (self.PROB_w * self.eta_D_w).sum()
                        ),
                        name='c_uD')

        self.model.addConstr(self.u_O
                        ==
                        (1 - self.beta_O) * (self.PROB_w * self.y_O).sum()  # Expected net utility
                        + self.beta_O
                        * (self.zeta_O  # The CVaR term,
                           - 1/(1-self.alpha)
                           * (self.PROB_w * self.eta_O_w).sum()
                        ),
                        name='c_uO')
#HERE
    def build_cvar_constrs(self) -> None:
        self.model.addConstr(self.eta_D_w
                        >=
                        self.zeta_D - self.y_D,  # Developer profit in scenario w (excl. CVaR term)
                        name='c_CVaR_D')

        self.model.addConstr(self.eta_O_w
                        >=
                        self.zeta_O - self.y_O,  # Off-taker utility in scenario w (excl. CVaR term)   
                        name='c_CVaR_O')

    def build_objective(self) -> None:
        self.model.setObjective(self.log_uD_dD + self.log_uO_dO  # <- actual objective :)
                                # - (self.y_ch + self.y_dch).sum() * self.eps_fake_cycling  # <- avoid the battery showing "fake cycling"! This does NOT change the solution!!
                                ,
                                sense=GRB.MAXIMIZE)

    def build_model(self) -> None:
        self.model = gp.Model("NBS")
        self.build_vars()  # General NBS vars
        self.build_profile_vars_constrs()  # Profile-specific vars and constrs
        self.build_aux_constrs()  # General NBS
        self.build_cvar_constrs()  # General NBS
        self.build_objective()  # General NBS

    def solve_model(self) -> None:
        self.model.Params.NonConvex = 2  # Enable non-convex solver
        self.model.write("NBS.lp")
        self.model.setParam("TimeLimit", 240)  # in seconds
        self.model.optimize()
        
        # save a specific result
        if self.BL and self.model.status == GRB.OPTIMAL:
            self.compliance_rates = self.v_min.X.sum(axis=0)/(self.T * self.M.X)

    def get_results(self) -> dict:
        vars_to_save = {
            "S": self.S,
            "eta_D_w": self.eta_D_w,
            "log_uD_dD": self.log_uD_dD,
            "u_D": self.u_D,
            "x_D": self.x_D,
            "y_D": self.y_D,
            "zeta_D": self.zeta_D,
        }
        if self.BL:
            vars_to_save["p_DA"] = self.p_DA
            vars_to_save["y_ch"] = self.y_ch
            vars_to_save["y_dch"] = self.y_dch
            vars_to_save["SOC"] = self.SOC
            vars_to_save["M"] = self.M

            if self.BL_compliance_perc > 0:
                vars_to_save["v_min"] = self.v_min
        else:
            vars_to_save["gamma"] = self.gamma

        results = {
            name: var_to_pandas(var, name=name)
            for name, var in vars_to_save.items()
        }
        results["d_D"] = self.d_D
        results["d_O"] = self.d_O

        return results

    def visualize_example_outcome(self, show_all_scens : bool = False):
        if self.model.status == GRB.OPTIMAL:
            max_hour_shown = 168
            S_X = self.S.X
            if self.BL:
                M_X = self.M.X
            elif self.PPA_profile in ['PaF', 'PaP']:
                gamma_X = self.gamma.X

            

            if show_all_scens:
                fig, axs = plt.subplots(2,2, figsize=(8,10))
                axs = axs.flatten()
                unify_palette_cyclers(axs)
                for w, ax in enumerate(axs):
                    ax.plot(self.P_fore_w[:max_hour_shown,w], label='Power forecasts', linestyle='-')
                    ax.plot(self.P_DA_w[:max_hour_shown,w], label='DA accepted offer', ls='--', alpha=.5)
                    ax.plot(self.L_t[:max_hour_shown], label='Off-taker load profile', linestyle='-', alpha=.8)

                    if self.BL:
                        ax.axhline(M_X, color='k', linestyle='-.', alpha=0.7,
                                   label='Agreed BL volume $M$'+f'={self.M.X:.2f} MW')
                    elif self.PPA_profile == 'PaF':
                        ax.plot(gamma_X * self.P_fore_w[:max_hour_shown,w], color='k', linestyle='-.', alpha=0.3,
                                label=r'Agreed PaF volume $\gamma$'+f'={self.gamma.X*100:.2f}% (P_fore)')
                    else:
                        ax.plot(gamma_X * self.P_DA_w[:max_hour_shown,w], color='k', linestyle='-.', alpha=0.3,
                                label=r'Agreed PaP volume $\gamma$'+f'={self.gamma.X*100:.2f}% (P_DA)')

                    if w % 2 == 0:
                        # Only the first plot in each row needs to have the label:
                        ax.set_ylabel('Power [MW]')
                    ax.set_title(f'Scen. {w+1}: Power generation and consumption', loc='left')
                prettify_subplots(axs)
                axs[0].legend_.remove()
                axs[2].legend_.remove()
                axs[3].legend_.remove()


            else:
                fig, ax = plt.subplots(figsize=(8,10))
                unify_palette_cyclers(ax)
                low_perc = 10
                high_perc = 90
                ax.plot(perc(self.P_fore_w, 50)[:max_hour_shown], label='Median power forecast', linestyle='-')
                ax.fill_between(range(self.T)[:max_hour_shown], perc(self.P_fore_w, low_perc)[:max_hour_shown], perc(self.P_fore_w, high_perc)[:max_hour_shown], alpha=0.4, label='Power forecast 10-90 percentile')
                ax.plot(perc(self.P_DA_w, 50)[:max_hour_shown], label='Median DA dev. acc. off.', ls='--', alpha=.5)
                ax.plot(self.L_t[:max_hour_shown], label='Off-taker load profile', linestyle='-', alpha=.8)
                if self.BL:
                    ax.axhline(M_X, color='k', linestyle='-.', alpha=0.7,
                                label='Agreed BL volume $M$'+f'={self.M.X:.2f} MW')
                elif self.PPA_profile == 'PaF':
                    ax.plot(gamma_X * perc(self.P_fore_w, 50)[:max_hour_shown], color='k', linestyle='-.', alpha=0.3,
                            label=r'Agreed PaF volume $\gamma$'+f'={self.gamma.X*100:.2f}% (P_fore)')
                else:
                    ax.plot(gamma_X * perc(self.P_DA_w, 50)[:max_hour_shown], color='k', linestyle='-.', alpha=0.3,
                            label=r'Agreed PaP volume $\gamma$'+f'={self.gamma.X*100:.2f}% (P_DA)')
                prettify_subplots(ax)
            plt.show()

            fig, ax = plt.subplots(figsize=(12,6))
            unify_palette_cyclers(ax)
            if show_all_scens:
                for w in range(self.W):
                    ax.plot(self.lambda_DA_w[:max_hour_shown,w], linestyle='-', label=f"Scen. {w+1}")
            else:
                ax.plot(perc(self.lambda_DA_w, 50)[:max_hour_shown], label='Median DA prices', alpha=.65, linestyle='-')
                ax.fill_between(range(self.T)[:max_hour_shown], perc(self.lambda_DA_w, low_perc)[:max_hour_shown], perc(self.lambda_DA_w, high_perc)[:max_hour_shown], alpha=0.4, label='DA price 10-90 percentile')
            ax.axhline(S_X, color='k', linestyle='-.', label=f'Optimal {self.PPA_profile} PPA strike price $S$')
            ax.set_ylabel('Price [€/MWh]')
            ax.set_title(fr"DA prices – Buyer with $\beta^B$={self.beta_O}, and Producer with $\beta^P$={self.beta_D}", loc='left')
            prettify_subplots(ax)
            plt.show()
        else:
            print("No results to show. No optimal solution was found.")

    def visualize_example_profit_dist(self, bars=False):
        if self.model.status == GRB.OPTIMAL:

            # # Compare their revenues distributions before and after
            # # Pure DA revenues
            # PI_D_w = [self.models[beta_O_chosen][beta_D_chosen].PI_D_w[w] for w in range(nbs_.W)]
            # PI_O_w = [self.models[beta_O_chosen][beta_D_chosen].PI_O_w[w] for w in range(nbs_.W)]
            # # VaR before PPA
            # VaR_D = self.models[beta_O_chosen][beta_D_chosen].VAR_D
            # VaR_O = self.models[beta_O_chosen][beta_D_chosen].VAR_O
            # Expected DA + PPA revenues
            power_costs_O_w = self.PI_O_w - (self.L_t * self.WTP).sum()
            power_costs_O_w_NBS = self.y_O.X - (self.L_t * self.WTP).sum()
            VAR_O_power_costs = self.VAR_O - (self.L_t * self.WTP).sum()
            zeta_O_power_costs = self.zeta_O.X - (self.L_t * self.WTP).sum()
            # VaR after PPA
            # zeta_D = self.models[beta_O_chosen][beta_D_chosen].zeta_D.X
            # zeta_O = self.models[beta_O_chosen][beta_D_chosen].zeta_O.X
            '''For debugging purposes, save as attrs:'''
            self.power_costs_O_w = power_costs_O_w
            self.power_costs_O_w_NBS =  power_costs_O_w_NBS
            self.VAR_O_power_costs = VAR_O_power_costs
            self.zeta_O_power_costs = zeta_O_power_costs 

            fig, ax = plt.subplots(1, 2, figsize=(10,6))
            unify_palette_cyclers(ax)

            w_idx = np.arange(1, self.W+1)
            bar_width = 0.35

            if bars:
                # --- Developer ---
                ax[0].bar(w_idx - bar_width/2, self.PI_D_w,
                        width=bar_width, alpha=0.6, label="Before")
                ax[0].bar(w_idx + bar_width/2, self.y_D.X,
                        width=bar_width, alpha=0.6, label="After PPA")
                ax[0].bar(x=w_idx + bar_width/2, height=self.eta_D_w.X,
                          bottom=self.y_D.X, width=bar_width, alpha=0.6, label="CVaR weight")
                ax[0].axhline(self.VAR_D, ls="--", label="VaR before")
                ax[0].axhline(self.zeta_D.X, ls="-.", label="VaR after")

                # --- Off-taker ---
                ax[1].bar(w_idx - bar_width/2, power_costs_O_w,
                        width=bar_width, alpha=0.6, label="Before")
                ax[1].bar(w_idx + bar_width/2, power_costs_O_w_NBS,
                        width=bar_width, alpha=0.6, label="After PPA")
                ax[1].bar(x=w_idx + bar_width/2, height=self.eta_O_w.X,
                          bottom=power_costs_O_w_NBS, width=bar_width, alpha=0.6, label="CVaR weight")
                ax[1].axhline(VAR_O_power_costs, ls="--", label="VaR before")
                if self.beta_O == 0:
                    ax[1].axhline(self.zeta_O.X, ls="-.", label="VaR after")
                else:
                    ax[1].axhline(zeta_O_power_costs, ls="-.", label="VaR after")
                ax[1].set_title(fr"Buyer POWER COSTS (no WTP) – $\beta^O=${self.beta_O}")
            else:  # histogram
                ax[0].hist(self.PI_D_w, color='r', alpha=0.5, label='Before')
                ax[0].axvline(x=self.VAR_D, color='r', linestyle='--', label='VaR before')
                ax[1].hist(self.PI_O_w, color='r', alpha=0.5, label='Before')
                ax[1].axvline(x=self.VAR_O, color='r', linestyle='--', label='VaR before')

                # PI_D_w_NBS = (nbs_.lambda_DA_w * nbs_.P_DA_w + (S_X - nbs_.lambda_DA_w) * M.X).sum(axis=0)
                # PI_O_w_NBS = ( (nbs_.WTP - nbs_.lambda_DA_w) * nbs_.L_t - (S_X - nbs_.lambda_DA_w) * M.X).sum(axis=0)
                ax[0].hist(self.y_D.X, alpha=0.5, label='after PPA')
                ax[0].axvline(x=self.zeta_D.X, linestyle='--', label='VaR after')
                ax[1].hist(self.y_O.X, alpha=0.5, label='after PPA')
                ax[1].axvline(x=self.zeta_O.X, linestyle='--', label='VaR after')
                ax[1].set_title(fr"Buyer UTILITY – $\beta^O=${self.beta_O}")

            ax[0].set_title(fr"{self.PPA_profile}{(f"({self.BL_compliance_perc})" if self.BL else "")}: Producer PROFITS – $\beta^D=${self.beta_D}   ")
            prettify_subplots(ax)
            ax[0].legend_.remove()
            for ax_ in ax:
                ax_.set_xlabel("Scenario")
            ax[0].set_ylabel("Profit or costs [€]")
            plt.tight_layout()
            plt.show()
        else:
            print("No results to show. No optimal solution was found.")
     
    def verify_behaviour(self, w_BESS=3, hours_shown = range(0,168)):
        if self.model.status == GRB.OPTIMAL:
            # Verify behaviour of p_DA and p_PPA. If lambda_DA >= 0 in all hour-scenarios, then we should always max out both!!
            if self.PPA_profile == 'PaP':
                for i in range(self.W)[:4]:
                    fig, ax = plt.subplots(figsize=(10,6))
                    unify_palette_cyclers(ax)
                    ax2 = ax.twinx()
                    unify_palette_cyclers(ax2)

                    # Plot and compare total power available and offered
                    # ax.plot(d.P_DA_w[:,i], alpha=.7, label=r"$\overline{P}^{DA}$")
                    ax.plot(self.P_fore_w[hours_shown, i], ls='-', alpha=.5, lw=3, label=r"$P^{fore}$")
                    ax.plot((1-self.gamma.X) * self.P_DA_w[hours_shown, i] + self.gamma.X * self.P_DA_w[hours_shown,i], ls=':', alpha=.5, label=r"$P^{DA} + p^{PPA}$")

                    # Plot and compare the power available and offered & remunerated at DA price
                    ax.plot((1-self.gamma.X) * self.P_fore_w[hours_shown, i], alpha=.3, lw=3, label=r"$\left(1-\gamma\right) \cdot P^{fore}$")
                    ax.plot((1-self.gamma.X) * self.P_DA_w[hours_shown, i], alpha=.5, ls='--', label=r"$P^{DA}$")

                    # Plot and compare power available for PPA and offered to comply with PPA
                    ax.plot(self.gamma.X * self.P_fore_w[hours_shown, i], alpha=.3, lw=3, label=r"$\gamma \cdot P^{fore}$")
                    ax.plot(self.gamma.X * self.P_DA_w[hours_shown, i], alpha=.5, ls='--', label=r"$p^{PPA}$")

                    # ax.plot(d.L_t, label=r"$L$")
                    # ax2.plot(d.lambda_DA_w[:, i], ls=':', c='k', label=r'$\lambda^{DA}$')
                    # ax2.axhline(d.S.X, c='k', label="S")
                    # ax2.legend(loc='upper right')
                    ax.set_title(f"w = {i}")
                    prettify_subplots(ax)
                    prettify_subplots(ax2)
                    plt.show()

            elif self.BL:
                # inspect the BESS behaviour to verify that HybridVRE has been correctly included in this model.
                fig, ax = plt.subplots(figsize=(10,6))
                unify_palette_cyclers(ax)

                ax.plot(self.P_fore_w[hours_shown, w_BESS], label="P_fore")
                ax.plot((self.p_DA.X + self.y_ch.X)[hours_shown, w_BESS], label="p_DA + y_ch", ls='--')
                ax.plot(self.p_DA.X[hours_shown, w_BESS], label="p_DA", ls='--', alpha=.5)
                if self.BL_compliance_perc > 0:
                    ax.plot(self.v_min.X[hours_shown, w_BESS], label="v_min", c='r', alpha=.5)
                ax.plot(self.SOC.X[hours_shown, w_BESS], label="SOC", ls=':', alpha=0.3)
                ax.axhline(self.M.X, c='k', label="BL volume", alpha=.4)
                if self.beta_D == 1.0:
                    ax.set_title(r'Beware! Nonsensical for $\beta_D=1.0$')
                # plt.plot(d.y_ch.X[: ,w] * d.y_dch.X[: ,w])
                prettify_subplots(ax)
        else:
            print("No results to show. No optimal solution was found.")


class NBSMultModel:
    '''
    This class is used to create and run multiple NBS model instances.
    The class contains multiple methods for inspecting these results.
    '''
    def __init__(self,
        PPA_profile: str = 'BL',  # Type of PPA profile ('BL' or 'PaP')
        BL_compliance_perc : float = 0, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
        P_fore_w : np.ndarray = None,
        P_batt : float = 1.0,
        batt_eta : float = float(np.sqrt(0.9)),
        batt_Crate : float = 1.0,
        L_t : np.ndarray = None,
        lambda_DA_w : np.ndarray = None,
        WTP : float = 0,
        add_batt: bool = None,
        S_LB : float = 0,  # Minimum PPA strike price
        S_UB : float = 1000,  # Maximum PPA strike price
        M_LB : float = 0,  # BL: Minimum baseload volume
        M_UB : float = 1,  # BL: Maximum baseload volume
        gamma_LB : float = 0, # PaP: Minimum PPA capacity share volume
        gamma_UB : float = 1, # PaP: Minimum PPA capacity share volume
        alpha : float = 0.75,  # CVaR: Tail of interest for CVaR
        nbs_mult_logger : logging.Logger | None = None
    ) -> None:
        self.nbs_mult_logger = nbs_mult_logger or utils.setup_logging(log_file="nbs.log")
        self.nbs_mult_logger.info(f"NBSMultModel: INITIALIZATION")

        self.PPA_profile = PPA_profile

        self.BL = False
        if self.PPA_profile == 'BL':
            self.BL = True

        if not (1 >= BL_compliance_perc >= 0):
            raise Exception(f"The chosen level of PPA BL compliance is not valid: {BL_compliance_perc}. It should be between [0,1].")
        else:
            self.BL_compliance_perc = BL_compliance_perc

        self.P_fore_w = P_fore_w
        self.P_batt = P_batt
        self.batt_eta = batt_eta
        self.batt_Crate = batt_Crate
        self.L_t = L_t
        self.lambda_DA_w = lambda_DA_w
        self.WTP = WTP

        self.add_batt = add_batt

        if self.add_batt is None:
            if PPA_profile == 'BL':
                self.add_batt = True
            else:
                self.add_batt = False

        self.S_LB = S_LB
        self.S_UB = S_UB
        self.M_LB = M_LB
        self.M_UB = M_UB
        self.gamma_LB = gamma_LB
        self.gamma_UB = gamma_UB
        self.alpha = alpha

    def run_multiple_NBS_models(self, beta_O_list, beta_D_list):
        self.nbs_mult_logger.info(f"NBSMultModel: SOLVING multiple NBS models using NBSMultModel instance")

        self.beta_O_list = beta_O_list
        self.beta_D_list = beta_D_list

        # Initialize result dictionaries
        results_S = {}
        results_volume = {}
        models = {}
        hybrid_plant_model = None  # save the disagreement point of the developer after running the first model since this value does not change between risk aversion

        # Generate synthetic data only for the purpose of demonstrating the NBS behaviour.
        

        ##### LOOP OVER BETAS #####
        for beta_O in beta_O_list:
            results_S[beta_O] = {}
            results_volume[beta_O] = {}
            models[beta_O] = {}

            for beta_D in beta_D_list:        
                # Initialize NBS instance.
                t0b = time.time()
                nbs_model = NBSModel(
                    PPA_profile=self.PPA_profile,  # BL or PaP
                    BL_compliance_perc=self.BL_compliance_perc,
                    P_fore_w=self.P_fore_w,
                    P_batt = self.P_batt,
                    batt_eta=self.batt_eta,
                    batt_Crate=self.batt_Crate,
                    L_t=self.L_t,
                    lambda_DA_w=self.lambda_DA_w,
                    WTP=self.WTP,
                    # add_batt=False, # automatically defined as "True" if 'BL' and no bool is given.
                    hp=hybrid_plant_model,  # the optimal solution of the hybrid plant in DA math. model
                    S_LB=self.S_LB, S_UB=self.S_UB,  # PPA price
                    M_LB=self.M_LB, M_UB=self.M_UB,  # BL volume
                    gamma_LB=self.gamma_LB, gamma_UB=self.gamma_UB,  # PaP volume
                    beta_D=beta_D, beta_O=beta_O, alpha=self.alpha,
                    nbs_model_logger=self.nbs_mult_logger,
                )
                hybrid_plant_model = nbs_model.hp
                models[beta_O][beta_D] = nbs_model
                # Build the mathematical model.
                # print(f"\nSolving for β_D = {beta_D}, β_O = {beta_O} ...")
                tb = time.time()
                # Solve the optimization problem.
                nbs_model.solve_model()
                ts = time.time()
                self.nbs_mult_logger.info(f"Building time: {tb-t0b:.2f}. Solving time: {ts-tb:.2f}.")

                # Save the results of PPA price and volume explicitly if it was solved to optimality.
                if nbs_model.model.status == GRB.OPTIMAL:
                    self.nbs_mult_logger.info("Solved to optimality!")
                    results_S[beta_O][beta_D] = nbs_model.S.X
                    if nbs_model.BL:
                        results_volume[beta_O][beta_D] = nbs_model.M.X
                    elif nbs_model.PPA_profile in ['PaF', 'PaP']:
                        results_volume[beta_O][beta_D] = nbs_model.gamma.X
                elif nbs_model.model.status == GRB.TIME_LIMIT:
                    self.nbs_mult_logger.info(f"Stopped due to time limit: S: {nbs_model.S.Xn:.2f}")
                    results_S[beta_O][beta_D] = np.nan
                    results_volume[beta_O][beta_D] = np.nan
                else:
                    self.nbs_mult_logger.info("Model was infeasible or unbounded...")
                    results_S[beta_O][beta_D] = np.nan
                    results_volume[beta_O][beta_D] = np.nan

        self.results_S, self.results_volume, self.models, self.hybrid_plant_model = results_S, results_volume, models, hybrid_plant_model

    def visualize_risk_impact_heatmap(self):
        # beta_D_grid, beta_O_grid = np.meshgrid(self.beta_D_list, self.beta_O_list)
        S_vals = np.array([[self.results_S[bO][bD] for bD in self.beta_D_list] for bO in self.beta_O_list])
        volume_vals = np.array([[self.results_volume[bO][bD] for bD in self.beta_D_list] for bO in self.beta_O_list])

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # unify_palette_cyclers(axs)
        im1 = axs[0].imshow(S_vals, origin='lower', cmap='coolwarm',
                            extent=[min(self.beta_D_list), max(self.beta_D_list), min(self.beta_O_list), max(self.beta_O_list)], aspect='auto')
        axs[0].set_title("Strike price (S in [€/MWh])")
        axs[0].set_xlabel(r"$\beta_D$")
        axs[0].set_ylabel(r"$\beta_O$")
        fig.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(volume_vals, origin='lower', cmap='viridis',
                            extent=[min(self.beta_D_list), max(self.beta_D_list), min(self.beta_O_list), max(self.beta_O_list)], aspect='auto')
        axs[1].set_title(f"{self.PPA_profile} volume {"(M in [MW])" if self.BL else "($\\gamma$ in [pu])"}")
        axs[1].set_xlabel(r"$\beta_D$")
        axs[1].set_ylabel(r"$\beta_O$")
        fig.colorbar(im2, ax=axs[1])

        prettify_subplots(axs)
        for ax in axs:
            ax.grid(False)
            ax.legend_.remove()
        plt.tight_layout()
        plt.show()


#%%
if __name__ == "__main__":
    load_plot_configs()
    t0 = time.time()
    # Fixed parameters:
    alpha = 0.75  # CVaR: tail of interest

    # Capture price VRE: (d.P_DA_w * d.lambda_DA_w).sum() / d.P_DA_w.sum() = 96.38 €/MWh
    # Capture price load: - (d.L_t * d.lambda_DA_w).sum() / (d.W * d.L_t.sum()) = -98.1 €/MWh
    S_LB, S_UB = 96.38 * 0.5, 98.1*1.5  # PPA strike price bounds
    M_LB, M_UB = 0.01, 0.99  # BL volume bounds.
    gamma_LB, gamma_UB = 0, 1  # PaP capacity share bounds.

    # Profile type
    PPA_profile = 'BL'
    BL_compliance_perc = 0.1

    # Define ranges for betas
    beta_D_list = np.round(np.arange(0.2, 0.3, 0.2), 2)  # avoid floating point issues
    beta_O_list = np.round(np.arange(0.2, 0.3, 0.2), 2)  # avoid floating point issues

    P_fore_w, lambda_DA_w, L_t, WTP = generate_data()
    P_batt, batt_eta, batt_Crate = specify_battery_data()

    runner = NBSMultModel(
        PPA_profile=PPA_profile,  # Type of PPA profile ('PaF', 'PaP', or 'BL')
        BL_compliance_perc=BL_compliance_perc, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
        P_fore_w=P_fore_w,
        P_batt=P_batt,
        batt_eta=batt_eta,
        batt_Crate=batt_Crate,
        L_t=L_t,
        lambda_DA_w=lambda_DA_w,
        WTP=WTP,
        # add_batt=True,
        S_LB=S_LB,  # Minimum PPA strike price
        S_UB=S_UB,  # Maximum PPA strike price
        M_LB=M_LB,  # BL: Minimum baseload volume
        M_UB=M_UB,  # BL: Maximum baseload volume
        gamma_LB=gamma_LB, # PaP: Minimum PPA capacity share volume
        gamma_UB=gamma_UB, # PaP: Minimum PPA capacity share volume
        alpha=alpha,  # CVaR: Tail of interest for CVaR
    )
    #%%
    runner.run_multiple_NBS_models(beta_O_list=beta_O_list,
                                   beta_D_list=beta_D_list)

    runner.visualize_risk_impact_heatmap()

    beta_O_chosen=beta_O_list[0]
    beta_D_chosen=beta_D_list[0]

    # For debugging
    d = runner.models[beta_O_chosen][beta_D_chosen]
    # end

    d.visualize_example_outcome()
    d.visualize_example_profit_dist()
    d.verify_behaviour()
    print(f"Total time elapsed: {time.time()-t0:.2f}")
