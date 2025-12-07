import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import hybrid_vre_in_da as hv
import logging
import enlight.utils as utils
from pathlib import Path


def generate_data():
    np.random.seed(42)
    num_weeks = 1/7
    num_days = int(7 * num_weeks)
    T = 24 * num_days  # num hours
    W = 10  # num scenarios

    PROB_w = np.full(shape=W, fill_value=1/W)  # all scenarios are equiprobable

    # Generate forecasts
    dist = 0.5*np.random.weibull(1.5, size=(T, W))
    P_fore_w = dist/np.max(dist)  # normalized forecast between 0 and 1

    # Generate DA prices
    lambda_DA_day = np.array([
        89.33, 89.14, 87.95, 86.89, 88.69, 98.73,
        113.97, 117.38, 108.84, 100.01, 72.64, 64.23,
        40.25, -23.12, 39.33, 71.01, 83.13, 110.93,
        125.91, 220.25, -195.33, 119.71, 108.31, 97.7
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
        alpha : float = 0.9,  # CVaR: Tail of interest for CVaR
        nbs_model_logger : logging.Logger | None = None,
    ) -> None:
    
        '''
        Initialize the NBS model with necessary parameters, variables, and constraints.
        '''
        self.nbs_model_logger = nbs_model_logger or utils.setup_logging(log_file="nbs.log")
        self.nbs_model_logger.info(f"INITIALIZING NBS MODEL with betas=(O:{beta_O}, D:{beta_D})")
        
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
            self.batt_power, self.batt_eta, self.batt_Crate = specify_battery_data()

            if self.hp is None:
                print("RUNNING HYBRID VRE")
                hp = hv.HybridVRE(
                    P_fore_w = self.P_fore_w,
                    lambda_DA_w = self.lambda_DA_w,
                    # model = None,
                    add_batt = self.add_batt,
                    batt_power = self.batt_power,
                    batt_eta = self.batt_eta,
                    batt_Crate = self.batt_Crate,
                )
                hp.build_model()
                hp.run_model()
                hp.get_results()

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
        
        #CVaR vars:
        self.S = self.model.addVar(lb=self.S_LB, ub=self.S_UB, vtype=GRB.CONTINUOUS, name="S")        
        self.zeta_D = self.model.addVar(vtype=GRB.CONTINUOUS, name="zeta_D")
        self.zeta_O = self.model.addVar(vtype=GRB.CONTINUOUS, name="zeta_O")
        self.eta_D_w = self.model.addVars(self.W, lb=0, vtype=GRB.CONTINUOUS, name="eta_D_w")
        self.eta_O_w = self.model.addVars(self.W, lb=0, vtype=GRB.CONTINUOUS, name="eta_O_w")

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
            batt_power = self.batt_power,
            batt_eta = self.batt_eta,
            batt_Crate = self.batt_Crate,
        )

        # add the vars and cons for batt to NBS BL model formulation
        # and save the new variables for easy access and manipulation
        self.model, self.p_DA, self.SOC, self.y_dch, self.y_ch = hp_BL.build_and_extract_model_no_obj()

    def build_profile_vars_constrs(self) -> None:
        '''
        This method builds the variables and constraints that are profile-specific.
        Meaning that they differ between a BL and a PaP electricity profile.
        All the other methods are general for a NBS mathematical model.
        '''
        # Utility/profit variables that EXCLUDE the CVaR term
        self.y_D = self.model.addVars(self.W, vtype=GRB.CONTINUOUS, name="y_D")  # developer profit excluding CVaR
        self.y_O = self.model.addVars(self.W, vtype=GRB.CONTINUOUS, name="y_O")  # off-taker net utility excluding CVaR

        # Profile-specific variables
        if self.PPA_profile in ['PaF', 'PaP']:
            self.gamma = self.model.addVar(lb=self.gamma_LB, ub=self.gamma_UB, vtype=GRB.CONTINUOUS, name="gamma")
            # Further if it is the future settlement, then the PPA payment and power delivery is COUPLED:
            if self.PPA_profile == 'PaP':
                # Power offered in DA but compensated in the PPA
                self.p_PPA = self.model.addMVar(shape=(self.T, self.W), lb=0, vtype=GRB.CONTINUOUS, name="p_PPA")

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
            # Currently it is not useful because we assume that the Producer
            # always earns net the PPA price, however, if the Buyer would choose
            # not to settle negative prices then the Producer's offering price would change.
            # including 'p_PPA' always to easily adapt the model to capture this behaviour.
            self.model.addConstrs((self.p_PPA[t, w]
                                    <=
                                    self.gamma * self.P_fore_w[t, w]
                                    for t in range(self.T) for w in range(self.W)),
                                    name="c_p_PPA")
            
            # Write objective function of the profit-maximizing developer
            #   and utility-maximizing off-taker.
            self.model.addConstrs((self.y_D[w] ==
                                gp.quicksum(
                                    # (1 - self.gamma) is implicit in p_DA because of
                                    #   the upper bound (1-gamma)*P_fore.
                                    (1 - self.gamma) * self.P_DA_w[t, w] * self.lambda_DA_w[t, w]  # DA revenues
                                    # self.gamma is implicit in p_PPA because of
                                    #   the upper bound gamma*P_fore
                                    + self.p_PPA[t, w] * self.S # PaP PPA revenues
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yD_link_PaP')
            self.model.addConstrs((self.y_O[w] ==
                                gp.quicksum(
                                    # They only pay for what the developer actually produces: p_DA
                                    #   not just for the forecast: P_fore_w.
                                    # Note that we use P_fore_w instead of P_DA_w, because the offers
                                    #   are now optimized as well, as for P_DA_w the assumption of MC=0 is made.
                                    self.p_PPA[t, w] * (self.WTP - self.S)  # The PPA volume is paid at the strike price
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
            self.model.addConstrs((self.y_D[w] ==
                                gp.quicksum(
                                    # OLD, plain P_DA_w (= forecast in non-negative price hours):
                                    # self.lambda_DA_w[t,w] * self.P_DA_w[t,w]  # DA revenues
                                    # NEW, p_DA optimized with BESS
                                    self.lambda_DA_w[t,w] * self.p_DA[t,w]  # DA revenues
                                    + (self.S - self.lambda_DA_w[t,w]) * self.M  # BL PPA revenues
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yD_link_BL')
            self.model.addConstrs((self.y_O[w] ==
                                gp.quicksum(
                                    self.L_t.ravel()[t] * (self.WTP - self.lambda_DA_w[t,w])  # Costs in DA
                                    - self.M * (self.S - self.lambda_DA_w[t,w])  # Costs in BL PPA
                                    for t in range(self.T)
                                ) for w in range(self.W)), name='c_yO_link_BL')
            # further, add compliance:
            if self.BL:
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

        else:
            raise Exception(f"{self.PPA_profile} is not a valid profile type. Please choose 'PaF', 'PaP', 'BL', or 'BL-COMPLIANCE'." )

    def build_aux_constrs(self) -> None:
        # Linear linking constraints
        self.model.addConstr(self.x_D == self.u_D - self.d_D, name="c_diff_D")
        self.model.addConstr(self.x_O == self.u_O - self.d_O, name="c_diff_O")

        self.model.addGenConstrLog(self.x_D, self.log_uD_dD, name="c_log_uD_dD")
        self.model.addGenConstrLog(self.x_O, self.log_uO_dO, name="c_log_uO_dO")

        self.model.setObjective(self.log_uD_dD + self.log_uO_dO, sense=GRB.MAXIMIZE)

        # Add aux constraints to make obj readable
        self.model.addConstr(self.u_D
                        == (1 - self.beta_D) * gp.quicksum(  # Expected profit
                            self.PROB_w[w] * #gp.quicksum(
                                self.y_D[w]
                                #for t in range(self.T)
                            #) 
                            for w in range(self.W)
                        )
                        + self.beta_D * (  # The CVaR term
                            self.zeta_D - 1/(1-self.alpha) * gp.quicksum(
                                self.PROB_w[w] * self.eta_D_w[w] for w in range(self.W)
                            )  
                        ),
                        name='c_uD')

        self.model.addConstr(self.u_O
                        == (1 - self.beta_O) * gp.quicksum( # Expected net utility
                            self.PROB_w[w] * #gp.quicksum(
                                self.y_O[w]
                                #for t in range(self.T)
                            #)
                            for w in range(self.W)
                        )
                        + self.beta_O * (  # The CVaR term,
                            self.zeta_O - 1/(1-self.alpha) * gp.quicksum(
                                    self.PROB_w[w] * self.eta_O_w[w] for w in range(self.W)
                            )
                        ),
                        name='c_uO')

    def build_cvar_constrs(self) -> None:
        self.model.addConstrs((self.eta_D_w[w]
                        >=
                        self.zeta_D - #gp.quicksum(
                                self.y_D[w]  # Developer profit in scenario w (excl. CVaR term)
                                #for t in range(self.T)
                        #)
                        for w in range(self.W)),
                        name='c_CVaR_D')

        self.model.addConstrs((self.eta_O_w[w]
                        >=
                        self.zeta_O - #gp.quicksum(
                                self.y_O[w]  # Off-taker utility in scenario w (excl. CVaR term)
                                #for t in range(self.T)
                        #)
                        for w in range(self.W)),
                        name='c_CVaR_O')

    def build_objective(self) -> None:
        self.model.setObjective(self.log_uD_dD + self.log_uO_dO
                                # - self.eps * gp.quicksum(self.y_ch[t, w] + self.y_dch[t, w] for t in range(self.T) for w in range(self.W))
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
        self.model.optimize()

    def visualize_example_outcome(self):
        if self.model.status == GRB.OPTIMAL:
            S_X = self.S.X
            low_perc = 10
            high_perc = 90

            fig, ax = plt.subplots(2,1, figsize=(8,10))

            ax[0].plot(perc(self.P_fore_w, 50), label='Median power forecast', color='r', linestyle='-')
            ax[0].fill_between(range(self.T), perc(self.P_fore_w, low_perc), perc(self.P_fore_w, high_perc), color='r', alpha=0.4, label='Power forecast 10-90 percentile')
            ax[0].plot(self.L_t, label='Off-taker load profile', color='b', linestyle='-')
            ax[0].plot(perc(self.P_DA_w, 50), label='Median DA dev. acc. off.', color='g', ls='--')

            if self.BL:
                M_X = self.M.X
                ax[0].axhline(M_X, color='k', linestyle='-.', alpha=0.7,
                            label='Agreed BL volume $M$')
            elif self.PPA_profile in ['PaF', 'PaP']:
                gamma_X = self.gamma.X
                ax[0].plot(gamma_X * perc(self.P_fore_w, 50), color='k', linestyle='-.', alpha=0.7,
                        label=r'Agreed PaP volume $\gamma$ (shown as % of median power forecast)')

            ax[0].set_ylabel('Power [MW]')

            ax[1].plot(perc(self.lambda_DA_w, 50), label='Median DA prices', color='r', linestyle='-')
            ax[1].fill_between(range(self.T), perc(self.lambda_DA_w, low_perc), perc(self.lambda_DA_w, high_perc), color='r', alpha=0.4, label='DA price 10-90 percentile')
            ax[1].axhline(S_X, color='k', linestyle='-.', label=f'Optimal {self.PPA_profile} PPA strike price $S$')
            ax[1].set_ylabel('Price [€/MWh]')

            ax[0].legend()
            ax[1].legend()
            plt.suptitle(fr"Off-taker with $\beta^O$={self.beta_O}, and developer with $\beta^D$={self.beta_D}")
            plt.tight_layout()
            plt.show()
        else:
            print("No results to show. No optimal solution was found.")

    def visualize_example_profit_hists(self):
        if self.model.status == GRB.OPTIMAL:

            # # Compare their revenues distributions before and after
            # # Pure DA revenues
            # PI_D_w = [self.models[beta_O_chosen][beta_D_chosen].PI_D_w[w] for w in range(nbs_.W)]
            # PI_O_w = [self.models[beta_O_chosen][beta_D_chosen].PI_O_w[w] for w in range(nbs_.W)]
            # # VaR before PPA
            # VaR_D = self.models[beta_O_chosen][beta_D_chosen].VAR_D
            # VaR_O = self.models[beta_O_chosen][beta_D_chosen].VAR_O
            # Expected DA + PPA revenues
            PI_D_w_NBS = [self.y_D[w].X for w in range(self.W)]
            PI_O_w_NBS = [self.y_O[w].X for w in range(self.W)]
            # VaR after PPA
            # zeta_D = self.models[beta_O_chosen][beta_D_chosen].zeta_D.X
            # zeta_O = self.models[beta_O_chosen][beta_D_chosen].zeta_O.X


            fig, ax = plt.subplots(1, 2, figsize=(8,6))
            ax[0].hist(self.PI_D_w, color='r', alpha=0.5, label='Before')
            ax[0].axvline(x=self.VAR_D, color='r', linestyle='--', label='VaR before')
            ax[1].hist(self.PI_O_w, color='r', alpha=0.5, label='Before')
            ax[1].axvline(x=self.VAR_O, color='r', linestyle='--', label='VaR before')

            # PI_D_w_NBS = (nbs_.lambda_DA_w * nbs_.P_DA_w + (S_X - nbs_.lambda_DA_w) * M.X).sum(axis=0)
            # PI_O_w_NBS = ( (nbs_.WTP - nbs_.lambda_DA_w) * nbs_.L_t - (S_X - nbs_.lambda_DA_w) * M.X).sum(axis=0)
            ax[0].hist(PI_D_w_NBS, color='b', alpha=0.5, label='after NBS')
            ax[0].axvline(x=self.zeta_D.X, color='b', linestyle='--', label='VaR after')
            ax[1].hist(PI_O_w_NBS, color='b', alpha=0.5, label='after NBS')
            ax[1].axvline(x=self.zeta_O.X, color='b', linestyle='--', label='VaR after')

            ax[0].set_title(fr"Developer with $\beta^D=${self.beta_D}")
            ax[0].legend()
            ax[1].set_title(fr"Off-taker with $\beta^O=${self.beta_O}")
            ax[1].legend()
            plt.show()
        else:
            print("No results to show. No optimal solution was found.")
     
    def verify_behaviour(self, w_BESS=3):
        if self.model.status == GRB.OPTIMAL:

            # Verify behaviour of p_DA and p_PPA. If lambda_DA >= 0 in all hour-scenarios, then we should always max out both!!
            if self.PPA_profile == 'PaP':
                for i in range(d.W)[:4]:
                    fig, ax = plt.subplots(figsize=(10,6))
                    ax2 = ax.twinx()

                    # Plot and compare total power available and offered
                    # ax.plot(d.P_DA_w[:,i], alpha=.7, label=r"$\overline{P}^{DA}$")
                    ax.plot(self.P_fore_w[:,i], ls='-', alpha=.5, lw=3, label=r"$P^{fore}$")
                    ax.plot((1-self.gamma.X) * self.P_DA_w[:,i] + self.p_PPA.X[:,i], ls=':', alpha=.5, label=r"$P^{DA} + p^{PPA}$")

                    # Plot and compare the power available and offered & remunerated at DA price
                    ax.plot((1-self.gamma.X) * self.P_fore_w[:, i], alpha=.3, lw=3, label=r"$\left(1-\gamma\right) \cdot P^{fore}$")
                    ax.plot((1-self.gamma.X) * self.P_DA_w[:,i], alpha=.5, ls='--', label=r"$P^{DA}$")

                    # Plot and compare power available for PPA and offered to comply with PPA
                    ax.plot(self.gamma.X * self.P_fore_w[:,i], alpha=.3, lw=3, label=r"$\gamma \cdot P^{fore}$")
                    ax.plot(self.p_PPA.X[:,i], alpha=.5, ls='--', label=r"$p^{PPA}$")

                    # ax.plot(d.L_t, label=r"$L$")
                    # ax2.plot(d.lambda_DA_w[:, i], ls=':', c='k', label=r'$\lambda^{DA}$')
                    # ax2.axhline(d.S.X, c='k', label="S")
                    ax.legend(loc='upper left')
                    # ax2.legend(loc='upper right')
                    ax.set_title(f"w = {i}")
                    plt.show()

            elif self.BL:
                # inspect the BESS behaviour to verify that HybridVRE has been correctly included in this model.
                w_BESS=3
                plt.plot(self.P_fore_w[:, w_BESS], label="P_fore")
                plt.plot((self.p_DA.X + self.y_ch.X)[:, w_BESS], label="p_DA + y_ch", ls='--')
                plt.plot(self.p_DA.X[:, w_BESS], label="p_DA", ls='--', alpha=.5)
                if self.BL_compliance_perc > 0:
                    plt.plot(self.v_min.X[:, w_BESS], label="v_min", c='r', alpha=.5)
                plt.plot(self.SOC.X[:, w_BESS], label="SOC", ls=':', alpha=0.3)
                plt.axhline(self.M.X, c='k', label="BL volume", alpha=.4)
                plt.legend()
                if self.beta_D == 1.0:
                    plt.title(r'Beware! Nonsensical for $\beta_D=1.0$')
                # plt.plot(d.y_ch.X[: ,w] * d.y_dch.X[: ,w])
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
        alpha : float = 0.9,  # CVaR: Tail of interest for CVaR
        nbs_mult_logger : logging.Logger | None = None
    ) -> None:
        self.nbs_mult_logger = nbs_mult_logger or utils.setup_logging(log_file="nbs.log")
        self.nbs_mult_logger.info(f"INITIALIZE NBSMultModel instance")

        self.PPA_profile = PPA_profile

        self.BL = False
        if self.PPA_profile == 'BL':
            self.BL = True

        if not (1 >= BL_compliance_perc >= 0):
            raise Exception(f"The chosen level of PPA BL compliance is not valid: {BL_compliance_perc}. It should be between [0,1].")
        else:
            self.BL_compliance_perc = BL_compliance_perc

        self.P_fore_w = P_fore_w
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
        self.nbs_mult_logger.info(f"SOLVING multiple NBS models using NBSMultModel instance")

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
                nbs_model = NBSModel(
                    PPA_profile=self.PPA_profile,  # BL or PaP
                    BL_compliance_perc=self.BL_compliance_perc,
                    P_fore_w=self.P_fore_w,
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

                # Solve the optimization problem.
                nbs_model.solve_model()

                # Save the results of PPA price and volume explicitly if it was solved to optimality.
                if nbs_model.model.status == GRB.OPTIMAL:
                    results_S[beta_O][beta_D] = nbs_model.S.X
                    if nbs_model.BL:
                        results_volume[beta_O][beta_D] = nbs_model.M.X
                    elif nbs_model.PPA_profile in ['PaF', 'PaP']:
                        results_volume[beta_O][beta_D] = nbs_model.gamma.X
                else:
                    results_S[beta_O][beta_D] = np.nan
                    results_volume[beta_O][beta_D] = np.nan

        self.results_S, self.results_volume, self.models, self.hybrid_plant_model = results_S, results_volume, models, hybrid_plant_model

    def visualize_risk_impact_heatmap(self):
        # beta_D_grid, beta_O_grid = np.meshgrid(self.beta_D_list, self.beta_O_list)
        S_vals = np.array([[self.results_S[bO][bD] for bD in self.beta_D_list] for bO in self.beta_O_list])
        volume_vals = np.array([[self.results_volume[bO][bD] for bD in self.beta_D_list] for bO in self.beta_O_list])

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axs[0].imshow(S_vals, origin='lower', cmap='coolwarm',
                            extent=[min(self.beta_D_list), max(self.beta_D_list), min(self.beta_O_list), max(self.beta_O_list)], aspect='auto')
        axs[0].set_title("Strike price (S)")
        axs[0].set_xlabel(r"$\beta_D$")
        axs[0].set_ylabel(r"$\beta_O$")
        fig.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(volume_vals, origin='lower', cmap='viridis',
                            extent=[min(self.beta_D_list), max(self.beta_D_list), min(self.beta_O_list), max(self.beta_O_list)], aspect='auto')
        axs[1].set_title(f"{self.PPA_profile} volume {"(M)" if self.BL else "($\\gamma$)"}")
        axs[1].set_xlabel(r"$\beta_D$")
        axs[1].set_ylabel(r"$\beta_O$")
        fig.colorbar(im2, ax=axs[1])

        plt.tight_layout()
        plt.show()


#%%
if __name__ == "__main__":
    # Fixed parameters:
    alpha = 0.9  # CVaR: tail of interest

    # Capture price VRE: (d.P_DA_w * d.lambda_DA_w).sum() / d.P_DA_w.sum() = 96.38 €/MWh
    # Capture price load: - (d.L_t * d.lambda_DA_w).sum() / (d.W * d.L_t.sum()) = -98.1 €/MWh
    S_LB, S_UB = 96.38 * 0.5, 98.1*1.5  # PPA strike price bounds
    M_LB, M_UB = 0.01, 0.99  # BL volume bounds.
    gamma_LB, gamma_UB = 0, 1  # PaP capacity share bounds.

    # Profile type
    PPA_profile = 'PaF'
    BL_compliance_perc = 0.0

    # Define ranges for betas
    beta_D_list = np.round(np.arange(0.0, 0.31, 0.1), 2)  # avoid floating point issues
    beta_O_list = np.round(np.arange(0.0, 0.31, 0.1), 2)  # avoid floating point issues

    P_fore_w, lambda_DA_w, L_t, WTP = generate_data()

    runner = NBSMultModel(
        PPA_profile=PPA_profile,  # Type of PPA profile ('PaF', 'PaP', or 'BL')
        BL_compliance_perc=BL_compliance_perc, # indicates the enforced compliance of the producer: meaning the % of PPA volume where the producer has to match the BL volume on an hourly basis
        P_fore_w=P_fore_w,
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

    beta_O_chosen=beta_O_list[3]
    beta_D_chosen=beta_D_list[2]

    # For debugging
    d = runner.models[beta_O_chosen][beta_D_chosen]
    # end

    d.visualize_example_outcome()
    d.visualize_example_profit_hists()
    d.verify_behaviour()
