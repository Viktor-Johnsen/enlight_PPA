import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

def initialize_data():
    np.random.seed(42)  # for reproducibility
    T=24
    W=10
    dist = 0.5*np.random.weibull(1.5, size=(T, W))
    P_fore_w = dist/np.max(dist)  # normalized forecast between 0 and 1

    # Generate DA prices
    lambda_DA_day = np.array([
        89.33, 89.14, 87.95, 86.89, 88.69, 98.73,
        113.97, 117.38, 108.84, 100.01, 72.64, 64.23,
        40.25, -23.12, 39.33, 71.01, 83.13, 110.93,
        125.91, 220.25, -195.33, 119.71, 108.31, 97.7
        ]).reshape(T, 1)  # September 1st prices

    num_days = 1

    lambda_DA = np.vstack([lambda_DA_day for _ in range(num_days)])
    lambda_DA = lambda_DA.ravel().reshape(T, 1)
    lambda_DA_coeffs = np.random.uniform(low=0.5, high=1.5, size=(T, W))
    lambda_DA_w = lambda_DA * lambda_DA_coeffs  # scenario-based prices

    return T, W, lambda_DA_w, P_fore_w

class HybridVRE:
    '''
    This class is used to shorthand the formulation of a
    VRE producer with a BESS (hybrid plant) participating
    in the DA market.

    It is similar to the option of 'current BL' in 'PPA_modeling',
    however, this is written in gurobipy and includes multiple scenarios.
    '''

    # Prepare model inputs
    def __init__(self,
        # no additional constraints: 998.6 €, PPA cov. 66.6%
        P_fore_w : np.ndarray,  # power forecast in MW
        lambda_DA_w : np.ndarray,  # €/MWh, DA prices
        model : gp.Model = None,  # use an existing model to build on top of that one
        add_batt : bool = True,  # boolean to include battery or not
        batt_power : float = 0.25, # MW
        batt_eta : float = float(np.sqrt(0.9)),  # round-trip efficiency
        batt_Crate : float = 1, # C-rate (1/C-rate = hours of storage

    ) -> None:

        self.P_fore_w = P_fore_w
        self.lambda_DA_w = lambda_DA_w
        self.model = model
        self.add_batt = add_batt
        self.batt_power = batt_power
        self.batt_eta = batt_eta

        self.T, self.W = P_fore_w.shape
        self.batt_energy = batt_power / batt_Crate  # assume 4 hours of storage
        self.PROB_w = np.full(shape=self.W, fill_value=1/self.W)  # all scenarios are equiprobable

        # self.build_model()
        # self.build_model_standalone()
        # self.build_model_extendable()

    def build_vars(self) -> None:
        # DA offer
        self.p_DA = self.model.addMVar(shape=(self.T, self.W),
                                       lb = 0,
                                       ub=self.P_fore_w + self.add_batt * self.batt_power,
                                       name="p_DA")  # DA market bid in MW
        # Battery variables
        self.SOC = self.model.addMVar(shape=(self.T, self.W),
                                      lb=0,
                                      ub=self.add_batt * self.batt_energy,  # assume 1 MWh
                                      name="SOC")  # State of Charge in MWh
        self.y_dch = self.model.addMVar(shape=(self.T, self.W),
                                        lb=0,
                                        ub=self.add_batt * self.batt_power,
                                        name="y_dch")  # discharge power in MW
        self.y_ch = self.model.addMVar(shape=(self.T, self.W),
                                        lb=0,
                                        ub=self.add_batt * self.batt_power,
                                        name="y_ch")  # charge power in MW

    def build_cons(self) -> None:
        self.batt_bal = self.model.addConstrs((self.SOC[t, w] - self.SOC[t-1, w]
                                               ==
                                               self.y_ch[t, w] * self.batt_eta
                                               - self.y_dch[t, w] / self.batt_eta
                                               for t in range(1, self.T)
                                               for w in range(self.W)),
                                              name="batt_bal")
        
        self.batt_bal_init = self.model.addConstrs((self.SOC[0, w]
                                                     ==
                                                     self.y_ch[0, w] * self.batt_eta
                                                     - self.y_dch[0, w] / self.batt_eta
                                                     for w in range(self.W)),
                                                    name="batt_bal_init")
        
        self.pow_bal = self.model.addConstrs((self.p_DA[t, w] + self.y_ch[t, w]
                                              <=
                                              self.P_fore_w[t, w] + self.y_dch[t, w]
                                              for t in range(self.T)
                                              for w in range(self.W)),
                                             name="pow_bal")

    def build_obj(self) -> None:
        self.model.setObjective(
            gp.quicksum(
                self.PROB_w[w]
                * self.p_DA[t, w] * self.lambda_DA_w[t, w]
                for t in range(self.T)
                for w in range(self.W)),
            sense=GRB.MAXIMIZE
        )

    def build_model(self) -> None:
        self.model = gp.Model("model")

        # Define simple hybrid vre model
        self.build_vars()
        self.build_cons()
        self.build_obj()

    def build_and_extract_model_no_obj(self) -> gp.Model:
        if self.model is None:
            self.model = gp.Model("model")

        self.build_vars()
        self.build_cons()

        return self.model, self.p_DA, self.SOC, self.y_dch, self.y_ch

    def run_model(self) -> None:
        self.model.optimize()

    def get_results(self) -> None:
        results_dfs = {}
        results_dfs['p_DA'] = pd.DataFrame(self.p_DA.X)
        if self.add_batt:
            results_dfs['SOC'] = pd.DataFrame(self.SOC.X)
            results_dfs['y_dch'] = pd.DataFrame(self.y_dch.X)
            results_dfs['y_ch'] = pd.DataFrame(self.y_ch.X)

        self.results_dfs = results_dfs

    def plot_results(self, scenario: int) -> None:
        w = scenario

        fig, ax = plt.subplots(figsize=(10,6))
        ax2 = ax.twinx()

        if not hasattr(self, 'results_df'):
            self.get_results()
        
        ax.plot(self.P_fore_w[:, w], label="P_fore")
        (self.results_dfs['p_DA'] + self.results_dfs['y_ch']).loc[:, w].plot(ax=ax, label="p_DA + y_ch", ls='--', alpha=.8)
        self.results_dfs['p_DA'].loc[:, w].plot(ax=ax, ls='--', alpha=.5, label="p_DA")
        self.results_dfs['SOC'].loc[:, w].plot(ax=ax, ls=':', alpha=.3, label="SOC")
        ax2.plot(self.lambda_DA_w[:, w], label="DA prices", c='k')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title(f"Scenario {w} results")
        plt.show()


if __name__ == "__main__":
    # Generate forecasts

    T, W, lambda_DA_w, P_fore_w = initialize_data()
    betas = np.array([0.0, 0.3, 0.6, 0.9])
    alpha = 0.9

    hp = hybrid_vre(
        P_fore_w = P_fore_w,
        lambda_DA_w = lambda_DA_w,
        add_batt = True,
        batt_power = .25,
        batt_Crate = 1,
    )
    hp.build_model()
    hp.run_model()
    hp.get_results()
    w=3
    hp.plot_results(scenario=w)

    PI_D_w = (hp.p_DA.X * hp.lambda_DA_w).sum(axis=0)  # expected power revenues in scenario w
    VAR_idx = round(W*(1-alpha)-1)  # "-1" because of 0-indexing
    VAR_D = np.sort(PI_D_w)[VAR_idx]
    ETA_D_w = VAR_D - PI_D_w
    print("Expected Power Incomes per scenario:\n", PI_D_w)
    print(f"Value at Risk at {alpha*100}% confidence level:\n", VAR_D)
    print("Excess over VaR per scenario:\n", ETA_D_w)

    ###
    # print(np.array(hp.model.ObjVal) * (1 - np.array(betas)) + np.array(betas) * np.min((hp.p_DA.X * hp.lambda_DA_w).sum(axis=0)))
    print((1 - betas) * sum([PI_D_w[w] * hp.PROB_w[w] for w in range(W)])
            + betas * (VAR_D - 1/(1-alpha) * sum([ETA_D_w[w] * hp.PROB_w[w] for w in range(W) if ETA_D_w[w] >= 0])))
    print((1 - betas) * hp.model.ObjVal
            + betas * (VAR_D - 1/(1-alpha) * sum([ETA_D_w[w] * hp.PROB_w[w] for w in range(W) if ETA_D_w[w] >= 0])))
    # print(np.array(hp.model.ObjVal) * (1 - np.array(betas)) + np.array(betas) * np.sort((hp.p_DA.X * hp.lambda_DA_w).sum(axis=0))[VAR_idx])
    ###
    d_D = (
                (1 - betas) * hp.model.ObjVal  # Expected profit
                + betas  # CVaR term in objective
                    * (VAR_D
                       - 1/(1-alpha)
                         # probability-weighted profits BELOW CVaR. 0 if above:
                       * sum([ETA_D_w[w] * hp.PROB_w[w] for w in range(W) if ETA_D_w[w] >= 0])
                       )
            )
    print("d_D:", d_D)
    print("ObjVal:", hp.model.ObjVal)