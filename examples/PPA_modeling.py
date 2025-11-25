import numpy as np
# import pandas as pd
import linopy
import matplotlib.pyplot as plt

def load_plot_configs() -> None:
    palette1 = ["#7782f0", "#70e0aa", "#f8dd7a", "#fc9a67", "#f9ccc3", "#ef7a81", "#4ab877", "#a566b5"]
    nature_pastel1 = [
        "#595959",  # soft black / gray
        "#F3C76B",  # pastel orange
        "#89CAF0",  # pastel sky blue
        "#44C1A1",  # pastel bluish green
        "#F6EB88",  # pastel yellow
        "#4C97C9",  # pastel blue
        "#E98A4C",  # pastel vermillion
        "#DA95BF",  # pastel reddish purple
    ]
    nature_pastel2 = [
    "#B3B3B3",  # light gray
    "#F9DB9C",  # light orange
    "#A7D8F4",  # light sky blue
    "#7FD3BB",  # light bluish green
    "#FAF2B8",  # light yellow
    "#7FB6DA",  # light blue
    "#F2A97E",  # light vermillion
    "#E4B5D0",  # light reddish purple
    ]
    # Set default color palette, font sizes, and font family.

    # chose palette1 or nature_pastel1
    chosen_palette = nature_pastel1
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=chosen_palette)  # matplotlib.pyplot
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 13
    })

    return chosen_palette
    
def unify_palette_cyclers(axs):  # run BEFORE plotting
    for ax in axs: # Unify palette cyclers for different plot types
        ax._get_patches_for_fill = ax._get_lines

    return axs
    
def prettify_subplots(axs):  # run AFTER plotting
    for ax in axs:
        ax.spines[['right', 'top']].set_visible(False)  # Remove spines
        ax.grid(alpha=.25)  # Add opaque gridlines
        ax.margins(0.005)  # Remove whitespace inside each plot
        ax.spines[['bottom','left']].set_alpha(0.5)  # Introduce opacity to the x- and y-axes spines
        ax.legend(bbox_to_anchor=[1, 1.05])
    
    return axs

def make_negative_DA_price_mask(times, lambda_DA):
    # Initialize arrays
    mask = lambda_DA.ravel() < 0

    # Detect whether the price changes from negative to positive or positive to negative
    transitions_from = np.where(np.diff(mask.astype(int)) != 0)[0]
    transitions_to = np.where(np.diff(mask.astype(int)) != 0)[0] + 1

    # We want to insert points only "after" a negative hour
    insert_points_idx = [t for t in transitions_to if mask[t-1]]

    # Build new extended lists
    mask_ext = mask.astype(int).tolist()
    times_ext = times.tolist()

    for t in reversed(insert_points_idx):  # we reverse to avoid that the first insert's affect later indices
        mask_ext.insert(t, True)
        times_ext.insert(t, times[t])

    mask_ext = np.append(mask_ext, mask_ext[-1])
    times_ext = np.append(times_ext, times_ext[-1])

    return mask_ext, times_ext

class PPA_modeling:
    '''
    This class is used to model the behaviour of a profit-maximizing
    developer who has entered a PPA.

    By setting the PPA_profile to either 'BL' or 'PaP' the class can be used
    to investigate the behaviour of a developer under both types of PPA profiles.
    The net settlement of a physical and virtual PPA are identical, so this class
    can be used to investigate both types of PPA structures as well.

    It is possible to choose between two objective functions:
        - "Financial" which is a normal/current BL meaning that
        the objective function is identical to a developer with no PPA,
        since her PPA payment is independent of her power delivery.
            - This objective function comes with the opportunity of specifying
            a required level of PPA coverage to avoid very low coverage.
        - "Coupled" which is a new BL settlement mechanism where the PPA
        payment depends on the power delivery. Thus, the PPA-DA price margin
        is not received for the power deficit.
            - This objective function allows for two additional constraints:
            1) Disallowing charging in deficit hours
            2) Disallowing offering more than V_BL in deficit hours
            Both restrict the usage of the battery.
    '''

    # Prepare model inputs
    def __init__(self,
        # no additional constraints: 998.6 €, PPA cov. 66.6%
        PPA_profile : str,  # the PPA electricity profile.
        PaP_coupled : bool = False,  # changes the ¡PaP! OBJECTIVE
        financial_settlement : bool = False,  # changes the ¡BL! OBJECTIVE. Decoupled or coupled.
        enforce_no_charge_in_deficit : bool = False,  # changes the ¡BL! CONSTRAINTS. Restricts a coupled PPA.
        annual_compliance_percentage : bool = False,  # changes the ¡BL! CONSTRAINTS. Restrics a DEcoupled PPA.
        compliance_perc : float = 0.5,
        add_batt : bool = True,
    ) -> None:

        # Load the chosen PPA profile (if valid)
        self.PPA_profile = PPA_profile    

        # Let all bools that are not relevant for the chosen profile remain FALSE as given in the default statement.
        self.financial_settlement = financial_settlement
        self.enforce_no_charge_in_deficit = enforce_no_charge_in_deficit
        self.annual_compliance_percentage = annual_compliance_percentage

        # A financial settlement should not include any forced behaviour on an hourly basis.
        if self.financial_settlement and self.enforce_no_charge_in_deficit:
            raise Exception("It makes no conceptual sense to enforce any kind of offering behaviour in deficit hours if the settlement is financial since a financial settlement by definition decouples delivery and PPA payment. Thus deficit do not exist.")

        # A coupled settlement should not include any forced behaviour on an annual basis.
        if not self.financial_settlement and self.annual_compliance_percentage:
            raise Exception("The settlement is already coupled on an hourly basis. A yearly coupling would only make sense for a decoupled settlement.")

        self.compliance_perc = compliance_perc
        self.add_batt = add_batt
        
        self.PaP_coupled = PaP_coupled

        if self.PPA_profile == 'BL':
            # Overwrite any PaP bools that have mistakenly been set to True to False.
            self.PaP_coupled = False

        elif self.PPA_profile == 'PaP':
            # Overwrite any BL bools that have mistakenly been set to True to False.
            self.financial_settlement = False
            self.enforce_no_charge_in_deficit = False
            self.annual_compliance_percentage = False

            # It does not make sense to get a PaP for a hybrid BESS power plant.
            self.add_batt = False
        else:
            raise Exception(f"{self.PPA_profile} is not a valid PPA electricity profile. Please try baseload ('BL') or Pay-as-Produced ('PaP')")

        # Immediately run data and model-building methods.
        self.initialize_data()
        self.build_model()

        # Prepare to plot
        self.palette = load_plot_configs()

    def initialize_data(self) -> None:
        seed = np.random.seed(42)
        self.T = 24
        dist = 0.5*np.random.weibull(1.5, size=24)
        self.P_fore = dist/np.max(dist)  # normalized forecast between 0 and 1
        self.lambda_DA = np.array([
            89.33+10, 89.14+10, 87.95-10, 86.89-10, 88.69+10, 98.73,
            113.97, 117.38, 108.84, 100.01, 72.64, 64.23,
            40.25, 23.12, 39.33, 71.01, 83.13, 110.93,
            125.91, 220.25, 195.33, 119.71, 108.31, 97.7
            ]).reshape(self.T, 1)  # September 1st prices
        
        # Overwrite with new prices to include (very) negative prices
        np.random.seed(43)
        self.lambda_DA = ((np.random.random((24, 1))*2 - .75) * 220)
        
        # Assume a PPA price of 90% of the VRE capture price
        VRE_capture_price = (self.P_fore * self.lambda_DA.ravel()).sum() / self.P_fore.sum()
        self.lambda_PPA = float(0.9 * VRE_capture_price) # €/MWh
        self.V_BL = 0.3 # MW
        self.times = np.arange(self.T)
        self.bidding_zones = ['DK1']
        self.eta = float(np.sqrt(0.9))

        batt_to_VRE_capacity_ratio = 0.54 / 2
        self.P_batt = float(batt_to_VRE_capacity_ratio * np.max(self.P_fore))  #  maximum 1 MW
        self.E_batt = 3.1 * self.P_batt
        self.lambda_max = np.maximum(self.lambda_PPA, self.lambda_DA)

        # xi is an auxiliary parameter that helps the model choose to pool p_DA
        # into v_min in cases where lambda_max = lambda_DA.
        # If kept extremely low: it does NOT affect the optimal solution/value.
        self.xi = 1e-5

        # Build a boolean array used to ensure that we do NOT try to earn
        # high DA prices without complying with the PPA in the same hour...
        self.power_deficit = 1 * (self.P_fore < self.V_BL)  # 1: power deficit, 0: power sufficiency/surplus
        
        ### DATA ONLY USED FOR PLOTTING
        # If it's a financial settlement, then the developer can't be in a deficit.
        # This is relevant only for plotting.
        if self.financial_settlement:
            self.power_deficit = 0 * self.power_deficit

        # compute time step (assuming hourly or regular intervals)
        dt = self.times[1] - self.times[0]
        self.times_ext = np.append(self.times, self.times[-1] + dt)  # extend one more interval

    # Build the model
    def build_vars(self) -> None:
        # p_DA: All of the power offered into the DA market in hour t.
        self.p_DA = self.model.add_variables(  #  y_dch is implicit in this
            lower=0,
            # The upper limit is set to:
            #   - PaP: P_fore
            #   - BL: P_fore + P_batt       <- BL is assumed to allow for a battery but it works either way.
            upper=self.P_fore.reshape(self.T, 1) + self.add_batt * self.P_batt,
            coords=[self.times, self.bidding_zones],
            dims=["T","Z"],
            name="p_DA")

        # The only variable needed in the PaP case is the DA offer. The developer does not
        #   have to meet a certain volume (V_BL for BL) so v_min is not needed. Further,
        #   it does not make sense to use a BESS for a VRE plant with a PaP PPA, since the
        #   value of the power is constant across all hours.
        if self.PPA_profile == 'BL':
            # v_min: The first V_BL of p_DA in hour t.
            self.v_min = self.model.add_variables(  # part of y_dch is implicit in this
                lower=0,
                coords=[self.times, self.bidding_zones],
                dims=["T","Z"],
                name="v_min")

            self.SOC = self.model.add_variables(
                lower=0,
                upper=(self.E_batt if self.add_batt else 0), #1,
                coords=[self.times, self.bidding_zones],
                dims=["T","Z"],
                name="SOC")

            self.y_charge = self.model.add_variables(
                lower=0,
                upper=self.P_batt,
                coords=[self.times, self.bidding_zones],
                dims=["T","Z"],
                name="y_charge")

            self.y_dch = self.model.add_variables(
                lower=0,
                upper=self.P_batt,
                coords=[self.times, self.bidding_zones],
                dims=["T","Z"],
                name="y_dch")

    def build_constrs(self) -> None:
        # 1 Power balance
        self.pow_bal = self.model.add_constraints(
            self.p_DA + self.y_charge <= self.y_dch + self.P_fore,
            name="pow_bal")

        # 2 Battery balance
        self.batt_bal = self.model.add_constraints(
            self.SOC.diff(n=1, dim="T")
            # == y_charge * eta - (y_dch_DA + y_dch) / eta,
            == self.y_charge * self.eta - self.y_dch / self.eta,
            name="batt_bal")

        # 3 Developer is only remunerated at the PPA price for the power delivered under the BL contract.
        #   Powered delivered "directly" is also generated by D in hour h. Power can be delivered "indirectly" by using the BESS.
        self.vmin1 = self.model.add_constraints(
            self.v_min <= self.V_BL, #p_PPA + y_dch,
            name="v_min__V_BL")

        # 4 The power offered in the DA contract can either be under the PPA or purely in the DA market.
        #   The second option allows the developer to earn the higher DA price in some hours.
        self.vmin2 = self.model.add_constraints(
            self.v_min <= self.p_DA,
            name="v_min__p_DA"
        )

        if self.enforce_no_charge_in_deficit:
            # 5 The developer is NOT free to charge the battery is she is not compliant in a given hour.
            self.PPA_first = self.model.add_constraints(
                self.v_min >= (self.V_BL * (1-self.power_deficit) + self.P_fore * self.power_deficit),
                name="PPA_first"
            )

        # if self.enforce_no_DA_in_deficit:
        #     # 6 Can only offer in DA market if there is a power surplus
        #     self.no_DA_def = self.model.add_constraints(
        #         # p_DA + y_dch_DA <= np.maximum(P_fore - V_BL, 0),
        #         self.p_DA - self.v_min <= np.maximum(self.P_fore - self.V_BL, 0),
        #         name="Power forecast, DA")

        if self.annual_compliance_percentage:
            # 7 Ensure that the developer is only free to participate in the DA market if there is an abundance of power
            self.model.add_constraints(
                self.v_min.sum("T")
                >=
                min(sum(self.P_fore), self.T*self.V_BL) * self.compliance_perc,
                name="Restrict DA participation"
            )

    def build_obj(self) -> None:
        # decoupled settlement. BL & PaP:
        if self.financial_settlement or (self.PPA_profile == 'PaP' and not self.PaP_coupled):
            self.model.add_objective(
                        # No PPA term because the obj is independent of it since there's no coupling
                        (self.p_DA * self.lambda_DA).sum(dim="T"),
                        sense='max'
            )
        # PaP coupled:
        elif self.PaP_coupled:  # and consequently also self.PPA_profile == 'PaP'
            self.model.add_objective(
                (self.p_DA * (self.lambda_DA + self.lambda_PPA)).sum(dim="T"),
                sense='max'
            )
        # BL coupled:
        else:
            self.model.add_objective(
                (self.p_DA * self.lambda_DA
                + self.v_min * (self.lambda_max - self.lambda_DA + self.xi)
                ).sum(dim="T"),
                sense='max'
            )

    def build_model(self) -> None:
        self.model = linopy.Model()
        self.build_vars()

        # The only constraint needed in a PaP is the power forecast as
        #   an upper limit to the power offered in the DA market.
        if self.PPA_profile == 'BL':
            self.build_constrs()

        self.build_obj()
        print("Done building model")

    # Run model
    def run_model(self) -> None:
        self.model.solve(solver_name='gurobi')

    # Visualize BL results
    def plot_BL_power_allocation(self, axs, axs_idx=0, add_to_title=''):
        '''
        plot power allocation
        '''
        times_ext = self.times_ext

        presentation = True

        # extend data arrays by repeating last value
        P_fore_ext = np.append(self.P_fore, self.P_fore[-1])
        v_min_ext = np.append(self.v_min[:, 0].solution, self.v_min.sol.sel(T=self.T-1))
        y_dch_ext = np.append(self.y_dch[:, 0].solution, self.y_dch.sol.sel(T=self.T-1))
        p_DA_ext = np.append(self.p_DA[:, 0].solution, self.p_DA.sol.sel(T=self.T-1))
        y_charge_ext = np.append(self.y_charge[:, 0].solution, self.y_charge.sol.sel(T=self.T-1))
        # Following is used to separate the power delivered in the PPA from the battery in hour t from the power delivered AND generated in hour t.
        batt_component_of_PPA_delivery = [-(self.p_DA.sol.sel(T=t) - self.y_dch.sol.sel(T=t) - min(self.V_BL, self.v_min.sol.sel(T=t))).item() if (self.y_dch.sol.sel(T=t) > 0 and self.power_deficit[t] == 1) else 0 for t in range(self.T)]
        batt_component_of_PPA_delivery_ext = np.append(batt_component_of_PPA_delivery, batt_component_of_PPA_delivery[-1])
        v_min_excl_batt_ext = v_min_ext-batt_component_of_PPA_delivery_ext

        # plot as step (stair) plot
        axs[axs_idx].step(times_ext, P_fore_ext, where='post', label='P_fore', linestyle='--')
        # axs[0].fill_between(times_ext, 0, v_min_ext, step='post', alpha=0.5, label='PPA')
        # Instead of the code right above, don't plot the batt. dch. part of the PPA delivery
        if not self.financial_settlement:
            axs[axs_idx].fill_between(times_ext, 0, v_min_excl_batt_ext, step='post', alpha=.8, label='PPA')
            axs[axs_idx].fill_between(times_ext, v_min_excl_batt_ext, p_DA_ext - y_dch_ext, step='post', alpha=.8,label='DA')
        else:
            # skip the PPA color
            axs[axs_idx].fill_between(times_ext, 0, 0, step='post', alpha=.8, label='PPA')
            # actual plot
            axs[axs_idx].fill_between(times_ext, 0, p_DA_ext - y_dch_ext, step='post', alpha=.8,label='DA')
        axs[axs_idx].fill_between(times_ext, p_DA_ext - y_dch_ext,
                        p_DA_ext, step='post', alpha=.8, label='Batt. discharge')
        axs[axs_idx].fill_between(times_ext, p_DA_ext,
                        p_DA_ext + y_charge_ext, 
                        step='post', alpha=.8, label='Batt. charge')
        # baseload line
        axs[axs_idx].axhline(self.V_BL, linestyle='-', c='k', label='V_BL')

        # shade hours of negative DA prices
        mask_ext, times_ext_new = make_negative_DA_price_mask(self.times, self.lambda_DA)
        axs[axs_idx].fill_between(times_ext_new, 0, np.max(self.p_DA.sol + self.y_charge.sol), where=mask_ext, hatch="/", facecolor=self.palette[-1], edgecolor=self.palette[-1], alpha=.1, label=r"$\lambda^{DA} < 0$", zorder=0)
        
        # title and xlabel
        # PPAcov2 = 100 * self.v_min.sol.sum(dim='T').item() / (self.V_BL*self.T) # doesn't work for the "current" settlement mechanism
        PPA_coverage = 100 * np.minimum(self.V_BL, self.p_DA.sol).sum(dim="T").item() / (self.V_BL*self.T)
        obj = self.model.objective.value
        DA_revs = (self.p_DA.sol * self.lambda_DA).sum()
        if not self.financial_settlement:
            PPA_revs = (self.v_min.sol * (self.lambda_max - self.lambda_DA)).sum().item()
            PPA_revs_corr = PPA_revs - (self.V_BL * (self.lambda_max - self.lambda_PPA)).sum()
            obj_corr = DA_revs + PPA_revs_corr
        else:
            PPA_revs = 0  # by definition. However, v_min is positive in "current_compliance" because it's used to count the compliant MWh...
            PPA_revs_corr = PPA_revs + (self.V_BL * (self.lambda_PPA - self.lambda_DA)).sum()
            obj_corr = DA_revs + PPA_revs_corr
        if presentation:
            axs[axs_idx].set_title(f'{add_to_title}\n\t PPA coverage: {PPA_coverage:.1f}% \n\t Developer revenues: €{obj_corr:.1f} - of which DA/PPA: €{DA_revs:.1f}/€{PPA_revs_corr:.1f})\n[MW]', loc='left')
        else:
            axs[axs_idx].set_title(f'{add_to_title}Power allocation (PPA BL-coverage: {PPA_coverage:.1f}%)\nobj: €{obj:.1f} - of which DA/PPA: €{DA_revs:.1f}/€{PPA_revs:.1f})\nobj: €{obj_corr:.1f} - of which DA/PPA: €{DA_revs:.1f}/€{PPA_revs_corr:.1f})\n[MW]', loc='left')
        axs[axs_idx].set_xlabel('Time')

        return axs

    def plot_prices_max(self, axs, axs_idx=1):
        times_ext = self.times_ext

        # plot prices
        lambda_DA_ext = np.append(self.lambda_DA, self.lambda_DA[-1])
        lambda_max_ext = np.append(self.lambda_max, self.lambda_max[-1])
        axs[axs_idx].axhline(self.lambda_PPA, linestyle='-', c='k', label=r'$\lambda^{PPA}$')
        axs[axs_idx].step(times_ext, lambda_DA_ext, where='post', label=r'$\lambda^{DA}$', ls='--')
        axs[axs_idx].step(times_ext, lambda_max_ext, where='post', label=r'$\lambda^{max}$', ls='--')
        # title and xlabel
        axs[axs_idx].set_title('Market & PPA Prices\n[€/MWh]', loc='left')
        axs[axs_idx].set_xlabel('Time')

        return axs
    
    def plot_battery_operations(self, axs, axs_idx=2):
        times_ext = self.times_ext

        # plot battery operations
        y_dch_ext_pre = np.append(self.y_dch[:, 0].solution, self.y_dch.sol.sel(T=0))
        y_charge_ext_pre = np.append(self.y_charge[:, 0].solution, self.y_charge.sol.sel(T=0))
        axs[axs_idx].plot(self.times, self.SOC[:, 0].solution, label='SOC', )
        axs[axs_idx].step(times_ext, y_dch_ext_pre, where='pre', label='y_dch', linestyle='-')
        axs[axs_idx].step(times_ext, y_charge_ext_pre, where='pre', label='y_charge', linestyle='-')
        # title and xlabel
        axs[axs_idx].set_title(f'State-of-Charge (SOC) - maximum is {self.E_batt:.1f} MWh\n[MWh]', loc='left')
        axs[axs_idx].set_xlabel('Time')

        return axs

    def plot_BL_results(self) -> None:
        '''
        This method plots the:
        1) Power allocation
        2) Prices
        3) Battery operations
        '''
        # load_plot_configs()

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

        axs = unify_palette_cyclers(axs)

        axs = self.plot_power_allocation(axs)
        axs = self.plot_prices_max(axs)
        axs = self.plot_battery_operations(axs)
        print(axs, type(axs))
        axs = prettify_subplots(axs)

        plt.show()

    # Visualize PaP results
    def plot_PaP_power(self, axs, axs_idx, add_to_title=''):
        '''
        plot power production under PaP
        '''
        times_ext = self.times_ext

        presentation = True

        # DA offers in negative price hours
        # extend data arrays by repeating last value
        P_fore_ext = np.append(self.P_fore, self.P_fore[-1])
        p_DA_ext = np.append(self.p_DA[:, 0].solution, self.p_DA.sol.sel(T=self.T-1))
        
        lambda_DA_ext = np.append(self.lambda_DA, self.lambda_DA[-1])
        p_DA_negp_ext = p_DA_ext * (lambda_DA_ext < 0)
        
        # plot as step (stair) plot
        axs[axs_idx].step(times_ext, P_fore_ext, where='post', label='P_fore', linestyle='--')
        
        # # skip the PPA color
        # axs[axs_idx].fill_between(times_ext, 0, 0, step='post', alpha=.8)#, label='PPA')
        # actual plot
        axs[axs_idx].fill_between(times_ext, 0, p_DA_negp_ext, step='post', alpha=.8,label=r'DA $\left(\lambda^{DA} < 0\right)$')
        axs[axs_idx].fill_between(times_ext, p_DA_negp_ext, p_DA_ext, step='post', alpha=.8,label='DA')

        # shade hours of negative DA prices
        mask_ext, times_ext_new = make_negative_DA_price_mask(self.times, self.lambda_DA)
        axs[axs_idx].fill_between(times_ext_new, 0, np.max(self.p_DA.sol), where=mask_ext, hatch="/", facecolor=self.palette[-1], edgecolor=self.palette[-1], alpha=.1, label=r"$\lambda^{DA} < 0$", zorder=0)
        
        # title and xlabel
        obj = self.model.objective.value
        DA_revs = (self.p_DA.sol * self.lambda_DA).sum()
        if self.PaP_coupled:
            PPA_revs = (self.p_DA.sol * self.lambda_PPA).sum().item()
            PPA_revs_corr = PPA_revs - (self.P_fore * self.lambda_DA.ravel()).sum()
            obj_corr = DA_revs + PPA_revs_corr
        else:
            PPA_revs = 0  # by definition. However, v_min is positive in "current_compliance" because it's used to count the compliant MWh...
            PPA_revs_corr = PPA_revs + ( self.P_fore * (self.lambda_PPA - self.lambda_DA.ravel()) ).sum()
            obj_corr = DA_revs + PPA_revs_corr
        
        if presentation:
            axs[axs_idx].set_title(f'{add_to_title}\n\t Developer revenues: €{obj_corr:.1f} - of which DA/PPA: €{DA_revs:.1f}/€{PPA_revs_corr:.1f}\n[MW]', loc='left')
        else:
            axs[axs_idx].set_title(f'{add_to_title}Power production\nobj: €{obj:.1f} - of which DA/PPA: €{DA_revs:.1f}/€{PPA_revs:.1f}\nobj: €{obj_corr:.1f} - of which DA/PPA: €{DA_revs:.1f}/€{PPA_revs_corr:.1f} (corrected)\n[MW]', loc='left')
        axs[axs_idx].set_xlabel('Time')

        return axs

    def plot_prices_sum(self, axs, axs_idx=1):
        times_ext = self.times_ext

        # plot prices
        lambda_DA_ext = np.append(self.lambda_DA, self.lambda_DA[-1])
        lambda_sum_ext = lambda_DA_ext + self.lambda_PPA
        axs[axs_idx].axhline(self.lambda_PPA, linestyle='-', c='k', label=r'$\lambda^{PPA}$')
        axs[axs_idx].step(times_ext, lambda_DA_ext, where='post', label=r'$\lambda^{DA}$', ls='--')
        axs[axs_idx].step(times_ext, lambda_sum_ext, where='post', label=r'$\lambda^{DA}+\lambda^{PPA}$', ls='--')
        # title and xlabel
        axs[axs_idx].set_title('Market & PPA Prices\n[€/MWh]', loc='left')
        axs[axs_idx].set_xlabel('Time')

        return axs

#%% 
# d = PPA_modeling(PPA_profile='BL',
#                  PaP_coupled=False,
#                  financial_settlement=True,
#                  enforce_no_charge_in_deficit=False,
#                  annual_compliance_percentage=True,
#                  compliance_perc=0.9,  # caution: too high a compliance rate may be infeasible
#                  add_batt=True,
#     )

# d.run_model()

# d.plot_BL_results()

#%% Solutions:
BL_settlements = {}
BL_settlements["current"] = [True, False, False, 0]
BL_settlements["coupled"] = [False, False, False, 0]
BL_settlements["coupled_restricted"] = [False, True, False, 0]
BL_settlements["current_compliance"] = [True, False, True, .612] #.827]
# BL_settlements["current_compliance2"] = [True, False, True, .827]
 
PaP_settlements = {}
PaP_settlements["current"] = [False]
PaP_settlements["coupled"] = [True]

profile_type = 'BL'
settlements =   BL_settlements

models_dict = {}
for i, (k, v) in enumerate(settlements.items()):
    if profile_type == 'BL':
        m = PPA_modeling(
                PPA_profile='BL',
                financial_settlement=v[0],
                enforce_no_charge_in_deficit=v[1],
                annual_compliance_percentage=v[2],
                compliance_perc=v[3],
            )

    elif profile_type == 'PaP':
        m = PPA_modeling(
            PPA_profile='PaP',
            PaP_coupled=v[0]
        )

    m.run_model()
    
    # Save the model for easier handling and debugging
    models_dict[k] = m

#%% Visualize results for different settlements mechanisms
# load_plot_configs()
fig, axs = plt.subplots(len(settlements)+1, 1, figsize=(10, 8+2*len(settlements)), constrained_layout=True)
axs = unify_palette_cyclers(axs)
for i, (k, v) in enumerate(settlements.items()):
    # Plot production curves
    if profile_type == 'BL':
        axs = models_dict[k].plot_BL_power_allocation(axs, axs_idx=i, add_to_title=fr"{profile_type} – {k.upper()}")
    elif profile_type == 'PaP':
        axs = models_dict[k].plot_PaP_power(axs, axs_idx=i, add_to_title=fr"{profile_type} – {k.upper()}")

# Plot price curves
if profile_type == 'BL':
    axs = models_dict["current"].plot_prices_max(axs, axs_idx=-1)
elif profile_type == 'PaP':
    axs = models_dict["current"].plot_prices_sum(axs, axs_idx=-1)

axs = prettify_subplots(axs)

plt.show()
# %%
d = models_dict["current"]

# %%
