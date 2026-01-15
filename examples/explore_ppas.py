import numpy as np
import linopy
import matplotlib.pyplot as plt

from enlight_PPA.models.ppa_exploration import PPAModeling
from enlight_PPA.utils.nbs_utils import unify_palette_cyclers, prettify_subplots, remove_mult_suffix


if __name__ == "__main__":
    # example
    # d = PPAModeling(PPA_profile='PaF',
    #                  BL_enforce_no_charge_in_deficit=False,
    #                  BL_annual_compliance_percentage=False,
    #                  BL_compliance_perc=0.0,  # caution: too high a compliance rate may be infeasible
    #                  add_batt=False,
    # )

    # d.run_model()
    # d.calculate_net_revenues()
    # d.model.objective.value

    # Solutions:

    profile_types_PaX = ['no_PPA', 'PaF', 'PaP']
    # In the list below the first C-BL is vanilla and the 2nd will include a further constraint.
    profile_types_BL = ['BL', 'BL–COMPLIANCE', 'C-BL', 'C-BL–RESTRICTED_CHARGING', 'AC-BL', 'AC-BL–RESTRICTED_CHARGING']
    profile_types_BL = ['BL', 'AC-BL', 'BL–COMPLIANCE']

    include_price_curves = False

    settlements = profile_types_BL

    models_dict = {}
    for p_ in settlements:
        # Remove any numbering, so e.g. "C-BL_2" is simplified to "C-BL"
        p = remove_mult_suffix(string=p_, suffixes=['–RESTRICTED_CHARGING','–COMPLIANCE'])

        m = PPAModeling(
                PPA_profile=p,
                BL_enforce_no_charge_in_deficit=(True if p_.endswith('–RESTRICTED_CHARGING') else False),
                BL_annual_compliance_percentage=(True if p_.endswith('–COMPLIANCE') else False),
                BL_compliance_perc=0.612,
                add_batt=(True if p in ['BL', 'C-BL', 'AC-BL'] else False)
            )

        m.run_model()
        m.calculating_net_revenues()
        
        # Save the model for easier handling and debugging
        # Change e.g. "C-BL_2" to "C_BL–RESTRICTED" and leave C-BL unchanged.
        models_dict[p_] = m

    #%% Visualize results for different settlements mechanisms
    # load_plot_configs()
    fig, axs = plt.subplots(len(settlements)+(1 if include_price_curves else 0), 1, figsize=(10, 8+2*len(settlements)), constrained_layout=True)
    axs = unify_palette_cyclers(axs)
    for i, p in enumerate(settlements):
        # Plot production curves
        if models_dict[p].BL:
            axs = models_dict[p].plot_BL_power_allocation(axs, axs_idx=i, add_to_title=f"{p}")
        else:
            axs = models_dict[p].plot_PaP_power(axs, axs_idx=i, add_to_title=f"{p.replace("_"," ")}")

    # Plot price curves
    if include_price_curves:
        if models_dict[p].BL:
            axs = models_dict[settlements[0]].plot_prices_max(axs, axs_idx=-1)
        else:
            axs = models_dict[settlements[0]].plot_prices_sum(axs, axs_idx=-1)

    axs = prettify_subplots(axs)

    plt.show()
    # %%
    d = models_dict[settlements[0]]
