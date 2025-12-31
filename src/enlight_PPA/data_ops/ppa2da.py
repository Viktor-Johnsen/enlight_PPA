from logging import Logger
from dataclasses import dataclass

@dataclass
class PaP2DA:
    '''
    Purely inputs.

    Pre-selects the data/results needed as inputs to the DA model
    in order to include PPAs in the model.

    To include a PaP all we need to pass are:
    - Bidding zone: Z
    - PPA price: S
    - PPA volume share: gamma
    - capacities of the techs in the Producer portfolio
        - off_wind_el_cap
        - on_wind_el_cap
        - solar_pv_el_cap
    '''
    # Bidding zone of producer and buyer
    z : str = "DK1"

    # Producer specs
    s : float = 5.0  # €/MWh -- PPA price
    gamma : float = 0.5  # p.u. of Producer total VRE capacity
    solar_pv_el_cap : float = 0
    on_wind_el_cap : float = 0
    off_wind_el_cap : float = 0

@dataclass
class BL2DA:
    '''
    Purely inputs.

    Pre-selects the data/results needed as inputs to the DA model
    in order to include PPAs in the model.

    To include a BL all we need to pass are:
    - Bidding zone: Z
    - PPA price: S
    - PPA volume: M
    - capacities of the techs in the Producer portfolio
        - off_wind_el_cap
        - on_wind_el_cap
        - solar_pv_el_cap
        - P_batt
        - E_batt
    '''
    # Bidding zone of producer and buyer
    z : str = "DK1"
    compl_rate : float = 0.0

    # Producer specs
    s : float = 5.0  # €/MWh -- PPA price
    m : float = 0.5  # MW -- BL volume
    solar_pv_el_cap : float = 1
    on_wind_el_cap : float = 1
    off_wind_el_cap : float = 1
    P_batt : float = 1
    E_batt : float = 1
