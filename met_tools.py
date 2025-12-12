import constants

import numpy as np



def q_to_h2ovmr(q_data: np.ndarray):
    return constants.m_mol_air*q_data / (constants.mw_h2o*(1.0 - q_data))