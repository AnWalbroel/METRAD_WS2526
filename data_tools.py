import os
import pdb

import numpy as np
import xarray as xr


def encode_time(DS: xr.Dataset):
    DS['time'] = DS['time'].values.astype("datetime64[s]").astype(np.float64)
    DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
    DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    DS['time'].encoding['dtype'] = 'double'
    
    return DS
