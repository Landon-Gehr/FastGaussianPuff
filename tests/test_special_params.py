import pytest
import numpy as np
import pandas as pd
from utils import params, grid_params, sensor_params
from FastGaussianPuff import GaussianPuff as GP

def test_wind_skip():

    gp = GP(**params, **grid_params)
    gp.simulate()
    ch4_nonzero = gp.ch4_obs.reshape((gp.n_out, *gp.grid_dims))

    ws = np.array([0.2]*61)
    params["wind_speeds"] = ws
    params["skip_low_wind"] = True
    params["low_wind_cutoff"] = 0.5
    gp = GP(**params, **grid_params)
    gp.simulate()
    ch4_zero = gp.ch4_obs.reshape((gp.n_out, *gp.grid_dims))

    assert np.all(ch4_zero == 0)
    assert not np.all(ch4_nonzero == 0)

def test_zero_wind():
    ws = np.array([0.5]*61)
    ws[20] = 0
    params["wind_speeds"] = ws
    params["skip_low_wind"] = False
    # params["low_wind_cutoff"] = 0.1
    with pytest.raises(ValueError, match="[FastGaussianPuff]*"):
        gp = GP(**params, **grid_params)