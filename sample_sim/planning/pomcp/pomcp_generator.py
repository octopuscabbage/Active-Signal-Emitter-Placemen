import numpy as np

from sample_sim.data_model.gp_wrapper import TorchExactGp
from sample_sim.planning.pomcp.pomcp_utilities import PomcpExtraData
from sample_sim.planning.reward_functions import calculate_reward
import logging


def Generator(s, a, s_history, extra_data: PomcpExtraData, remaining_budget):
    sprime_idx = extra_data.grid.get_transition_matrix()[s][a]
    if sprime_idx == -1 or remaining_budget == 0 or (len(s_history) > 0 and s_history[-1][1] is None):
        s_history.append((None,None))
        return s, s, 0

    o = sprime_idx
    Sreal_ndarrays = extra_data.grid.get_Sreal_ndarrays()

    model = extra_data.data_model
    if s_history != []:
        s_ndarray_history = np.concatenate(list(map(lambda s: s[0], s_history)),axis=0)
        s_ndarray_history_ys = np.concatenate(list(map(lambda s: s[1], s_history)),axis=0)

        X = np.vstack((extra_data.data_model.Xs, s_ndarray_history.reshape(-1,extra_data.environment.dimensionality())))
        Y = np.concatenate((extra_data.data_model.Ys, s_ndarray_history_ys.reshape(-1)))
        assert isinstance(gp,TorchExactGp)
        model.update_prior(X,Y)
    else:
        X = extra_data.data_model.Xs
        Y = extra_data.data_model.Ys

    s_loc = Sreal_ndarrays[s]
    sprime_loc = Sreal_ndarrays[sprime_idx]

    sensor_points = extra_data.grid.get_samples_traveling_from(s_loc,sprime_loc,extra_data.sensor)

    mean,stdv = model.predict(sensor_points,return_std=True)


    rw = np.sum(abs(
        calculate_reward(mean, stdv, extra_data.objective_function,
                         extra_data.quantiles, extra_data.grid, extra_data.data_model, sensor_points, X, Y)))
    rw += extra_data.objective_c * (np.sum(stdv) / sensor_points.size)
    rw = np.sum(rw)
    s_history.append((sensor_points,mean))
    return sprime_idx, o, rw

