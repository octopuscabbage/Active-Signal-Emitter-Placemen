import logging

import utm
import numpy as np
import scipy
from joblib import Memory, hashing

from sample_sim.action.actions import ActionModel
from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.environments.base import BaseEnvironment
import pandas as pd

from sample_sim.environments.workspace import RectangularPrismWorkspace


class EcomapperCSVEnvironment(BaseEnvironment):
    def __init__(self,filename, workspace, model: TorchExactGPBackedDataModel):
        super().__init__(workspace)
        self.filename = filename
        self.model = model

    def sample(self, xs):
        return self.model.query_many(xs,return_std=False)

    def action_model(self):
        return ActionModel.XYZ

    def dimensionality(self):
        return 3

    def get_starting_location(self):
       return  (int(np.mean(self.workspace.get_bounds(), 1)[0]),
                     int(np.mean(self.workspace.get_bounds(), 1)[1]),
                     self.workspace.zmin)

    def __hash__(self):
        return hashing.hash(self.filename)

    def __repr__(self):
        return f"Ecomapper CSV: {self.filename}"

    def get_parameters(self):
        return {"file": self.filename}

memory = Memory("cache",verbose=0)


@memory.cache
def load_from_ecomapper_data(csv_filename, first_waypoint_id, last_waypoint_id, reading_type="YSI-Chl ug/L"):
    data = pd.read_csv(csv_filename, sep=";")
    lat = data["Latitude"].values
    long = data["Longitude"].values
    waypoints = data["Current Step"].values
    assert len(lat) == len(long)

    first_idx = 0
    if first_waypoint_id is not None:
        first_idx = np.min(np.where(waypoints == first_waypoint_id)[0])

    last_idx = lat.size
    if last_waypoint_id is not None:
        last_idx = np.min(np.where(waypoints == last_waypoint_id)[0])

    lat = lat[first_idx:last_idx]
    long = long[first_idx:last_idx]

    m_xs = []
    m_ys = []
    for cur_lat, cur_lon in zip(lat, long):
        m_x, m_y, zone_number, zone_letter = utm.from_latlon(cur_lat, cur_lon)
        m_xs.append(m_x)
        m_ys.append(m_y)
    m_x = np.array(m_xs)
    m_y = np.array(m_ys)
    m_x = m_x - np.min(m_x)
    m_y = m_y - np.min(m_y)

    height = data["DTB Height (m)"].values
    height = height[first_idx:last_idx]
    values = data[reading_type].values
    values = values[first_idx:last_idx]

    assert len(values) == len(height) == len(m_x) == len(m_y)
    X = np.stack((m_x, m_y, height), axis=-1)

    cdists = scipy.spatial.distance.cdist(X, X)
    dists = np.sum(cdists, axis=0)

    mean_dist = np.mean(dists)
    std_dists = np.std(dists)

    Y = values
    # Y = np.log(np.clip(Y,1e-12,None))
    X_out = []
    Y_out = []
    removed = 0
    kept = 0
    for i, dist in enumerate(dists):
        if dist < (mean_dist + .5 * std_dists):
            X_out.append(X[i, :])
            Y_out.append(Y[i])
            kept += 1
        else:
            removed += 1
    print(f"Remove {removed} Kept {kept}")
    X = np.array(X_out)
    Y = np.array(Y_out)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    workspace = RectangularPrismWorkspace(np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1]),
                                          np.min(X[:, 2]), np.max(X[:, 2]))

    model = TorchExactGPBackedDataModel(X, Y, "default", force_cpu=True)
    model.update(X, Y)
    # Number of iterations to train for
    model.fit(1 * 10 ** 5)

    return EcomapperCSVEnvironment(csv_filename,workspace, model)

if __name__ == "__main__":
    non_experiment_logger = logging.getLogger("default")
    non_experiment_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    non_experiment_logger.addHandler(handler)
    env = load_from_ecomapper_data("data/ecomapper/20170518_163636.csv",15,50)
    env.model.model.model.eval()
    grid = env.meshgrid(30)
    values = env.sample(grid)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(values)
    plt.show()
    print(f"N({np.mean(values)},{np.std(values)}")

