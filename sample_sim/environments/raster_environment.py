import logging
import math
import subprocess

import numpy as np
from joblib import Memory

from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.sensors.camera_sensor import FixedHeightCamera

try:
    import rasterio
    from rasterio import MemoryFile
except ModuleNotFoundError:
    pass

from sample_sim.action.actions import ActionModel
from sample_sim.environments.base import BaseEnvironment

from sample_sim.environments.workspace import RasterWorkspace, RasterWorkspace3d
import random


class DroneFixedHeightRasterEnvironment(BaseEnvironment):
    def __init__(self, workspace, raster_image_fname, channel):
        super().__init__(workspace)
        self.raster_image_fname = raster_image_fname
        self.channel = channel
        self.open_raster = None
        #ys = self.get_all_values()
        #self.ymin = np.min(ys)
        #self.ymax = np.max(ys)

    def get_parameters(self):
        return {"image_fname": self.raster_image_fname,
                "channel": self.channel}

    def __hash__(self):
        return hash((self.raster_image_fname, self.channel))

    def open_dataset_in_memory(self):
        with open(self.raster_image_fname, 'rb') as f:
            self.open_filehandle = MemoryFile(f)
            self.open_raster = self.open_filehandle.open()

    # The following are to allow pickling because you can't pickle an open raster
    def __getstate__(self):
        if self.open_raster is not None:
            self.open_raster.close()
            self.open_raster = None
        if self.open_filehandle is not None:
            self.open_filehandle.close()
            self.open_filehandle = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def get_all_values(self):
        if self.open_raster is None:
            self.open_dataset_in_memory()

        values =  self.open_raster.read(self.channel).ravel()
        mask = np.isin(values,[0,255])
        return values[~mask]


    def sample(self, xs):
        if self.open_raster is None:
            self.open_dataset_in_memory()
        Y = np.array(list(self.open_raster.sample(xs[:,:2], self.channel))).ravel() / 255.0
        return Y

    def sample_unnormalized(self,xs):
        if self.open_raster is None:
            self.open_dataset_in_memory()
        Y = np.array(list(self.open_raster.sample(xs[:,:2], self.channel))).ravel()
        return Y



    def action_model(self):
        return ActionModel.XY

    def dimensionality(self):
        return 2

    def get_starting_location(self):
        r = random.Random(0)

        starting_point = (int(np.mean(self.workspace.get_bounds(), 1)[0]),
                int(np.mean(self.workspace.get_bounds(), 1)[1]),
                0.0)

        while not self.is_inside(starting_point):
            x = r.uniform(self.workspace.get_bounds()[0,0],self.workspace.get_bounds()[0,1])
            y = r.uniform(self.workspace.get_bounds()[1,0],self.workspace.get_bounds()[1,1])
            z = 0.0
            starting_point = (int(x),int(y),z)
        return starting_point
    def __repr__(self):
        return f"Drone Fixed Height Raster Environment {self.raster_image_fname}"

class DroneFixedHeightRasterEnvironment3d(DroneFixedHeightRasterEnvironment):
    def __init__(self, workspace:RasterWorkspace, raster_image_fname, channel):
        real_workspace = RasterWorkspace3d(workspace)
        super().__init__(real_workspace,raster_image_fname,channel)

    def dimensionality(self):
        return 3

    def action_model(self):
        return ActionModel.XYZ

    def __repr__(self):
        return f"Drone Fixed Height Raster Environment 3d {self.raster_image_fname}"

class DroneFixedHeightRasterEnvironment3dGP(DroneFixedHeightRasterEnvironment3d):

    def __init__(self, workspace:RasterWorkspace, raster_image_fname, channel):
        super().__init__(workspace,raster_image_fname,channel)
        if self.open_raster is None:
            self.open_dataset_in_memory()
        band1 = self.open_raster.read(channel)
        print('Band1 has shape', band1.shape)
        # height = band1.shape[0]
        # width = band1.shape[1]
        # cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        # xs, ys = rasterio.transform.xy(self.open_raster.transform, rows, cols)
        # lons = np.array(xs)
        # lats = np.array(ys)
        # X = np.concatenate((lons.ravel().reshape(-1,1),lats.ravel().reshape(-1,1)),axis=1)

        X = []
        Y = []
        band_arr = self.open_raster.read(channel)
        for x in range(0,band_arr.shape[0],15):
            for y in range(0,band_arr.shape[1],15):
                value = band_arr[x,y]
                if value in [0,255]:
                    continue
                else:
                    lat, lon = rasterio.transform.xy(self.open_raster.transform, x, y)
                    X.append([lat,lon])
                    Y.append(value)
        X = np.array(X)
        Y = np.array(Y) / 254.0

        #Y = np.array(list(self.open_raster.sample(X, self.channel))).ravel()
        print(X.shape)


        self.model = TorchExactGPBackedDataModel(X, Y, "default", force_cpu=True)
        self.model.update(X, Y)
        # Number of iterations to train for
        self.model.fit(1 * 10 ** 5)
        self.model.model.model.eval()

    def sample(self, xs):
        return self.model.query_many(xs[:,:2],return_std=False)

memory = Memory("cache",verbose=0)


@memory.cache
def load_from_egret_data(picture_fname, camera, channel=7,use_3d = True, use_gp = True):
    p = subprocess.Popen(
        ["gdalinfo", picture_fname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    try:

        out, err = p.communicate(timeout=60 * 5)
        out = out.decode()
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        print("gladinfo timed out.")
        print("STDOUT:\n{}".format(out))
        print("STDERR:\n{}".format(err))
        raise RuntimeError

    # TODO: use better pattern matching
    ur = list(map(float, out[out.find("Upper Right") + 15:
                             out.find("Upper Right") + 38].split(",")))
    ll = list(map(float, out[out.find("Lower Left") + 15:
                             out.find("Lower Left") + 38].split(",")))
    workspace = RasterWorkspace(picture_fname, camera, min(ll[0], ur[0]), max(ll[0], ur[0]),
                                min(ll[1], ur[1]), max(ll[1], ur[1]), channel=channel)

    if not use_3d:
        if use_gp:
            raise Exception("Not Implemented")
        else:
            return DroneFixedHeightRasterEnvironment(workspace, picture_fname, channel)
    else:
        if use_gp :
           return DroneFixedHeightRasterEnvironment3dGP(workspace,picture_fname,channel)
        else:
            return DroneFixedHeightRasterEnvironment3d(workspace,picture_fname,channel)


if __name__ == "__main__":
    non_experiment_logger = logging.getLogger("default")
    non_experiment_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    non_experiment_logger.addHandler(handler)

    sensor_x = 3  # 6.3
    sensor_y = 3  # 4.7
    focal_length = 3  # 4.49
    fov_x = math.degrees(2 * math.atan2(sensor_x, 2 * focal_length)) / 2
    fov_y = math.degrees(2 * math.atan2(sensor_y, 2 * focal_length)) / 2

    sensor = FixedHeightCamera(fov_x, fov_y, 12, 12,
                               14)
    env = load_from_egret_data("data/orthophotos/2020-08-06_16-56-15-cropped.tif",sensor,use_gp=False)
    grid = env.meshgrid((30,30,1))
    values = env.sample(grid)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(values)
    plt.show()
    print(f"N({np.mean(values)},{np.std(values)}")