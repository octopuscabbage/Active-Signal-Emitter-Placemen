import os
import os
import random
import sqlite3
import traceback
import logging
from math import inf

import numpy as np
import ray
import scipy
import utm
from joblib import Memory, hashing

#from learning.caching import RotatingCache
from sample_sim.action.actions import ActionModel
from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.environments.base import BaseEnvironment
from sample_sim.environments.utilities.rotator import RotateSpatialDataToAxesAlignedBox
from sample_sim.environments.workspace import RectangularPrismWorkspace
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR



class NOAADBCommunicator():
    def __init__(self, location):
        self.location = location


    def __single_query_to_list(self, res):
        return list(map(lambda x: x[0], res))

    def all_robot_keys(self):
        con = sqlite3.connect(self.location)
        cursor = con.cursor()
        return self.__single_query_to_list(cursor.execute("SELECT robot_key FROM robots"))

    def get_single_trajectory(self, robot_key, type_of_data):
        con = sqlite3.connect(self.location)
        cursor = con.cursor()
        query = "SELECT latitude,longitude,depth,(SELECT value FROM sensor_data WHERE sensor_data.type_of_data == ? and sensor_data.position_key == position.position_key) FROM position WHERE robot_key == ?"
        dataset = cursor.execute(query, (type_of_data, robot_key))
        dataset = list(map(list, dataset))
        np_locations = np.array(dataset, dtype=np.float64)
        return np_locations

class NOAADBEnvironment(BaseEnvironment):
    def __init__(self, robot_key, sensor_type, workspace, model: SVR, X,Y, index):
        super().__init__(workspace)
        self.robot_key = robot_key
        self.sensor_type = sensor_type
        self.model = model
        self.X = X
        self.Y = Y
        self.index = index

    def sample(self, xs):
        return self.model.predict(xs)
        #return self.model.query_many(xs,return_std=False)


    def action_model(self):
        return ActionModel.XYZ

    def dimensionality(self):
        return 3

    def get_starting_location(self):
        return (int(np.mean(self.workspace.get_bounds(), 1)[0]),
                int(np.mean(self.workspace.get_bounds(), 1)[1]),
                self.workspace.zmin)

    def __hash__(self):
        return hashing.hash((self.robot_key, self.sensor_type))

    def __repr__(self):
        return f"NOAA DB : Key {self.robot_key} Sensor {self.sensor_type}"

    def get_parameters(self):
        return {"key": self.robot_key,
                "sensor_type": self.sensor_type,
                "index": self.index}


class NOAADBRandomLoader():
    def __init__(self, noaa_db_location, exclusion_data_location="noaa-db/noaa-db-checks", only_use_sensor_types=None, logger_name=None, training_cache_ray: RotatingCache = None, sample_new_training_probability=0.2, seed=None):
        self.logger_name = logger_name
        self.sensor_exclusion_dictionary = dict()
        if only_use_sensor_types is None:
            for filename in os.listdir(exclusion_data_location):
                self.load_sensor_exclusion_from_file(os.path.join(exclusion_data_location, filename))
        else:
            for sensor_type in only_use_sensor_types:
                self.load_sensor_exclusion(sensor_type, exclusion_data_location)

        self.db_helper = NOAADBCommunicator(noaa_db_location)
        self.robot_keys = self.db_helper.all_robot_keys()
        self.all_sensor_types = list(self.sensor_exclusion_dictionary.keys())

        training_split = int(.8 * len(self.robot_keys))
        evaluation_split = training_split + int(.1 * len(self.robot_keys))
        logging.getLogger(self.logger_name).info(f"Training Idx {training_split} evaluation {evaluation_split}")


        if seed is not None:
            self.random = random.Random(seed)
        else:
            self.random = random.Random()

        self.training_keys = self.robot_keys[:training_split]
        self.evaluation_keys = self.robot_keys[training_split:evaluation_split]
        self.test_keys = self.robot_keys[evaluation_split:]

        self.training_cache_ray = training_cache_ray
        self.sample_new_training_probability =  sample_new_training_probability


    def load_sensor_exclusion(self, name, checks_location):
        with open(os.path.join(checks_location, f"{name}_bad.txt")) as f:
            self.sensor_exclusion_dictionary[name] = set(map(int, f.readlines()))

    def load_sensor_exclusion_from_file(self, filename):
        with open(filename) as f:
            name = filename.split("/")[-1].split("_")[0]
            self.sensor_exclusion_dictionary[name] = set(map(int, f.readlines()))

    def load_training_dataset(self):
        if self.training_cache_ray is not None:
            logging.getLogger(self.logger_name).info("Cache " + ray.get(self.training_cache_ray.to_string.remote()))
            if random.random() < self.sample_new_training_probability or ray.get(self.training_cache_ray.cached_filled.remote()):
                new_dataset = self.load_random_dataset(self.training_keys,allow_retry=False) #Don't allow retry because i'll just retry from cache
                if new_dataset is None:
                    return self.load_training_dataset()
                else: 
                    ray.get(self.training_cache_ray.put_randomly.remote(new_dataset))
                    logging.getLogger(self.logger_name).info("Cache After Update " + ray.get(self.training_cache_ray.to_string.remote()))
                    return new_dataset
            else:
                logging.getLogger(self.logger_name).info("Returning cached element")
                return ray.get(self.training_cache_ray.get_random_element.remote())
        else:
            return self.load_random_dataset(self.training_keys)

    def load_evaluation_dataset(self):
        return self.load_random_dataset(self.evaluation_keys)

    def load_test_dataset(self):
        return self.load_random_dataset(self.test_keys)

    def load_random_dataset(self,keys=None,allow_retry=True):
        found_good_dataset = False
        i = 0
        if keys is None:
            used_keys = self.robot_keys
        else:
            used_keys = keys
        while not found_good_dataset:
            logging.getLogger(self.logger_name).warning(f"Retrying {i}")
            i+=1
            robot_key = self.random.choice(used_keys)
            sensor_type = self.random.choice(self.all_sensor_types)
            while robot_key in self.sensor_exclusion_dictionary[sensor_type]:
                robot_key = self.random.choice(used_keys)
                sensor_type = self.random.choice(self.all_sensor_types)
            try:
                dataset = load_single_dataset(robot_key, sensor_type, db_communicator=self.db_helper,logger_name=self.logger_name,random_obj=self.random)
                found_good_dataset = True
            except Exception as e:
                traceback.print_exc()
                if not allow_retry:
                    return None
        return dataset


def load_single_dataset(robot_key, type_of_data, num_points=5000, db_communicator=None, db_location=None,logger_name=None, random_obj=None):
    if db_communicator is None:
        db_communicator = NOAADBCommunicator(db_location)
    dataset = db_communicator.get_single_trajectory(robot_key, type_of_data)
    dataset = dataset[~np.isnan(dataset).any(axis=1)]
    dataset = dataset[dataset[:,3] != 0.0]


    number_of_rows = dataset.shape[0]
    if number_of_rows > num_points:
        if random_obj is not None:
            start_idx = random_obj.randint(0, number_of_rows - num_points)
        else:
            start_idx = random.randint(0, number_of_rows - num_points)


        dataset = dataset[start_idx:start_idx+num_points, :]
    else:
        start_idx = 0

    lat = dataset[:, 0]
    long = dataset[:, 1]

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

    height = dataset[:, 2]
    values = dataset[:, 3]

    assert len(values) == len(height) == len(m_x) == len(m_y)
    X = np.stack((m_x, m_y, height), axis=-1)
    preprocessor = RotateSpatialDataToAxesAlignedBox()
    X[:, :2] = preprocessor.process_points(X[:, :2])

    Y = values
    origin = np.array([np.min(X[:,0]), np.min(X[:,1]), np.min(X[:,2])])
    close_rows = np.linalg.norm(X - origin, ord=1,axis=1) < 1000
    X = X[close_rows,:]
    Y = Y[close_rows]
    if Y.shape[0] <= 200:
        logging.getLogger(logger_name).error("Dataset is too sparse")
        raise Exception("Dataset is too sparse")

    X[:,0] = X[:,0] - np.min(X[:,0])
    X[:,1] = X[:,1] - np.min(X[:,1])

    #X = X[:1000,:]
    #Y = Y[:1000]
    
    logging.getLogger(logger_name).info(f"Remove {close_rows.shape} Kept {Y.shape}")



    workspace = RectangularPrismWorkspace(np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1]),
                                          np.min(X[:, 2]), np.max(X[:, 2]))

    # Linear rescale to 0 - 1
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    if np.isnan(Y).any():
        logging.getLogger(logger_name).error("Dataset contains NaN")
        raise Exception("Dataset contains nan")


    #model = TorchExactGPBackedDataModel(X, Y, logger_name, force_cpu=True)
    #model.update(X, Y)
    # Number of iterations to train for
    #model.fit(50)
    #model.model.eval_model()

    #model = KNeighborsRegressor(weights="distance",n_jobs=-1, )
    model = SVR(kernel="rbf")
    model.fit(X,Y)



    return NOAADBEnvironment(robot_key, type_of_data, workspace, model, X, Y, start_idx)



