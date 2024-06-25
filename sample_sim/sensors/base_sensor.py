import abc
import numpy as np
from joblib import hashing


class BaseSensor(abc.ABC):
    @abc.abstractmethod
    def get_sensed_locations_from_point(self,p):
        pass
    def get_sensed_locations_from_trajectory(self, trajectory):
        out = []
        for point in trajectory:
            out.append(self.get_sensed_locations_from_point(point))
        return np.stack(out,axis=1)

class PointSensor(BaseSensor):
    def get_sensed_locations_from_point(self,p):
        return np.array([p],dtype=np.float64)
    def get_sensed_locations_from_trajectory(self, trajectory):
        return trajectory
    def __hash__(self):
        return hashing.hash(0)
    def __repr__(self):
        return f"Point Sensor"

