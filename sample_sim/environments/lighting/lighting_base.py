import abc 
import numpy as np


class AnalyticalLightingModel(abc.ABC):
    def compute_light(self, xs, light_location,light_brightnesses):
        if not isinstance(light_brightnesses, np.ndarray):
            light_brightnesses = np.ones(light_location.shape[0]) * light_brightnesses
        return self.__compute_lights__impl__(xs,light_location,light_brightnesses)


    @abc.abstractmethod
    def __compute_lights__impl__(self, xs, light_locations, light_brightnesses):
        pass

