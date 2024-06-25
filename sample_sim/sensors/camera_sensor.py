import math

import numpy as np
from joblib import hashing

from sample_sim.general_utils import stacked_meshgrid
from sample_sim.sensors.base_sensor import BaseSensor


class FixedHeightCamera(BaseSensor):
    def __init__(self, fov_x_degrees, fov_y_degrees, pixels_x, pixels_y, altitude_meters):
        self.fov_x_degrees = fov_x_degrees
        self.fov_y_degrees = fov_y_degrees
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.altitude_meters = altitude_meters

    def __hash__(self):
        return hashing.hash((self.fov_y_degrees,self.fov_y_degrees,self.pixels_x,self.pixels_y,self.altitude_meters))

    def get_sensed_locations_from_point(self,p):
        '''
        Does a simplified geometric calculation to calculate all the points the drone can see from a position
        :param altitude_meters:
        :param camera:
        :return:
        '''
        current_x_meters = p[0]
        current_y_meters = p[1]
        leg_x = self.altitude_meters * math.tan(math.radians(self.fov_x_degrees))
        leg_y = self.altitude_meters * math.tan(math.radians(self.fov_y_degrees))
        return stacked_meshgrid(np.linspace(current_x_meters - leg_x, current_x_meters + leg_x, self.pixels_x),
                                np.linspace(current_y_meters - leg_y, current_y_meters + leg_y, self.pixels_y),np.array([0]))

