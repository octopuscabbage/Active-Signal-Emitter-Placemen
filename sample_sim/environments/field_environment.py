import logging
from time import sleep

import utm

from sample_sim.action.actions import ActionModel
from sample_sim.environments.base import BaseEnvironment
from sample_sim.environments.workspace import FieldWorkspace3d
from sample_sim.seabridge.api_utils import APIUtils
import      numpy as np


class FieldEnvirnoment(BaseEnvironment):
    def __init__(self,easting1,northing1,easting2,northing2,api_url):
        super().__init__(FieldWorkspace3d(easting1,northing1,easting2,northing2))
        self.easting1 = easting1
        self.easting2 = easting2
        self.northing1 = northing1
        self.northing2 = northing2
        self.api_url = api_url

    def sample(self, xs):
        raise NotImplementedError()

    def action_model(self):
        return ActionModel.XYZ

    def dimensionality(self):
        return 3

    def get_starting_location(self):
        first_id = 1
        r_json = APIUtils.get_robotstate_json(id=first_id, api_url=self.api_url)
        logging.info("Attempting to get first robot state...")
        while not APIUtils.request_succeeded(r_json):
            sleep(0.5)
            r_json = APIUtils.get_robotstate_json(id=first_id, api_url=self.api_url)
        x, y, self.zone_number, self.zone_letter = utm.from_latlon(r_json['lat'], r_json['lon'])

        auv_location = np.array([x, y, 0])
        return auv_location


    def __repr__(self):
        return f"Field Experiment E1 {self.easting1} N1 {self.northing1} E2 {self.easting2} N2 {self.northing2}"

    def get_parameters(self):
        return {"easting1":self.easting1,
                "northing1": self.northing1,
                "easting2": self.easting2,
                "northign2": self.northing2
                }
