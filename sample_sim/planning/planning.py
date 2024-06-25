import abc
import numpy as np

from sample_sim.action.grid import FinitePlanningGrid
from sample_sim.environments.base import BaseEnvironment
from sample_sim.data_model.data_model import DataModel
from sample_sim.sensors.base_sensor import BaseSensor




class PlanningAgent(abc.ABC):
    @abc.abstractmethod
    def next_step(self, auv_location, data_model: DataModel, environment:BaseEnvironment, grid:FinitePlanningGrid, sensor:BaseSensor):
        pass

