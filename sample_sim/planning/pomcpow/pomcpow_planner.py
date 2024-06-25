import enum
import json
import math
import random
from copy import deepcopy

from scipy.stats import norm
import logging

from sample_sim.action.actions import action_enum
from sample_sim.action.grid import FinitePlanningGrid
from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.factor_graph import FactorGraphDataModel
from sample_sim.environments.base import BaseEnvironment
from sample_sim.environments.lighting_scene import LightingScene2d
from sample_sim.planning.planning import PlanningAgent
from sample_sim.planning.pomcp.pomcp_utilities import get_uct_c, get_default_low_param, get_default_hi_param
from sample_sim.planning.pomcpow.pomcpow import RolloutStrategy, SamplingStateHistory, POMCPOW, DiscreteWeightedBelief, HashableNumpyArray
from sample_sim.planning.reward_functions import calculate_reward
from sample_sim.sensors.base_sensor import BaseSensor
import numpy as np


class ObservationSamplingStrategy(enum.Enum):
    MEAN = 0,
    RANDOM_NORMAL = 1,
    CONFIDENCE_INTERVALS = 2

def gaussian_log_liklihood(data,mu,sigma):
    return np.sum(norm(mu,sigma).logpdf(data))

class PomcpowExtraData():
    def __init__(self, data_model,
                 objective_function, environment: BaseEnvironment, sensor, grid: FinitePlanningGrid,
                 observation_sampling_strategy: ObservationSamplingStrategy,base_gp, desired_environment: LightingScene2d, rollout_strategy: RolloutStrategy):
        self.data_model = data_model
        self.objective_function = objective_function
        self.environment = environment
        self.sensor = sensor
        self.grid = grid
        self.observation_sampling_strategy = observation_sampling_strategy
        self.base_gp = base_gp
        self.desired_environment = desired_environment 
        self.rollout_strategy = rollout_strategy


def calculate_observation(sensor_points: np.ndarray, mean: np.ndarray, stdv: np.ndarray,
                          observation_sampling_strategy: ObservationSamplingStrategy):
    if observation_sampling_strategy == ObservationSamplingStrategy.MEAN:
        return (sensor_points, mean, 1)
    elif observation_sampling_strategy == ObservationSamplingStrategy.RANDOM_NORMAL:
        sample = np.random.normal(mean, stdv)
        ll = 0
        for cur_sample,c_mean,c_stdv in zip(sample,mean,stdv):
            ll += gaussian_log_liklihood(cur_sample,c_mean,c_stdv)
        if not isinstance(sample,np.ndarray):
            sample = np.array([sample])

        return (sensor_points, sample, np.exp(ll))
    elif observation_sampling_strategy == ObservationSamplingStrategy.CONFIDENCE_INTERVALS:
        indexes = np.round(np.random.normal(0, 1, size=mean.shape[0]))
        ll = 0
        for index in indexes:
             ll += gaussian_log_liklihood(index,0,1)
        return (sensor_points, mean + indexes * stdv, np.exp(ll))
    else:
        raise Exception(f"Unkown Observation Sampling Strategy {observation_sampling_strategy}")


def Generator(belief_history: SamplingStateHistory, a, extra_data: PomcpowExtraData):
    loc, _, _ = belief_history.get_last_state()
    loc_prime = extra_data.grid.get_next_state_applying_action(loc, a)
    
    sensor_points = extra_data.grid.get_samples_traveling_from(loc, loc_prime, extra_data.sensor)
    model = extra_data.data_model
    model.update_prior(belief_history.xs, belief_history.ys)
    mean,stdv = model.query_many(sensor_points,return_std=True)



    sensor_points_obs, observed_values,likelihood = calculate_observation(sensor_points, mean, stdv,
                                                               extra_data.observation_sampling_strategy)

    rw = np.sum(abs(
        calculate_reward(observed_values, stdv, extra_data.objective_function,
                         extra_data.grid, extra_data.data_model, sensor_points_obs, belief_history.xs,
                         belief_history.ys,extra_data.desired_environment)))

    s_prime = (loc_prime, sensor_points_obs, observed_values)
    rw = np.sum(rw)
    return s_prime, (HashableNumpyArray(sensor_points_obs), HashableNumpyArray(observed_values)), rw, likelihood


def next_action(belief: SamplingStateHistory, extra_data: PomcpowExtraData):
    loc, _, _ = belief.get_last_state()

    chosen_action = random.choice(list(action_enum(extra_data.environment.action_model())))
    next_loc = extra_data.grid.get_next_state_applying_action(loc, chosen_action)
    while next_loc is None or not extra_data.environment.is_inside(next_loc):
        chosen_action = random.choice(list(action_enum(extra_data.environment.action_model())))
        next_loc = extra_data.grid.get_next_state_applying_action(loc, chosen_action)
    return chosen_action

def all_valid_actions(belief: SamplingStateHistory, extra_data: PomcpowExtraData):
    loc, _, _ = belief.get_last_state()
    possible_actions = action_enum(extra_data.environment.action_model())
    acceptable_actions = []
    for action in possible_actions:
        next_loc = extra_data.grid.get_next_state_applying_action(loc, action)
        if next_loc is not None and extra_data.environment.is_inside(next_loc):
            acceptable_actions.append(action)
    return acceptable_actions




    

def reward(belief_history: SamplingStateHistory, action, next_state, extra_data: PomcpowExtraData):
    loc, _, _ = belief_history.get_last_state()
    loc_prime = extra_data.grid.get_next_state_applying_action(loc, action)

    sensor_points = extra_data.grid.get_samples_traveling_from(loc, loc_prime, extra_data.sensor)

    model = extra_data.data_model
    model.update_prior(belief_history.xs, belief_history.ys)
    mean,stdv = model.query_many(sensor_points,return_std=True)


    rw = np.sum(abs(
        calculate_reward(mean, stdv, extra_data.objective_function,
                         extra_data.grid, extra_data.data_model, sensor_points, belief_history.xs,
                         belief_history.ys,extra_data.desired_environment)))

    # rw += extra_data.objective_c * (np.sum(stdv) / sensor_points.size)
    # rw = np.sum(rw)
    return rw


class POMCPOWPlanner(PlanningAgent):
    def __init__(self, logger_name, objective_function, 
                 observation_sampling_strategy: ObservationSamplingStrategy,fignum=None):
        self.logger_name = logger_name
        self.objective_function = objective_function
        self.observation_sampling_strategy = observation_sampling_strategy

        self.rollouts_per_step = None
        self.gamma = None
        self.planner_c = None
        self.objective_c = None
        self.max_planning_depth = None
        self.c_lo = None
        self.c_hi = None
        self.extra_data = None
        self.fignum = fignum
        self.rew_max =  None
        self.rew_min = None
        self.planner_c_set = False


    def __set_if_new_non_none(self, new, old):
        if new is not None:
            return new
        else:
            return old

    def set_parameters(self, rollouts_per_step=None, gamma=None, planner_c=None, objective_c=None,
                       max_planning_depth=None):
        # TODO make sure these get applied (esp planner_c)
        self.rollouts_per_step = self.__set_if_new_non_none(rollouts_per_step, self.rollouts_per_step)
        self.gamma = self.__set_if_new_non_none(gamma, self.gamma)
        self.planner_c = self.__set_if_new_non_none(planner_c, self.planner_c)
        self.objective_c = self.__set_if_new_non_none(objective_c, self.objective_c)
        self.max_planning_depth = self.__set_if_new_non_none(max_planning_depth, self.max_planning_depth)
    
    def set_extra_data(self, extra_data: PomcpowExtraData):
        self.extra_data = extra_data


    def next_step(self, auv_location, data_model: DataModel, environment: BaseEnvironment, grid: FinitePlanningGrid,
                  sensor: BaseSensor):


        if self.rew_max is not None and self.rew_min is not None and self.rew_max - self.rew_min == 0:
            raise Exception("Planner C is set to 0")
        
        if self.extra_data is None:
            raise Exception("You have to call set extra data")

        planner = POMCPOW(self.logger_name, next_action, Generator, reward, self.extra_data, max_depth=self.max_planning_depth,
                          check_actions_repeated=True,
                          check_observations_repeated=self.observation_sampling_strategy == ObservationSamplingStrategy.MEAN,
                          exploration_weight_upper=self.rew_max, exploration_weight_lower=self.rew_min, gamma=self.gamma,all_actions_fn=all_valid_actions)

        belief_history = SamplingStateHistory(np.array(auv_location,dtype=np.float64),data_model.Xs[-1:,:],data_model.Ys[-1:])
        initial_belief = DiscreteWeightedBelief([1],[belief_history])
        best_action = planner.plan(initial_belief,self.rollouts_per_step)

        self.save_uct_c(planner,str(environment))
        if self.fignum is not None:
        #     #logging.getLogger(self.logger_name).info(planner.to_string(belief_history))
            planner.draw_tree_igraph(belief_history, fignum=self.fignum)
        return [grid.get_next_state_applying_action(auv_location,best_action)]

    def save_uct_c(self,planner,name):
        self.rew_min = planner.exploration_weight_lower
        self.rew_max = planner.exploration_weight_upper
