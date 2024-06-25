import enum

import numpy as np
from joblib import Memory
from tqdm import tqdm

from sample_sim.action.actions import ActionModel, action_enum, apply_action_to_state
from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.environments.base import BaseEnvironment
from sample_sim.environments.ecomapper_csv import EcomapperCSVEnvironment
from sample_sim.environments.field_environment import FieldEnvirnoment
from sample_sim.environments.lighting_scene import LightingScene2d
#from sample_sim.environments.noaa_db import NOAADBEnvironment

from sample_sim.environments.raster_environment import DroneFixedHeightRasterEnvironment
from sample_sim.motion_models.DubinsAirplane.DubinsWrapper import euc_dist
from sample_sim.sensors.base_sensor import BaseSensor
from sample_sim.timeit import timeit
import itertools
from collections import defaultdict

MAX_POINT_PRECISION = 5
def tuple_to_max_precision(t):
    return tuple(map(lambda r: round(r,MAX_POINT_PRECISION), t))

class FinitePlanningGrid():
    def __init__(self, Sreal_ndarrays, Sreal, S, S_real_pts_to_idxs, transition_matrix, edge_samples,
                 sensed_locations, sensed_values,environment):
        self.Sreal = Sreal
        self.Sreal_ndarrays = Sreal_ndarrays
        self.S = S
        self.S_real_pts_to_idxs = S_real_pts_to_idxs
        self.transition_matrix = transition_matrix
        self.edge_samples = edge_samples
        self.sensed_locations = sensed_locations
        self.sensed_values = sensed_values
        self.environment = environment

    def get_samples_traveling_from(self, source, dest, sensor: BaseSensor):
        if self.edge_samples is not None:
            src_tuple = tuple_to_max_precision( source)
            dest_tuple = tuple_to_max_precision(dest)
            try:
                next_locations = self.edge_samples[src_tuple][dest_tuple]
            except KeyError:
                try:
                    next_locations = self.edge_samples[dest_tuple][src_tuple]
                except KeyError:
                    raise Exception(f"Couldn't find edge samples between points {source} {dest}")
            return sensor.get_sensed_locations_from_trajectory(next_locations)
        else:
            locations = sensor.get_sensed_locations_from_point(dest)
            if isinstance(self.environment, DroneFixedHeightRasterEnvironment):
                sampled_values = self.environment.sample_unnormalized(locations).squeeze()

                locations = locations[(sampled_values < 255) & (sampled_values > 0), :]
                return locations
            else:
                return locations
                

    def get_next_state_applying_action(self, point,action):
        state_idx = self.S_real_pts_to_idxs[tuple_to_max_precision(point)]
        next_state_idx = self.transition_matrix[state_idx,action.value]
        if next_state_idx == -1:
            return None
        return self.Sreal_ndarrays[next_state_idx]



    def snap_to_grid(self,point):
        S_real_pts_to_idxs = self.get_Sreal_pts_to_idxs()

        # TODO i think this is caused by a bug, figure out what causes it
        if tuple(point) in S_real_pts_to_idxs:
            return tuple(point)
        else:
            # TODO make this not happen
            #TODO use kdtree
            best_idx = None
            best_dist = float("inf")
            best_key = None
            for i, state_space_point in enumerate(S_real_pts_to_idxs.keys()):
                dist = euc_dist(np.array(state_space_point), np.array(point))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_key = state_space_point
            #assert best_dist < 1  # f"closest point {best_key} should be closer than 1 to our auv state {auv.get_current_state()}, but best dist is {best_dist}"
            return tuple(best_key)

    def get_Sreal(self):
        return self.Sreal

    def get_Sreal_ndarrays(self):
        return self.Sreal_ndarrays

    def get_Sreal_pts_to_idxs(self):
        return self.S_real_pts_to_idxs

    def get_S(self):
        return self.Sreal_ndarrays

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_sensed_locations(self):
        return self.sensed_locations

    def get_sensed_values(self):
        return self.sensed_values

    def __repr__(self):
        if self.edge_samples is not None:
            edge_sampling_part = " is doing edge sampling"
        else:
            edge_sampling_part = " is not doing edge sampling"
        return f"Grid with {len(self.S)} states, {self.sensed_values.shape[0]} sensed states," + edge_sampling_part

def generate_finite_grid(environment: BaseEnvironment, state_space_dimensionality, sensor: BaseSensor,
                         number_edge_samples_ecomapper=5):
    workspace = environment.workspace
    action_model = environment.action_model()

    x_step = round((workspace.xmax - workspace.xmin) / state_space_dimensionality[0], 2)
    y_step = round((workspace.ymax - workspace.ymin) / state_space_dimensionality[1], 2)
    if environment.dimensionality() == 3:
        z_step = round((environment.workspace.zmax - environment.workspace.zmin) / state_space_dimensionality[2], 2)
        step_sizes = (x_step, y_step, z_step)

        adjacency_dict = create_adjacency_dict(environment.get_starting_location(seed=0),
                                               workspace, step_sizes, action_model=action_model)

    elif environment.dimensionality() == 2:
        step_sizes = (x_step, y_step)
        adjacency_dict = create_adjacency_dict(environment.get_starting_location(seed=0), workspace, step_sizes,
                                               action_model=action_model)
    else:
        raise Exception()

    Sreal = np.array(list(adjacency_dict.keys()))
    Sreal_ndarrays = np.array(list(map(lambda x: np.array(x), Sreal)))
    S = np.array(list(range(len(Sreal))))
    S_real_pts_to_idxs = {tuple_to_max_precision(Sreal[v]): v for v in S}
    transition_matrix = adjacency_dict_to_numpy(adjacency_dict, S_real_pts_to_idxs, step_sizes,
                                                action_model=action_model)

    if isinstance(environment, LightingScene2d):
        edge_samples_dict = create_edge_sample_dict(adjacency_dict, number_edge_samples_ecomapper,environment.dimensionality())
        all_nonunique_locations = np.concatenate(
            list(itertools.chain(*list(map(lambda k: list(k.values()), edge_samples_dict.values())))), axis=0)
        all_locations = np.unique(all_nonunique_locations, axis=0)
        sensed_values = environment.sample(all_locations)
        sensed_locations = all_locations
    else:
        raise Exception(f"Sensed locations and values not defined for {environment}")
    return FinitePlanningGrid(Sreal=Sreal, Sreal_ndarrays=Sreal_ndarrays, S=S, S_real_pts_to_idxs=S_real_pts_to_idxs,
                              transition_matrix=transition_matrix, edge_samples=edge_samples_dict,
                              sensed_locations=sensed_locations, sensed_values=sensed_values,
                              environment=environment
                              )


@timeit
def create_adjacency_dict(start_state, workspace, step_sizes, action_model: ActionModel):
    # if not workspace.is_inside(start_state):
    #     raise Exception("Starting point is not inside workspace")
    queue = [start_state]
    visited = set()
    adjacency_dict = defaultdict(dict)
    i = 0
    while queue != []:
        i += 1
        cur_state = queue.pop()
        for action in action_enum(action_model):
            next_state = tuple_to_max_precision(tuple(apply_action_to_state(cur_state, action, step_sizes)))
            if workspace.is_inside(next_state):
                adjacency_dict[tuple_to_max_precision(cur_state)][action] = next_state
                if next_state not in visited:
                    queue.append(next_state)
                    visited.add(next_state)
    return adjacency_dict


def create_edge_sample_dict(adjacency_dict, num_samples_between_points, dimensionality):
    edge_sample_dict = dict()
    for edge1 in adjacency_dict.keys(): #tqdm(adjacency_dict.keys(), desc="Creating edge sample dictionary", total=len(adjacency_dict.keys())):
        edge_sample_dict[edge1] = dict()
        action_dict = adjacency_dict[edge1]
        for action, edge2 in action_dict.items():
            if edge1 != edge2:
                xs = np.linspace(edge1[0], edge2[0], num_samples_between_points+2)[1:]
                ys = np.linspace(edge1[1], edge2[1], num_samples_between_points+2)[1:]
                if dimensionality == 3: 
                    zs = np.linspace(edge1[2], edge2[2], num_samples_between_points+2)[1:]
                    points = np.vstack((xs, ys, zs)).T
                else:
                    points = np.vstack((xs, ys)).T

                edge_sample_dict[edge1][edge2] = points
    return edge_sample_dict


def adjacency_dict_to_numpy(adjacency_dict, S_real_pts_to_idxs, step_sizes, action_model: ActionModel):
    action_enums = action_enum(action_model)
    transition_matrix = np.zeros((len(adjacency_dict.keys()), len(action_enums)), dtype="int32") + -1
    for i, state in enumerate(adjacency_dict.keys()):
        assert i == S_real_pts_to_idxs[state]
        # try:
        #     transition_matrix[S_real_pts_to_idxs[state], :] = S_real_pts_to_idxs[
        #         tuple(apply_action_to_state(state, action_enums.STAY_STILL, step_sizes))]
        # except KeyError:
        #     # This is when you get to the end, just go to the same state?
        #     transition_matrix[S_real_pts_to_idxs[state], :] = -1
        for action in action_enums:
            # if action not in adjacency_dict[state]:
            #     # 4 is the "do nothing"
            #     action = action_enums.STAY_STILL
            try:
                if adjacency_dict[state][action] in S_real_pts_to_idxs:
                    transition_matrix[S_real_pts_to_idxs[state], action.value] = S_real_pts_to_idxs[
                        adjacency_dict[state][action]]
            except (KeyError, IndexError):
                pass
                # not a valid move

    return transition_matrix
