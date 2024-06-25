import logging
from typing import List

import kdtree
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from numpy.random.mtrand import RandomState
from sklearn.cluster import AgglomerativeClustering
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from tqdm import tqdm

from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace
from sample_sim.noise_models import get_depth_modified_covariance, get_y_modified_covariance
from sample_sim.motion_models.motion_models import MotionModels, LinearMotionModel, DubinsMotionModel
from sample_sim.planning.planning import PlanningAgent
from sample_sim.planning.utils import unit_vector_between_two_points
from sample_sim.robot_manager import RobotManager
from sample_sim.vis.base_vis import Visualizer
from sample_sim.vis.util import set_axes_equal


class Node():
    def __init__(self, position, cost, information, parent, children=None):
        self.position = position
        self.cost = cost
        self.information = information
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children

    # Makes it usable in kdtree?
    def __getitem__(self, item):
        return self.position[item]

    def __len__(self):
        return len(self.position)


class RIG(PlanningAgent):
    def __init__(self, budget, iterations, batch_size, logger, step_size=1,
                 motion_model: MotionModels = MotionModels.Linear,
                 verbose=False, query_points_with_noise_model=False, seed=1,current_model=None):
        self.budget = budget
        self.iterations = iterations
        self.batch_size = batch_size
        self.step_size = step_size
        self.motion_model_type = motion_model
        if motion_model == motion_model.Linear:
            self.motion_model = LinearMotionModel()
        elif motion_model == motion_model.DubinsAirplane:
            self.motion_model = DubinsMotionModel()
        self.verbose = verbose
        self.query_points_with_noise_model = query_points_with_noise_model
        self.logger_name = logger
        self.rs = RandomState(seed)
        self.current_model = current_model

    def next_step(self, auv: RobotManager, data_model: DataModel, workspace: Workspace):
        if workspace.dimensions() == 3:
            self.initial_node = Node(auv.get_current_state()[:3], 0, 0, parent=None)
        else:
            self.initial_node = Node(auv.get_current_state()[:2], 0, 0, parent=None)
        #print(self.initial_node)
        tree = kdtree.create(dimensions=data_model.Xs.shape[1])
        tree.add(self.initial_node)
        closed = set()
        best_information = float("-inf")
        best_node = None

        nodes = []
        positions = []
        noises = []
        for i in tqdm(range(self.iterations), file=TqdmToLogger(logging.getLogger(self.logger_name)), desc="RIG",
                      disable=not self.verbose):
            workspace_sample = workspace.get_point_inside(rs=self.rs)
            k_nearest = tree.search_knn(workspace_sample, k=self.batch_size)
            nearests=[]
            for nearest in k_nearest:
                nearest_node, distance = nearest  # Ignore distance
                nearest = nearest_node.data
                if nearest in closed:
                    continue

                nearest_position_array = np.array(nearest.position)
                workspace_sample_array = np.array(workspace_sample)
                sample = unit_vector_between_two_points(nearest_position_array, workspace_sample_array) * self.step_size
                sample += nearest_position_array
                cov = get_depth_modified_covariance(sample,workspace.zmin,workspace.zmax)

                if self.motion_model.is_feasible(nearest.position, sample):
                    cost = nearest.cost + np.linalg.norm(sample - nearest.position, ord=2)
                    new_node = Node(sample, cost, None, parent=nearest)
                    nearest.children.append(new_node)
                    tree.add(new_node)

                    nodes.append(new_node)
                    positions.append(sample)
                    noises.append([cov,cov,cov])

                    if self.budget < cost < self.budget + self.step_size * 1.5:
                        closed.add(new_node)

        positions = np.array(positions)
        means,stds = data_model.query_many(positions)

        for node,mean,std in zip(nodes,means,stds):
            information = 1/2 * np.log(2 * np.pi * std)
            if node.parent is not None:
                node.information = node.parent.information + information
            else:
                node.information = information


        if closed:
            best_node = None
            best_information = float("-inf")
            for end_node in closed:
                if end_node.information > best_information:
                    best_node = end_node
                    best_information = end_node.information

        out_path = []
        cur_node = best_node
        # Walk tree to get nodes in reverse order
        while cur_node.parent is not None:
            out_path.append(cur_node.position)
            cur_node = cur_node.parent
        path = np.array(list(reversed(out_path)))

        if path.shape[0] < 2:
            return np.array([np.append(path, 0)])
        else:
            if self.motion_model_type == MotionModels.Linear and self.step_size == 1:
                return np.append(np.array(path), np.zeros((len(path), 1)), 1)
            else:
                return self.motion_model.construct_path(path)


class RIGVisualizer(Visualizer):
    def __init__(self, spatial_dimensions, planner: RIG, screen_size=(20, 10)):
        self.fig = plt.figure(figsize=screen_size)
        self.spatial_dimensions = spatial_dimensions
        if spatial_dimensions == 3:
            self.tree_visualizer_ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.tree_visualizer_ax = self.fig.add_subplot(111)
        self.planner = planner

    def update(self):
        self.tree_visualizer_ax.clear()
        stack = [self.planner.initial_node]
        points = [self.planner.initial_node.position[:3]]
        colors = [self.planner.initial_node.information]

        while stack != []:
            elem = stack.pop()
            points.append(elem.position[:3])
            colors.append(elem.information)

            for child in elem.children:
                plottable = np.array([elem, child.position])
                if self.spatial_dimensions == 3:
                    self.tree_visualizer_ax.plot(plottable[:, 0], plottable[:, 1], plottable[:, 2])
                else:
                    self.tree_visualizer_ax.plot(plottable[:, 0], plottable[:, 1])
            stack.extend(elem.children)
        points = np.array(points)
        if self.spatial_dimensions == 3:
            self.tree_visualizer_ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
            set_axes_equal(self.tree_visualizer_ax)
        else:
            self.tree_visualizer_ax.scatter(points[:, 0], points[:, 1], c=colors)

        self.tree_visualizer_ax.set_title("RIG")
