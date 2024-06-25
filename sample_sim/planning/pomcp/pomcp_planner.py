import json
import logging
import random
import unittest
from statistics import mode, variance

import numpy as np
import scipy.stats as stats

from pomcp.pomcp import POMCP, average_or_0_on_empty
from sample_sim.action.actions import ActionModel, action_enum
from sample_sim.action.grid import FinitePlanningGrid, create_adjacency_dict, adjacency_dict_to_numpy
from sample_sim.data_model.data_model import DataModel
from sample_sim.environments.base import BaseEnvironment
from sample_sim.environments.workspace import RectangularPrismWorkspace
from sample_sim.motion_models.DubinsAirplane.DubinsWrapper import euc_dist
from sample_sim.planning.planning import PlanningAgent
from sample_sim.planning.pomcp.pomcp_generator import Generator
from sample_sim.planning.pomcp.pomcp_utilities import get_uct_c, PomcpExtraData, get_default_low_param, \
    get_default_hi_param
from sample_sim.planning.pomcp_top_level_action_selection.action_selectors import ActionSelectors
from sample_sim.planning.pomcp_top_level_action_selection.ugap import UGapEb
from sample_sim.sensors.base_sensor import BaseSensor


class POMCPPlanner(PlanningAgent):
    def __init__(self, grid, budget: int, logger_name, seed,
                 objective_c, action_model: ActionModel, state_space_dimensionality,
                 filename, rollouts_per_step, quantiles,
                 use_t_test,
                 t_test_value, max_planning_depth, objective_function, gamma):
        self.logger_name = logger_name
        self.budget = budget
        self.state_space_dimensionality = state_space_dimensionality
        self.action_enum = action_enum(action_model)

        self.rollouts_per_step = rollouts_per_step
        self.quantiles = quantiles
        self.max_planning_depth = max_planning_depth
        self.objective_function = objective_function
        self.gamma = gamma

        self.use_t_test = use_t_test
        self.t_test_value = t_test_value

        S = grid.get_S()
        self.A = [a.value for a in self.action_enum]
        self.O = S[:]
        self.logger_name = logger_name
        self.rs = np.random.RandomState(seed)
        self.objective_c = objective_c

        self.log_sum_stdvs = []
        self.used_budgets = []
        self.used_rollouts = []
        self.reward_stds = []
        self.filename = filename
        self.objectives = []
        self.state_space_dimensionality = state_space_dimensionality
        self.c_lo = None
        self.c_hi = None
        self.planner_c = None
        self.generator_calls_last_search = 0

        self.arm_selection_algorithm = ActionSelectors.UCB

    def __set_if_new_non_none(self, new, old):
        if new is not None:
            return new
        else:
            return old

    def set_parameters(self, rollouts_per_step=None, gamma=None, planner_c=None, objective_c=None,
                       max_planning_depth=None, t_test_value=None, arm_selection_algorithm=None):
        # TODO make sure these get applied (esp planner_c)
        self.rollouts_per_step = self.__set_if_new_non_none(rollouts_per_step, self.rollouts_per_step)
        self.gamma = self.__set_if_new_non_none(gamma, self.gamma)
        self.planner_c = self.__set_if_new_non_none(planner_c, self.planner_c)
        self.objective_c = self.__set_if_new_non_none(objective_c, self.objective_c)
        self.max_planning_depth = self.__set_if_new_non_none(max_planning_depth, self.max_planning_depth)
        self.t_test_value = self.__set_if_new_non_none(t_test_value, self.t_test_value)
        self.arm_selection_algorithm = self.__set_if_new_non_none(arm_selection_algorithm, self.arm_selection_algorithm)

    # take current state (AUV and world). Runs POMCP. Outputs traj of poses (where next to take samples from)
    def next_step(self, auv_location, data_model: DataModel, environment: BaseEnvironment, grid: FinitePlanningGrid,
                  sensor: BaseSensor, max_generator_calls=None):
        start_state_idx = grid.S_real_pts_to_idxs[grid.snap_to_grid(auv_location)]
        # S_real_pts_to_idxs = grid.get_Sreal_pts_to_idxs()
        #
        # # TODO i think this is caused by a bug, figure out what causes it
        # try:
        #     logging.getLogger(self.logger_name).info(f"AUV position {auv_location}")
        #     start_state_idx = S_real_pts_to_idxs[tuple(auv_location)]
        # except KeyError:  # A bit unsure why this happens, maybe floating point error???
        #     #TODO make this not happen
        #     best_idx = None
        #     best_dist = float("inf")
        #     best_key = None
        #     for i, state_space_point in enumerate(S_real_pts_to_idxs.keys()):
        #         dist = euc_dist(np.array(state_space_point), np.array(auv_location))
        #         if dist < best_dist:
        #             best_dist = dist
        #             best_idx = i
        #             best_key = state_space_point
        #     start_state_idx = best_idx
        #     assert best_dist < 1 # f"closest point {best_key} should be closer than 1 to our auv state {auv.get_current_state()}, but best dist is {best_dist}"
        #     logging.getLogger(self.logger_name).debug(
        #         f"Current state not found in state space, using {best_key} for {auv_location}, Dist {best_dist}")

        self.used_budgets.append(self.budget)
        # self.log_sum_stdvs.append(np.log(np.sum(stdv)))

        if self.c_lo is None:
            c_low, c_hi = get_uct_c(self.objective_c, self.filename, self.logger_name, self.objective_function)
            self.c_lo = c_low
            self.c_hi = c_hi

        if self.planner_c is not None:
            planner_c = self.planner_c
        else:
            planner_c = self.c_hi - self.c_lo


        if self.arm_selection_algorithm == ActionSelectors.UCB:
            arm_selection_algorithm_instance = None
        elif self.arm_selection_algorithm == ActionSelectors.UGapEb:
            arm_selection_algorithm_instance = UGapEb(self.rollouts_per_step,0.01,planner_c,self.logger_name,self.action_enum)
        else:
            raise Exception(f"Action selection alrogirthm not understood {self.arm_selection_algorithm}")

        cur_plan = POMCP(Generator, logger=self.logger_name, gamma=self.gamma, start_states=[start_state_idx],
                         c=planner_c,
                         random_state=self.rs,
                         action_enum=self.action_enum,
                         extra_generator_data=PomcpExtraData(
                             objective_c=self.objective_c,
                             data_model=data_model, filename=self.filename,
                             state_space_dimensionality=self.state_space_dimensionality,
                             quantiles=self.quantiles,
                             objective_function=self.objective_function,
                             grid=grid, environment=environment, sensor=sensor
                         ),
                         num_rollouts=self.rollouts_per_step,
                         total_budget=self.budget,
                         max_planning_depth=self.max_planning_depth,
                         arm_selection_algorithm=arm_selection_algorithm_instance)
        cur_plan.initialize(self.O, self.A, self.O)
        self.cur_plan = cur_plan

        cur_plan.Search(max_generator_calls=max_generator_calls)
        self.generator_calls_last_search = cur_plan.get_generator_calls_from_last_search()

        # populate best actions from tree
        next_actions = []
        curr_state_tree_idx = -1  # start from root

        logging.getLogger(self.logger_name).debug(self.pomcp_tree_to_string(cur_plan.tree.nodes, grid.get_Sreal()))
        while not cur_plan.tree.isUnvisitedObsNode(curr_state_tree_idx):
            action_idx, action_tree_idx = cur_plan.SearchBest(curr_state_tree_idx, UseUCB=False)
            if action_tree_idx is None:
                raise Exception(
                    f"Action tree idx is none! {self.pomcp_tree_to_string(cur_plan.tree.nodes, grid.get_Sreal())}")
            if self.use_t_test:
                child_obs_tree_idx = random.choice(
                    list(cur_plan.tree.get_children(action_tree_idx).values()))  # sample from children nodes
                # caveat: take the first action always
                if len(next_actions) == 0:
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3]))  # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                    continue

                if cur_plan.tree.get_visits(curr_state_tree_idx) < len(self.action_enum) + 1:
                    # you haven't gone down each path at least once, cannot apply ugap theory
                    break

                action_idx_2ndBest, action_tree_2ndBest_idx = cur_plan.SearchSecondBest(curr_state_tree_idx,
                                                                                        UseUCB=False)
                rewards_best = cur_plan.tree.get_reward_history(action_tree_idx)
                rewards_2ndBest = cur_plan.tree.get_reward_history(action_tree_2ndBest_idx)
                try:
                    tstat, self.plan_ttest_pval = stats.ttest_ind(rewards_best, rewards_2ndBest, equal_var=False)
                except RuntimeWarning:
                    break
                # assert not np.isnan(self.plan_ttest_pval)
                # assert not np.isnan(tstat)
                if np.isnan(self.plan_ttest_pval):
                    # maybe not enough data to run ttest? skip it
                    break

                if self.plan_ttest_pval < self.t_test_value:  # if p(null hypothesis) is small enough
                    # we can reject the null hypothesis
                    # best arm is statistically better than 2nd best. Append action
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3]))  # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                else:  # stop appending from this point onwards
                    break

            else:
                next_actions.append(self.action_enum(action_idx))
                self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3]))  # append reward
                break

        # logging.getLogger(self.logger_name).debug(self.pomcp_tree_to_string(cur_plan.tree.nodes, Sreal))
        assert len(next_actions) > 0

        # Populate next states based on planners choice of actions
        logging.getLogger(self.logger_name).warning(
            f"Next Actions {next_actions}, current state {auv_location}")
        next_states = []
        curr_state_coord = auv_location
        for action in next_actions:
            next_state_coord = grid.get_next_state_applying_action(curr_state_coord, action)
            # print(curr_state_coord,next_state_coord,action,environment.workspace.get_bounds())
            # if not environment.is_inside(next_state_coord):
            #         raise Exception("If this happens write some code to make the robot go the opposite way")

            # TODO: better way to update curr_state_coord instead of calling it in special cases?
            next_states.append(next_state_coord)
            curr_state_coord = next_state_coord  # continue loop

        #self.save_uct_c()

        rewards = [reward for arm_reward in self.get_root_rewards() for reward in arm_reward]
        # self.c_lo = min(rewards)
        # self.c_hi = max(rewards)

        assert len(next_states) > 0

        return next_states

    def get_generator_calls_from_last_search(self):
        return self.generator_calls_last_search

    def pomcp_tree_to_string(self, nodes, state_space, depth=3, include_action_nodes=True, include_belief_nodes=False):
        # I should've just done this recursively, my haskeller is leaving
        out_str = "\n"
        stack = [("", nodes[-1], depth)]
        while stack != []:
            action, node, remaining_depth = stack.pop()
            is_action_node = node[-1] == -1
            if is_action_node:
                #assert len(node[1]) in [0, 1], "I don't know how to interpret non deterministic action nodes"
                if include_action_nodes and node[2] != 0:
                    out_str += ("\t" * (
                            depth - remaining_depth)) + f"{self.action_enum(action)} R - N({average_or_0_on_empty(node[3]):.2E},{np.std(node[3]):.2E}), N: {node[2]} \n"
                for parent, child in node[1].items():
                    stack.append((action, nodes[child], remaining_depth))
            else:  # Belief node
                if include_belief_nodes:
                    out_str += ("\t" * (
                            depth - remaining_depth)) + f"  {node[0]} R: {round(node[3], 2)}, N: {node[2]} B: {node[-1]} \n"
                # else:
                #     out_str += ("\t" * (
                #             depth - remaining_depth)) + f"{self.action_enum(action)} - Key: {node[0]} R: {round(node[3], 2)}, N: {node[2]} B: {set(map(lambda state_index: state_space[state_index], node[-1]))} \n"
                if remaining_depth > 0:
                    for action, child_idx in node[1].items():
                        stack.append((action, nodes[child_idx], remaining_depth - 1))
        return out_str

    def save_uct_c(self):
        cur_low_param, cur_hi_param = get_uct_c(self.objective_c, self.filename, self.logger_name,self.objective_function)
        rewards = [reward for arm_reward in self.get_root_rewards() for reward in arm_reward]
        cur_low = min(filter(lambda x: x != 0, rewards))
        cur_hi = max(rewards)


        if cur_low_param != get_default_low_param():
            new_low_param = min(cur_low, cur_low_param)
        else:
            new_low_param = cur_low
        if cur_hi_param != get_default_hi_param():
            new_hi_param = max(cur_hi, cur_hi_param)
        else:
            new_hi_param = cur_hi
        if isinstance(new_hi_param, np.ndarray):
            new_hi_param = float(new_hi_param[0])
        if isinstance(new_low_param, np.ndarray):
            new_low_param = float(new_low_param[0])

        with open(f"params/{self.objective_function}-{self.filename.replace('/', '').replace(':','')}.json", "w") as f:
            json.dump({"low": new_low_param, "high": new_hi_param}, f)

    def get_root_rewards(self):
        try:
            return self.cur_plan.get_root_rewards()
        except AttributeError:
            logging.getLogger(self.logger_name).warning("Get Reward called without a computed plan")
            return [0 for _ in range(len(self.action_enum))]

    def get_critical_path(self, grid: FinitePlanningGrid,coordinates: bool = False):
        """
        Returns the critical path
        :param coordinates: Whether to return the path as a list
                            of (observation node, reward) or (coordinates, reward) tuples
        """

        # Exit if the plan hasn't been made yet
        if not hasattr(self, "cur_plan") or not self.cur_plan:
            return []

        # First, iterate through current plan to find most recent critical path
        critical_path = []
        tree = self.cur_plan.tree
        work = [tree.nodes[n] for n in tree.get_children(-1).values()]
        best_node = None
        best_reward = None

        while work:
            # Get info about one of the nodes
            cur_node = work.pop()

            if cur_node[3]:  # Some nodes don't have reward assigned yet, just skip them

                cur_reward = tree.get_node_reward(cur_node)

                if not best_node:
                    # Should only happen on first run at each depth
                    best_node = cur_node
                    best_reward = cur_reward

                elif cur_reward > best_reward:
                    # Found a better node
                    best_node = cur_node
                    best_reward = cur_reward

                elif cur_reward == best_reward and len(tree.get_node_value(cur_node)) > 1 and \
                        len(tree.get_node_value(best_node)) > 1 and variance(cur_node[3]) < variance(best_node[3]):
                    # Choose the node with lower variance if rewards are equal
                    best_node = cur_node
                    best_reward = cur_reward

            # Check if we're done with nodes at this depth.
            if not work and best_node:
                # If so, get the child observation node
                obs_node = tree.nodes[list(best_node[1].values())[0]]

                work = [tree.nodes[n] for n in obs_node[1].values()]  # Go to next depth based on chosen node

                if coordinates:
                    if obs_node[4]:
                        # Only add node if it has been visited bc we dont have a position estimate otherwise
                        position = grid.get_Sreal_ndarrays()[mode(obs_node[4])]
                        critical_path.append((position, best_reward))
                else:
                    critical_path.append((obs_node, best_reward))

                # Reset values
                best_node = None
                best_reward = None

        return critical_path

class TestAdjacencyMatrix(unittest.TestCase):
    def test_adjacency_matrix(self):
        workspace = RectangularPrismWorkspace(0, 4, 0, 4, 0, 4)
        adjacency_dict = create_adjacency_dict([0, 0, 0], workspace)
        self.assertIn((0, 0, 0), adjacency_dict)
        self.assertEqual((1, 0, 1), adjacency_dict[(0, 0, 0)][0])
        self.assertEqual((0, 0, 1), adjacency_dict[(0, 0, 0)][4])
        # self.assertEquals((0, 0, 1), adjacency_dict[(0, 0, 0)][1])
        # self.assertEquals((0, 0, 1), adjacency_dict[(0, 0, 0)][3])
        self.assertEqual((0, 1, 1), adjacency_dict[(0, 0, 0)][2])

        self.assertEqual(dict(), adjacency_dict[(0, 0, 4)])

    def test_adjacency_matrix_to_numpy(self):
        workspace = RectangularPrismWorkspace(0, 4, 0, 4, 0, 4)
        adjacency_dict = create_adjacency_dict([0, 0, 0], workspace)
        Sreal = list(adjacency_dict.keys())
        S = list(range(len(Sreal)))
        S_real_pts_to_idxs = {tuple(Sreal[v]): v for v in S}

        matrix = adjacency_dict_to_numpy(adjacency_dict, S_real_pts_to_idxs)
        self.assertIn((0, 0, 0), adjacency_dict)
        self.assertEqual(S_real_pts_to_idxs[(1, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 0])
        self.assertEqual(S_real_pts_to_idxs[(0, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 4])
        self.assertEqual(S_real_pts_to_idxs[(0, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 1])
        self.assertEqual(S_real_pts_to_idxs[(0, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 3])
        self.assertEqual(S_real_pts_to_idxs[(0, 1, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 2])

        print(matrix)


if __name__ == "__main__":
    unittest.main()
