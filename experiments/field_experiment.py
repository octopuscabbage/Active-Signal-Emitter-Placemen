import json
import logging
import os
import random
import time
import typing
from copy import deepcopy

import PIL
import numpy as np
import utm
from smallab.experiment_types.overlapping_output_experiment import (OverlappingOutputCheckpointedExperimentReturnValue,
                                                                    OverlappingOutputCheckpointedExperiment)
from smallab.name_helper.dict import dict2name
from smallab.smallab_types import Specification, ExpProgressTuple

from sample_sim.action.actions import ActionXYZ, action_enum
from sample_sim.action.grid import generate_finite_grid, tuple_to_max_precision
from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.environments.field_environment import FieldEnvirnoment
from sample_sim.environments.workspace import FieldWorkspace3d
from sample_sim.parameterization.base_parameterization import IPPMetaData
from sample_sim.parameterization.policy import BasePomcpParameterPolicy, BaseActionPolicy
from sample_sim.parameterization.policy_dispatch import policy_dispatch
from sample_sim.planning.pilot_surveys import quantize_to_sensed_locations
from sample_sim.planning.pomcp.pomcp_planner import POMCPPlanner
from sample_sim.planning.reward_functions import calculate_reward
from sample_sim.seabridge.api_utils import APIUtils
from sample_sim.sensors.camera_sensor import FixedHeightCamera

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


class FieldExperiment(OverlappingOutputCheckpointedExperiment):
    def initialize(self, specification: Specification):
        self.saved_before_first_action = False
        logger = logging.getLogger(self.get_logger_name())
        logger.setLevel(logging.DEBUG)
        self.specification = specification
        logger.info(f"Begin {specification}")

        drone_altitude = specification.get("drone_altitude", None)
        camera_fov_x = specification.get("camera_fov_x", None)
        camera_fov_y = specification.get("camera_fov_y", None)

        camera_pixels_x = specification.get("camera_pixels_x", None)
        camera_pixels_y = specification.get("camera_pixels_y", None)

        self.sensor = FixedHeightCamera(camera_fov_x, camera_fov_y, camera_pixels_x, camera_pixels_y,
                                        drone_altitude)

        self.plot = specification["plot"]
        self.seed = specification["seed"]
        self.objective_c = specification["objective_c"]
        self.state_space_dimensionality = specification.get("state_space_dimensionality", None)
        self.physical_step_size = specification.get("physical_step_size", None)
        if self.state_space_dimensionality is not None and self.physical_step_size is not None:
            raise Exception("You can only set either state space dimensionality or physical step size")

        self.quantiles = specification["quantiles"]
        self.max_budget = specification["planning_steps"]  # self.state_space_dimensionality[2]
        self.rollouts_per_step = specification["rollouts_per_step"]
        self.objective_function = specification["objective_function"]
        self.lengthscale = specification["lengthscale"]
        self.parameter_policy_name = specification["parameter_policy"]

        self.total_rolluts = specification["number_of_rollouts"]
        self.remaining_rollouts = specification["number_of_rollouts"]

        self.planner_gamma = specification["planner_gamma"]
        self.max_planner_depth = specification["max_planner_rollout_depth"]

        self.max_generator_calls = specification.get("max_generator_calls", None)

        self.api_url = specification["api_url"]
        self.images_folder = specification["images_folder"]


        input("Press Enter when the workspace is defined...")

        workspaces_json = APIUtils.get_all_workspaces_json(api_url=self.api_url)
        logger.info(f"Attempt to get most recent workspace...")
        while not APIUtils.request_succeeded(workspaces_json):
            time.sleep(0.5)
            workspaces_json = APIUtils.get_all_workspaces_json(api_url=self.api_url)
        print(workspaces_json)
        most_recent_workspace = sorted(workspaces_json["items"], key=lambda w: w["timestamp"])[-1]

        eastings = sorted([most_recent_workspace["easting1"], most_recent_workspace["easting2"]])
        northings = sorted([most_recent_workspace["northing1"], most_recent_workspace["northing2"]])

        #workspace = FieldWorkspace3d(eastings[0],northings[0],eastings[1],northings[1])
        self.environment = FieldEnvirnoment(eastings[0],northings[0],eastings[1],northings[1],self.api_url)

        self.grid = generate_finite_grid(self.environment, tuple(self.state_space_dimensionality), self.sensor,
                                         0)
        self.file = "field"

        # TODO make good repr for these
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Grid: {self.grid}")

        self.current_traj = []
        self.data_X = []
        self.data_Y = []
        self.noises = []
        self.budgets_that_were_used = []
        self.used_budget = 0
        self.used_rollouts = 0
        self.planner = POMCPPlanner(grid=self.grid, budget=self.max_budget,
                                    action_model=self.environment.action_model(),
                                    logger_name=self.get_logger_name(), seed=self.seed,
                                    objective_c=self.objective_c,
                                    state_space_dimensionality=self.state_space_dimensionality,
                                    filename=self.file, rollouts_per_step=self.rollouts_per_step,
                                    quantiles=self.quantiles,
                                    use_t_test=specification["use_t_test"], t_test_value=specification["t_test_value"],
                                    objective_function=self.objective_function, gamma=self.planner_gamma,
                                    max_planning_depth=self.max_planner_depth)

        logger.debug("creating auv data model")



        first_id = 1
        r_json = APIUtils.get_robotstate_json(id=first_id, api_url=self.api_url)
        logger.info("Attempting to get first robot state...")
        while not APIUtils.request_succeeded(r_json):
            time.sleep(0.5)
            r_json = APIUtils.get_robotstate_json(id=first_id, api_url=self.api_url)

        i_json = APIUtils.get_image_json(id=first_id, api_url=self.api_url)
        logger.info(f"Attempt to get first image...")
        while not APIUtils.request_succeeded(i_json):
            time.sleep(0.5)
            i_json = APIUtils.get_image_json(id=first_id, api_url=self.api_url)

        x, y, self.zone_number, self.zone_letter = utm.from_latlon(r_json['lat'], r_json['lon'])

        self.auv_location = np.array([x,y,0])

        view, samples = self.get_view_and_samples_from_robotstate_image(r_json, i_json)

        self.auv_data_model = TorchExactGPBackedDataModel(view, samples,
                                                          logger=self.logger)

        self.auv_data_model.update(view, samples)
        self.auv_data_model.model.model.covar_module.base_kernel.lengthscale = self.lengthscale
        self.auv_data_model.model.eval_model()
        self.auv_data_model.model.model.mean_module.mean = np.mean(self.auv_data_model.Ys)

        self.robot_states = [self.auv_location]
        self.parameter_policy = policy_dispatch(self.parameter_policy_name)

        try:
            self.parameter_policy.get_settings_from_config(config=self.specification)
        except Exception:
            logger.error("Couldn't load parameters for policy", exc_info=True)

        self.last_parameter_control_object = None
        self.cumulative_environment_reward = 0


    def calculate_progress(self) -> ExpProgressTuple:
        return self.used_budget, self.max_budget

    def calculate_return(self) -> OverlappingOutputCheckpointedExperimentReturnValue:
        logger = logging.getLogger(self.logger)
        logger.debug(f"Budget Remaining: {self.planner.budget}")
        should_continue = self.planner.budget > 1
        out_specification = deepcopy(self.specification)
        out_specification["budget"] = self.used_budget

        mean, stdv = self.auv_data_model.query_many(self.grid.sensed_locations, return_std=True)

        rewards = list(sorted(map(np.average, self.planner.get_root_rewards()), reverse=True))
        highest_reward = rewards[0]
        second_highest_reward = rewards[1]
        auv_points = mean

        # Point selection
        self.curr_estimated_Q = np.quantile(auv_points, self.quantiles)
        logger.info(f"curr_estimated_Q: {self.curr_estimated_Q}")

        try:
            _, predicted_std = self.auv_data_model.query_many(self.next_sensed_locations)

            environment_reward = np.sum(
                calculate_reward(self.next_sensed_values.squeeze(), predicted_std, self.objective_function,
                                 self.quantiles, self.grid, self.auv_data_model, self.next_sensed_locations,
                                 self.auv_data_model.Xs, self.auv_data_model.Ys, subsample_amount=None))
        except AttributeError:
            logger.warning(f"couldn't calculate environment reward. this is okay if this is the 0th step")
            environment_reward = 0

        self.cumulative_environment_reward += environment_reward

        # logger.info("Completed: {stats}".format(stats=stats))
        return_dict = {"Xs": self.data_X,
                       "Ys": self.data_Y,
                       "means": mean,
                       "used_budget": self.used_budget,
                       "total_used_rollouts": self.used_rollouts,
                       "remaining_rollouts": self.remaining_rollouts,
                       "reward_gap": highest_reward - second_highest_reward,
                       "generator_calls": self.planner.get_generator_calls_from_last_search(),
                       "reward": highest_reward,
                       "environment_reward": environment_reward,
                       "cumulative_environment_reward": self.cumulative_environment_reward,
                       "pomcp_traj_length": len(self.current_traj),
                       "estimated_quantiles": dict(zip(self.quantiles,
                                                       self.curr_estimated_Q)),
                       "robot_states": self.robot_states,
                       # Use dict so we don't have to have this class acccesible to load it.
                       "parameters_for_pomcp": dict() if self.last_parameter_control_object is None else self.last_parameter_control_object.__dict__,
                       }
        progress, outof = self.calculate_progress()
        return_value = OverlappingOutputCheckpointedExperimentReturnValue(should_continue, out_specification,
                                                                          return_dict, progress, outof)
        return return_value

    def step(self) -> typing.Union[ExpProgressTuple, OverlappingOutputCheckpointedExperimentReturnValue]:
        if self.saved_before_first_action == False:
            self.saved_before_first_action = True
            return self.calculate_return()

        logger = logging.getLogger(self.get_logger_name())
        logger.debug(f"Remaining Budget: {self.planner.budget} / {self.max_budget}")
        logger.info(f"Current Traj Length: {len(self.current_traj)}")
        self.budgets_that_were_used.append(self.planner.budget)
        if self.current_traj == []:
            parameterization_for_policy = self.parameter_policy.get_parameterization()
            metadata = IPPMetaData(remaining_budget=self.planner.budget, remaining_rollouts=self.remaining_rollouts,
                                   total_budget=self.max_budget, total_rollouts=self.total_rolluts,
                                   objective_function=self.objective_function)
            if isinstance(self.parameter_policy, BasePomcpParameterPolicy):
                self.last_parameter_control_object = self.parameter_policy.apply_policy(parameterization_for_policy,
                                                                                        self.auv_location,
                                                                                        self.auv_data_model,
                                                                                        self.environment, self.grid,
                                                                                        metadata, self.sensor,
                                                                                        self.planner, self.logger)

                self.current_traj = self.planner.next_step(self.auv_location, self.auv_data_model, self.environment,
                                                           self.grid, self.sensor,
                                                           max_generator_calls=self.max_generator_calls)

                used_rollouts_this_iteration = self.planner.cur_plan.rollouts_used_this_iteration
            elif isinstance(self.parameter_policy, BaseActionPolicy):
                self.last_parameter_control_object = None
                self.current_traj = self.parameter_policy.apply_policy(parameterization_for_policy, self.auv_location,
                                                                       self.auv_data_model, self.environment, self.grid,
                                                                       metadata, self.sensor, self.logger)
                used_rollouts_this_iteration = 0
            else:
                raise Exception("Unknown parameter policy base class for " + str(self.parameter_policy))
            self.used_rollouts += used_rollouts_this_iteration
            self.remaining_rollouts -= used_rollouts_this_iteration
        next_step = self.current_traj.pop(0)

        while not self.environment.is_inside(next_step):
            logger.warning("Bad next step, taking random step")
            action = random.choice([e.value for e in action_enum(self.environment.action_model())])
            next_step = self.grid.get_next_state_applying_action(self.auv_location, action)

        lat, lon = utm.to_latlon(next_step[0], next_step[1],
                                 self.zone_number, self.zone_letter)
        waypoint = {"lat": lat, "lon": lon, "height": self.sensor.altitude_meters}
        w_json = APIUtils.post_waypoint_json(waypoint, api_url=self.api_url)
        assert APIUtils.request_succeeded(w_json)
        # Now wait to receive next robot state and image
        next_id = w_json['id'] + 1
        r_json = APIUtils.get_robotstate_json(next_id, api_url=self.api_url)
        logger.debug(f"Attempting to get RobotState {next_id}...")
        while not APIUtils.request_succeeded(r_json):
            time.sleep(0.5)
            r_json = APIUtils.get_robotstate_json(id=next_id, api_url=self.api_url)
        x, y, _, _ = utm.from_latlon(r_json['lat'], r_json['lon'])

        i_json = APIUtils.get_image_json(next_id, api_url=self.api_url)
        logger.debug(f"Attempting to get Image {next_id}...")
        while not APIUtils.request_succeeded(i_json):
            time.sleep(0.5)
            i_json = APIUtils.get_image_json(id=next_id, api_url=self.api_url)
        time.sleep(1) # Just in case image hasn't finished loading on disk

        next_view, next_samples = self.get_view_and_samples_from_robotstate_image(r_json, i_json)

        #next_sensed_values = self.environment.sample(next_sensed_locations)
        next_sensed_locations = next_view
        self.next_sensed_locations = next_view
        next_sensed_values = next_samples

        self.next_sensed_values = next_samples

        self.auv_location = next_step

        try:
            logger.debug(
                "Sampled Data Stats: MAX: {} MIN: {} STD: {} MEAN: {} ".format(np.max(next_sensed_values),
                                                                               np.min(next_sensed_values),
                                                                               np.std(next_sensed_values),
                                                                               np.mean(next_sensed_values)))
        except:
            pass  # this can happen if the drone goes outside the bounds

        self.auv_data_model.update(next_sensed_locations, next_sensed_values)

        self.robot_states.append(self.auv_location)
        self.data_X.append(next_sensed_locations)
        self.data_Y.append(next_sensed_values)
        self.planner.budget -= 1
        self.used_budget += 1
        logger.debug(f"Remaining Budget: {self.planner.budget}")

        return self.calculate_return()

    def steps_before_checkpoint(self):
        # return int(self.max_budget / 8)
        return 10000000000000

    def max_iterations(self, specification):
        return specification["planning_steps"]

    def format_name(self, specification):
        name = dict()
        name["policy"] = specification["parameter_policy"]
        name["quantiles"] = specification["quantiles"]
        name["fn"] = specification["objective_function"]
        name["seed"] = specification["seed"]
        name["hash"] = str(hash(json.dumps(specification)))
        return name

    def get_name(self, specification):
        return dict2name(self.format_name(specification))

    def get_current_name(self, specification):
        name = self.format_name(specification)
        name["budget"] = self.used_budget
        return dict2name(name)

    def get_view_and_samples_from_robotstate_image(self, r_json, i_json):
        lat, lon, height = r_json['lat'], r_json['lon'], r_json['height']
        x, y, _, _ = utm.from_latlon(lat, lon)
        view = self.sensor.get_sensed_locations_from_point([x, y, height])
        print(view)
        width, height = self.sensor.pixels_x, self.sensor.pixels_y
        img = PIL.Image.open(
            self.images_folder + i_json['filename'].split("/")[-1]).resize((width, height),

                resample=PIL.Image.LANCZOS)
        assert img.mode == "RGB"

        img_array = np.asarray(img)
        img_array = img_array.reshape(-1, 3)[:, 2] #TODO parameterize which channel you choose, default to green

        return view, img_array