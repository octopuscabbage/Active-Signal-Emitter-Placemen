import json
import logging
import os
import random
import time
import typing
from copy import deepcopy

import numpy as np
from smallab.experiment_types.overlapping_output_experiment import (OverlappingOutputCheckpointedExperimentReturnValue,
                                                                    OverlappingOutputCheckpointedExperiment)
from smallab.smallab_types import Specification, ExpProgressTuple
from lighting_placement.algorithm import BOLightingOptimizer, CMAESLightingOptimizer, ScipyLightingOptimizer, get_first_light_placement
from lighting_placement.monte_carlo import compute_mc_variance
from lighting_placement.triggers import compute_logprob, every_n_trigger, logprob_fraction_trigger, logprob_percent_trigger

from sample_sim.action.grid import generate_finite_grid, tuple_to_max_precision
from sample_sim.data_model.additive_lighting_model import AdditiveLightingModel
from sample_sim.data_model.conditional_factor_graph import ConditionalFactorGraphDataModel
from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.data_model.factor_graph import FactorGraphDataModel
from sample_sim.environments.ecomapper_csv import load_from_ecomapper_data
from sample_sim.environments.lighting.from_lights import FromLightLightingComputer
from sample_sim.environments.lighting_scene import AdditiveLightingScene2d, create_sdf_environment
from sample_sim.environments.raster_environment import load_from_egret_data
from sample_sim.planning.pilot_surveys import cross_survey, quantize_to_sensed_locations
from sample_sim.planning.pomcpow.pomcpow import HashableNumpyArray, RolloutStrategy
from sample_sim.planning.pomcpow.pomcpow_planner import POMCPOWPlanner, ObservationSamplingStrategy, PomcpowExtraData
from sample_sim.sensors.base_sensor import PointSensor
from sample_sim.sensors.camera_sensor import FixedHeightCamera
from smallab.name_helper.dict import dict2name
from sample_sim.planning.reward_functions import calculate_reward
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


class PlanningExperiment(OverlappingOutputCheckpointedExperiment):
    def initialize(self, specification: Specification):
        self.saved_before_first_action = False
        logger = logging.getLogger(self.get_logger_name())
        logger.setLevel(logging.DEBUG)
        self.specification = specification
        logger.info(f"Begin {specification}")

        self.sensor = PointSensor()

        self.plot = specification["plot"]
        self.seed = specification["seed"]
        self.objective_c = specification["objective_c"]

        self.state_space_dimensionality = specification["state_space_dimensionality"]
        self.physical_step_size = specification["physical_step_size"]

        # self.state_space_dimensionality[2]
        self.max_budget = specification["planning_steps"]
        self.rollouts_per_step = specification["rollouts_per_step"]
        self.do_pilot_survey = specification["do_pilot_survey"] and specification["pilot_survey_len"] != 0
        self.pilot_survey_len = specification["pilot_survey_len"]
        self.objective_function = specification["objective_function"]

        number_of_edge_samples = specification["number_of_edge_samples"]

        self.planner_gamma = specification["planner_gamma"]
        self.max_planner_depth = specification["max_planner_rollout_depth"]

        self.max_generator_calls = None

        self.ambient_light_brightness = specification["ambient_light_brightness"]
        self.placed_light_brightness = specification["placed_light_brightness"]
        self.desired_light_brightness = specification["desired_light_brightness"]
        self.raytracing_steps = specification["ray_steps"]
        self.ground_truth_reflections = specification["gt_reflections"]
        self.model_reflections = specification["model_reflections"]

        xmax = self.state_space_dimensionality[0] * self.physical_step_size
        ymax = self.state_space_dimensionality[1] * self.physical_step_size

        self.target_environment = create_sdf_environment(
            specification["environment_seed"],specification["environment_seed"], generator_name=specification["target_sdf_environment"], 
            x_size=xmax, y_size=ymax, preset_light_intensities=self.desired_light_brightness,
            num_lights=specification["target_lights"])
        self.target_environment.lighting_computer.set_max_reflections(self.ground_truth_reflections)
        self.ambient_light_environment = create_sdf_environment(
            specification["environment_seed"], specification["seed"],generator_name=specification["ground_truth_sdf_environment"], 
            x_size=xmax, y_size=ymax, preset_light_intensities=self.ambient_light_brightness,
            num_lights=specification["ambient_lights"])
        self.ambient_light_environment.lighting_computer.set_max_reflections( self.ground_truth_reflections)
        self.gt_grid = generate_finite_grid(self.target_environment, tuple(self.state_space_dimensionality), self.sensor,
                                            number_of_edge_samples)

        self.grid = generate_finite_grid(self.ambient_light_environment, tuple(self.state_space_dimensionality), self.sensor,
                                         number_of_edge_samples)
        self.target_sensed_points = self.target_environment.sample(
            self.grid.get_sensed_locations())
        self.target_sensed_points_gt = self.target_environment.sample(
            self.gt_grid.get_sensed_locations())
        
        self.target_environment.bake(self.grid.sensed_locations)
        self.ambient_light_environment.bake(self.grid.sensed_locations)



        # TODO make good repr for these
        logger.info(f"Environment: {self.ambient_light_environment}")
        logger.info(f"Grid: {self.grid}")

        self.current_traj = []
        self.data_X = []
        self.data_Y = []
        self.noises = []
        self.budgets_that_were_used = []
        self.used_budget = 0
        self.used_rollouts = 0

        #self.measurement_noise = specification["measurement_noise"]

        if specification["observation_sampling_strategy"] == "mean":
            self.observation_sampling_strategy = ObservationSamplingStrategy.MEAN
        elif specification["observation_sampling_strategy"] == "random_normal":
            self.observation_sampling_strategy = ObservationSamplingStrategy.RANDOM_NORMAL
        elif specification["observation_sampling_strategy"] == "confidence_interval":
            self.observation_sampling_strategy = ObservationSamplingStrategy.CONFIDENCE_INTERVALS
        else:
            raise Exception()
        
        if specification["rollout_strategy"] == "random":
            self.rollout_strategy = RolloutStrategy.RANDON
        elif specification["rollout_strategy"] == "random_weighted":
            self.rollout_strategy = RolloutStrategy.REWARD_WEIGHTED

        if self.plot:
            fignum = 111
        else:
            fignum = None
        self.planner = POMCPOWPlanner(self.get_logger_name(
        ), self.objective_function, self.observation_sampling_strategy,fignum=fignum)
        self.planner.set_parameters(rollouts_per_step=self.rollouts_per_step, objective_c=self.objective_c,
                                    max_planning_depth=self.max_planner_depth, gamma=self.planner_gamma)

        logger.debug("creating auv data model")

        self.auv_location = self.ambient_light_environment.get_starting_location(specification["seed"])

        initial_locations = self.sensor.get_sensed_locations_from_point(
            self.auv_location)
        # if not isinstance(initial_locations, np.ndarray):
        #     initial_locations_np = np.array([initial_locations])
        # else:
        #     initial_locations_np = initial_locations
        logger.debug(f"initial locations shape {initial_locations.shape}")
        initial_samples = self.ambient_light_environment.sample(
            np.array(initial_locations))

        sensed_value_linkage = specification["fg.sensed_value_linkage"]
        minimum_distance_sigma = specification["fg.minimum_distance_sigma"]
        distance_sigma_scaling = specification["fg.distance_sigma_scaling"]
        measurement_uncertainty = specification["fg.measurement_uncertainty"]
        lighting_model_uncertainty = specification["fg.lighting_model_uncertainty"]

        if specification["data_model"] == "additive_gaussian_process":
            base_data_model = TorchExactGPBackedDataModel(initial_locations, initial_samples,
                                                          logger=self.logger)
            base_data_model.model.model.covar_module.base_kernel.lengthscale = specification[
                "gp.lengthscale"]
            base_data_model.model.eval_model()
            self.auv_data_model = AdditiveLightingModel(
                base_data_model, self.logger)
        elif specification["data_model"] == "conditional_factor_graph":
            self.auv_data_model =  ConditionalFactorGraphDataModel(self.grid, self.logger, sensed_value_linkage=sensed_value_linkage, minimum_distance_sigma=minimum_distance_sigma,
                                                   distance_sigma_scaling=distance_sigma_scaling, measurement_uncertainty=measurement_uncertainty, lighting_model_uncertainty=lighting_model_uncertainty,
                                                   residual_prior_uncertainty=specification["fgc.residual_prior_uncertainty"])

            
        elif specification["data_model"] == "additive_factor_graph":
            base_data_model = FactorGraphDataModel(self.grid, self.logger, sensed_value_linkage=sensed_value_linkage, minimum_distance_sigma=minimum_distance_sigma,
                                                   distance_sigma_scaling=distance_sigma_scaling, measurement_uncertainty=measurement_uncertainty, lighting_model_uncertainty=lighting_model_uncertainty)
            self.auv_data_model = AdditiveLightingModel(
                base_data_model, self.logger)
        else:
            raise Exception()

       

        self.robot_states = [self.auv_location]

        if specification["data_model"] == "gaussian_process":
            self.auv_data_model.model.model.mean_module.mean = np.mean(
                self.auv_data_model.Ys)

        self.ambient_light_prediction_errors = list()
        self.gt_prediction_errors = list()
        self.desired_prediction_error = []

        self.cumulative_environment_reward = 0

        self.num_lights_to_place = specification["num_lights_to_place"]
        self.first_placed_lights = False
        self.trigger = specification["lighting_trigger"]
        self.replaced_lights = True
        self.overall_errors = []

        self.cem_iterations = specification["cem.iterations"]
        self.cem_population = specification["cem.population"]
        self.cem_alpha = specification["cem.alpha"]
        self.cem_objective = specification["cem.objective"]
        self.lighting_placement = None
        self.logprobs = []
        self.placement_iters = []

        self.light_placement_variance = specification["light_placement_variance"]

        self.light_placement_mc_iters = specification["light_placement_mc_iters"]
        self.lighting_model = FromLightLightingComputer(self.ambient_light_environment.workspace.sdf_fn,self.physical_step_size / (number_of_edge_samples+1),self.raytracing_steps,self.model_reflections)
        if specification["cem.optimizer"] == "cmaes":
            self.lighting_optimizer = CMAESLightingOptimizer(logger_name=self.logger,num_lights=self.num_lights_to_place,sensed_locations=self.grid.sensed_locations, 
                                                        desired_lighting=self.target_sensed_points, lighting_model=self.lighting_model,num_iters=self.cem_iterations,population_size=self.cem_population,
                                                        workspace=self.ambient_light_environment.workspace,objective=self.cem_objective,lighting_upper_bound=self.placed_light_brightness)
        elif specification["cem.optimizer"] == "bo":
            self.lighting_optimizer = BOLightingOptimizer(logger_name=self.logger,num_lights=self.num_lights_to_place,sensed_locations=self.grid.sensed_locations, 
                                                        desired_lighting=self.target_sensed_points, lighting_model=self.lighting_model,num_iters=self.cem_iterations * self.cem_population,method_name=specification["cem.optimizer"],
                                                        workspace=self.ambient_light_environment.workspace,objective=self.cem_objective,lighting_upper_bound=self.placed_light_brightness)


        else:
            self.lighting_optimizer = ScipyLightingOptimizer(logger_name=self.logger,num_lights=self.num_lights_to_place,sensed_locations=self.grid.sensed_locations, 
                                                        desired_lighting=self.target_sensed_points, lighting_model=self.lighting_model,num_iters=self.cem_iterations * self.cem_population,method_name=specification["cem.optimizer"],
                                                        workspace=self.ambient_light_environment.workspace,objective=self.cem_objective,lighting_upper_bound=self.placed_light_brightness)


        if specification["place_at_beginning"]:
            
            self.lighting_placement, self.lighting_brightnesses, self.best_lighting = get_first_light_placement(logger_name="smallab",num_lights=self.num_lights_to_place,sensed_locations=HashableNumpyArray(self.grid.sensed_locations), 
                                                    desired_lighting=HashableNumpyArray(self.target_sensed_points), num_iters=self.cem_iterations,population_size=self.cem_population,
                                                    objective=self.cem_objective,lighting_upper_bound=self.placed_light_brightness,
                                                    environment_seed=specification["environment_seed"], generator_name=specification["ground_truth_sdf_environment"],x_size=xmax,y_size=ymax,physical_step_size=specification["physical_step_size"],number_of_edge_samples=specification["number_of_edge_samples"],
                                                    raytracing_steps=specification["ray_steps"],model_reflections=specification["model_reflections"],optimizer_name=specification["cem.optimizer"])
            self.lighting_optimizer.light_locations = self.lighting_placement
            self.lighting_optimizer.light_brightnesses = self.lighting_brightnesses
            if isinstance(self.lighting_optimizer,ScipyLightingOptimizer):
                last_sample = []
                for location, brightness in zip(self.lighting_placement,self.lighting_brightnesses):
                    last_sample.append(location[0])
                    last_sample.append(location[1])
                    last_sample.append(brightness)
                self.lighting_optimizer.last_results = np.array(last_sample)

            self.placed_light_environment = AdditiveLightingScene2d(
                self.ambient_light_environment.workspace, self.ambient_light_environment, self.lighting_placement, self.lighting_brightnesses)
            self.placed_light_environment.bake(self.grid.sensed_locations)
            self.first_placed_lights = True

        if specification["do_pilot_survey"] and self.pilot_survey_len != 0:
            # raw_grid = self.ambient_light_environment.meshgrid(
            #     self.pilot_survey_len)
            raw_points = cross_survey(self.pilot_survey_len,self.grid)
            
            inside_workspace_points = self.ambient_light_environment.filter_by_inside(raw_points,already_in_box_constraints=False)

            initial_locations = quantize_to_sensed_locations(
                inside_workspace_points, self.grid.sensed_locations)
            for point in initial_locations:
                self.robot_states.append(point)
            if self.first_placed_lights:
                initial_samples = self.placed_light_environment.sample(initial_locations)

            else:
                initial_samples = self.ambient_light_environment.sample(initial_locations)

            if not isinstance(self.auv_data_model,ConditionalFactorGraphDataModel):
                self.auv_data_model.update(initial_locations,initial_samples)
        
        else:
            if self.first_placed_lights:
                initial_samples = self.placed_light_environment.sample(
                            np.array(initial_locations))
            else:
                initial_samples = self.ambient_light_environment.sample(
                            np.array(initial_locations))
            if not isinstance(self.auv_data_model,ConditionalFactorGraphDataModel):
                self.auv_data_model.update(initial_locations,initial_samples)

        if specification["place_at_beginning"]:
            if self.light_placement_variance != 0.0:
                #lighting_variance = compute_mc_variance(self.lighting_placement,self.light_placement_mc_iters,self.light_placement_variance,self.grid.sensed_locations,self.ambient_light_environment.sdf_fn,self.placed_light_brightness,self.estimated_hardness)
                raise Exception()
            else:
                lighting_variance = np.zeros(self.best_lighting.shape)
            if isinstance(self.auv_data_model,ConditionalFactorGraphDataModel):
                self.auv_data_model.on_replacing_lights_first(self.grid.sensed_locations,self.best_lighting,lighting_variance, 
                                    initial_locations, initial_samples)
            else:
                self.auv_data_model.on_replacing_lights(
                    self.grid.sensed_locations, self.best_lighting, lighting_variance)
            #self.auv_data_model.age_measurements()
        #for i in range(10):
        if isinstance(self.auv_data_model,ConditionalFactorGraphDataModel):
            self.auv_data_model.update(initial_locations,initial_samples)


        #    self.auv_data_model.isam.update()
    def calculate_progress(self) -> ExpProgressTuple:
        return self.used_budget, self.max_budget

    def calculate_return(self) -> OverlappingOutputCheckpointedExperimentReturnValue:
        logger = logging.getLogger(self.logger)
        logger.debug(f"Budget Remaining: {self.max_budget - self.used_budget}")
        should_continue = self.used_budget <= self.max_budget
        out_specification = deepcopy(self.specification)
        out_specification["budget"] = self.used_budget

        
        if self.first_placed_lights:
            gt_lighting_rmse = np.sqrt(mean_squared_error(self.target_sensed_points, self.placed_light_environment.sample(self.grid.get_sensed_locations())))
        else:
            gt_lighting_rmse = np.sqrt(mean_squared_error(self.target_sensed_points, self.ambient_light_environment.sample(self.grid.get_sensed_locations())))

        predictive_error = np.sqrt(mean_squared_error(self.mean, self.target_sensed_points))

        self.auv_data_model.clear_prior()
        #Need to append twice for timing to make sense in plots
        if self.overall_errors == []:
            self.overall_errors.append(gt_lighting_rmse)
            self.ambient_light_prediction_errors.append(np.sqrt(mean_squared_error(self.auv_data_model.predict_ambient(self.grid.get_sensed_locations()), self.ambient_light_environment.sample(self.grid.get_sensed_locations()))))
            self.gt_prediction_errors.append(np.sqrt(mean_squared_error(self.auv_data_model.query_many(self.grid.get_sensed_locations(),return_std=False), self.placed_light_environment.sample(self.grid.get_sensed_locations()))))
        self.ambient_light_prediction_errors.append(np.sqrt(mean_squared_error(self.auv_data_model.predict_ambient(self.grid.get_sensed_locations()), self.ambient_light_environment.sample(self.grid.get_sensed_locations()))))
        
        mean = self.auv_data_model.query_many(self.grid.get_sensed_locations(),return_std=False)
        self.gt_prediction_errors.append(np.sqrt(mean_squared_error(mean, self.placed_light_environment.sample(self.grid.get_sensed_locations()))))

        self.desired_prediction_error.append(np.sqrt(mean_squared_error(mean, self.target_sensed_points)))

        self.overall_errors.append(gt_lighting_rmse)

        try:
            _, predicted_std = self.auv_data_model.query_many(
                self.next_sensed_locations)

            environment_reward = np.sum(calculate_reward(self.next_sensed_values.squeeze(), predicted_std, self.objective_function,
                                                         self.grid, self.auv_data_model, self.next_sensed_locations, self.auv_data_model.Xs, self.auv_data_model.Ys, subsample_amount=None,desired_lighting_scene=self.target_environment))
        except AttributeError:
            logger.warning(
                f"couldn't calculate environment reward. this is okay if this is the 0th step")
            environment_reward = 0

        self.cumulative_environment_reward += environment_reward

        # TODO log desired lighting and current lighting
        return_dict = {"Xs": self.data_X,
                       "Ys": self.data_Y,
                       "means": self.mean,
                       "used_budget": self.used_budget,
                       "total_used_rollouts": self.used_rollouts,
                       "environment_reward": environment_reward,
                       "cumulative_environment_reward": self.cumulative_environment_reward,
                       "pomcp_traj_length": len(self.current_traj),
                       "robot_states": self.robot_states,
                       "gt_lighting_rmse": gt_lighting_rmse,
                       "replaced_lights": self.replaced_lights,
                       "lighting_placement": self.lighting_placement,
                       "lighting_brightnesses": self.lighting_brightnesses,
                       "logprob": self.logprob,
                       "predictive_error": predictive_error,
                       #"ambient_light_prediction_error": self.ambient_light_prediction_errors[-1]
                       }
        if isinstance(self.auv_data_model,AdditiveLightingModel):
            return_dict["ambient_light_prediction_error"] =  self.ambient_light_prediction_errors[-1]

        progress, outof = self.calculate_progress()
        should_serialize = self.replaced_lights or not should_continue
        return_value = OverlappingOutputCheckpointedExperimentReturnValue(should_continue, out_specification,
                                                                          return_dict, progress, outof,should_serialize=should_serialize)
        return return_value

    def step(self) -> typing.Union[ExpProgressTuple, OverlappingOutputCheckpointedExperimentReturnValue]:
        if self.saved_before_first_action == False:
            self.saved_before_first_action = True
            self.auv_data_model.clear_prior()
            self.mean, self.std = self.auv_data_model.query_many(self.grid.sensed_locations, return_std=True)
            self.logprob = compute_logprob(self.target_sensed_points,self.mean,self.std)
            self.logprobs.append(self.logprob)
            self.previous_replan_logprob = self.logprob
            self.starting_logprob = self.logprob
            if self.plot:
                self.draw_matplotlib()
            return self.calculate_return()

        logger = logging.getLogger(self.get_logger_name())
        logger.debug(
            f"Remaining Budget: {self.max_budget - self.used_budget} / {self.max_budget}")
        logger.info(f"Current Traj Length: {len(self.current_traj)}")
        self.budgets_that_were_used.append(self.used_budget)

        pomcpow_extra_data = PomcpowExtraData(data_model=self.auv_data_model, objective_function=self.objective_function,
                                              environment=self.ambient_light_environment, sensor=self.sensor, grid=self.grid, 
                                              observation_sampling_strategy=self.observation_sampling_strategy,base_gp=None,
                                              desired_environment=self.target_environment,rollout_strategy=self.rollout_strategy)
        self.planner.set_extra_data(pomcpow_extra_data)
        self.current_traj = self.planner.next_step(self.auv_location, self.auv_data_model, self.ambient_light_environment,
                                                   self.grid, self.sensor)

        next_step = self.current_traj.pop(0)

        good_next_step = False
        while not good_next_step:
            try:
                next_sensed_locations = self.grid.get_samples_traveling_from(
                    self.auv_location, next_step, self.sensor)
                self.next_sensed_locations = next_sensed_locations
                good_next_step = True
            except Exception:
                logger.error("Taking a random step", exc_info=True)
                next_step = random.choice(
                    list(self.grid.edge_samples[tuple_to_max_precision(self.auv_location)].keys()))
        if self.first_placed_lights:
            next_sensed_values = self.placed_light_environment.sample(
                next_sensed_locations)
        else:
            next_sensed_values = self.ambient_light_environment.sample(
                next_sensed_locations)

        #next_sensed_values += np.random.normal(0,self.measurement_noise,next_sensed_values.shape)
        self.next_sensed_values = next_sensed_values


        self.auv_location = next_step

        try:
            logger.debug(
                "Sampled Data Stats: MAX: {} MIN: {} STD: {} MEAN: {} ".format(np.max(next_sensed_values),
                                                                               np.min(
                                                                                   next_sensed_values),
                                                                               np.std(
                                                                                   next_sensed_values),
                                                                               np.mean(next_sensed_values)))
        except:
            pass  # this can happen if the drone goes outside the bounds

        self.auv_data_model.update(next_sensed_locations, next_sensed_values)

        self.robot_states.append(self.auv_location)
        self.data_X.append(next_sensed_locations)
        self.data_Y.append(next_sensed_values)
        self.used_budget += 1
        logger.debug(f"Remaining Budget: {self.max_budget - self.used_budget}")

        self.auv_data_model.clear_prior()
        self.mean, self.std = self.auv_data_model.query_many(
            self.grid.sensed_locations, return_std=True)
        self.logprob = compute_logprob(self.target_sensed_points,self.mean,self.std)
        if self.logprob > self.previous_replan_logprob:
            self.previous_replan_logprob = self.logprob
        self.logprobs.append(self.logprob)

        if self.should_place_lights():
            self.lighting_placement, self.lighting_brightnesses, self.best_lighting = self.lighting_optimizer.minimize_lighting(self.auv_data_model,True)

            self.placed_light_environment = AdditiveLightingScene2d(
                self.ambient_light_environment.workspace, self.ambient_light_environment, self.lighting_placement, self.lighting_brightnesses)
            self.placed_light_environment.bake(self.grid.sensed_locations)

            if self.light_placement_variance != 0.0:
                lighting_variance = compute_mc_variance(self.lighting_placement,self.light_placement_mc_iters,self.light_placement_variance,self.grid.sensed_locations,self.ambient_light_environment.sdf_fn,self.placed_light_brightness,self.estimated_hardness)
            else:
                lighting_variance = np.zeros(self.best_lighting.shape)
            self.auv_data_model.on_replacing_lights(
                self.grid.sensed_locations, self.best_lighting, lighting_variance)
            #self.auv_data_model.age_measurements()
            self.first_placed_lights=True
            self.replaced_lights=True
            self.auv_data_model.clear_prior()
            self.mean, self.std = self.auv_data_model.query_many(
            self.grid.sensed_locations, return_std=True)
            self.previous_replan_logprob = compute_logprob(self.target_sensed_points,self.mean,self.std)
            self.placement_iters.append(self.used_budget)
        else:
            self.replaced_lights=False
        if self.plot:
            self.draw_matplotlib()

        return self.calculate_return()

    def steps_before_checkpoint(self):
        #return int(self.max_budget / 8)
        return 10000000000000

    def max_iterations(self, specification):
        return specification["planning_steps"]

    def format_name(self, specification):
        name=dict()
        #name["sdf_env"]=specification["ground_truth_sdf_environment"]
        name["objective"] = str(specification["objective_function"]).replace(",","")
        name["model"]=specification["data_model"]
        name["seed"]=specification["seed"]
        name["environment_seed"]=specification["environment_seed"]
        name["trigger"]=str(specification["lighting_trigger"]).replace(",","")
        name["hash"]=str(hash(json.dumps(specification)))
        return name

    def get_name(self, specification):
        return dict2name(self.format_name(specification))

    def get_current_name(self, specification):
        name=self.format_name(specification)
        name["budget"]=self.used_budget
        return dict2name(name)

    def should_place_lights(self):
        if self.trigger == "every":
            return True
        elif self.trigger[0] == "every_n":
            return every_n_trigger(self.trigger[1], self.used_budget)
        elif self.trigger[0] == "every":
            return True
        elif self.trigger[0] == "last":
            return self.used_budget == self.max_budget
        elif self.trigger[0] == "logprob_percent":
            return logprob_percent_trigger(self.trigger[1], self.previous_replan_logprob,self.logprob)
        elif self.trigger[0] == "logprob_fraction":
            return logprob_fraction_trigger(self.trigger[1], self.previous_replan_logprob,self.logprob)
        elif self.trigger[0] == "wait_until_percent": 
            if self.used_budget / self.max_budget > self.trigger[1]:
                self.trigger = self.trigger[2]
                return True
            else:
                return False


        else:
            raise Exception()

    def draw_matplotlib(self):
        plt.ion()
        #self.auv_data_model.save_graphviz(self.used_budget)
        
        plt.figure(1)
        plt.clf()
        plt.title("Desired Lighting")
        max=np.max(self.target_sensed_points)
        min=np.min(self.target_sensed_points)
        sensed_locations=self.gt_grid.get_sensed_locations()
        a=plt.scatter(
            x = sensed_locations[:, 0], y = sensed_locations[:, 1], c = self.target_sensed_points_gt)
        plt.colorbar()
        plt.figure(2)
        plt.clf()
        plt.title("Ambient Lighting")
        sensed_locations=self.grid.get_sensed_locations()
        ambient_lighting=self.ambient_light_environment.sample(
            self.grid.get_sensed_locations())
        a=plt.scatter(
            x = sensed_locations[:, 0], y = sensed_locations[:, 1], c = ambient_lighting)
        plt.colorbar(a)

        plt.figure(44)
        plt.clf()
        plt.title("Ambient Prediction Error")
        plt.plot(self.ambient_light_prediction_errors)
        plt.gca().ticklabel_format(useOffset=False)

        plt.figure(45)
        plt.clf()
        plt.title("Error between Prediction and real life")
        plt.plot(self.gt_prediction_errors)
        plt.gca().ticklabel_format(useOffset=False)

        plt.figure(46)
        plt.clf()
        plt.title("Error between Prediction and desired")
        plt.plot(self.desired_prediction_error)
        plt.gca().ticklabel_format(useOffset=False)

            

        plt.figure(3)
        plt.clf()
        plt.title("Mean and Location")
        self.auv_data_model.clear_prior()
        mean, std=self.auv_data_model.query_many(
            self.grid.sensed_locations, return_std = True)
        a = plt.scatter(
            sensed_locations[:, 0], sensed_locations[:, 1], c = mean)
        plt.colorbar(a)
        np_robot_states=np.array(self.robot_states)
        for s0, s1 in zip(np_robot_states[:, :], np_robot_states[1:, :]):
             plt.arrow(s0[0], s0[1], s1[0]-s0[0], s1[1]-s0[1], length_includes_head = True,
                       fc = "r", ec = "r", head_width = 1.0, head_length = 1.0)
        plt.scatter([np_robot_states[-1,0]],[np_robot_states[-1,1]],c="r",marker="x")
        # TODO plot sensed locations and values
        plt.figure(4)
        plt.clf()
        plt.title("Std and location")
        #ax = plt.axes(projection ="3d")
        a = plt.scatter( sensed_locations[:, 0],
                    sensed_locations[:, 1], c=std)
        plt.colorbar(a)
        plt.figure(444)
        plt.clf()
        plt.title("logStd and location")
        #ax = plt.axes(projection ="3d")
        a = plt.scatter( sensed_locations[:, 0],
                    sensed_locations[:, 1], c=np.log(std))
        plt.colorbar(a)
        #ax.scatter(np_robot_states[:,0],np_robot_states[:,1],np.zeros(np_robot_states.shape[1]) + 5,c="r")
        #
        # for s0, s1 in zip(np_robot_states[:, :], np_robot_states[1:, :]):
        #     plt.arrow(s0[0], s0[1], s1[0]-s0[0], s1[1]-s0[1], length_includes_head = True,
        #               fc = "r", ec = "r", head_width = 0.05, head_length =0.1)
        plt.figure(11)
        plt.clf()
        plt.title("logprob")
        plt.plot(self.logprobs)
        plt.gca().ticklabel_format(useOffset=False)

        if self.placement_iters != []:
            for iter in self.placement_iters:
                plt.axvline(iter)
        if self.trigger[0] == "logprob_fraction":
            fractional_amount = self.previous_replan_logprob * self.trigger[1]
            plt.axhline(fractional_amount)



        if self.first_placed_lights:
            plt.figure(5)
            plt.clf()
            plt.title("Current Environment")
            sensed_locations=self.grid.get_sensed_locations()
            placed_environment=self.placed_light_environment.sample(
                self.grid.get_sensed_locations())
            a = plt.scatter(
                x=sensed_locations[:, 0], y=sensed_locations[:, 1], c=placed_environment)
            plt.colorbar(a)
            plt.scatter(self.lighting_placement[:, 0],
                        self.lighting_placement[:, 1], marker="x", c="r")
            
            plt.figure(55)
            plt.clf()
            plt.title("Error between prediction and environment")
            sensed_locations=self.grid.get_sensed_locations()
            placed_environment=self.placed_light_environment.sample(
                self.grid.get_sensed_locations())
            a = plt.scatter(
                x=sensed_locations[:, 0], y=sensed_locations[:, 1], c=placed_environment - mean)
            plt.colorbar(a)
            plt.scatter(self.lighting_placement[:, 0],
                        self.lighting_placement[:, 1], marker="x", c="r")
            

            plt.figure(6)
            plt.clf()
            plt.title("Error between Environment and Desired")
            error = self.target_sensed_points - placed_environment
            a = plt.scatter(
                x=sensed_locations[:, 0], y=sensed_locations[:, 1], c=error)
            plt.colorbar(a)
            plt.scatter(self.lighting_placement[:, 0],
                        self.lighting_placement[:, 1], marker="x", c="r")

            plt.figure(7)
            plt.clf()
            plt.title("RMSE Between placed and desired")
            plt.plot(self.overall_errors)
            plt.gca().ticklabel_format(useOffset=False)
            for iter in self.placement_iters:
                plt.axvline(iter)


            plt.figure(8)
            plt.clf()
            plt.title("Error between Predicted and Desired")
            error = self.target_sensed_points - mean
            a = plt.scatter(
                x=sensed_locations[:, 0], y=sensed_locations[:, 1], c=error)
            plt.colorbar(a)
            plt.scatter(self.lighting_placement[:, 0],
                        self.lighting_placement[:, 1], marker="x", c="r")
            for s0, s1 in zip(np_robot_states[:, :], np_robot_states[1:, :]):
             plt.arrow(s0[0], s0[1], s1[0]-s0[0], s1[1]-s0[1], length_includes_head = True,
                       fc = "r", ec = "r", head_width = 1.0, head_length = 1.0)

            plt.figure(9)
            plt.clf()
            plt.title("Predicted Ambient Lighting")

            #ambient = mean - self.best_lighting
            a = plt.scatter(
                x=sensed_locations[:, 0], y=sensed_locations[:, 1], c=self.auv_data_model.predict_ambient(sensed_locations))
            plt.colorbar(a)

            # plt.figure(99)
            # plt.clf()
            # plt.title("Predicted Analytical Lighting")

            # #ambient = mean - self.best_lighting
            # a = plt.scatter(
            #     x=sensed_locations[:, 0], y=sensed_locations[:, 1], c=self.auv_data_model.predict_analytical(sensed_locations))
            # plt.colorbar(a)

            plt.figure(10)
            plt.clf()
            plt.title("Error in Ambient Lighting")
            ambient_error = ambient_lighting - (mean - self.best_lighting)
            a = plt.scatter(
                x=sensed_locations[:, 0], y=sensed_locations[:, 1], c=ambient_error)
            for s0, s1 in zip(np_robot_states[:, :], np_robot_states[1:, :]):
             plt.arrow(s0[0], s0[1], s1[0]-s0[0], s1[1]-s0[1], length_includes_head = True,
                       fc = "r", ec = "r", head_width = 1.0, head_length = 1.0)
            plt.colorbar(a)

        plt.pause(5.0)
        print()
