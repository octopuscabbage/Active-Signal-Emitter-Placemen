import datetime
import logging
import math
import os
import sys
from copy import deepcopy
from itertools import product
from os import mkdir
from os.path import exists, abspath, join, dirname

import numpy as np
from smallab.runner.runner import ExperimentRunner
from smallab.runner_implementations.main_process_runner import MainRunner
from smallab.runner_implementations.multiprocessing_runner import MultiprocessingRunner
from smallab.specification_generator import SpecificationGenerator
import random

from experiments.plannin_experiment import PlanningExperiment

DEBUG = False
if not DEBUG:
    import ray
    from my_multiprocessing_runner import RayMultiprocessingRunner
    ray.init()

    print(sys.argv)
    name = "test_system_extended_new_intensity"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import warnings
    warnings.filterwarnings("error")
    import time
    name = str(hash(time.time()))

import pprint
#from create_cache import create_ecomapper_cache, create_drone_cache
from sample_sim.sensors.camera_sensor import FixedHeightCamera
from sample_sim.sensors.base_sensor import PointSensor


np.set_printoptions(threshold=5)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MP_NUM_THREADS"] = "1"
#:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.getLogger("smallab").propogate = False



if "experiments" in os.getcwd():
    os.chdir("../..")

this_dir = dirname(abspath(__file__))
for dir_name in ('.cache', '.params'):
    path = join(this_dir, dir_name)
    if not exists(path):
        mkdir(path)
non_experiment_logger = logging.getLogger("default")
non_experiment_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
handler.setFormatter(formatter)
non_experiment_logger.addHandler(handler)

if len(sys.argv) == 1:
    seeds = [0]
else:
    seeds = list(map(int, sys.argv[1:]))


specifications = []
base_specs = {
    "plot": [False],
    "seed": list(range(15)),

    "environment_seed": [1,2,5,6,7],
    ## ENVIRONMENT
    "num_lights_to_place": 7,
    "ambient_light_brightness": .5,
    "placed_light_brightness": 1.0,
    "desired_light_brightness": 1.0,
    "target_lights": 3,
    "ambient_lights":5,
    "ray_steps": 30,
    "gt_reflections": 5,
    "model_reflections": 1,
    "target_sdf_environment": [["free_space"]],
    "ground_truth_sdf_environment": [["rectangles",8]],
    "do_pilot_survey": [False],
    "pilot_survey_len": 20,
    "number_of_edge_samples": 0,
    "state_space_dimensionality": [[13,13]],
    "physical_step_size": 5,
    "place_at_beginning": [True],

    ## LIGHTING PLACEMENT ALGORITHM
    "cem.iterations": 200,
    "cem.population": 1,
    "cem.alpha":0.8,
    "cem.objective": ["rmse"],
    "cem.optimizer": ["Nelder-Mead"],

    ## DATA MODEL
    "age_measurements": [False],


    "gp.lengthscale": [15], #Equal to 3x physical step size

    "fg.sensed_value_linkage": 5,
    "fg.minimum_distance_sigma":0.01,
    "fg.distance_sigma_scaling":1, 
    "fg.measurement_uncertainty":0.1,
    "fg.lighting_model_uncertainty": 0.1,
    "fgc.residual_prior_uncertainty": 10,

     
    "light_placement_variance": [0],
    "light_placement_mc_iters": 30,

    ##PLANNING
    "objective_c": [1E-6],
    "planning_steps": [100],
    "rollouts_per_step": [50],
    "planner_gamma": [.95],
    "max_planner_rollout_depth": [15],

    "use_t_test": [True],
    "t_test_value": [0.3],
    "objective_function": ["entropy"],
    "observation_sampling_strategy": ["mean"],
    "rollout_strategy": "random",

}


for difficulty in ["easy","medium","hard"]:
    #easy is the default
    if difficulty == "medium":
        base_specs["num_lights_to_place"] = 15
        base_specs["ground_truth_sdf_environment"] = [["rectangles",12]]
        base_specs["ambient_lights"] = 15
        base_specs["target_lights"] =  9
    if difficulty == "hard":
        base_specs["environment_seed"] =  [2,6,7,8,9]
        base_specs["num_lights_to_place"] = 20
        base_specs["ground_truth_sdf_environment"] = [["rectangles",16]]
        base_specs["ambient_lights"] = 30
        base_specs["target_lights"] =  12


    baseline_spec = {
        "data_model": ["additive_gaussian_process"],
        "lighting_trigger": [["every_n",10]]
    }
    proposed_spec = {
        "data_model": ["conditional_factor_graph"],
        "lighting_trigger": [["logprob_fraction",1.1]]
    }

    baselines = deepcopy(base_specs)
    baselines.update(baseline_spec)
    proposeds = deepcopy(base_specs)
    proposeds.update(proposed_spec)

    specifications.extend(SpecificationGenerator().generate(baselines))

    specifications.extend(SpecificationGenerator().generate(proposeds))



if __name__ == '__main__':
    random.shuffle(specifications)

    pp = pprint.PrettyPrinter(indent=4)
    for spec in specifications:
        pp.pprint(spec)
    print(f"Running {len(specifications)} experiments. I hope it works.")
    runner = ExperimentRunner()
    if DEBUG:
        runner.run(name, specifications, PlanningExperiment(),
                   propagate_exceptions=True, specification_runner=MainRunner(),
                   use_dashboard=False, force_pickle=True, context_type="spawn",
                   )
    else:
        runner.run(name, specifications, PlanningExperiment(),
                   propagate_exceptions=False,
                   specification_runner=RayMultiprocessingRunner(int(sys.argv[1])),
                   context_type="spawn",
                   use_dashboard=True,
                   force_pickle=True
                   )
