from sample_sim.environments.lighting_scene import AdditiveLightingScene2d, create_sdf_environment
from sample_sim.sensors.base_sensor import PointSensor
from sample_sim.action.grid import generate_finite_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.patches as patches
import seaborn as sns
from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
from collections import defaultdict

name = "test_system"
#scheme = "Greys_r"
scheme = "cividis"
interpolation = "gaussian"
boxcolor = "tab:blue"
environment_seed = 5
seed = 2
ambient_light_brightness= 0.5
desired_light_brightness = 1.0
placed_light_brightness = 1.0
#target_lights =  3
#ambient_lights = 5
ray_steps =  30
ground_truth_reflections = 5
model_reflections =  1
target_sdf_environment =  ["free_space"]
#ground_truth_sdf_environment =  ["rectangles",8]
number_of_edge_samples = 0
state_space_dimensionality =  [13,13]
physical_step_size =  .35
sensor = PointSensor()

real_point_size = 10
small_point_size = 3
dense_points_per_axis = 150

included = set()
difference = defaultdict(lambda: 0)
for experiment_instance in experiment_iterator(name,use_tqdm=True):

    # ambient_lights = experiment_instance["specification"]["ambient_lights"]
    # if ambient_lights != 5:
    #     continue

    if experiment_instance["result"] != []:
        if experiment_instance["specification"]["budget"] - 1 == experiment_instance["specification"]["planning_steps"]:
            error = experiment_instance["result"]["gt_lighting_rmse"]
            is_baseline = experiment_instance["specification"]["data_model"] == "additive_gaussian_process"
            cur_environment_seed = experiment_instance["specification"]["environment_seed"] 
            cur_seed = experiment_instance["specification"]["seed"]
            cur_ground_truth_sdf_environment = tuple(experiment_instance["specification"]['ground_truth_sdf_environment'])
            cur_ambient_lights  = experiment_instance["specification"]["ambient_lights"]
            cur_target_lights = experiment_instance["specification"]["target_lights"]

            key = (cur_environment_seed,cur_seed,cur_ground_truth_sdf_environment,cur_ambient_lights,cur_target_lights)
            if is_baseline:
                difference[key] += error
            else:
                difference[key] -= error

            if key in included:
                included.remove(key)
            else:
                included.add(key)



environment_seed = None
seed = None
# best_difference = None
# for key,value in difference.items():
#      if best_difference is None or value > best_difference and key not in included:
#          best_difference = value
#          #print(best_difference)
#          environment_seed = key[0]
#          seed = key[1]
#          ground_truth_sdf_environment = key[2]
#          ambient_lights = key[3]
#          target_lights = key[4]
# print(best_difference)
sorted = sorted(difference.items(), key=lambda item: item[1])
idx = -1
key = sorted[idx][0]
print(sorted[idx])
environment_seed = key[0]
seed = key[1]
ground_truth_sdf_environment = key[2]
ambient_lights = key[3]
target_lights = key[4]

specification = dict()
specification["environment_seed"] = environment_seed
specification["target_sdf_environment"] = ["free_space"]
specification["ground_truth_sdf_environment"] = ground_truth_sdf_environment
specification["desired_light_brightness"] = 1.0
specification["target_lights"]= target_lights
specification["ambient_lights"] = ambient_lights    
specification["seed"] = seed

xmax = state_space_dimensionality[0] * physical_step_size
ymax = state_space_dimensionality[1] * physical_step_size


target_environment = create_sdf_environment(
    specification["environment_seed"],specification["environment_seed"], generator_name=specification["target_sdf_environment"], 
    x_size=xmax, y_size=ymax, preset_light_intensities=1.0,
    num_lights=specification["target_lights"])
target_environment.lighting_computer.set_max_reflections(ground_truth_reflections)
ambient_light_environment = create_sdf_environment(
    specification["environment_seed"], specification["seed"],generator_name=specification["ground_truth_sdf_environment"], 
    x_size=xmax, y_size=ymax, preset_light_intensities=0.3,
    num_lights=specification["ambient_lights"])

ambient_light_environment.lighting_computer.set_max_reflections( ground_truth_reflections)


gt_grid = generate_finite_grid(target_environment, tuple(state_space_dimensionality), sensor,
                                    number_of_edge_samples)

grid = generate_finite_grid(ambient_light_environment, tuple(state_space_dimensionality), sensor,
                                    number_of_edge_samples)
target_sensed_points = target_environment.sample(
    grid.get_sensed_locations())
target_sensed_points_gt = target_environment.sample(
    gt_grid.get_sensed_locations())



dense_meshgrid = ambient_light_environment.meshgrid(dense_points_per_axis)
dense_target_sensed_points = (target_environment.sample(dense_meshgrid))
dense_ambient_sensed_points = (ambient_light_environment.sample(dense_meshgrid))

min_light_brightness = float("inf")
max_light_brightness = 0
# for experiment_instance in experiment_iterator(name, use_tqdm=True):
#     if experiment_instance["result"] != []:
#         if experiment_instance["specification"]["environment_seed"] == environment_seed and experiment_instance["specification"]["seed"] == seed and experiment_instance["specification"]["ambient_lights"] == ambient_lights and experiment_instance["specification"]["budget"] - 1 == experiment_instance["specification"]["planning_steps"]: 
#             lighting_placement = experiment_instance["result"]["lighting_placement"]
#             lighting_brightnesses = experiment_instance["result"]["lighting_brightnesses"]
#             is_baseline = experiment_instance["specification"]["data_model"] == "additive_gaussian_process"
#             placed_light_environment = AdditiveLightingScene2d(
#                     ambient_light_environment.workspace, ambient_light_environment, lighting_placement, lighting_brightnesses)
#             dense_final_measurements = placed_light_environment.sample(dense_meshgrid)
#             max_light_brightness = max(max_light_brightness, np.max(dense_final_measurements))
#             min_light_brightness = min(min_light_brightness, np.min(dense_final_measurements))
min_light_brightness = 0 #float("inf")
max_light_brightness = 30700
print(f"max {max_light_brightness} min {min_light_brightness}")

plt.figure()
title = f"Desired"
#plt.title(title)
plt.imshow(dense_target_sensed_points.reshape(dense_points_per_axis,dense_points_per_axis),vmin=min_light_brightness,vmax=max_light_brightness,extent=(0,xmax,0,ymax),origin="lower",cmap=scheme,interpolation=interpolation)
plt.xticks([])
plt.yticks([])
sns.despine(plt.gcf(),plt.gca(),True,True,True,True)


plt.figure()
title = f"Ambient"
#plt.title(title)
plt.imshow(dense_ambient_sensed_points.reshape(dense_points_per_axis,dense_points_per_axis),vmin=min_light_brightness,vmax=max_light_brightness,extent=(0,xmax,0,ymax),origin="lower",cmap=scheme,interpolation=interpolation)

generator = np.random.default_rng(environment_seed)
max_size = .35 * 2#max(np.sqrt(ymax),np.sqrt(xmax))
center=generator.uniform([0,0],[xmax,ymax])
size=generator.uniform(max_size/2,max_size)

rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor)
plt.gca().add_patch(rec)
for i in range(ground_truth_sdf_environment[1]):
    #max_size = .35#max(np.sqrt(ymax),np.sqrt(xmax))
    center=generator.uniform([0,0],[xmax,ymax])
    size=generator.uniform(max_size/2,max_size)
    rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor)
    plt.gca().add_patch(rec)

plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.xticks([])
plt.yticks([])
sns.despine(plt.gcf(),plt.gca(),True,True,True,True)



print(environment_seed, seed)

for experiment_instance in experiment_iterator(name, use_tqdm=True):
    if experiment_instance["result"] != []:
        if experiment_instance["specification"]["environment_seed"] == environment_seed and experiment_instance["specification"]["seed"] == seed and experiment_instance["specification"]["ambient_lights"] == ambient_lights and experiment_instance["specification"]["budget"] - 1 == experiment_instance["specification"]["planning_steps"]:
            lighting_placement = experiment_instance["result"]["lighting_placement"]
            lighting_brightnesses = experiment_instance["result"]["lighting_brightnesses"]
            is_baseline = experiment_instance["specification"]["data_model"] == "additive_gaussian_process"
            placed_light_environment = AdditiveLightingScene2d(
                    ambient_light_environment.workspace, ambient_light_environment, lighting_placement, lighting_brightnesses)
            dense_final_measurements = (placed_light_environment.sample(dense_meshgrid))

            plt.figure()
            plt.imshow(dense_final_measurements.reshape(dense_points_per_axis,dense_points_per_axis),extent=(0,xmax,0,ymax),origin="lower",cmap=scheme,vmin=min_light_brightness,vmax=max_light_brightness,interpolation=interpolation)
            generator = np.random.default_rng(environment_seed)
            max_size = .35 * 2#max(np.sqrt(ymax),np.sqrt(xmax))
            center=generator.uniform([0,0],[xmax,ymax])
            size=generator.uniform(max_size/2,max_size)

            rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor)
            plt.gca().add_patch(rec)
            for i in range(ground_truth_sdf_environment[1]):
               # max_size = max(np.sqrt(ymax),np.sqrt(xmax))
                #max_size = .35 / 2#max(np.sqrt(ymax),np.sqrt(xmax))
                center=generator.uniform([0,0],[xmax,ymax])
                size=generator.uniform(max_size/2,max_size)
                rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor)
                plt.gca().add_patch(rec)
            np_robot_states = np.array(experiment_instance["result"]["robot_states"])
            for s0, s1 in zip(np_robot_states[:, :], np_robot_states[1:, :]):
                plt.arrow(s0[0], s0[1], s1[0]-s0[0], s1[1]-s0[1], length_includes_head = True,
                        fc = "tab:red", ec = "tab:red", head_width = 0.05, head_length = 0.05, alpha=0.8)
            lights_not_inside = (ambient_light_environment.workspace.sdf_fn(lighting_placement) > 0).squeeze()
            plt.scatter(lighting_placement[lights_not_inside,0],lighting_placement[lights_not_inside,1],c=lighting_brightnesses[lights_not_inside],vmin=0,vmax=placed_light_brightness,cmap="Greys_r",marker="o",facecolor=None) 
            plt.xlim(0,xmax)
            plt.ylim(0,ymax)
            plt.xticks([])
            plt.yticks([])
            sns.despine(plt.gcf(),plt.gca(),True,True,True,True)
            #plt.title(f"Result {'Baseline' if is_baseline else 'Proposed'}")


            
    #wall_idxs = ambient_light_environment.workspace.sdf_fn(dense_meshgrid) <= 0.0
    #dense_points_in_walls = dense_meshgrid[wall_idxs.squeeze(),:]
        #plt.scatter(dense_points_in_walls[:,0],dense_points_in_walls[:,1],s=5,c="b")
plt.show()
