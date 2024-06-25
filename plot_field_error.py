from copy import deepcopy
from ros.constants import *
import os

from ros.planner_wrapper import create_field_environment
from sample_sim.action.grid import generate_finite_grid
from sample_sim.sensors.base_sensor import PointSensor
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sample_sim.environments.lighting import sdf_primitives
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar



base_folder = "field_results/asp_exp/"
run_length = 70
from collections import defaultdict
barplot_df = []
boxcolor = "tab:blue"
cmap = "cividis"
save_location = "/home/cdennist/octop/Pictures/thesis pics/active_light/field_res"

def rename_trial(trial_number):
    if trial_number == 1:
        return "No Unknown Lights"
    elif trial_number in [2,3]:
        return "Two Unknown Lights"
    else:
        raise Exception("trial_number")


def get_field_description(trial_number):
    if trial_number in [1,2,3]:
        OBSTACLE_SIZE_m = 0.5
        obstacles = {     
            "0": [-0.088956,1.38497,0.460561],
            "1": [0.90359,-0.729829,0.455748],
            "2": [-1.23724,-1.24825,0.452176],
            #"3": [2.98215,-1.46349,0.473282],
            "4": [2.5063,0.712156,0.464254],
            "5": [-2.87594,0.813079,0.453357],
            "6": [0.484188,-3.20853,0.440682]
        }      
        sdf_description = []
        for _,location in obstacles.items():
            sdf_description.append(RectangleObstacle(location[0],location[1],OBSTACLE_SIZE_m))
        EMISSIVITY = 0.65
        base_spec = {
        "plot": False,
        "seed": 0,


        "environment_seed": 1,
        ## ENVIRONMENT
        "num_lights_to_place": 5,
        "ambient_light_brightness": 0.5,
        "placed_light_brightness": 1.0,
        "desired_light_brightness": 1.0,
        "target_lights": 3,
        "ambient_lights":1,
        "raytracing_steps": 30,
        "ray_steps": 30,
        "gt_reflections": 5,
        "model_reflections": 1,
        "target_sdf_environment": ["free_space"],
        "ground_truth_sdf_environment": ["rectangles",8],
        "do_pilot_survey": False,
        "pilot_survey_len": 20,
        "number_of_edge_samples": 0,
        "state_space_dimensionality": [13,13],
        "physical_step_size": .35,
        "place_at_beginning": True,


        ## LIGHTING TRIGGER
        #"lighting_trigger": [["every_n",10],["every_n",30],["last"], ["logprob_fraction",20], ["logprob_fraction",30],],


        ## LIGHTING PLACEMENT ALGORITHM
        "cem_iterations": 200,
        "cem.population": 1,
        "cem.alpha":0.8,
        "cem_objective": "rmse",
        "cem.optimizer": "Nelder-Mead",


        ## DATA MODEL
        "age_measurements": False,




        "gp.lengthscale": 3 * .35, #Equal to 3x physical step size


        "fg.sensed_value_linkage": 5,
        "fg.minimum_distance_sigma":0.01,
        "fg.distance_sigma_scaling":1.0,
        "fg.measurement_uncertainty":0.1,
        "fg.lighting_model_uncertainty": 1,
        "fgc.residual_prior_uncertainty": 10,


        
        "light_placement_variance": 0,
        "light_placement_mc_iters": 30,


        ##PLANNING
        "objective_c": 1E-6,
        "planning_steps": 70,
        "rollouts_per_step": 50,
        #"rollouts_per_step":[4],
        "gamma": .95,
        "max_planning_depth": 7,
        #"max_planner_rollout_depth": [1],


        "use_t_test": True,
        "t_test_value": 0.3,
        "objective_function": "entropy",
        # "observation_sampling_strategy": ["mean"],
        # "rollout_strategy": "random",
        }


        baseline_spec = deepcopy(base_spec)
        baseline_spec.update({
        "data_model": "additive_gaussian_process",
        "lighting_trigger": ["every_n",10]
        })
        proposed_spec = deepcopy(base_spec)
        proposed_spec.update({
        "data_model": "conditional_factor_graph",
        "lighting_trigger": ["logprob_fraction",1.1]
        })

        target_environment = create_field_environment(
            [],
            base_spec["environment_seed"],  
            x_max=WORKSPACE_XMAX, y_max=WORKSPACE_YMAX,x_min=WORKSPACE_XMIN,y_min=WORKSPACE_YMIN, 
            preset_light_intensities=base_spec["desired_light_brightness"], num_lights=base_spec["target_lights"])
        target_environment.lighting_computer.set_max_reflections(base_spec["gt_reflections"])
        ambient_light_environment = create_field_environment(
            sdf_description,
            base_spec["environment_seed"] , 
            x_max=WORKSPACE_XMAX, y_max=WORKSPACE_YMAX,x_min=WORKSPACE_XMIN,y_min=WORKSPACE_YMIN, 
            preset_light_intensities=0, num_lights=base_spec["ambient_lights"])
        ambient_light_environment.lighting_computer.set_max_reflections(base_spec["gt_reflections"])
        sensor = PointSensor()
        grid = generate_finite_grid(ambient_light_environment, tuple(base_spec["state_space_dimensionality"]), sensor,0)
        #ambient_light_environment.bake(grid.sensed_locations)
        #target_environment.bake(grid.sensed_locations)
    return sdf_description,EMISSIVITY,baseline_spec,proposed_spec,grid,target_environment

def plot_obstacles(obstacles):
    #rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor)
    needs_legend_entry = True
    for obstacle in obstacles:
        # max_size = max(np.sqrt(ymax),np.sqrt(xmax))
        #max_size = .35 / 2#max(np.sqrt(ymax),np.sqrt(xmax))
        #center=generator.uniform([0,0],[xmax,ymax])
        #size=generator.uniform(max_size/2,max_size)
        center = [obstacle.x ,obstacle.y]
        size = obstacle.size
        if needs_legend_entry:
            rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor,label="Obstacle")
            needs_legend_entry = False
        else:
            rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor)
        plt.gca().add_patch(rec)

all_locations= []
trial_locations = dict()
df = []
for run_type in os.listdir(os.path.join(base_folder)):
    #for trial in os.listdir(os.path.join(base_folder,run_type)):
    for trial in ["1","3"]:
        sdf, emissivity, baseline_spec, proposed_spec, grid, target_environment = get_field_description(int(trial))
        dense_grid = target_environment.workspace.get_meshgrid(400)

        #obstacles = SDF_DESCRIPTION
        obstacles = sdf
        sdf_fn = sdf_primitives.rectangle(center=[obstacles[0].x,obstacles[0].y],size=obstacles[0].size) 
        for rectangle in obstacles:
            sdf_fn = sdf_primitives.rectangle(center=[rectangle.x,rectangle.y],size=rectangle.size)  | sdf_fn
        #obstacle_locations = dense_grid[(sdf_fn(dense_grid).T <= 0)[0],:]
        
        robot_locations = []
        light_locations = []
        for i in range(1,run_length):
            with open(os.path.join(base_folder,run_type,trial,"run",f"{i}.pkl"),"rb") as f:
                robot_data = pickle.load(f)
                robot_locations.append(robot_data["location"])
                light_locations.append(robot_data["light_locations"])
                # if i % 10 == 0:
                #     mean = robot_data["mean"]
                #     var = robot_data["std"]
                #     plt.figure()
                #     plt.title(f"{i} Mean")
                #     plt.scatter(locations[:,0], locations[:1],c=mean)
                #     robot_locations_np = np.array(robot_locations)
                #     plt.plot(robot_locations_np[:,0],robot_locations_np[:,1])
                #     plt.figure()
                #     plt.title(f"{i} Var")
                #     plt.scatter(locations[:,0], locations[:,1],c=var)
                #     plt.plot(robot_locations_np[:,0],robot_locations_np[:,1])


        
        light_location = light_locations[-1]
        data = []
        locations = []
        able_to_be_measured_locations = set()
        for gt_point_file in sorted(os.listdir(os.path.join(base_folder,run_type,trial,"gt")),key=lambda s:int(s.split(".")[0].split("_")[1])):
            with open(os.path.join(base_folder,run_type,trial,"gt",gt_point_file)) as f:
                gt_data = json.load(f) 
                #grid_distances = np.linalg.norm(grid.get_sensed_locations() - np.array(gt_data["location"]),axis=1)
                #print(np.min(grid_distances))
                #closest_grid_point = grid.get_sensed_locations()[np.argmin(grid_distances),:]

                location = gt_data["location"]
                all_locations.append(location)
                able_to_be_measured_locations.add(tuple(location))

                #print(gt_point_file,location)
                dist = np.linalg.norm(np.array(location) - light_location,axis=1)
                t_dist = np.linalg.norm(np.array(location) - target_environment.light_locations,axis=1)
                        #plt.scatter(target_environment.light_locations[:,0], target_environment.light_locations[:,1],marker="x",c="r")

                #print(dist)
                #if min(np.min(dist),np.min(t_dist)) > 0.7:
                locations.append(location)
                    #desired = target_environment.sample(np.array([location]))
                data.append([location[0],location[1],np.median(gt_data["sensor_buffer"])])
        #print(able_to_be_measured_locations)
        if trial in trial_locations:
            print('in ', trial)
            print("before", len(trial_locations[trial]))
            trial_locations[trial] = trial_locations[trial].intersection(able_to_be_measured_locations)
            print("after", len(trial_locations[trial]))
        else:
            print(trial)
            trial_locations[trial] = able_to_be_measured_locations

        data = np.array(data)
        locations = np.array(locations)
        desired = target_environment.sample(locations)
        data = np.append(data,np.zeros([desired.shape[0],1]),1)
        data[:,3] = desired
        vmax = np.max(data)
        vmin = np.min(data)

        plt.figure()
        #plt.title(f"{run_type.capitalize()} {rename_trial(int(trial))}")
        print(f"Figure {plt.gcf().number} {run_type.capitalize()} {rename_trial(int(trial))}")

        print(np.sqrt(data[:,0].shape))
        x_min = np.min(data[:,0])
        x_max = np.max(data[:,0])
        y_min = np.min(data[:,1])
        y_max = np.max(data[:,1])

        img_data = np.zeros((13,13))

        img_data[img_data==0] = float("NaN")
        # for i in range(13):
        #     for j in range(13):

        x_idxs = ((data[:,0] + 3.5 ) * 13) / (3.5 * 2) 
        y_idxs = ((data[:,1] + 3.5 ) * 13) / (3.5 * 2) 
        offset = np.abs(data[0,0] - data[1,1]) / 2
        print(offset)
        for x_idx,y_idx,c in zip(x_idxs,y_idxs,data[:,2]):
            img_data[int(x_idx),int(y_idx)] = c
        ax = plt.imshow(img_data.T,extent=[x_min-offset,x_max+offset,y_min-offset,y_max+offset],origin="lower",vmin=vmin,vmax=vmax,label="GT Measurements",cmap=cmap)
        #sns.despine()
        sns.despine(None,None,True,True,True,True)

        #ax = plt.scatter(data[:,0],data[:,1],c=data[:,2],vmin=vmin,vmax=vmax,label="GT Measurements")
       # ax = plt.scatter(data[:,0],data[:,1],c='g',vmin=vmin,vmax=vmax,label="GT Measurements")

        plt.scatter(light_location[:,0],light_location[:,1],c="r",marker="x",label="Placed Lights")
        if int(trial) == 3:
            unknown_light_locations = np.array([[-3.7,3.7],[1.1,3.7]])
            plt.scatter(unknown_light_locations[:,0],unknown_light_locations[:,1],c="b",marker="x",label="Unknown Lights")

        #plt.scatter(obstacle_locations[:,0],obstacle_locations[:,1],c="b",marker=".")
        plot_obstacles(sdf)
        #robot_locations = np.array(robot_locations)
        needs_legend_entry = True
        np_robot_states = np.array(robot_locations)
        for s0, s1 in zip(np_robot_states[:, :], np_robot_states[1:, :]):
            if needs_legend_entry:
                plt.arrow(s0[0], s0[1], s1[0]-s0[0], s1[1]-s0[1], length_includes_head = True,
                        fc = "tab:red", ec = "tab:red", head_width = 0.1, head_length = 0.1, alpha=0.8,label="Robot Path")
                needs_legend_entry = False
            else:
                plt.arrow(s0[0], s0[1], s1[0]-s0[0], s1[1]-s0[1], length_includes_head = True,
                        fc = "tab:red", ec = "tab:red", head_width = 0.1, head_length = 0.1, alpha=0.8)

        #plt.plot(robot_locations[:,0],robot_locations[:,1])
        sns.despine(None,None,True,True,True,True)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        if trial == "3" and run_type == "baseline":
            plt.legend(loc="lower right",framealpha=0.99)
        if trial == "1"  and run_type == "baseline":
            scalebar = ScaleBar(1.0,"m")
            plt.gca().add_artist(scalebar)

        if trial == "3":
            plt.savefig(os.path.join(save_location,f"{run_type}_2.pdf"))

        else:
            plt.savefig(os.path.join(save_location,f"{run_type}_{trial}.pdf"))
        #plt.colorbar(ax)
        
                # plt.colorbar(ax)
        # plt.figure()
        # plt.title(f"{run_type} {trial} Abs Error")
        # ax = plt.scatter(data[:,0],data[:,1],c=np.abs(data[:,2] - data[:,3]))
        # plt.scatter(obstacle_locations[:,0],obstacle_locations[:,1],c="b",marker=".")

        # plt.colorbar(ax)
        # plt.figure()
        # plt.title(f"{run_type} {trial} Error")
        # #ax = plt.scatter(data[:,0],data[:,1],c=np.abs(data[:,2] - data[:,3]))
        # plt.plot(data[:,3],label="Desired")
        # plt.plot(data[:,2],label="Measured")
        # plt.plot(np.abs(data[:,2] - data[:,3]),label="Error")
        # plt.legend()
        # plt.figure()
        # plt.title(f"{run_type} {trial} Error rmse {mean_squared_error(data[:,2],data[:,3],squared=False)}")
        # plt.hist(np.abs(data[:,2] - data[:,3]),label="Error")
        print(f"{run_type} {trial} {mean_squared_error(data[:,2],data[:,3],squared=False)}")
        for l,err in zip(zip(data[:,0],data[:,1]), np.abs(data[:,2] - data[:,3])):
            df.append({"Type": run_type.capitalize(), "Trial": rename_trial(int(trial)), "Error": err, "Location":l, "T_num": trial})
        barplot_df.append({"Type": run_type.capitalize(), "Trial": rename_trial(int(trial)), "RMSE": mean_squared_error(data[:,2],data[:,3],squared=False)})
#print(trial_locations)
# df_out = []
# for d in df:
#     if d["Location"] in trial_locations[d["T_num"]]:
#         #print(l)
#         df_out.append(d)
#     else:
#         print("...")
# print(len(df),len(df_out))
df = pd.DataFrame(df)
#print(df)
plt.figure()
sns.boxplot(x="Trial",hue="Type",y="Error",data=df,order=["No Unknown Lights", "Two Unknown Lights"],hue_order=["Baseline","Proposed"],showfliers=False)
print(df.groupby(["Trial","Type"]).median())
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(save_location,"results.pdf"))
# plt.figure()
# barplot_df = pd.DataFrame(barplot_df)

plt.figure()
#plt.title(f"Desired")
locations = np.array(all_locations)

desired = target_environment.sample(locations)
data = locations
x_min = np.min(data[:,0])
x_max = np.max(data[:,0])
y_min = np.min(data[:,1])
y_max = np.max(data[:,1])

img_data = np.zeros((13,13))
img_data[img_data==0] = float("NaN")
# for i in range(13):
#     for j in range(13):

x_idxs = ((data[:,0] + 3.5 ) * 13) / (3.5 * 2) 
y_idxs = ((data[:,1] + 3.5 ) * 13) / (3.5 * 2) 
offset = np.abs(data[0,0] - data[1,1]) / 2
print(offset)
for x_idx,y_idx,c in zip(x_idxs,y_idxs,desired):
    img_data[int(x_idx),int(y_idx)] = c
ax = plt.imshow(img_data.T,extent=[x_min-offset,x_max+offset,y_min-offset,y_max+offset],origin="lower",vmin=vmin,vmax=vmax,cmap=cmap)

#ax = plt.scatter(locations[:,0],locations[:,1],c=desired,vmax=vmax,vmin=vmin)
plt.scatter(target_environment.light_locations[:,0], target_environment.light_locations[:,1],marker="x",c="r")



#plt.scatter(obstacle_locations[:,0],obstacle_locations[:,1],c="b",marker=".")
plot_obstacles(sdf)
#sns.despine()
sns.despine(None,None,True,True,True,True)
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(save_location,"desired.pdf"))

plt.show()