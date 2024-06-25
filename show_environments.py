from sample_sim.environments.lighting_scene import create_sdf_environment
from sample_sim.sensors.base_sensor import PointSensor
from sample_sim.action.grid import generate_finite_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.patches as patches
import seaborn as sns

environment_seeds = [1,2,5,6]
environment_names = {1: "A",
                     2: "B",
                     5: "C",
                     6: "D",
                     }
ambient_light_brightness= 50
desired_light_brightness = 150
target_lights =  3
ambient_lights = 5
ray_steps =  30
ground_truth_reflections = 5
model_reflections =  1
target_sdf_environment =  ["free_space"]
ground_truth_sdf_environment =  ["rectangles",8]
number_of_edge_samples = 0
state_space_dimensionality =  [13,13]
physical_step_size =  5
sensor = PointSensor()

real_point_size = 10
small_point_size = 3
dense_points_per_axis = 100

for environment_seed in tqdm(environment_seeds):
        xmax = state_space_dimensionality[0] * physical_step_size
        ymax = state_space_dimensionality[1] * physical_step_size

        target_environment = create_sdf_environment(
            environment_seed, generator_name=target_sdf_environment, 
            x_size=xmax, y_size=ymax, preset_light_intensities=desired_light_brightness,
            num_lights=target_lights)
        target_environment.lighting_computer.set_max_reflections(ground_truth_reflections)
        ambient_light_environment = create_sdf_environment(
            environment_seed, generator_name=ground_truth_sdf_environment, 
            x_size=xmax, y_size=ymax, preset_light_intensities=ambient_light_brightness,
            num_lights=ambient_lights)
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
        dense_target_sensed_points = target_environment.sample(dense_meshgrid)
        dense_ambient_sensed_points = ambient_light_environment.sample(dense_meshgrid)
        plt.figure()
        title = f"{environment_names[environment_seed]}, Desired"
        plt.title(title)
        plt.imshow(dense_target_sensed_points.reshape(dense_points_per_axis,dense_points_per_axis),vmin=0,vmax=desired_light_brightness,extent=(0,xmax,0,ymax),origin="lower",cmap="gray")
        plt.scatter(grid.get_sensed_locations()[:,0],grid.get_sensed_locations()[:,1],c="r",marker="x",s=real_point_size)#,cmap="gray")
        plt.xticks([])
        plt.yticks([])
        sns.despine(plt.gcf(),plt.gca(),True,True,True,True)


        plt.figure()
        title = f"{environment_names[environment_seed]}, Ambient"
        plt.title(title)
        plt.imshow(dense_ambient_sensed_points.reshape(dense_points_per_axis,dense_points_per_axis),vmin=0,vmax=ambient_light_brightness,extent=(0,xmax,0,ymax),origin="lower",cmap="gray")
        plt.scatter(grid.get_sensed_locations()[:,0],grid.get_sensed_locations()[:,1],c="r",marker="x",s=real_point_size)#,cmap="gray")

        generator = np.random.default_rng(environment_seed)
        max_size = max(np.sqrt(ymax),np.sqrt(xmax))
        center=generator.uniform([0,0],[xmax,ymax])
        size=generator.uniform(max_size/2,max_size)

        rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor="dimgray",edgecolor="dimgray")
        plt.gca().add_patch(rec)
        for i in range(8):
            max_size = max(np.sqrt(ymax),np.sqrt(xmax))
            center=generator.uniform([0,0],[xmax,ymax])
            size=generator.uniform(max_size/2,max_size)
            rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor="dimgray",edgecolor="dimgray")
            plt.gca().add_patch(rec)

        plt.xlim(0,xmax)
        plt.ylim(0,ymax)
        plt.xticks([])
        plt.yticks([])
        sns.despine(plt.gcf(),plt.gca(),True,True,True,True)

        #wall_idxs = ambient_light_environment.workspace.sdf_fn(dense_meshgrid) <= 0.0
        #dense_points_in_walls = dense_meshgrid[wall_idxs.squeeze(),:]
        #plt.scatter(dense_points_in_walls[:,0],dense_points_in_walls[:,1],s=5,c="b")
plt.show()
