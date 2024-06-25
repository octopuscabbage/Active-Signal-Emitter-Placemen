from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter

from tqdm import tqdm
from collections import defaultdict
import numpy as np

try:
    from sample_sim.environments.lighting.lighting_base import AnalyticalLightingModel
    from sample_sim.environments.lighting.sdf.sdf.d2 import *
except ModuleNotFoundError:
    from sdf.sdf.d2 import * 
    class AnalyticalLightingModel:
        pass
import typing
import time
import seaborn as sns
import matplotlib.patches as patches


BROADPHASE = 'rec'
METHOD = 'new'


#def intensity(dist,intensity):
#    return  (1.0 / (1.05 + 0.035 * dist + 0.025 * dist * dist)) *intensity

A = 3070.811
def intensity(dist,intensity):
    if dist <= 1e-7:
        p =  A * intensity / (1e-14)
    else:
        p = A * intensity / (dist * dist)
    if p > A * 10 * intensity:
        return (A * 10) * intensity
    else:
        return p


def compute_sphere_direction(num_rays, iteration):
    sphere_angle = 2 * np.pi / num_rays * iteration
    sphere_direction = np.array([np.cos(sphere_angle),np.sin(sphere_angle)])

    ray_direction = sphere_direction / np.linalg.norm(sphere_direction)
    return ray_direction

def compute_gradient_many(locs, scene, h):
    x_pos = np.array(locs,copy=True)
    x_pos[:,0] += h
    x_min = np.array(locs,copy=True)
    x_min[:,0] -= h
    y_pos = np.array(locs,copy=True)
    y_pos[:,1] += h
    y_min = np.array(locs,copy=True)
    y_min[:,1] -= h

    out = np.array(locs,copy=True)
    out[:,0] = scene(x_pos).squeeze() - scene(x_min).squeeze() / 2 * h
    out[:,1] = scene(y_pos).squeeze() - scene(y_min).squeeze() / 2 * h
    return out
def reflect_rays(current_locs,ray_directions,scene):
    #If this is expensive, we can not compute it on the last iter
    #Reflect intersected rays
    normals = compute_gradient_many(current_locs,scene,h=0.01)
    
    #Set up for new iteration
    #I'm not smart enough to figure out how to do this vectorized but it's pretty cheap :/
    #ray_directions = (ray_directions  - 2 * (np.sum(ray_directions * normalized_gradients,axis=1))[:,np.newaxis] * normalized_gradients)
    normalizing_values = np.linalg.norm(normals,ord=2,axis=1) 
    normalized_gradients = normals / normalizing_values[:,np.newaxis]
    ray_directions = (ray_directions  - 2 * (np.sum(ray_directions * normalized_gradients,axis=1))[:,np.newaxis] * normalized_gradients)
    return ray_directions

def row_wise_dot_product(a,b):
    return np.sum(a * b, axis=1)

def line_point_distance(points,start,end):
    x_0 = points[:,0]
    y_0 = points[:,1]
    x_1 = start[0]
    y_1 = start[1]
    x_2 = end[0]
    y_2 = end[1]
    distance = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2) 
    projection = np.abs((x_2 - x_1) * (y_1 * y_0) - (x_1 - x_0) * (y_2 - y_1))

    #u = u1 / (LineMag * LineMag)
    u = projection / (np.square(distance))

    out_distance = np.zeros(x_0.shape)
    #need to find closest to both endpoints
    outside_mask = (u < 0) | (u > 1)
    out_distance[outside_mask] =  np.min(np.stack((np.linalg.norm(points[outside_mask,:] - start,axis=1),np.linalg.norm(points[outside_mask,:] - end,axis=1)),axis=1),axis=1)
    inside_mask = np.logical_not(outside_mask)
    projected_points = np.zeros(points[inside_mask].shape)
    projected_points[:,0] = x_1 + u[inside_mask] * (x_2 - x_1)
    projected_points[:,1] = y_1 + u[inside_mask] * (y_2 - y_1)
    out_distance[inside_mask]  = np.linalg.norm(points[inside_mask] - projected_points,axis=1)
    return  out_distance

def lineseg_dists(p, a, b):
    # Handle case where p is a single point, i.e. 1d array.
    p = np.atleast_2d(p)

    # TODO for you: consider implementing @Eskapp's suggestions
    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)

def kd_broadphase(t, start_location,line_length,coordinates):
    results = t.query_ball_point(start_location,line_length)
    if len(results) == 0:
        return None
            
    idxs = np.array(results)
    results_to_coordinates = coordinates[idxs]
    return results_to_coordinates, idxs

def rec_broadphase(start_location,end_location,coordinates,all_idxs):
    x_0 = min(start_location[0],end_location[0])
    y_0 = min(start_location[1],end_location[1])
    x_1 = max(start_location[0],end_location[0])
    y_1 = max(start_location[1],end_location[1])

    mask = (x_0 <= coordinates[:,0]) & (coordinates[:,0] <= x_1) & (y_0 <= coordinates[:,1]) & (coordinates[:,1] <= y_1)
    idxs = all_idxs[mask]
    results_to_coordinates = coordinates[mask,:]
    return results_to_coordinates, idxs


def compute_pixel_color_from_rays(colors,t,coordinates,ray_origins,current_loc,previous_ray_distances,reflection_number,coordinate_spacing_x,coordinate_spacing_y,light_idxs,light_brightnesses,emissivity):
    if reflection_number == 0:
        new_colors = defaultdict(lambda: 0)

    emissivity = emissivity ** reflection_number
    all_idxs = np.array(range(coordinates.shape[0]))
    #Color based on rays
    for start_location, end_location,previous_ray_distance,original_light_idx in zip(ray_origins,current_loc,previous_ray_distances,light_idxs):
        #This triggers if the light was inside the sdf
        if reflection_number == 0 and (start_location == end_location).all():
            continue

        #Points which are valid can be at most this distance away 
        line_length = np.linalg.norm(start_location - end_location)
        if line_length == 0:
            continue
        if BROADPHASE=="kd":
            res = kd_broadphase(t,start_location,line_length, coordinates)
            if res is None:
                continue
            results_to_coordinates, idxs = res
        if BROADPHASE == "rec":
            results_to_coordinates, idxs = rec_broadphase(start_location,end_location,coordinates,all_idxs)
        

        line_point_distances = lineseg_dists(results_to_coordinates, start_location, end_location)
        valid_coordinates = results_to_coordinates[line_point_distances <= coordinate_spacing_x]
        idxs = idxs[line_point_distances <= coordinate_spacing_x]
        distance_to_origin = np.linalg.norm(valid_coordinates - start_location,axis=1) + previous_ray_distance

        #Need to treat the rays from the lights specially since they will all add too much at the source
        if reflection_number > 0:
            new_colors = defaultdict(lambda: 0)
        #this prevents recomputing norms a billion times
        for idx, distance in zip(idxs,distance_to_origin):
            color = emissivity  * intensity( distance, light_brightnesses[original_light_idx])
            #Only want to update color once per ray (or once per light source on the first ray)
            new_colors[idx] = max(color, new_colors[idx])
        #Need to treat the rays from the lights specially since they will all add too much
        if reflection_number != 0:
            for idx, color in new_colors.items():
                colors[idx] += color
    #Need to treat the rays from the lights specially since they will all add too much

    if reflection_number == 0:
        for idx, color in new_colors.items():
            colors[idx] += color
    return colors






def compute_pixel_color_from_rays_old(colors,t,coordinates,ray_origins,current_loc,previous_ray_distances,reflection_number,coordinate_spacing_x,coordinate_spacing_y,light_idxs,light_brightnesses,emissivity):
    if reflection_number == 0:
        new_colors = defaultdict(lambda: 0)

    emissivity = emissivity ** reflection_number
    #Color based on rays
    for start_location, end_location,previous_ray_distance,original_light_idx in zip(ray_origins,current_loc,previous_ray_distances,light_idxs):
        #try:
        samples = int(np.ceil(max(abs(end_location[0] - start_location[0]) / coordinate_spacing_x / 2,
                            abs(end_location[1] - start_location[1]) / coordinate_spacing_y / 2)))
        xs = np.linspace(start_location[0],end_location[0],samples)
        ys = np.linspace(start_location[1],end_location[1],samples)

        sample_points = np.stack((xs,ys),axis=1)
        line_tree = KDTree(sample_points)
        results = line_tree.query_ball_tree(t,r=max(coordinate_spacing_x,coordinate_spacing_y))
        #Need to treat the rays from the lights specially since they will all add too much
        if reflection_number > 0:
            new_colors = defaultdict(lambda: 0)
        #this prevents recomputing norms a billion times
        seen_coordiates = set()
        for sample_idx,idxs in enumerate(results):
            distances = np.linalg.norm(coordinates[idxs]-start_location,axis=1) + previous_ray_distance

            #color_update = emissivity * intensity(distances,10**5)
            for idx,distance in zip(idxs,distances):
                if idx not in seen_coordiates:
                    seen_coordiates.add(idx)
                    color = emissivity  * intensity( distance, light_brightnesses[original_light_idx])
                    #Only want to update color once per ray (or once per light source on the first ray)
                    new_colors[idx] = max(color, new_colors[idx])
        #Need to treat the rays from the lights specially since they will all add too much
        if reflection_number != 0:
            for idx, color in new_colors.items():
                colors[idx] += color
    #Need to treat the rays from the lights specially since they will all add too much

    if reflection_number == 0:
        for idx, color in new_colors.items():
            colors[idx] += color
    return colors


def light_at_locations_from_lights_fast(coordinates,scene,coordinate_spacing,light_locations,light_brightnesses, reflections=1, num_rays=2**7,record_path=True,steps=30,emissivity=.35):
    paths = []

    colors = np.zeros(coordinates.shape[0])
    x_min = np.min(coordinates[:,0])
    x_max = np.max(coordinates[:,0])
    y_min = np.min(coordinates[:,1])
    y_max = np.max(coordinates[:,1])
    # coordinate_spacing_x = ((x_max - x_min) / x_values) 
    # coordinate_spacing_y = ((y_max - y_min) / y_values) 
    coordinate_spacing_x = coordinate_spacing
    coordinate_spacing_y = coordinate_spacing


    t = KDTree(coordinates)
    ray_directions = []
    ray_locations = []
    previous_ray_distances = []
    total_ray_number = 0
    light_idxs = []
    for i,light_location in enumerate(light_locations):
        for iteration in range(num_rays):
            paths.append([light_location])
            ray_directions.append(compute_sphere_direction(num_rays,iteration))
            ray_locations.append(light_location)
            previous_ray_distances.append(0)
            total_ray_number += 1
            light_idxs.append(i)

    assert len(paths) == total_ray_number
    ray_directions = np.array(ray_directions)
    ray_locations = np.array(ray_locations)
    previous_ray_distances = np.array(previous_ray_distances)

    ray_origins = np.array(ray_locations,copy=True)
    previous_ray_distances = np.zeros(total_ray_number)

    active = np.ones(total_ray_number,dtype=bool)
    active_before = np.array(active,dtype=bool)

    ray_progress = np.zeros(total_ray_number)# + EPS #This moves it off the original light and off the surface 
    for reflection_number in range(0,reflections+1):

        #March the rays
        intersections = np.zeros(total_ray_number,dtype=bool)
        current_loc = np.array(ray_origins,copy=True)
        for ray_iter in range(0, steps):
            current_loc[active,:] = ray_origins[active,:] + (ray_directions[active,:] * ray_progress[active,np.newaxis])
            if record_path:
                for i,loc in enumerate(current_loc):
                    if active[i]:
                    #assert ray_iter == 0 or np.linalg.norm(paths[i][-1] - loc,ord=2) != 0
                        paths[i].append(np.array(loc,copy=True))
            #assert ray_iter == 0 or (old_current_loc != current_loc).any()
            #This should be active masked
            scene_dist = scene(current_loc[active,:]).squeeze()

            ray_progress[active] += scene_dist#[:,0]
            #intersections[active] = np.logical_or(intersections[active], (scene_dist <= 0)[:,0]) #Need to bounce ray
            intersections[active] = np.logical_or(intersections[active], (scene_dist <= 1e-4)) #Need to bounce ray

            #If not intersected, stay active
            active = np.logical_and(active,np.logical_not(intersections))

            #Check the rays are still inside workspace
            active = active &  (x_min <= current_loc[:,0]) &  (current_loc[:,0] <= x_max) &( y_min <= current_loc[:,1]) & (current_loc[:,1] <= y_max)
            if (np.logical_not(active)).all():
                break
        # colors = compute_pixel_color_from_rays(colors,t,coordinates,
        #             ray_origins[active_before,:],current_loc[active_before,:],previous_ray_distances[active_before],
        #             reflection_number,coordinate_spacing_x,coordinate_spacing_y,num_rays, 
        #             light_idxs,light_idxs,light_brightnesses, emissivity)
        if METHOD == "new":
            colors = compute_pixel_color_from_rays(colors,t,coordinates,
                        ray_origins[active_before,:],current_loc[active_before,:],previous_ray_distances[active_before],
                        reflection_number,coordinate_spacing_x,coordinate_spacing_y,
                        light_idxs,light_brightnesses,emissivity)
        if METHOD == 'old':
            colors = compute_pixel_color_from_rays_old(colors,t,coordinates,
                        ray_origins[active_before,:],current_loc[active_before,:],previous_ray_distances[active_before],
                        reflection_number,coordinate_spacing_x,coordinate_spacing_y,
                        light_idxs,light_brightnesses,emissivity)

        #Update Distances
        previous_ray_distances = previous_ray_distances  + ray_progress

        #If this is expensive, we can not compute it on the last iter
        #Reflect intersected rays
        ray_directions = reflect_rays(current_loc,ray_directions,scene)

        ray_progress = np.zeros(total_ray_number) + 0.01 #This moves it off the original light and off the surface 
        ray_origins = current_loc
        active_before = np.array(active,copy=True)
        active = np.array(intersections,copy=True)

            
    return colors, paths

class FromLightLightingComputer(AnalyticalLightingModel):
    def __init__(self, sdf, coordinate_spacing, max_ray_steps=20,reflections=3) -> None:
        super().__init__()
        self.max_ray_steps = max_ray_steps
        self.reflections = reflections
        self.sdf = sdf
        self.coordinate_spacing = coordinate_spacing
    def __compute_lights__impl__(self, xs, light_locations, light_brightnesses):
        colors, paths = light_at_locations_from_lights_fast(xs,self.sdf,self.coordinate_spacing,light_locations,light_brightnesses,reflections=self.reflections,steps=self.max_ray_steps)
        #import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(150):
        #     p = np.array(paths[i])
        #     plt.plot(p[:,0],p[:,1])
        # plt.show()
        return colors
    def set_max_reflections(self, reflections):
        self.reflections = reflections
if __name__ == "__main__":
    animation_save_location = "/home/cdennist/octop/Pictures/thesis pics/active_light/ray_animation"
    import os
    y_size = 3.5
    x_size = 3.5
    scale = 0.35
    def vec2(a,b):
        return np.array([a,b])
    # def scene_dist(ps):
    # #return #(rectangle(center=vec2(20 ,20 ),size=10 ) | rectangle(center=vec2(80 ,20 ),size=10 )
    #     return  (circle(10  ,center=vec2(50 ,50 )) | rectangle(center=vec2(20,80),size=10) 
    #             | rectangle(center=vec2(80,80),size=10) | circle(3,center=(80,50)))(ps) 
    import sdf_primitives
    max_size = 1.0
    def random_rectangle(generator,xmin,xmax,ymin,ymax):
    #max_size = max(np.sqrt(xmax-xmin),np.sqrt(ymax-ymin))
        return sdf_primitives.rectangle(center=generator.uniform([xmin,ymin],[xmax,ymax]),size=generator.uniform(max_size/2,max_size))
    def wall_generator(generator,xmin,xmax,ymin,ymax,num):
        scene = random_rectangle(generator,xmin,ymax,ymin,ymax)
        for i in range(num):
            scene =  random_rectangle(generator,xmin,xmax,ymin,ymax) | scene
        return scene
    def plot_obstacles(ax,generator,num):
        obstacles = []

        center = generator.uniform([-3.5,-3.5],[3.5,3.5])
        size = generator.uniform([max_size/2],[max_size])
        #print(np.append(center,size))
        obstacles.append(np.append(center,size))
        for i in range(num):
            obstacles.append(np.append(generator.uniform([-3.5,-3.5],[3.5,3.5]),generator.uniform([max_size],[max_size])))

        boxcolor = "tab:blue"

        for obstacle in obstacles:
            center = [obstacle[0] ,obstacle[1]]
            size = obstacle[2]
            rec = patches.Rectangle((center[0]-size/2,center[1]-size/2),size,size,facecolor=boxcolor,edgecolor=boxcolor)
            ax.add_patch(rec)
        
    generator = np.random.default_rng(0)
    scene_dist = wall_generator(generator,-3.5,3.5,-3.5,3.5,8)
    #dists = np.zeros((x_size,y_size))
    #lights = np.zeros((x_size,y_size))
    # coords = []
    # for x in range(0,x_size):
    #     for y in range(0,y_size):
    # #         #dists[x,y] = scene_dist_singular(vec2(float(x),float(y)))
    #          coords.append([float(x) / scale,float(y) / scale])
#         lights[x,y] = light_at_location(vec2(float(x),float(y)),scene_dist_singular,light_locations)
    dense_points_per_axis = 50
    test_x_range = np.linspace(-3.5, 3.5, num=dense_points_per_axis)
    test_y_range = np.linspace(-3.5, 3.5, num=dense_points_per_axis)

    test_x, test_y = np.meshgrid(test_x_range, test_y_range)
    test_x = test_x.flatten()
    test_y = test_y.flatten()

    coords = np.stack((test_x, test_y), axis=-1)
    #coords = np.array(coords)
    dists = scene_dist(coords).reshape(-1)
    inside_shape_coords = coords[dists < 0,:]
    #^lights = light_at_locations(coords,scene_dist,light_locations).reshape((x_size,y_size))
    import matplotlib.pyplot as plt
    # plt.figure()
    # ax = plt.imshow(dists)
    # plt.colorbar(ax)
    # plt.title("Distances")
    #plt.ion()

    r = np.random.default_rng(0)
    #for i in range(10):
    if True:

        num_lights = 8
        light_locations = r.uniform(-3.5,3.5,size=(num_lights,2))
        while (scene_dist(light_locations) <= 0).any():
             light_locations = r.uniform(-3.5,3.5,size=(num_lights,2))
        #light_locations = np.array([[48,64]])

            
        light_locations  = np.array(light_locations,dtype=float)
        time_start = time.time()
        lights_flat = []
        lights_flat_reflected = []
        light_differences = []
        reflected_paths = []

        brightness = np.zeros(light_locations.shape[0])+ 150
        plt.figure(1,figsize=(6,4))
        #lights = light_at_locations(coords,scene_dist,light_locations,light_brightnesses=10**5)
        for i in range(20):
            print(i)
            start_fast = time.time()
            lights_flat_reflected_fast,reflected_paths_fast = light_at_locations_from_lights_fast(coords,scene_dist,1,light_locations, brightness,reflections=3,num_rays=2**10,steps=i)
            #lights_flat_reflected_fast_more_rays,_ = light_at_locations_from_lights_fast(coords,scene_dist,1,light_locations,brightness,reflections=3,num_rays=2**10,record_path=False)
            end_fast = time.time()
            #exit()
            # start_slow = time.time()
            # lights_flat_reflected,reflected_paths = light_at_locations_from_lights(coords,scene_dist,x_size,y_size,light_locations,reflections=1)
            # end_slow = time.time()
            #exit()


            #lights_flat_reflected_blurred = gaussian_filter(lights_flat_reflected.reshape(x_size,y_size),sigma=1).reshape(-1)
            #ight_flat_reflected_blurred_fast = gaussian_filter(lights_flat_reflected_fast.reshape(x_size,y_size),sigma=1).reshape(-1)
            #light_flat_reflected_blurred_more_rays = gaussian_filter(lights_flat_reflected_fast_more_rays.reshape(x_size,y_size),sigma=1).reshape(-1)
            light_flat_reflected_blurred_fast = lights_flat_reflected_fast

            #lights_flat_reflected_blurred = lights_flat_reflected
            #light_flat_reflected_blurred_fast  = lights_flat_reflected_fast
    

            #fig, (ax1, ax2) = plt.subplots(1, 2,num=1)
            ax1 = plt.gca()
            ax2 = plt.gca()
            #plt.clf()
            #ax = ax1.scatter(coords[:,0],coords[:,1],c=(light_flat_reflected_blurred_fast))
            scheme = "cividis"
            interpolation = "gaussian"
            ax1.imshow(light_flat_reflected_blurred_fast.reshape(dense_points_per_axis,dense_points_per_axis),extent=(-3.5,3.5,-3.5,3.5),origin="lower",cmap=scheme,interpolation=interpolation)

            ax1.scatter(light_locations[:,0],light_locations[:,1],marker="*",c="r")
            #ax1.scatter(inside_shape_coords[:,0],inside_shape_coords[:,1],marker="x",c="b")
            plot_obstacles(ax1, np.random.default_rng(0),8)

            #plt.title(f"Lights with reflection fast blur {end_fast - start_fast}")
            #plt.colorbar(ax)

            #plt.clf()
            paths_to_display = 25
            cm = plt.get_cmap('gist_rainbow')

            NUM_COLORS = paths_to_display
            LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
            NUM_STYLES = len(LINE_STYLES)
            sns.reset_orig()  # get default matplotlib styles back
            #ax2.imshow(np.zeros((dense_points_per_axis,dense_points_per_axis)),extent=(-3.5,3.5,-3.5,3.5),origin="lower",cmap=scheme,interpolation=interpolation)

            clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples
            #cm = plt.get_cmap('gist_rainbow')
            #ax2.set_prop_cycle(color=[cm(1.*i/paths_to_display) for i in range(paths_to_display)])

            idxs = np.random.default_rng(0).integers(0,len(reflected_paths_fast)-1,paths_to_display)
            for clr_i, idx in enumerate(idxs):
                p_array = np.array(reflected_paths_fast[idx])
                lines = ax2.plot(p_array[:,0],p_array[:,1],marker=".")
                lines[0].set_color(clrs[clr_i])
                lines[0].set_linestyle(LINE_STYLES[clr_i%NUM_STYLES])
            #ax2.scatter(inside_shape_coords[:,0],inside_shape_coords[:,1],marker="x",c="b")
            #ax2.scatter(light_locations[:,0],light_locations[:,1],marker="*",c="r",s=100)
            #plt.title("Fast")
            ax1.set_xlim([-3.5,3.5])
            ax1.set_ylim([-3.5,3.5])

            #ax2.set_xlim([-3.5,3.5])
            #ax2.set_ylim([-3.5,3.5])
            #plot_obstacles(ax2, np.random.default_rng(0),8)


            #sns.despine(None,ax1,True,True,True,True)

            sns.despine(None,ax2,True,True,True,True)
            plt.tight_layout()

            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

            plt.pause(0.1)
            plt.draw()
            plt.savefig(os.path.join(animation_save_location,f"{i}.png"))
