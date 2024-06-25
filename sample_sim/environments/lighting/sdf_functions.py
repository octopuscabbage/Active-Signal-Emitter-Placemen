from asyncio.format_helpers import _format_args_and_kwargs
import numpy as np
from sample_sim.environments.lighting.sdf.sdf.d2 import *
import typing
import time
from matplotlib import colors

#https://www.ronja-tutorials.com/post/037-2d-shadows/

def vec2(a,b):
    return np.array([a,b])

STEPS = 20
SCALE = 3
class Ray():
    def __init__(self,o,t) -> None:
        self.o = o
        self.t = t
    def length(self):
        return np.linalg.norm(self.t - self.o)
    def unit_direction(self):
        if self.length() == 0:
            return np.zeros(self.o.shape)
        else:
            return (self.t - self.o) / self.length()


def scene_dist(ps):
    return (rectangle(center=vec2(20 ,20 ),size=10 ) | rectangle(center=vec2(80 ,20 ),size=10 )
            | circle(10  ,center=vec2(50 ,50 )) | rectangle(center=vec2(20,80),size=10) |
            rectangle(center=vec2(80,80),size=10) | circle(3,center=(80,50)))(ps)

def scene_dist_singular(p):
    return scene_dist(np.array([p]))[0]
           

def make_ray (origin, target):
    r = Ray(origin, target)
    return r;

def hard_march(r: Ray,scene):
    ray_distance = r.length()
    ray_direction = r.unit_direction()

    ray_progress = 0
    for i in range(0, STEPS):
        current_loc = r.o + ray_direction * ray_progress
        scene_dist = scene(current_loc)
        if scene_dist <= 0:
            return True, current_loc
        if ray_progress > ray_distance:
            return False, current_loc
        ray_progress += scene_dist
    return True, None

def soft_march(r: Ray, scene,hardness=20):
    ray_distance = r.length()
    ray_direction = r.unit_direction()
    
    ray_progress = 10**-5
    nearest = float("inf")
    for i in range(0, STEPS):
        current_loc = r.o + ray_direction * ray_progress
        scene_dist = scene(current_loc)
        if scene_dist <= 0:
            return True, current_loc
        if ray_progress > ray_distance:
            return False, np.clip(nearest,0,1)#clamp(nearest,0,1)
        nearest = min(nearest,scene_dist * hardness / ray_progress)
        ray_progress += scene_dist
    return True, None

def soft_march_many(rs : typing.List[Ray], scene, hardness=10,EPS=10**-8):
    ray_distances = list(map(lambda x: x.length(),rs))
    ray_directions = np.array(list(map(lambda x: x.unit_direction(),rs)))
    origins = np.array(list(map(lambda x: x.o, rs)))
    current_ray_locations = origins

    ray_progress = np.zeros(len(rs)) + 10**-3
    nearest = np.zeros(len(rs)) + float("inf")
    active_mask = np.ones(len(rs),dtype=bool) #Marks collided rays
    #intersected = np.ones(len(rs)) #Marks intersected rays
    saturations = np.ones(len(rs))
    for i in range(0, STEPS):
        current_ray_locations[active_mask] = origins[active_mask]  + ray_directions[active_mask,:] * ray_progress[active_mask,np.newaxis]
        scene_distances = scene(current_ray_locations[active_mask]).squeeze()

        next_active_mask = active_mask.copy()
        saturations[active_mask] = scene_distances > EPS
        next_active_mask[active_mask]  = scene_distances > EPS

        saturations[next_active_mask] *= np.clip(nearest[next_active_mask],0,1)
        nearest[active_mask] = np.min(np.vstack((nearest[active_mask],scene_distances * hardness / ray_progress[active_mask])),axis=0)
        ray_progress[active_mask] += scene_distances

        active_mask[active_mask] = (scene_distances > EPS)
        active_mask = np.logical_and(active_mask, (ray_progress < ray_distances) )
        if not(active_mask.any()):
            #print(f"Broke at {i} iters")
            return saturations
    #print(f"didn't break early {np.count_nonzero(active_mask)} active")
    return saturations


def intensity(dist,intensity):
    return  (1.0 / (1.05 + 0.035 * dist + 0.025 * dist * dist)) *intensity

light_locations = [vec2(0,0),vec2(20,90), vec2(30,30), vec2(50,50)]

def light_at_location(fragCoord,scene_dist,light_locations):
    uv = fragCoord 
    col = 0
    for light_location in light_locations:
        r0 = make_ray (uv, light_location);
        intersection, sat = soft_march(r0,scene_dist);
        if intersection:
            col += 0
        else:
            col += np.log(sat * intensity(r0.length(),10**5))
    return col

def light_at_locations(fragCoords, scene_dist, light_locations,light_brightnesses,hardness):
    if not isinstance(light_brightnesses,np.ndarray):
        light_brightnesses = np.zeros(light_locations.shape[0]) + light_brightnesses
    cols = np.zeros(fragCoords.shape[0])
    rays = []
    for light_location in light_locations: #shold reformat this to do all the lights in parallel too
        for uv in fragCoords:
            rays.append(make_ray (uv, light_location))

    ray_lengths = np.array(list(map(lambda x: x.length(),rays)))
    saturations = soft_march_many(rays,scene_dist,hardness)
    for i in range(light_locations.shape[0]):
        start = i * fragCoords.shape[0]
        end = start + fragCoords.shape[0]
        cols += saturations[start:end] * intensity(ray_lengths[start:end],light_brightnesses[i])

    return cols
if __name__ == "__main__":
    y_size = 150
    x_size = 150
    scale = 1.5
    #dists = np.zeros((x_size,y_size))
    lights = np.zeros((x_size,y_size))
    coords = []
    for x in range(0,x_size):
        for y in range(0,y_size):
            #dists[x,y] = scene_dist_singular(vec2(float(x),float(y)))
            coords.append([float(x) / scale,float(y) / scale])
            #lights[x,y] = light_at_location(vec2(float(x),float(y)),scene_dist_singular,light_locations)
    coords = np.array(coords)
    #dists = scene_dist(coords).reshape((x_size,y_size))
    #lights = light_at_locations(coords,scene_dist,light_locations).reshape((x_size,y_size))
    import matplotlib.pyplot as plt
    plt.figure()
    # ax = plt.imshow(dists)
    # plt.colorbar(ax)
    # plt.title("Distances")
    plt.figure()
    plt.ion()
    for i in range(100):
        plt.clf()

        light_locations  = np.array(light_locations,dtype=float)
        time_start = time.time()
        lights = light_at_locations(coords,scene_dist,light_locations).reshape((x_size,y_size))
        ax = plt.imshow(lights)# norm=colors.LogNorm(clip=True))
    
        time_end = time.time()
        #plt.scatter(light_locations[:,1],light_locations[:,0],marker="^")
        plt.colorbar(ax)
        plt.title(f"Lights, computed in {time_end-time_start}s")
        light_locations = np.random.uniform(0,x_size / scale,size=(np.random.randint(1,7),2))
        plt.pause(0.01)



    plt.show()