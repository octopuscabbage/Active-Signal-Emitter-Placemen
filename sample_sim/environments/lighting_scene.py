from sample_sim.environments.base import BaseEnvironment
import numpy.random as nr
import numpy as np
import matplotlib.pyplot as plt
from sample_sim.environments.lighting.from_lights import FromLightLightingComputer
from sample_sim.environments.lighting.lighting_base import AnalyticalLightingModel
import sample_sim.environments.lighting.sdf_functions as sdf
import sample_sim.environments.lighting.sdf_primitives as sdf_primitives
from sample_sim.action.actions import ActionModel
import random

from sample_sim.environments.workspace import SDFWorksapce2d
from sample_sim.planning.pomcpow.pomcpow import HashableNumpyArray

class OmniLight():
    def __init__(self,x,y,strength,brightness):
        self.x = x
        self.y = y
        self.strength = strength
        self.brightness = brightness

class LightingScene2d(BaseEnvironment):
    def __init__(self, workspace: SDFWorksapce2d, lighting_computer:AnalyticalLightingModel,light_locations,light_brightnesses):
        super().__init__(workspace)
        self.light_locations = light_locations
        self.light_brightnesses = light_brightnesses
        self.baked = False
        self.lighting_computer = lighting_computer
    
    def bake(self, sensed_locations):
        values = self.sample(sensed_locations)
        self.locations_to_values = dict()
        for location,value in zip(sensed_locations,values):
            self.locations_to_values[HashableNumpyArray(location)] = value
        self.baked = True


    def sample(self, xs):
        if self.baked:
            results = []
            for location in xs:
                key = HashableNumpyArray(location)
                if key not in self.locations_to_values:
                    lights = self.lighting_computer.compute_light(np.array([location]),self.light_locations,self.light_brightnesses)

                    results.append(lights[0])
                    raise Exception()
                else:
                    results.append(self.locations_to_values[key])
            return np.array(results)
        else:
            lights = self.lighting_computer.compute_light(xs,self.light_locations,self.light_brightnesses)
            return lights

    def get_parameters(self):
        #TODO add sdf function stringification
        return {"light_locations": self.light_locations,
                "light_brightnesses": self.light_brightnesses}
    
    def action_model(self):
        return ActionModel.XY

    def dimensionality(self):
        return 2

    def get_starting_location(self,seed):
        r = random.Random(seed)


        starting_point = (int(np.mean(self.workspace.get_bounds(), 1)[0]),
                int(np.mean(self.workspace.get_bounds(), 1)[1])
                )

        while not self.workspace.is_inside(starting_point):
            x = r.uniform(self.workspace.get_bounds()[0,0],self.workspace.get_bounds()[0,1])
            y = r.uniform(self.workspace.get_bounds()[1,0],self.workspace.get_bounds()[1,1])
            starting_point = (int(x),int(y))
        return starting_point

class AdditiveLightingScene2d(BaseEnvironment):
    def __init__(self, workspace, base_lighting_scene:LightingScene2d, light_locations,light_brightnesses):
        super().__init__(workspace)

        self.base_lighting_scene = base_lighting_scene
        self.moving_scene = LightingScene2d(workspace,self.base_lighting_scene.lighting_computer,light_locations,light_brightnesses)

    def sample(self, xs):
        lights = self.moving_scene.sample(xs)
        base_lights = self.base_lighting_scene.sample(xs)
        return lights + base_lights

    def bake(self,sensed_locations):
        self.moving_scene.bake(sensed_locations)
        self.base_lighting_scene.bake(sensed_locations)

    def get_parameters(self):
        #TODO add sdf function stringification
        return {"light_locations": self.light_locations,
                "light_brightnesses": self.light_brightnesses}
    
    def action_model(self):
        return ActionModel.XY

    def dimensionality(self):
        return 2

    def get_starting_location(self):
        raise Exception()

# def random_rectangle(generator,xmin,xmax,ymin,ymax):
#     #max_size = max(np.sqrt(xmax-xmin),np.sqrt(ymax-ymin)) * 0.35
#     max_size = 0.35 * 2
#     return sdf_primitives.rectangle(center=generator.uniform([xmin,ymin],[xmax,ymax]),size=generator.uniform(max_size/2,max_size))

# def wall_generator(generator,xmin,xmax,ymin,ymax,num):
#     scene = random_rectangle(generator,xmin,ymax,ymin,ymax)
#     for i in range(num):
#         scene =  random_rectangle(generator,xmin,xmax,ymin,ymax) | scene 
#     return scene


# def generate_random_sdf(seed, generator_name,xmin,xmax,ymin,ymax):
#     generator = np.random.default_rng(seed)
#     if generator_name[0] == "rectangles":
#         return wall_generator(generator,xmin,xmax,ymin,ymax,generator_name[1])
#     elif generator_name[0] == "free_space":
#         return lambda xs: np.zeros(shape=xs.shape[0]) + max(xmax,ymax)
#     else:
#         raise Exception(f"Unknown SDF Generator {generator_name}")


# def create_sdf_environment(environment_seed,seed,generator_name,x_size,y_size,num_lights=30,preset_light_intensities=None):
#     sdf_fn = generate_random_sdf(environment_seed,generator_name,0,x_size,0,y_size)
#     lighting_computer = FromLightLightingComputer(sdf_fn,1)
#     workspace = SDFWorksapce2d(sdf_fn,0,x_size,0,y_size)
#     generator = np.random.default_rng(seed)
#     light_locations = generator.uniform([0,0],[x_size,y_size],size=(num_lights,2))
#     if preset_light_intensities == None:
#         light_intensities = np.zeros(shape=num_lights) + np.sqrt(max(x_size,y_size))
#     else:
#         light_intensities = np.zeros(shape=num_lights) + preset_light_intensities


#     return LightingScene2d(workspace,lighting_computer,light_locations,light_intensities)


def random_rectangle(generator,xmin,xmax,ymin,ymax):
    #max_size = max(np.sqrt(xmax-xmin),np.sqrt(ymax-ymin))
    max_size = 0.35 * 2
    return sdf_primitives.rectangle(center=generator.uniform([xmin,ymin],[xmax,ymax]),size=generator.uniform(max_size/2,max_size))


def wall_generator(generator,xmin,xmax,ymin,ymax,num):
    scene = random_rectangle(generator,xmin,ymax,ymin,ymax)
    for i in range(num):
        scene =  random_rectangle(generator,xmin,xmax,ymin,ymax) | scene
    return scene


def generate_random_sdf(seed, generator_name,xmin,xmax,ymin,ymax):
    generator = np.random.default_rng(seed)
    if generator_name[0] == "rectangles":
        return wall_generator(generator,xmin,xmax,ymin,ymax,generator_name[1])
    elif generator_name[0] == "free_space":
        return lambda xs: np.zeros(shape=xs.shape[0]) + max(xmax,ymax)
    else:
        raise Exception(f"Unknown SDF Generator {generator_name}")


def create_sdf_environment(environment_seed,seed,generator_name,x_size,y_size,num_lights=30,preset_light_intensities=None):
    sdf_fn = generate_random_sdf(environment_seed,generator_name,0,x_size,0,y_size)
    lighting_computer = FromLightLightingComputer(sdf_fn,1)
    workspace = SDFWorksapce2d(sdf_fn,0,x_size,0,y_size)
    generator = np.random.default_rng(seed)
    light_locations = generator.uniform([0,0],[x_size,y_size],size=(num_lights,2))
    if preset_light_intensities == None:
        light_intensities = np.zeros(shape=num_lights) + np.sqrt(max(x_size,y_size))
    else:
        light_intensities = np.zeros(shape=num_lights) + preset_light_intensities


    return LightingScene2d(workspace,lighting_computer,light_locations,light_intensities)

class FieldLightingScene2d(LightingScene2d):
    def get_starting_location(self, seed):
        if not self.workspace.is_inside((0,0)):
            raise Exception()
        return (0,0)