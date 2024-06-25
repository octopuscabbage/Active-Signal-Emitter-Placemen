from ros_driver import base_spec
from sample_sim.action.grid import generate_finite_grid
from ros.planner_wrapper import create_field_environment
from ros.constants import *
from sample_sim.environments.lighting.from_lights import FromLightLightingComputer
import os
import json 
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sample_sim.environments.lighting import sdf_primitives

from sample_sim.sensors.base_sensor import PointSensor
data_folder = "em_cal/"
specification = base_spec

# A = 0.00050833 
# B = -0.00101397 
# C = 0.00064646
# B = -3.31327760e-04
#C = 3.68974824e-04
# A,B,C = [ 8.54972929e-05, -3.31327760e-04,  3.68974824e-04]

#A,B,C= [0.00050833, -0.00101397,  0.00064646]
#A = 11844.43319393 / 3.75
OBSTACLES = {     
     "0": [-0.088956,1.38497,0.460561],
     "1": [0.90359,-0.729829,0.455748],
     "2": [-1.23724,-1.24825,0.452176],
     #"3": [2.98215,-1.46349,0.473282],
     "4": [2.5063,0.712156,0.464254],
     "5": [-2.87594,0.813079,0.453357],
     "6": [0.484188,-3.20853,0.440682]
}      

OBSTACLE_SIZE_m = 0.5
SDF_DESCRIPTION = []
for _,location in OBSTACLES.items():
   SDF_DESCRIPTION.append(RectangleObstacle(location[0],location[1],OBSTACLE_SIZE_m))
LIGHT_PLACEMENT  = np.array([[-3.21,-0.935],[-0.885,-2.669],[3.154,-0.059],[-0.0659,2.095]])
#A =1.87312489e+03
A = 3070
EMISSIVITY = 0.61
def intensity(dist,_):
    # if dist*dist <= 1e-7:
    #     return A / 1e-14
    # else:
    #p =  (X[0] / (dist * dist))  + (X[1] * dist)
    p = A / (dist * dist)
        #if p > 10000:
        #    return 10000
        #else: 
    return p 

#LIGHT_LOCATIONS = np.array([[-0.6069,2.95],[2.201,2.0639],[2.642,0.7725],[-3.3106,0.993]])
#LIGHT_LOCATIONS = np.array([[2.262,2.00],[-3.34,-0.87],[2.66,0.70],[-0.57,3.00]])
#LIGHT_LOCATIONS  = np.array([[1.74,2.33],[2.25,0.1657],[-2.18,-0.8],[-1.21,2.16]])

# def intensity(dist,_):
#     #return np.clip(1.0 / (A + B * dist + C * dist * dist),0,10000) 
#     return 1.0 / (A + B * dist + C * dist * dist)
def intensity_1d(dist):
    #return np.clip(1.0 / (A + B * dist + C * dist * dist),0,10000) 
# if dist*dist <= 1e-7:
    #     return A / 1e-7
    # else:
    #return (X[0] / (dist * dist))  + (X[1] * dist)
    return intensity(dist,0)

x = np.linspace(0,3.5)
y = intensity_1d(x)
plt.figure()
plt.plot(x,y)
#plt.show()
sensor = PointSensor()
ambient_light_environment = create_field_environment(
            SDF_DESCRIPTION,
            specification["environment_seed"] , 
            x_max=WORKSPACE_XMAX, y_max=WORKSPACE_YMAX,x_min=WORKSPACE_XMIN,y_min=WORKSPACE_YMIN, 
            preset_light_intensities=0, num_lights=specification["ambient_lights"])
grid = generate_finite_grid(ambient_light_environment, tuple(specification["state_space_dimensionality"]), sensor,
                                         0)
lighting_model = FromLightLightingComputer(ambient_light_environment.workspace.sdf_fn,specification['physical_step_size'],specification["ray_steps"],specification["model_reflections"],intensity_fn=intensity,emissitivity=EMISSIVITY)
dense_grid = ambient_light_environment.workspace.get_meshgrid(400)
#dense_prediction = lighting_model.compute_light(dense_grid,LIGHT_PLA,np.zeros(4))
# plt.figure()
# plt.scatter(dense_grid[:,0],dense_grid[:,1],c=dense_prediction)
# plt.show()

locations = []
measurements = []
for gt_file in sorted(os.listdir(data_folder)):
    with open(os.path.join(data_folder,gt_file)) as f:
        data = json.load(f)
        locations.append(data["location"])
        measurements.append(data["measurement"])
locations = np.array(locations)
measurements = np.array(measurements)
sensed_locations = grid.get_sensed_locations()

def global_sdf():
    obstacles = SDF_DESCRIPTION
    sdf_fn = sdf_primitives.rectangle(center=[obstacles[0].x,obstacles[0].y],size=obstacles[0].size) 
    for rectangle in obstacles:
        sdf_fn = sdf_primitives.rectangle(center=[rectangle.x,rectangle.y],size=rectangle.size)  | sdf_fn
    return sdf_fn
#distances = global_sdf()(dense_grid)
obstacle_locations = dense_grid[(global_sdf()(dense_grid).T <= 0)[0],:]
# def min_distance_to_light(loc, light_locations):
#     return np.min(np.linalg.norm(light_locations-loc,ord=2,axis=1))
# sensed_locations_out = []
# for sensed_location in sensed_locations:
#     if min_distance_to_light(sensed_location,LIGHT_LOCATIONS) > 0.4:
#         sensed_locations_out.append(sensed_location)
#sensed_locations = np.array(sensed_locations_out) 
# locations_out = []
# measurements_out = []

# for sensed_location in sensed_locations:
#     idx = np.where((locations == sensed_location).all(axis=1))[0]
#     locations_out.append(locations[idx[0],:])
    #measurements_out.append(measurements[idx[0]])
sensed_locations_out = []
for location in locations:
    idx = np.where((location == sensed_locations).all(axis=1))[0]
    assert idx.size == 1
    sensed_locations_out.append(sensed_locations[idx,:])
sensed_locations = np.array(sensed_locations_out).squeeze()

assert (sensed_locations == locations).all()
model_predictions = lighting_model.compute_light(sensed_locations,LIGHT_PLACEMENT,np.zeros(4))
rmse =  mean_squared_error(measurements,model_predictions,squared=False)
print(rmse)


error = np.sqrt((measurements - model_predictions) **2)
plt.figure()
plt.hist(error)

plt.figure()
ax = plt.scatter(sensed_locations[:,0],sensed_locations[:,1],c=error)
plt.scatter(obstacle_locations[:,0],obstacle_locations[:,1],marker=".",c="b")
plt.scatter(LIGHT_PLACEMENT[:,0],LIGHT_PLACEMENT[:,1],marker="x",c="r")
plt.colorbar(ax)
plt.title(f"Measured error {rmse}")
plt.figure()
ax = plt.scatter(sensed_locations[:,0],sensed_locations[:,1],c=model_predictions)
plt.scatter(obstacle_locations[:,0],obstacle_locations[:,1],marker=".",c="b")

plt.scatter(LIGHT_PLACEMENT[:,0],LIGHT_PLACEMENT[:,1],marker="x",c="r")
plt.colorbar(ax)
plt.title(f"Predicted")

plt.figure()
ax = plt.scatter(sensed_locations[:,0],sensed_locations[:,1],c=measurements)
plt.scatter(obstacle_locations[:,0],obstacle_locations[:,1],marker=".",c="b")

plt.scatter(LIGHT_PLACEMENT[:,0],LIGHT_PLACEMENT[:,1],marker="x",c="r")
plt.colorbar(ax)
plt.title(f"measured")

plt.figure()
plt.plot(error,label="Error")
plt.plot(measurements,label="Measurements")
plt.plot(model_predictions,label="Predictions")
plt.legend()
plt.show()

