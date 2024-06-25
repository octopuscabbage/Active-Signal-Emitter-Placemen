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

from sample_sim.sensors.base_sensor import PointSensor
from scipy.optimize import minimize_scalar, minimize

data_folder = "em_cal/"
specification = base_spec
X_0 = np.array([3070, 0.01])
#LIGHT_LOCATIONS  = np.array([[1.74,2.33],[2.25,0.1657],[-2.18,-0.8],[-1.21,2.16]])
OUTLIERS_QTILE = .9
OBSTACLES = {     
     "0": [-0.088956,1.38497,0.460561],
     "1": [0.90359,-0.729829,0.455748],
     "2": [-1.23724,-1.24825,0.452176],
     "3": [2.98215,-1.46349,0.473282],
     "4": [2.5063,0.712156,0.464254],
     "5": [-2.87594,0.813079,0.453357],
     "6": [0.484188,-3.20853,0.440682]
}      


OBSTACLE_SIZE_m = 0.5
SDF_DESCRIPTION = []
for _,location in OBSTACLES.items():
   SDF_DESCRIPTION.append(RectangleObstacle(location[0],location[1],OBSTACLE_SIZE_m))
LIGHT_PLACEMENT  = np.array([[-3.21,-0.935],[-0.885,-2.669],[3.154,-0.059],[-0.0659,2.095]])


sensor = PointSensor()
ambient_light_environment = create_field_environment(
            SDF_DESCRIPTION,
            specification["environment_seed"] , 
            x_max=WORKSPACE_XMAX, y_max=WORKSPACE_YMAX,x_min=WORKSPACE_XMIN,y_min=WORKSPACE_YMIN, 
            preset_light_intensities=0, num_lights=specification["ambient_lights"])
grid = generate_finite_grid(ambient_light_environment, tuple(specification["state_space_dimensionality"]), sensor,
                                         0)
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

sensed_locations_out = []
for location in locations:
    idx = np.where((location == sensed_locations).all(axis=1))[0]
    assert idx.size == 1
    sensed_locations_out.append(sensed_locations[idx,:])
sensed_locations = np.array(sensed_locations_out).squeeze()
assert (sensed_locations == locations).all()


def optimization_fn(X):
    def intensity(dist,_):
        return (X[0] / (dist * dist))  
    lighting_model = FromLightLightingComputer(ambient_light_environment.workspace.sdf_fn,specification['physical_step_size'],specification["ray_steps"],specification["model_reflections"],intensity_fn=intensity,emissitivity=X[1])
    model_predictions = lighting_model.compute_light(sensed_locations,LIGHT_PLACEMENT,np.zeros(4))
    squared_error = (measurements - model_predictions)  ** 2
    cutoff = np.quantile(squared_error,OUTLIERS_QTILE)
    squared_error = squared_error[squared_error <= cutoff]
    rmse =  np.sqrt(np.mean(squared_error))
    return rmse

res = minimize(optimization_fn, x0=X_0, options={"disp":True})
print(f"Successful? {res.success} {res.message}")
print(f"Best A {res.x}")