import logging
import numpy as np
from tqdm import tqdm
from sample_sim.data_model.additive_lighting_model import AdditiveLightingModel
from sample_sim.data_model.gp_wrapper import TorchExactGp
from sample_sim.environments.lighting.from_lights import FromLightLightingComputer
from sample_sim.environments.lighting.lighting_base import AnalyticalLightingModel
import sample_sim.environments.lighting.sdf_functions as sdf
from sklearn.metrics import mean_squared_error
from cmaes import CMA
from sample_sim.environments.lighting_scene import create_sdf_environment, generate_random_sdf
from sample_sim.environments.workspace import SDFWorksapce2d, Workspace
from lighting_placement.triggers import compute_logprob
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from joblib import Memory

from sample_sim.planning.pomcpow.pomcpow import HashableNumpyArray
from scipy.optimize import minimize
from ax.service.ax_client import AxClient, ObjectiveProperties

class CMAESLightingOptimizer():
    def __init__(self,logger_name,num_lights,sensed_locations,desired_lighting,lighting_model,num_iters,population_size,workspace,objective, lighting_upper_bound) -> None:
        self.logger_name = logger_name
        self.num_lights = num_lights
        self.sensed_locations = sensed_locations
        self.desired_lighting = desired_lighting
        self.lighting_model = lighting_model
        self.num_iters = num_iters
        self.population_size = population_size
        self.workspace = workspace
        self.objective = objective
        self.lighting_upper_bound = lighting_upper_bound
        self.light_locations = None
        self.light_brightnesses = None

        self.previous_prediction = None

    def minimize_lighting(self,data_model, use_model=True):
        means = np.zeros((self.num_lights * 3))
        means[0::3] = np.mean(self.sensed_locations)
        means[1::3] = np.mean(self.sensed_locations)
        means[2::3] = self.lighting_upper_bound / 2

        stds = np.std(self.sensed_locations)

        x_bounds, y_bounds = self.workspace.get_bounds()
        l_bounds = np.zeros(self.num_lights * 3)
        l_bounds[0::3] = x_bounds[0]
        l_bounds[1::3] = y_bounds[0]
        l_bounds[2::3] = 0.0
        

        u_bounds = np.zeros(self.num_lights * 3)
        u_bounds[0::3] = x_bounds[1]
        u_bounds[1::3] = y_bounds[1]
        u_bounds[2::3] = self.lighting_upper_bound

        bounds = np.stack((l_bounds,u_bounds),axis=1)
        popsize = self.population_size
        optimizer = CMA(mean=means,sigma=stds,population_size=popsize,bounds=bounds,seed=0)


        if use_model:
            if self.light_locations is None:
                predicted_ambient_lighting, predicted_ambient_std = data_model.query_many(self.sensed_locations,return_std=True) 
            else:
                predicted_ambient_lighting, predicted_ambient_std = data_model.predict_ambient(self.sensed_locations,return_std=True)
               
        else:
            predicted_ambient_lighting = np.zeros(self.sensed_locations.shape[0])
            predicted_ambient_std = np.ones(self.sensed_locations.shape[0])

        if self.previous_prediction is not None and (self.previous_prediction == predicted_ambient_lighting).all():
            raise Exception()
        self.previous_prediction = predicted_ambient_lighting
        best_error = float("inf")
        outer_iterator = tqdm(range(self.num_iters),desc="Computing Light Placement", file=TqdmToLogger(logging.getLogger(self.logger_name)))
        for i in outer_iterator:
            solutions = []
            for _ in range(optimizer.population_size):
                raw_sample = optimizer.ask()
                sample = raw_sample.reshape(self.num_lights,3)

                current_lighting = self.lighting_model.compute_light(self.sensed_locations,sample[:,:2],sample[:,2])
                #sdf.light_at_locations(sensed_locations,sdf_fn,sample.reshape(num_lights,2),light_intensities,hardness) 
                ambient_corrected = current_lighting + predicted_ambient_lighting
                if self.objective == "rmse":
                    error = np.sqrt(mean_squared_error(self.desired_lighting, ambient_corrected))
                elif self.objective == "logprob":
                    error = -1 * compute_logprob(self.desired_lighting, ambient_corrected, predicted_ambient_std)
                else:
                    raise Exception()
                #error[error == np.inf] = 0
                #mean_error = error.mean()
                mean_error = error
                if not np.isnan(mean_error):
                    if mean_error < best_error:
                        best_error = mean_error
                        best_light_placement = sample[:,:2]
                        best_brightnesses = sample[:,2]
                        best_lighting = current_lighting
                    solutions.append((raw_sample, mean_error))
            outer_iterator.set_postfix(best_error=best_error,popsize=popsize)
            optimizer.tell(solutions)
            if optimizer.should_stop():
                popsize = optimizer.population_size * 2
                #mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
                optimizer = CMA(mean=means, sigma=stds, population_size=popsize,seed=0)
        self.light_locations = best_light_placement
        self.light_brightnesses = best_brightnesses
        return self.light_locations, self.light_brightnesses, best_lighting

class ScipyLightingOptimizer():
    def __init__(self,logger_name,num_lights,sensed_locations,desired_lighting,lighting_model,num_iters,method_name,workspace,objective, lighting_upper_bound) -> None:
        self.logger_name = logger_name
        self.num_lights = num_lights
        self.sensed_locations = sensed_locations
        self.desired_lighting = desired_lighting
        self.lighting_model = lighting_model
        self.num_iters = num_iters
        self.method_name = method_name
        self.workspace = workspace
        self.objective = objective
        self.lighting_upper_bound = lighting_upper_bound
        self.light_locations = None
        self.light_brightnesses = None
        self.last_results = None

        self.previous_prediction = None
        
        self.random_seed = 0

    def get_random_initial(self,seed):
        r = np.random.default_rng(seed)
        if self.last_results is None:
            means = np.zeros((self.num_lights * 3))
            #means[0::3] = r.integers(x_bounds[0],x_bounds[1],self.num_lights)
            #means[1::3] = r.integers(y_bounds[0],y_bounds[1],self.num_lights)
            #means[2::3] = r.integers(self.lighting_upper_bound/2,self.lighting_upper_bound,self.num_lights)

            #Select n random locations with high concentration
            idxs = np.array(list(range(self.sensed_locations.shape[0])))
            chosen_idxs = r.choice(idxs,size=self.num_lights,p=self.desired_lighting/np.sum(self.desired_lighting))
            means[0::3] = self.sensed_locations[chosen_idxs,0]
            means[1::3] = self.sensed_locations[chosen_idxs,1]
            means[2::3] = r.uniform(self.lighting_upper_bound/2,self.lighting_upper_bound,self.num_lights)
        return means


    def minimize_lighting(self,data_model, use_model=True):

        x_bounds, y_bounds = self.workspace.get_bounds()
        #r = np.random.default_rng(0)

        if self.last_results is None:
            means = self.get_random_initial(self.random_seed)
            self.random_seed += 1
        else:
            means = self.last_results



        bounds = []
        for i in range(self.num_lights):
            bounds.append((x_bounds[0],x_bounds[1]))
            bounds.append((y_bounds[0],y_bounds[1]))
            bounds.append((0.0,self.lighting_upper_bound))

        if use_model:
            if self.light_locations is None:
                predicted_ambient_lighting, predicted_ambient_std = data_model.query_many(self.sensed_locations,return_std=True) 
            else:
                predicted_ambient_lighting, predicted_ambient_std = data_model.predict_ambient(self.sensed_locations,return_std=True)
        else:
            predicted_ambient_lighting = np.zeros(self.sensed_locations.shape[0])
            predicted_ambient_std = np.ones(self.sensed_locations.shape[0])

        if self.previous_prediction is not None and (self.previous_prediction == predicted_ambient_lighting).all():
            raise Exception()
        self.previous_prediction = predicted_ambient_lighting
        def f(raw_sample):
                sample = raw_sample.reshape(self.num_lights,3)

                current_lighting = self.lighting_model.compute_light(self.sensed_locations,sample[:,:2],sample[:,2])
                #sdf.light_at_locations(sensed_locations,sdf_fn,sample.reshape(num_lights,2),light_intensities,hardness) 
                ambient_corrected = current_lighting + predicted_ambient_lighting
                if self.objective == "rmse":
                    error = np.sqrt(mean_squared_error(self.desired_lighting, ambient_corrected))
                elif self.objective == "logprob":
                    error = -1 * compute_logprob(self.desired_lighting, ambient_corrected, predicted_ambient_std)
                else:
                    raise Exception()
                return error

        pbar = tqdm(range(self.num_iters),desc=f"Computing Light Placement {self.method_name}", file=TqdmToLogger(logging.getLogger(self.logger_name)))
        fvals = []
        best_result = float("inf")
        for i in range(4):
            res = minimize(f,means,method=self.method_name,bounds=bounds,callback=lambda x: pbar.update(1),options={"maxiter":int(self.num_iters/4)})        
            if res.fun < best_result:
                cur_best_sample = res.x
                best_result = res.fun
           
            means = self.get_random_initial(self.random_seed)
            self.random_seed += 1
            fvals.append(res.fun)
            logging.getLogger(self.logger_name).info(f"Final fval: {res.fun}! Method { self.method_name} terminated with {'success' if res.success else 'failure'} bc {res.message} after {res.nit} iters. ")
        logging.getLogger(self.logger_name).info(f"Fvals: {fvals}")
        best_sample = cur_best_sample.reshape(self.num_lights,3)
        self.last_results = cur_best_sample
        best_light_placement = best_sample[:,:2]
        best_light_brightnesses = best_sample[:,2]
        best_lighting = self.lighting_model.compute_light(self.sensed_locations,best_light_placement,best_light_brightnesses)
        self.light_locations = best_light_placement
        self.light_brightnesses = best_light_brightnesses
        return self.light_locations, self.light_brightnesses, best_lighting

class BOLightingOptimizer():
    def __init__(self,logger_name,num_lights,sensed_locations,desired_lighting,lighting_model,num_iters,method_name,workspace,objective, lighting_upper_bound) -> None:
        self.logger_name = logger_name
        self.num_lights = num_lights
        self.sensed_locations = sensed_locations
        self.desired_lighting = desired_lighting
        self.lighting_model = lighting_model
        self.num_iters = num_iters
        self.method_name = method_name
        self.workspace = workspace
        self.objective = objective
        self.lighting_upper_bound = lighting_upper_bound
        self.light_locations = None
        self.light_brightnesses = None
        self.last_results = None
        self.best_results = None

        self.previous_prediction = None

    def minimize_lighting(self,data_model, use_model=True):

        x_bounds, y_bounds = self.workspace.get_bounds()
        r = np.random.default_rng(0)

        # if self.last_results is None:
        #     means = np.zeros((self.num_lights * 3))
        #     #means[0::3] = r.integers(x_bounds[0],x_bounds[1],self.num_lights)
        #     #means[1::3] = r.integers(y_bounds[0],y_bounds[1],self.num_lights)
        #     #means[2::3] = r.integers(self.lighting_upper_bound/2,self.lighting_upper_bound,self.num_lights)

        #     #Select n random locations with high concentration
        #     idxs = np.array(list(range(self.sensed_locations.shape[0])))
        #     chosen_idxs = r.choice(idxs,size=self.num_lights,p=self.desired_lighting/np.sum(self.desired_lighting))
        #     means[0::3] = self.sensed_locations[chosen_idxs,0]
        #     means[1::3] = self.sensed_locations[chosen_idxs,1]
        #     means[2::3] = r.integers(self.lighting_upper_bound/2,self.lighting_upper_bound,self.num_lights)
        # else:
        #     means = self.last_results
        parameters = []
        for i in range(self.num_lights):
            parameters.append({"name": f"x_{i}",
                               "type": "range",
                               "bounds": [float(x_bounds[0]),float(x_bounds[1])],
                               })
            parameters.append({"name":f"y_{i}",
                               "type": "range",
                               "bounds": [float(y_bounds[0]),float(y_bounds[1])],
                               })

            parameters.append({"name":f"i_{i}",
                               "type": "range",
                               "bounds": [0.0,float(self.lighting_upper_bound)],
                               })


        if use_model:
            if self.light_locations is None:
                predicted_ambient_lighting, predicted_ambient_std = data_model.query_many(self.sensed_locations,return_std=True) 
            else:
                predicted_ambient_lighting, predicted_ambient_std = data_model.predict_ambient(self.sensed_locations,return_std=True)
        else:
            predicted_ambient_lighting = np.zeros(self.sensed_locations.shape[0])
            predicted_ambient_std = np.ones(self.sensed_locations.shape[0])

        if self.previous_prediction is not None and (self.previous_prediction == predicted_ambient_lighting).all():
            raise Exception()
        self.previous_prediction = predicted_ambient_lighting
        def f(parameterization):
                sample = []
                for i in range(self.num_lights):
                    sample.append([parameterization.get(f"x_{i}"),parameterization.get(f"y_{i}"),parameterization.get(f"i_{i}")])
                sample = np.array(sample)
                current_lighting = self.lighting_model.compute_light(self.sensed_locations,sample[:,:2],sample[:,2])
                #sdf.light_at_locations(sensed_locations,sdf_fn,sample.reshape(num_lights,2),light_intensities,hardness) 
                ambient_corrected = current_lighting + predicted_ambient_lighting
                if self.objective == "rmse":
                    error = np.sqrt(mean_squared_error(self.desired_lighting, ambient_corrected))
                elif self.objective == "logprob":
                    error = -1 * compute_logprob(self.desired_lighting, ambient_corrected, predicted_ambient_std)
                else:
                    raise Exception()
                return {"error": (error,0.0)}

        pbar = tqdm(range(self.num_iters),desc=f"Computing Light Placement {self.method_name}", file=TqdmToLogger(logging.getLogger(self.logger_name)))

        # def cb(res):
        #     pbar.update(1)
        #     pbar.set_postfix({"fval": res.fun})
        #res = minimize(f,means,method=self.method_name,bounds=bounds,callback=lambda x: pbar.update(1),options={"maxiter":self.num_iters})        
        #optimizer = BayesianOptimization(f=f,pbounds=bounds, verbose=0,random_state=0)
        # if self.last_results is not None:
        #     res = gp_minimize(f,
        #                     dimensions=bounds,
        #                     n_calls=self.num_iters,
        #                     random_state=0,
        #                     callback=cb,
        #                     x0 = self.last_results,
        #                     y0 = f(self.last_results),
        #                     acq_func = "EI",
        #                     ) 

        # else:
        #     res = gp_minimize(f,
        #                     dimensions=bounds,
        #                     n_calls=self.num_iters,
        #                     random_state=0,
        #                     callback=cb,
        #                     acq_func = "EI",
        #                     )
        ax_client = AxClient(random_seed=0)
        ax_client.create_experiment(parameters,name="TEST",objectives={"error": ObjectiveProperties(minimize=True)})
        best_error = float("inf")
        for i in range(self.num_iters):
            pbar.update(1)
            
            parameters,trial_index  = ax_client.get_next_trial()
            fval = f(parameters)
            best_error = min(best_error,fval["error"][0])
            pbar.set_postfix_str(f"F {best_error}")
            ax_client.complete_trial(trial_index,raw_data=fval)

        best_parameters, values = ax_client.get_best_parameters()
        
        best_sample = []
        for i in range(self.num_lights):
            best_sample.append([best_parameters.get(f"x_{i}"),best_parameters.get(f"y_{i}"),best_parameters.get(f"i_{i}")])
        best_sample = np.array(best_sample)
        means, covariances = values
    
        logging.getLogger(self.logger_name).info(f"Final fval: {means}, {covariances}!")
        self.last_results = best_sample
        best_light_placement = best_sample[:,:2]
        best_light_brightnesses = best_sample[:,2]
        best_lighting = self.lighting_model.compute_light(self.sensed_locations,best_light_placement,best_light_brightnesses)
        self.light_locations = best_light_placement
        self.light_brightnesses = best_light_brightnesses
        return self.light_locations, self.light_brightnesses, best_lighting



memory = Memory("cache",verbose=0)

@memory.cache
def get_first_light_placement(logger_name,num_lights,sensed_locations:HashableNumpyArray, desired_lighting:HashableNumpyArray, num_iters,population_size,objective,lighting_upper_bound,
                                environment_seed, generator_name, x_size,y_size,physical_step_size,number_of_edge_samples,raytracing_steps,model_reflections, optimizer_name):
    sdf_fn = generate_random_sdf(environment_seed,generator_name,0,x_size,0,y_size)
    workspace =  SDFWorksapce2d(sdf_fn,0,x_size,0,y_size)
    lighting_model = FromLightLightingComputer(sdf_fn,physical_step_size / (number_of_edge_samples+1),raytracing_steps,model_reflections)
    if optimizer_name == "cmaes":
        lighting_optimizer = CMAESLightingOptimizer(logger_name,num_lights,sensed_locations.xs, desired_lighting.xs, lighting_model,num_iters,population_size,workspace,objective,lighting_upper_bound)
    elif optimizer_name == "bo":
        lighting_optimizer = BOLightingOptimizer(logger_name,num_lights,sensed_locations.xs,desired_lighting.xs,lighting_model,num_iters*population_size,optimizer_name,workspace,objective,lighting_upper_bound)
    else:
        lighting_optimizer = ScipyLightingOptimizer(logger_name,num_lights,sensed_locations.xs,desired_lighting.xs,lighting_model,num_iters*population_size,optimizer_name,workspace,objective,lighting_upper_bound)

    lighting_placement, lighting_brightnesses, best_lighting = lighting_optimizer.minimize_lighting(None,False)
    return lighting_placement, lighting_brightnesses, best_lighting 
