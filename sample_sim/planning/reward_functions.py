import numpy as np
import scipy
from scipy.stats import norm
from scipy.stats.mstats_extras import mjci
from lighting_placement.triggers import compute_logprob
from sample_sim.data_model.data_model import DataModel

from sample_sim.environments.lighting_scene import LightingScene2d
import statistics


def subsample_sensed_locations(locs, num_rows=300):
    if num_rows == None:
        return locs
    else:
        idxs = np.random.randint(locs.shape[0], size=num_rows)
        return locs[idxs, :]

def mi_fast(predicted_stds,data_model:DataModel,all_sampled_locations):
    #From: "Gaussian Process Optimization with Mutual Information" Contal, Perchet, Vayatis
    lb_mi = 0
    _,std = data_model.query_many(all_sampled_locations)
    cumulative_variance = np.sum(np.square(std)) 
    penalty = np.sqrt(cumulative_variance)
    predicted_vars = np.square(predicted_stds)
    for predicted_var in predicted_vars:
        phi = np.sqrt(predicted_var + cumulative_variance) - penalty
        lb_mi += phi
    return lb_mi

def global_mi(next_point,next_point_predicted_value,grid,model, all_sampled_points,all_predicted_sampled_values,subsample_amount):
    locs = subsample_sensed_locations(grid.get_sensed_locations(), num_rows=subsample_amount)
    _,pre_std = model.query_many(locs, return_std=True)

    if next_point_predicted_value.size == 0 or next_point_predicted_value.ndim == 0 :
        return 0

    elif next_point_predicted_value.shape[0] > 1:
        post_Ys = np.concatenate((all_predicted_sampled_values, next_point_predicted_value))
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    else:
        post_Ys = np.append(all_predicted_sampled_values, next_point_predicted_value)
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    model.update_prior(Xs, post_Ys)

    _,post_std = model.query_many(locs, return_std=True)
    pre_sum = np.sum(pre_std)
    post_sum = np.sum(post_std)
    return pre_sum - post_sum

def ucb_ambient(next_locations, predicted_std, data_model,b=3):
    if np.sum(predicted_std) == 0:
        return 0
    else:
        ambient_mean = data_model.predict_ambient(next_locations)
        predicted_std *= b
        return ambient_mean + predicted_std

def ucb(predicted_value, predicted_std, b=3):
    if np.sum(predicted_std) == 0:
        return 0
    else:
        return predicted_value + b * predicted_std



def expected_improvement(next_locations,predicted_value, predicted_std, grid, data_model, subsample_amount, xi=0.1,residual=False):
    # TODO make xi a parameter
    if (predicted_std == 0.0).all():
        return 0.0
    else:
        locs = subsample_sensed_locations(grid.get_sensed_locations(), num_rows=subsample_amount)
        if not residual:
            predicted_value = predicted_value[predicted_std != 0.0]
            all_values = data_model.query_many(locs, return_std=False)
        else:
            predicted_value = data_model.predict_ambient(next_locations)[predicted_std != 0.0]
            all_values = data_model.predict_ambient(locs, return_std=False)
        predicted_std = predicted_std[predicted_std != 0.0]
        ymax = np.max(all_values)
        improvment = predicted_value - ymax - xi
        Z = improvment / predicted_std
        ei = improvment * norm.cdf(Z) + predicted_std * norm.pdf(Z)
        return np.sum(ei)

def weighted_bound_deviation(next_points, predicted_value, predicted_std, desired_lighting_scene: LightingScene2d):
    # TODO make xi a parameter

    predicted_value = predicted_value[predicted_std != 0.0]
    predicted_std = predicted_std[predicted_std != 0.0]

    desired_value = desired_lighting_scene.sample(next_points)
    ll = 0
    for cur_predicted_value, cur_predicted_std, cur_desired_value in zip(predicted_value,predicted_std,desired_value):
        #upper_bound = cur_predicted_value +  3 * cur_predicted_std
        #lower_bound = cur_predicted_value - 3 * cur_predicted_std
        if cur_desired_value >= cur_predicted_value:
            #Use lower bound
            bound = cur_predicted_value - cur_predicted_std 
            bound_deviation = cur_desired_value - bound  
        else:
            #Use upper bound
            bound = cur_predicted_value + cur_predicted_std
            bound_deviation = bound - cur_desired_value 
        deviation = cur_predicted_std * np.abs(bound_deviation)

        ll += np.abs(deviation)
    return ll 

def weighted_deviation(next_points, predicted_value, predicted_std, desired_lighting_scene: LightingScene2d):
    predicted_value = predicted_value[predicted_std != 0.0]
    predicted_std = predicted_std[predicted_std != 0.0]

    desired_value = desired_lighting_scene.sample(next_points)
    ll = 0
    for cur_predicted_value, cur_predicted_std, cur_desired_value in zip(predicted_value,predicted_std,desired_value):
        deviation = cur_predicted_std * np.abs(cur_predicted_value - cur_desired_value) 
        ll += np.abs(deviation)
    return ll 


def weighted_desirability(next_points, predicted_value, predicted_std, desired_lighting_scene: LightingScene2d):
    predicted_value = predicted_value[predicted_std != 0.0]
    predicted_std = predicted_std[predicted_std != 0.0]

    desired_value = desired_lighting_scene.sample(next_points)
    ll = 0
    for cur_predicted_value, cur_predicted_std, cur_desired_value in zip(predicted_value,predicted_std,desired_value):
        deviation = cur_predicted_std * cur_desired_value 
        ll += np.abs(deviation)
    return ll 




def logprob_change(next_point,next_point_predicted_value,grid,data_model, desired_lighting,all_sampled_points,all_predicted_sampled_values,subsample_amount):
    locs = subsample_sensed_locations(grid.get_sensed_locations(), num_rows=subsample_amount)

    model = data_model

    lighting = desired_lighting.sample(locs)
    pre_mean,pre_std = model.query_many(locs, return_std=True)

    pre_logprob = compute_logprob(lighting,pre_mean,pre_std)

    if next_point_predicted_value.size == 0 or next_point_predicted_value.ndim == 0 :
        return 0

    elif next_point_predicted_value.shape[0] > 1:
        post_Ys = np.concatenate((all_predicted_sampled_values, next_point_predicted_value))
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    else:
        post_Ys = np.append(all_predicted_sampled_values, next_point_predicted_value)
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    model.update_prior(Xs, post_Ys)

    post_mean,post_std = model.query_many(locs, return_std=True)
    post_logprob = compute_logprob(lighting,post_mean,post_std)

    return pre_logprob - post_logprob


def mjci_objective(data, mean, quantiles, next_point, all_sampled_points):
    pre_error = np.sum(mjci(data, quantiles))
    post_error = np.sum(mjci(np.append(data, mean), quantiles))
    return abs(pre_error - post_error)


def gaussian_kde_objective(next_point, next_point_predicted_value, grid, data_model, quantiles, all_sampled_points,
                           all_predicted_sampled_values, subsample_amount):
    locs = subsample_sensed_locations(grid.get_sensed_locations(), num_rows=subsample_amount)

    model = data_model

    pre_mean = gp.predict(locs, return_std=False)
    pre_quantiles = np.quantile(pre_mean, quantiles)
    pre_pdf = scipy.stats.gaussian_kde(pre_mean)
    pre_pdf_values = pre_pdf.pdf(pre_quantiles)
    pre_sqrt_n = np.sqrt(pre_mean.shape[0])

    if next_point_predicted_value.size == 0 or next_point_predicted_value.ndim == 0 :
        return 0

    elif next_point_predicted_value.shape[0] > 1:
        post_Ys = np.concatenate((all_predicted_sampled_values, next_point_predicted_value))
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    else:
        post_Ys = np.append(all_predicted_sampled_values, next_point_predicted_value)
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    model.update_prior(Xs, post_Ys)

    post_mean = model.query_many(locs, return_std=False)
    post_quantiles = np.quantile(post_mean, quantiles)
    post_pdf = scipy.stats.gaussian_kde(post_mean)
    post_pdf_values = post_pdf.pdf(post_quantiles)
    post_sqrt_n = np.sqrt(post_mean.shape[0])

    pre_error = 0
    post_error = 0

    for quantile, pre_pdf_value, post_pdf_value in zip(quantiles, pre_pdf_values, post_pdf_values):
        pre_error += (np.sqrt(quantile * (1 - quantile)) / (pre_sqrt_n * pre_pdf_value))
        post_error += (np.sqrt(quantile * (1 - quantile)) / (post_sqrt_n * post_pdf_value))

    return abs(pre_error - post_error)


def quantile_change(next_point, next_point_predicted_value, grid, data_model, quantiles, all_sampled_points,
                    predicted_sampled_values, subsample_amount):
    # Evaluate entire path segment at once
    locs = subsample_sensed_locations(grid.get_sensed_locations(), num_rows=subsample_amount)

    model = data_model
    pre = np.quantile(model.query_many(locs, return_std=False), quantiles)
    
    if next_point_predicted_value.size == 0 or next_point_predicted_value.ndim == 0 :
        return 0

    elif next_point_predicted_value.shape[0] > 1:
        post_Ys = np.concatenate((predicted_sampled_values, next_point_predicted_value))
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    else:
        post_Ys = np.append(predicted_sampled_values, next_point_predicted_value)
        Xs = np.concatenate((all_sampled_points, next_point), axis=0)

    model.query_many(Xs, post_Ys)

    post = np.quantile(model.predict(locs, return_std=False), quantiles)

    return np.sum(np.abs(pre - post)) / np.array(quantiles).size


def calculate_reward(predicted_value, predicted_std, objective_function_name,  grid, data_model,
                     next_point, all_sampled_points, predicted_samples_values, desired_lighting_scene, subsample_amount=None):
    if objective_function_name == "entropy":
        cur_rw = predicted_std
    elif objective_function_name == "fast_mi":
        cur_rw = mi_fast(predicted_std,data_model,all_sampled_points)
    elif objective_function_name == "global_mi":
        cur_rw = global_mi(next_point,predicted_value,grid,data_model,all_sampled_points,predicted_samples_values,subsample_amount)
    elif objective_function_name == "weighed_deviation":
        cur_rw = weighted_deviation(next_point, predicted_value,predicted_std,desired_lighting_scene)
    elif objective_function_name == "weighed_bound_deviation":
        cur_rw = weighted_bound_deviation(next_point, predicted_value,predicted_std,desired_lighting_scene)
    elif objective_function_name == "weighed_desirabilty":
        cur_rw = weighted_desirability(next_point, predicted_value,predicted_std,desired_lighting_scene)
    elif objective_function_name == "logprob":
        cur_rw = logprob_change(next_point, predicted_value, grid, data_model, desired_lighting_scene, all_sampled_points,
                                  predicted_samples_values, subsample_amount)
    elif isinstance(objective_function_name,list) and objective_function_name[0] == "ucb_ambient":
        cur_rw = ucb_ambient(next_point, predicted_std, data_model,objective_function_name[1])
    elif isinstance(objective_function_name,list) and objective_function_name[0] == "ucb":
        cur_rw = ucb(predicted_value, predicted_std, objective_function_name[1])
    elif isinstance(objective_function_name,list) and objective_function_name[0] == "ei":
        cur_rw = expected_improvement(next_point,predicted_value, predicted_std, grid, data_model, subsample_amount, xi=objective_function_name[1],residual=False)
    elif isinstance(objective_function_name,list) and objective_function_name[0] == "ei_ambient":
        cur_rw = expected_improvement(next_point,predicted_value, predicted_std, grid, data_model, subsample_amount, xi=objective_function_name[1],residual=True)
    else:
        raise Exception(f"Unknown objective function {objective_function_name}")
    return cur_rw
