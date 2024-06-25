from functools import partial
from typing import List, Optional
import gtsam
import numpy as np
from bidict import bidict
from scipy.spatial import KDTree

from sample_sim.action.grid import FinitePlanningGrid
from sample_sim.data_model.data_model import DataModel

from sample_sim.data_model.factor_graph import GRAPH_TYPE
from sample_sim.planning.pomcpow.pomcpow import HashableNumpyArray
from sample_sim.data_model.factor_graph import FactorGraphDataModel

from copy import deepcopy


def additive_factor(measurement: np.ndarray, this: gtsam.CustomFactor,
              values: gtsam.Values,
              jacobians: Optional[List[np.ndarray]]) -> float:
    #https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/CustomFactorExample.py
    """Additive Factor error function
    :param measurement: full light measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    key1 = this.keys()[0]
    key2 = this.keys()[1]
    estimate1 = values.atVector(key1)
    estimate2 = values.atVector(key2)
    # if estimate1 != 0.0 or estimate2 != 0.0:
    #     print(estimate1,estimate2)
    error = (estimate1 + estimate2) - measurement
    if jacobians is not None:
        jacobians[0] = np.eye(1)
        jacobians[1] = np.eye(1)
    return error


class ConditionalFactorGraphDataModel(FactorGraphDataModel):
    def __init__(self, grid: FinitePlanningGrid, logger, verbose=False, sensed_value_linkage=4, 
            minimum_distance_sigma=0.001, distance_sigma_scaling=0.01, measurement_uncertainty=0.01, 
            lighting_model_uncertainty=0.05, residual_prior_uncertainty=1.0):

            self.residual_prior_uncertainty = residual_prior_uncertainty
            super().__init__(grid,logger,verbose,sensed_value_linkage,minimum_distance_sigma,distance_sigma_scaling,measurement_uncertainty,lighting_model_uncertainty)
            self.last_residual_posterior_factors = []

    def create_factor_graph(self):
        #return super().create_factor_graph()
        self.all_symbols = []
        if GRAPH_TYPE=="linear":
            self.graph = gtsam.GaussianFactorGraph()
        elif GRAPH_TYPE=="nonlinear" :
            self.graph = gtsam.NonlinearFactorGraph()
        elif GRAPH_TYPE=="isam":
            self.graph = self.incremental_graph
        else:
            raise Exception()
        self.current_estimate = gtsam.Values()
        sensed_locations = self.grid.get_sensed_locations()
        self.sensed_locations_to_analytical_symbols = bidict()
        self.sensed_locations_to_residual_symbols = bidict()
        self.all_residual_symbols = []
        self.all_analytical_symbols = []


        self.sensed_locations_tree = KDTree(sensed_locations)

        # 1 add all the values
        for i, sensed_location in enumerate(sensed_locations):
            analytical_symbol = gtsam.Symbol("a", i).key()
            residual_symbol = gtsam.Symbol("r", i).key()
            assert not self.current_estimate.exists(analytical_symbol)
            assert not self.current_estimate.exists(residual_symbol)
            self.current_estimate.insert(analytical_symbol, np.array([0.0]))
            self.current_estimate.insert(residual_symbol, np.array([0.0]))

            self.sensed_locations_to_analytical_symbols[tuple(
                sensed_location)] = analytical_symbol

            self.sensed_locations_to_residual_symbols[tuple(
                sensed_location)] = residual_symbol

        # 2 link all the values to their k nearest neighbors
        # Need to make sure all values exist first
        # TODO add in uncertainty due to obstacles

        #self.add_topology_factors(self.graph)
        # #Add priors to the residuals
        # means = np.zeros(len(self.all_residual_symbols))
        # stds = np.ones(len(self.all_residual_symbols))  * self.residual_prior_uncertainty
        # for residual_symbol,mean,std in zip(self.all_residual_symbols,means,stds):
        #     noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([std ]))
        #     self.add_measurement_factor(self.graph,residual_symbol,np.array(
        #             [mean]), noise)

        # 3 Graph and values are ready to be conditioned
        if GRAPH_TYPE=="isam":
            self.need_to_do_first_update = True
        else:
            self.mesh_graph = self.graph.clone()
            self.base_graph = self.graph.clone()

    def add_topology_factors(self,graph):
        distances, indexes = self.sensed_locations_tree.query(
            self.grid.get_sensed_locations(), k=self.sensed_value_linkage)
        for i, sensed_location, linked_sensed_location_distances, linked_sensed_location_idxs in zip(range(len(self.grid.get_sensed_locations())), self.grid.get_sensed_locations(), distances, indexes):
            analytical_symbol = gtsam.Symbol("a", i)
            residual_symbol = gtsam.Symbol("r", i)

            self.all_symbols.append(analytical_symbol.key())
            self.all_analytical_symbols.append(analytical_symbol.key())
            self.all_symbols.append(residual_symbol.key())
            self.all_residual_symbols.append(residual_symbol.key())


            # TODO review this to make sure it makes sense
            min_dist = np.min(linked_sensed_location_distances[1:])
            for linked_sensed_location_distance, linked_sensed_location_idx in zip(linked_sensed_location_distances[1:], linked_sensed_location_idxs[1:]):
                # if linked_sensed_location_distance > min_dist:
                #     continue
                sigma = self.distance_to_noise_model(
                    linked_sensed_location_distance)
                linked_analytical_symbol = self.sensed_locations_to_analytical_symbols[tuple(
                    self.sensed_locations_tree.data[linked_sensed_location_idx])]
                linked_residual_symbol = self.sensed_locations_to_residual_symbols[tuple(
                    self.sensed_locations_tree.data[linked_sensed_location_idx])]

                # self.graph.add(gtsam.BetweenFactorVector(
                #     symbol.key(), linked_symbol, np.array([0.0]), sigma))
                self.add_between_factor(graph,analytical_symbol.key(),linked_analytical_symbol,np.array([0.0]),sigma)
                self.add_between_factor(graph,residual_symbol.key(),linked_residual_symbol,np.array([0.0]),sigma)


    
    def on_replacing_lights_first(self, sensed_locations,predicted_lighting, predicted_lighting_variance, 
                                    intial_locations, initial_values):
         # Need to condition model
        if GRAPH_TYPE!="isam":
            self.base_graph = self.mesh_graph.clone()
        # else:
        #     self.incremental_graph.resize(0)
        self.last_lightinging_factor_idxs = []    
        for sensed_location, analytical_predicted_measurement, cur_predicted_lighting_variance in zip(sensed_locations, predicted_lighting,predicted_lighting_variance):
            symbol = self.sensed_locations_to_analytical_symbols[tuple(
                sensed_location)]
            # self.base_graph.add(gtsam.PriorFactorVector(symbol, np.array(
            #     [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model))
            if GRAPH_TYPE=="isam":
                self.add_measurement_factor(self.incremental_graph,symbol,np.array(
                    [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model)
            else:
                self.add_measurement_factor(self.base_graph,symbol, np.array(
                    [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model)
                self.last_lightinging_factor_idxs.append(self.incremental_graph.size())
        self.add_topology_factors(self.incremental_graph)        
        if GRAPH_TYPE=="isam":
            #This makes it not go ILS
            self.condition_graph(self.incremental_graph,intial_locations,initial_values,np.ones(initial_values.shape))

        if GRAPH_TYPE == "isam":
            #TODO this is wrong since it will either contain too many factors (it contains the )
            results = self.isam.update(self.incremental_graph,self.current_estimate)

            self.need_to_do_first_update = False
            self.last_lightinging_factor_idxs = results.getNewFactorsIndices()
        else:
            # Need to recondition graph and erase cache
            self.condition_graph(self.base_graph, self.Xs, self.Ys,self.measurement_age)
            self.graph = self.base_graph.clone()
        self.cache = dict()
        # It has to be a numpy array
        self.cur_prior = (np.array([float('-inf')]), np.array([float("-inf")]))



    def on_replacing_lights(self, sensed_locations, predicted_lighting, predicted_lighting_variance):
        # Need  to condition model
        self.clear_prior()
        self.optimize()
        if GRAPH_TYPE!="isam":
            self.base_graph = self.mesh_graph.clone()
        else:
            self.incremental_graph.resize(0)
            
        for sensed_location, analytical_predicted_measurement, cur_predicted_lighting_variance in zip(sensed_locations, predicted_lighting,predicted_lighting_variance):
            symbol = self.sensed_locations_to_analytical_symbols[tuple(
                sensed_location)]
            # self.base_graph.add(gtsam.PriorFactorVector(symbol, np.array(
            #     [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model))
            if GRAPH_TYPE=="isam":
                self.add_measurement_factor(self.incremental_graph,symbol,np.array(
                    [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model)
            else:
                self.add_measurement_factor(self.base_graph,symbol, np.array(
                    [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model)

        #Turn the posterior on the analytical lighting model to the prior 
        if self.last_lightinging_factor_idxs is not None:

            #means,stds = self.compute_and_cache(None, self.all_residual_symbols, return_std=True, cache=False)
            Xs = np.unique(self.Xs,axis=0)
            for x in Xs:
            #for residual_symbol in self.all_residual_symbols:# in sel,mean,std in zip(self.all_residual_symbols,means,stds):
                residual_symbol = self.sensed_locations_to_residual_symbols[tuple(x)]
                mean = self.current_estimate.atVector(residual_symbol)[0] 
                std = self.isam.marginalCovariance(residual_symbol)[0][0]
                noise = gtsam.noiseModel.Diagonal.Variances(np.array([std]))
                if GRAPH_TYPE=="isam":
                    self.add_measurement_factor(self.incremental_graph,residual_symbol,np.array(
                        [mean]), noise)
                else:
                    self.add_measurement_factor(self.base_graph,residual_symbol,np.array(
                        [mean]), noise)
        #self.incremental_graph.saveGraph("priors_only.dot",self.current_estimate)
        self.add_topology_factors(self.incremental_graph)
        if GRAPH_TYPE == "isam":
            #Restart ISAM because we don't want to carry the old measurement factors with us (they're carried in the new priors on the residual)
            parameters = gtsam.ISAM2Params()
            parameters.enableRelinearization = True
            self.isam = gtsam.ISAM2(parameters)
            results = self.isam.update(self.incremental_graph,self.current_estimate)



            self.need_to_do_first_update = False
            self.last_lightinging_factor_idxs = results.getNewFactorsIndices()
                #residual_results = self.isam.update(self.residual_incremental_factor_graph,gtsam.Values())
            #self.last_residual_posterior_factors = residual_results.getNewFactorIndices()

                #dont condition the residual graph since this is the first lighting placement

        else:
            # Need to recondition graph and erase cache
            self.condition_graph(self.base_graph, self.Xs, self.Ys,self.measurement_age)
            self.graph = self.base_graph.clone()
        self.cache = dict()
        # It has to be a numpy array
        self.cur_prior = (np.array([float('-inf')]), np.array([float("-inf")]))
    
    def condition_graph(self,graph,X,Y,measurement_ages):
        for location, measurement, age in zip(X,Y,measurement_ages):
            analytical_symbol = self.sensed_locations_to_analytical_symbols[tuple(location)]
            residual_symbol = self.sensed_locations_to_residual_symbols[tuple(location)]

            graph.add(gtsam.CustomFactor(self.measurement_uncertainty_noise_model(age),
                        [analytical_symbol,residual_symbol],partial(additive_factor,np.array(measurement))))


    
    def find_symbols(self, xs):
        symbols = list(
            map(lambda x: (self.sensed_locations_to_analytical_symbols[tuple(x)],self.sensed_locations_to_residual_symbols[(tuple(x))]), xs))
        return symbols


    def compute_and_cache(self,hash_key,symbols,return_std,cache=True):
        # optimizer = gtsam.LevenbergMarquardtOptimizer(
        #         self.graph, self.current_estimate, self.get_lm_params())
        # self.current_estimate = optimizer.optimize()
        self.optimize()

        predicted_outputs = []
        for analytical_symbol,residual_symbol in symbols:
            if GRAPH_TYPE=="nonlinear" or GRAPH_TYPE=="isam":
                predicted_outputs.append(
                    self.current_estimate.atVector(analytical_symbol)[0] + self.current_estimate.atVector(residual_symbol)[0])
            elif GRAPH_TYPE=="linear":
                predicted_outputs.append(
                    self.current_estimate.at(analytical_symbol)[0] + self.current_estimate.at(residual_symbol)[0])
            else:
                raise Exception()

        if not return_std:
            return np.array(predicted_outputs)
        if return_std:
            predicted_stds, marginals_dict = self.compute_marginals(symbols)
            if cache:
                self.cache[hash_key] = (deepcopy(self.current_estimate), marginals_dict)
            return np.array(predicted_outputs), np.array(predicted_stds)

    def compute_marginals(self,symbols):
        predicted_stds = []
        marginals_dict = dict()
        if GRAPH_TYPE != "isam":
            marginals = gtsam.Marginals(self.graph, self.current_estimate)
        else:
            marginals = self.isam
        for analytical_symbol,residual_symbol in symbols:
            analytical_std = np.sqrt(marginals.marginalCovariance(analytical_symbol)[0][0])
            #try:
            residual_std = np.sqrt(marginals.marginalCovariance(residual_symbol)[0][0])
            # except IndexError as e:
            #     #TODO find out why this is, likely  it's because this isn't involved in enough factors
            #     #TODO 0.0 is probably not the correct value
            #     residual_std = 0.0

            predicted_stds.append(analytical_std + residual_std)
            marginals_dict[analytical_symbol] = analytical_std
            marginals_dict[residual_symbol] = residual_std

        for symbol in self.all_residual_symbols:
            if symbol not in marginals_dict:
                #try:
                std = np.sqrt(marginals.marginalCovariance(symbol)[0][0])
                # except IndexError as e:
                #     #TODO same problem as line 261
                #     std = 0.0
                marginals_dict[symbol] = std

        for symbol in self.all_analytical_symbols:
            if symbol not in marginals_dict:
                std = np.sqrt(marginals.marginalCovariance(symbol)[0][0])
                marginals_dict[symbol] = std
        
        return predicted_stds, marginals_dict

    def return_cached_version(self,hash_key,symbols,return_std):
        values,marginals_dict = self.cache[hash_key]
        predicted_outputs = []
        predicted_stds = []
        for residual_symbol,analytical_symbol in symbols:
            if GRAPH_TYPE=='nonlinear' or GRAPH_TYPE=="isam":
                predicted_outputs.append(values.atVector(analytical_symbol)[0] + values.atVector(residual_symbol)[0])
            elif GRAPH_TYPE=='linear':
                predicted_outputs.append(values.atVector(analytical_symbol)[0] + values.at(residual_symbol)[0])
            else:
                raise Exception()

            predicted_stds.append(marginals_dict[analytical_symbol] + marginals_dict[residual_symbol] )
        if not return_std:
            return np.array(predicted_outputs)
        if return_std:
            return np.array(predicted_outputs), np.array(predicted_stds)

    def predict_ambient(self,Xs,return_std=False):
        symbols = list(map(lambda x: self.sensed_locations_to_residual_symbols[tuple(x)], Xs))
        means = []
        stds = []
        if GRAPH_TYPE != "isam":
            marginals = gtsam.Marginals(self.graph, self.current_estimate)
        else:
            marginals = self.isam
        for symbol in symbols:
            means.append(self.current_estimate.atVector(symbol)[0])
            if return_std:
                stds.append(np.sqrt(marginals.marginalCovariance(symbol)[0][0]))

        if return_std:
            return np.array(means), np.array(stds)
        else:
            return np.array(means)
    
    def predict_analytical(self,Xs,return_std=False):
        symbols = list(map(lambda x: self.sensed_locations_to_analytical_symbols[tuple(x)], Xs))
        means = []
        stds = []
        if GRAPH_TYPE != "isam":
            marginals = gtsam.Marginals(self.graph, self.current_estimate)
        else:
            marginals = self.isam
        for symbol in symbols:
            means.append(self.current_estimate.atVector(symbol)[0])
            if return_std:
                stds.append(np.sqrt(marginals.marginalCovariance(symbol)[0][0]))

        if return_std:
            return np.array(means), np.array(stds)
        else:
            return np.array(means)