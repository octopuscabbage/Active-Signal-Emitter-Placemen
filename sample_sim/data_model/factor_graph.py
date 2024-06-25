import gtsam
from sample_sim.data_model.data_model import DataModel
from scipy.spatial import KDTree
from sample_sim.action.grid import FinitePlanningGrid
from bidict import bidict
import numpy as np

from sample_sim.planning.pomcpow.pomcpow import HashableNumpyArray
from sample_sim.data_model.requires_light_information import RequiresLightInformation
from copy import deepcopy
import logging
from igraph import Graph, plot
import matplotlib.pyplot as plt

GRAPH_TYPE = "isam"

class FactorGraphDataModel(DataModel, RequiresLightInformation):
    def __init__(self, grid: FinitePlanningGrid, logger, verbose=False, sensed_value_linkage=4, 
                minimum_distance_sigma=0.001, distance_sigma_scaling=0.01, measurement_uncertainty=0.01, 
                lighting_model_uncertainty=0.05):
        super().__init__(logger, verbose)
        self.grid = grid
        # Need to add one becuase the first is the node itself
        self.sensed_value_linkage = sensed_value_linkage + 1
        self.minimum_distance_sigma = minimum_distance_sigma
        self.distance_sigma_scaling = distance_sigma_scaling
        self.lighting_model_uncertainty = lighting_model_uncertainty
        self.measurement_uncertainty = measurement_uncertainty
        self.cache = dict()
        if GRAPH_TYPE == "isam":
            self.incremental_graph = gtsam.NonlinearFactorGraph()
            parameters = gtsam.ISAM2Params()
            parameters.enableRelinearization = True
            self.isam = gtsam.ISAM2(parameters)
            self.last_lightinging_factor_idxs = None
        self.create_factor_graph()
        self.measurement_age = []
        self.lighting_model_uncertainty_noise_model =  gtsam.noiseModel.Diagonal.Sigmas(np.array([self.lighting_model_uncertainty]))
        self.added_priors = None
        self.indexes_to_remove = []


    def num_cliques(self):
        cliques = int(self.isam.__str__().split("\n")[0].split(":")[2].split(",")[0])
        logging.getLogger(self.logger).info(f"Cliques {cliques}")
        return cliques


    def distance_to_sigma(self, distance):
        return self.minimum_distance_sigma + self.distance_sigma_scaling * distance

    def distance_to_noise_model(self, distance):
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([self.distance_to_sigma(distance)]))

    # def lighting_model_uncertainty_noise_model(self):
    #     return gtsam.noiseModel.Diagonal.Sigmas(np.array([self.lighting_model_uncertainty]))

    def measurement_uncertainty_noise_model(self,age):
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([self.measurement_uncertainty * (age**2)]))

    def add_between_factor(self,graph,k1,k2,measured,noise):
        if GRAPH_TYPE=="nonlinear" or GRAPH_TYPE=='isam':
            graph.add(gtsam.BetweenFactorVector(
                        k1, k2, measured, noise))
        elif GRAPH_TYPE == 'linear':
            graph.add(k1, np.eye(1), k2, -np.eye(1), measured, noise)
        else:
            raise Exception()

    def add_measurement_factor(self,graph,k1,measured,noise):
        if GRAPH_TYPE=="nonlinear" or GRAPH_TYPE=='isam':
            graph.add(gtsam.PriorFactorVector(k1, measured,noise))
            return graph.size()
        if GRAPH_TYPE=="linear":
            graph.add(k1, np.eye(1), measured, noise)
            return graph.size()

    

    def create_factor_graph(self):
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
        self.sensed_locations_to_gtsam_symbols = bidict()

        self.sensed_locations_tree = KDTree(sensed_locations)

        # 1 add all the values
        for i, sensed_location in enumerate(sensed_locations):
            symbol = gtsam.Symbol("s", i).key()
            self.current_estimate.insert(symbol, np.array([0.0]))
            self.sensed_locations_to_gtsam_symbols[tuple(
                sensed_location)] = symbol

        # 2 link all the values to their k nearest neighbors
        # Need to make sure all values exist first
        # TODO add in uncertainty due to obstacles

        distances, indexes = self.sensed_locations_tree.query(
            sensed_locations, k=self.sensed_value_linkage)
        for i, sensed_location, linked_sensed_location_distances, linked_sensed_location_idxs in zip(range(len(sensed_locations)), sensed_locations, distances, indexes):
            symbol = gtsam.Symbol("s", i)
            self.all_symbols.append(symbol.key())

            # TODO review this to make sure it makes sense
            min_dist = np.min(linked_sensed_location_distances[1:])
            for linked_sensed_location_distance, linked_sensed_location_idx in zip(linked_sensed_location_distances[1:], linked_sensed_location_idxs[1:]):
                if linked_sensed_location_distance > min_dist:
                    continue
                sigma = self.distance_to_noise_model(
                    linked_sensed_location_distance)
                linked_symbol = self.sensed_locations_to_gtsam_symbols[tuple(
                    self.sensed_locations_tree.data[linked_sensed_location_idx])]
                # self.graph.add(gtsam.BetweenFactorVector(
                #     symbol.key(), linked_symbol, np.array([0.0]), sigma))
                self.add_between_factor(self.graph,symbol.key(),linked_symbol,np.array([0.0]),sigma)
        # 3 Graph and values are ready to be conditioned
        if GRAPH_TYPE=="isam":
            self.need_to_do_first_update = True
        else:
            self.mesh_graph = self.graph.clone()
            self.base_graph = self.graph.clone()
    
    def __filter_unique(self,X,Y):
        _,idxs = np.unique(X,return_index=True,axis=0)
        return X[idxs,:], Y[idxs]


    # Update is non-reversible (used for real data)
    def update(self, X, Y):
        if self.Xs is not None:
            X,Y = self.find_new(X,Y)
        if not X.size == 0:
            super().update(X, Y)
            self.measurement_age.extend([1] * Y.shape[0])
            if GRAPH_TYPE!="isam":
                self.base_graph = self.mesh_graph.clone()
                self.condition_graph(self.base_graph, self.Xs, self.Ys,self.measurement_age)
                self.graph = self.base_graph.clone()
            else:
                #We can't initially update the graph without some measurement factors so we need to handle the first one specially
                if not self.need_to_do_first_update:
                    self.incremental_graph.resize(0)
                #TODO this should be elements of X and Y which are not already included
                #X,Y = self.__find_new_measurements(X,Y)
                self.condition_graph(self.incremental_graph,X,Y,np.ones(Y.shape))

                if self.need_to_do_first_update:
                    self.isam.update(self.incremental_graph,self.current_estimate)
                else:
                    self.isam.update(self.incremental_graph,gtsam.Values())


                self.need_to_do_first_update = False


        self.cache = dict()
        # It has to be a numpy array
        self.cur_prior = (np.array([float('-inf')]), np.array([float("-inf")]))
        self.added_priors = None

    # Used to condition the model (for hypothetical measurements)
    def clear_prior(self):
        if GRAPH_TYPE=="isam":
            self.incremental_graph.resize(0)
            self.isam.update(self.incremental_graph, gtsam.Values(), removeFactorIndices=self.indexes_to_remove)

            self.indexes_to_remove = []
        else:
            #X,Y = self.__filter_unique(X,Y)
            self.graph = self.base_graph.clone()
        #self.__condition_graph(self.graph, X, Y,np.ones(Y.shape))
        self.cur_prior = (np.array([float('-inf')]), np.array([float("-inf")]))
        self.added_priors = None
    
    def find_new(self,X,Y):
        X_out = []
        Y_out = []
        for x,y in zip(X,Y):
            include = True
            idxs = np.where((self.Xs == (x[0], x[1])).all(axis=1))[0]
            if idxs.size == 0:
                include = True
            else:
                for idx in idxs:
                    if self.Ys[idx] == y:
                        include = False
            if include:
                X_out.append(x)
                Y_out.append(y)
        return np.array(X_out),np.array(Y_out)
                    


    # Used to condition the model (for hypothetical measurements)
    def update_prior(self, X, Y):

        X,Y = self.find_new(X,Y)
        #self.graph = self.base_graph.clone()
        if GRAPH_TYPE=="isam":
            self.incremental_graph.resize(0)
            self.condition_graph(self.incremental_graph, X, Y,np.ones(Y.shape))
            results = self.isam.update(self.incremental_graph,gtsam.Values(),removeFactorIndices=self.indexes_to_remove)

            self.indexes_to_remove = results.getNewFactorsIndices()
        else:
            self.added_priors = self.condition_graph(self.base_graph, X, Y,np.ones(Y.shape),self.added_priors)
        self.cur_prior = (X, Y)
    
    def condition_graph(self,graph,X,Y,measurement_ages):
        for location, measurement, age in zip(X,Y,measurement_ages):
            symbol = self.sensed_locations_to_gtsam_symbols[tuple(location)]
            self.add_measurement_factor(graph,symbol,np.array(
                [measurement]), self.measurement_uncertainty_noise_model(age))
        #     added_factors.append((location,measurement,idx))
        # if GRAPH_TYPE!="isam":
        #     return added_factors 

    # def condition_graph(self, graph, X, Y, measurement_ages,previously_added_priors=None):
    #     to_add = []
    #     to_remove = []

    #     if previously_added_priors is not None:
    #         to_remove_idxs = set(range(len(previously_added_priors)))
    #         for x,y,measurement_age in zip(X,Y,measurement_ages):
    #             already_added = False
    #             for i,prior in enumerate(previously_added_priors):
    #                 x_prev,y_prev,_ = prior 
    #                 if (x_prev == x).all() and y==y_prev:
    #                     already_added=True
    #                     to_remove_idxs.discard(i)
    #             if not already_added:
    #                 to_add.append((x,y,measurement_age))
    #         for to_remove_idx in to_remove_idxs:
    #             to_remove.append(previously_added_priors[to_remove_idx][2])
    #     else:
    #         for x,y,measurement_age in zip(X,Y,measurement_ages):
    #             to_add.append((x,y,measurement_age))

    #     if GRAPH_TYPE!="isam":
    #         for factor_idx in to_remove:
    #             self.graph.remove(factor_idx)

    #     added_factors = []
    #     for location, measurement, age in to_add:
    #         symbol = self.sensed_locations_to_gtsam_symbols[tuple(location)]
    #         # graph.add(gtsam.PriorFactorVector(symbol, np.array(
    #         #     [measurement]), self.measurement_uncertainty_noise_model(age)))
    #         idx = self.add_measurement_factor(graph,symbol,np.array(
    #             [measurement]), self.measurement_uncertainty_noise_model(age))
    #         # if GRAPH_TYPE=="isam":
    #         #     self.isam.getFactorsUnsafe().size() - 1
    #         added_factors.append((location,measurement,idx))
    #     if GRAPH_TYPE!="isam":
    #         return added_factors 
    #     else:
    #         return added_factors, to_remove
            
    def on_replacing_lights(self, sensed_locations, predicted_lighting, predicted_lighting_variance):
        # if GRAPH_TYPE=="isam" and self.need_to_do_first_update:
        #     self.isam.update(self.incremental_graph,self.current_estimate)
            #self.need_to_do_first_update = False
        # Need to condition model
        if GRAPH_TYPE!="isam":
            self.base_graph = self.mesh_graph.clone()
        else:
            self.incremental_graph.resize(0)
            
        for sensed_location, analytical_predicted_measurement, cur_predicted_lighting_variance in zip(sensed_locations, predicted_lighting,predicted_lighting_variance):
            symbol = self.sensed_locations_to_gtsam_symbols[tuple(
                sensed_location)]
            # self.base_graph.add(gtsam.PriorFactorVector(symbol, np.array(
            #     [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model))
            if GRAPH_TYPE=="isam":
                self.add_measurement_factor(self.incremental_graph,symbol,np.array(
                    [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model)
            else:
                self.add_measurement_factor(self.base_graph,symbol, np.array(
                    [analytical_predicted_measurement]), self.lighting_model_uncertainty_noise_model)

        if GRAPH_TYPE == "isam":
            if self.last_lightinging_factor_idxs is not None:
                results = self.isam.update(self.incremental_graph, gtsam.Values(), removeFactorIndices=self.last_lightinging_factor_idxs)

            else:
                results = self.isam.update(self.incremental_graph,gtsam.Values())

            self.last_lightinging_factor_idxs = results.getNewFactorsIndices()
        else:
            # Need to recondition graph and erase cache
            self.condition_graph(self.base_graph, self.Xs, self.Ys,self.measurement_age)
            self.graph = self.base_graph.clone()
        self.cache = dict()
        # It has to be a numpy array
        self.cur_prior = (np.array([float('-inf')]), np.array([float("-inf")]))

    def age_measurements(self):
        self.measurement_age = list(map(lambda x: x+1, self.measurement_age))

    def return_cached_version(self,hash_key,symbols,return_std):
        values,marginals_dict = self.cache[hash_key]
        predicted_outputs = []
        predicted_stds = []
        for symbol in symbols:
            if GRAPH_TYPE=='nonlinear' or GRAPH_TYPE=="isam":
                predicted_outputs.append(values.atVector(symbol)[0])
            elif GRAPH_TYPE=='linear':
                predicted_outputs.append(values.at(symbol)[0])
            else:
                raise Exception()

            predicted_stds.append(marginals_dict[symbol])
        if not return_std:
            return np.array(predicted_outputs)
        if return_std:
            return np.array(predicted_outputs), np.array(predicted_stds)

    def get_lm_params(self):
        lm_params = gtsam.LevenbergMarquardtParams()
        # lm_params.setVerbosityLM("SUMMARY")
        lm_params.setMaxIterations(20)
        return lm_params


    def optimize(self):
        if GRAPH_TYPE=="nonlinear":
        # optimizer = gtsam.LevenbergMarquardtOptimizer(
        #         self.graph, self.current_estimate, self.get_lm_params())

            gn_params = gtsam.GaussNewtonParams()
            gn_params.setMaxIterations(10)
            gn_params.setAbsoluteErrorTol(1e-2)
            optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.current_estimate, gn_params)
            self.current_estimate = optimizer.optimize()
        elif GRAPH_TYPE=="linear":
            self.current_estimate = self.graph.optimize()#optimizer.optimize()
        elif GRAPH_TYPE == "isam":
            self.isam.update()
            self.current_estimate = self.isam.calculateEstimate()
        else:
            raise Exception()
    
    def compute_marginals(self,symbols):
        predicted_stds = []
        marginals_dict = dict()
        if GRAPH_TYPE != "isam":
            marginals = gtsam.Marginals(self.graph, self.current_estimate)
        else:
            marginals = self.isam
        for symbol in symbols:
            std = np.sqrt(marginals.marginalCovariance(symbol)[0][0])
            predicted_stds.append(std)
            marginals_dict[symbol] = std
        for symbol in self.all_symbols:
            if symbol not in marginals_dict:
                std = np.sqrt(marginals.marginalCovariance(symbol)[0][0])
                marginals_dict[symbol] = std
        
        return predicted_stds, marginals_dict


    def compute_and_cache(self,hash_key,symbols,return_std,cache=True):
        # optimizer = gtsam.LevenbergMarquardtOptimizer(
        #         self.graph, self.current_estimate, self.get_lm_params())
        # self.current_estimate = optimizer.optimize()
        self.optimize()

        predicted_outputs = []
        for symbol in symbols:
            if GRAPH_TYPE=="nonlinear" or GRAPH_TYPE=="isam":
                predicted_outputs.append(
                    self.current_estimate.atVector(symbol)[0])
            elif GRAPH_TYPE=="linear":
                predicted_outputs.append(
                    self.current_estimate.at(symbol)[0])
            else:
                raise Exception()

        if not return_std:
            return np.array(predicted_outputs)
        if return_std:
            predicted_stds, marginals_dict = self.compute_marginals(symbols)
            if cache:
                self.cache[hash_key] = (deepcopy(self.current_estimate), marginals_dict)
            return np.array(predicted_outputs), np.array(predicted_stds)
    
    def find_symbols(self, xs):
        symbols = list(
            map(lambda x: self.sensed_locations_to_gtsam_symbols[tuple(x)], xs))
        return symbols


    def query_many_implementation__(self, Xs, return_std=True):
        hash_key = (HashableNumpyArray(self.cur_prior[0]), HashableNumpyArray(self.cur_prior[1]))
        symbols = self.find_symbols(Xs)
        if hash_key in self.cache:
            return self.return_cached_version(hash_key,symbols,return_std)
        else:
            return self.compute_and_cache(hash_key,symbols,return_std) 

    def save_graphviz(self,budget):
        self.isam.getFactorsUnsafe().saveGraph(f"budget_{budget}.dot",self.current_estimate)
