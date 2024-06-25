import numpy as np

from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.factor_graph import FactorGraphDataModel
from sample_sim.data_model.requires_light_information import RequiresLightInformation
from sample_sim.planning.pomcpow.pomcpow import HashableNumpyArray

#This model just tries to predict the difference between the lighting model and reality
class AdditiveLightingModel(DataModel,RequiresLightInformation):
    def __init__(self, data_model: DataModel,logger, verbose=False):
        super().__init__(logger, verbose)
        self.model = data_model
        self.lighting_model = None
        self.lighting_variances = None

    #Update is non-reversible (used for real data)
    def update(self, X, Y):
        super().update(X, Y)
        self.model.update(X,self.__get_ambient_component_of_measurement__(X,Y))
    
    def clear_prior(self):
        try:
            self.model.clear_prior()
        except AttributeError:
            pass

    def __get_ambient_component_of_measurement__(self,X,Y):
        #This subtracts the lighting model computed by the analytical model so the model only models 
        if self.lighting_model is None:
            return Y
        Y_ambient = []
        for x,y in zip(X,Y):
            Y_ambient.append(y - self.lighting_model[HashableNumpyArray(x)])
        return np.array(Y_ambient)
    
    def predict_ambient(self,Xs,return_std=False):
        return self.model.query_many(Xs,return_std)

    def age_measurements(self):
        if isinstance(self.model, RequiresLightInformation):
            return self.model.age_measurements()

    #Used to condition the model (for hypothetical measurements)
    def update_prior(self,X,Y):
        self.model.update_prior(X,self.__get_ambient_component_of_measurement__(X,Y))

    def on_replacing_lights(self, sensed_locations, predicted_lighting, predicted_lighting_variance):
        self.lighting_model = dict()
        self.lighting_variances = dict()
        for location,lighting, variance in zip(sensed_locations,predicted_lighting, predicted_lighting_variance):
            self.lighting_model[HashableNumpyArray(location)] = lighting
            self.lighting_variances[HashableNumpyArray(location)] = variance
            
    def query_many_implementation__(self, Xs, return_std=True):
        if self.lighting_model is None:
            return self.model.query_many(Xs,return_std)
        if return_std:
            Y_ambient, Y_ambient_std = self.model.query_many(Xs,return_std)
        else:
            Y_ambient = self.model.query_many(Xs,return_std)
            Y_ambient_std = np.zeros(Y_ambient.shape)
        Y = [] 
        Y_std = []
        for x,y_ambient,y_ambient_std in zip(Xs,Y_ambient, Y_ambient_std):
            Y.append(y_ambient + self.lighting_model[HashableNumpyArray(x)]) 
            Y_std.append(np.sqrt((y_ambient_std**2) + self.lighting_variances[HashableNumpyArray(x)]))
        Y = np.array(Y)
        Y_std = np.array(Y_std)
        if return_std:
            return Y, Y_std
        else:
            return Y

        


