import abc

class RequiresLightInformation(abc.ABC):
    @abc.abstractmethod
    def on_replacing_lights(self, sensed_locations,predicted_lighting, predicted_lighting_variance):
        pass
    
    @abc.abstractmethod
    def age_measurements(self):
        pass
