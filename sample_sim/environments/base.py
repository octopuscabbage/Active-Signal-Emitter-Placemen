import abc
import numpy as np

class BaseEnvironment(abc.ABC):
    def __init__(self, workspace):
        self.workspace = workspace

    @abc.abstractmethod
    def sample(self, xs):
        pass

    @abc.abstractmethod
    def action_model(self):
        pass

    @abc.abstractmethod
    def dimensionality(self):
        pass

    @abc.abstractmethod
    def get_starting_location(self):
        pass

    def is_inside(self, p):
        return self.workspace.is_inside(p)

    def filter_by_inside(self, points, already_in_box_constraints=False):
        if already_in_box_constraints and self.workspace.is_rectangular():
            return points
        return np.array(list(filter(lambda point: self.workspace.is_inside(point), points)))

    def meshgrid(self, grid_spacing):
        grid = self.workspace.get_meshgrid(grid_spacing=grid_spacing)
        if self.workspace.is_rectangular():
            return grid
        else:
            return self.filter_by_inside(grid, True)

    def get_training_points_and_values(self,spacing=30):
        points = self.meshgrid(spacing)
        values = self.sample(points)
        return points, values


    @abc.abstractmethod
    def get_parameters(self):
        pass
    
    def show_matplotlib(self):
        print("Show matplotlib method not implemented")
        pass

    def show_ascii(self):
        print("Show ascii method not implemented")
        pass