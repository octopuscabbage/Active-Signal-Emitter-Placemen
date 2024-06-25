import typing

import abc
import numpy as np
try:
    import rasterio
    from rasterio.io import MemoryFile
except ModuleNotFoundError:
    pass

from numpy.random.mtrand import RandomState

from sample_sim.sensors.camera_sensor import FixedHeightCamera
from ros.constants import *


class Workspace(abc.ABC):
    # @abc.abstractmethod
    # def is_inside(self, x, y, z):
    #     passs
    @abc.abstractmethod
    def dimensions(self) -> typing.SupportsInt:
        pass

    @abc.abstractmethod
    def get_bounds(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_point_inside(self, rs: RandomState = None):
        pass

    @abc.abstractmethod
    def get_meshgrid(self, grid_spacing):
        pass


    @abc.abstractmethod
    def get_meshgrid_with_resolution(self, resolution):
        pass

    @abc.abstractmethod
    def is_inside(self,p):
        pass

    @abc.abstractmethod
    def is_rectangular(self):
        pass

class TemporalRectangularPlaneWorkspace(Workspace):
    def __init__(self,xmin,xmax,ymin,ymax,tmin,tmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.tmin = tmin
        self.tmax = tmax

    def is_rectangular(self):
        return True


    def dimensions(self) -> typing.SupportsInt:
        return 3

    def get_bounds(self) -> np.ndarray:
        return np.array([[self.xmin, self.xmax], [self.ymin, self.ymax],
            [self.tmin, self.tmax]])

    def get_point_inside(self, rs=None):
        if rs is None:
            rs = np.random.RandomState()
        return rs.uniform(self.xmin, self.xmax), \
               rs.uniform(self.ymin, self.ymax), \
               rs.uniform(self.tmin, self.tmax)

    def get_meshgrid(self, grid_spacing):
        if isinstance(grid_spacing, tuple):
            grid_spacing_x, grid_spacing_y, grid_spacing_t = grid_spacing
        else:
            grid_spacing_x = grid_spacing
            grid_spacing_y = grid_spacing
            grid_spacing_t = grid_spacing
        test_x_range = np.linspace(self.xmin, self.xmax, num=grid_spacing_x)
        test_y_range = np.linspace(self.ymin, self.ymax, num=grid_spacing_y)
        test_t_range = np.linspace(self.tmin, self.tmax, num=grid_spacing_t)

        test_x, test_y, test_t = np.meshgrid(test_x_range, test_y_range, test_t_range)
        test_x = test_x.flatten()
        test_y = test_y.flatten()
        test_t = test_t.flatten()

        t_X = np.stack((test_x, test_y, test_t), axis=-1)
        return t_X

    def get_meshgrid_with_resolution(self, resolution):
        if isinstance(resolution,tuple):
            x_resolution = resolution[0]
            y_resolution = resolution[1]
            z_resolution = resolution[2]
        else:
            x_resolution = resolution
            y_resolution = resolution
            z_resolution = resolution

        num_x_points = int((self.xmax - self.xmin) / x_resolution)
        num_y_points = int((self.ymax - self.ymin) / y_resolution)
        num_t_points = int((self.tmax - self.tmin) / z_resolution)
        return self.get_meshgrid((num_x_points, num_y_points, num_t_points))

    def is_inside(self,p):
        return self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax and self.tmin <= p[2] <= self.tmax



class RectangularPlaneWorkspace(Workspace):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def is_rectangular(self):
        return True

    def dimensions(self) -> typing.SupportsInt:
        return 2

    def get_bounds(self) -> np.ndarray:
        return np.array([[self.xmin, self.xmax], [self.ymin, self.ymax]])
    

    def is_inside(self, p):
        return self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax

    def get_point_inside(self, rs=None):
        # TODO uniform is on a closed interval, while workspace is on open interval, make sure it actually lies inside
        if rs is None:
            rs = np.random.RandomState()
        return rs.uniform(self.xmin, self.xmax), \
               rs.uniform(self.ymin, self.ymax)

    def get_meshgrid(self, grid_spacing):
        if isinstance(grid_spacing, tuple):
            grid_spacing_x, grid_spacing_y = grid_spacing
        else:
            grid_spacing_x = grid_spacing
            grid_spacing_y = grid_spacing
        test_x_range = np.linspace(self.xmin, self.xmax, num=grid_spacing_x)
        test_y_range = np.linspace(self.ymin, self.ymax, num=grid_spacing_y)

        test_x, test_y = np.meshgrid(test_x_range, test_y_range)
        test_x = test_x.flatten()
        test_y = test_y.flatten()

        t_X = np.stack((test_x, test_y), axis=-1)
        return t_X

    def get_meshgrid_with_resolution(self, resolution):
        num_x_points = int((self.xmax - self.xmin) / resolution)
        num_y_points = int((self.ymax - self.ymin) / resolution)
        return self.get_meshgrid((num_x_points, num_y_points))

class RasterWorkspace(RectangularPlaneWorkspace):
    def __init__(self,raster_fname, camera:FixedHeightCamera,  xmin,xmax,ymin,ymax, nodata_values=[0,255], channel=7):
        super().__init__(min(xmin,xmax),max(xmin,xmax),min(ymin,ymax),max(ymin,ymax))
        self.raster_fname= raster_fname
        self.nodata_values= nodata_values
        self.channel = channel
        self.open_raster = None
        self.open_filehandle = None
        self.camera = camera

    def is_rectangular(self):
        return False

    def open_dataset_in_memory(self):
        with open(self.raster_fname, 'rb') as f:
            self.open_filehandle = MemoryFile(f)
            self.open_raster = self.open_filehandle.open()
            

    def is_inside(self, p):
        if not super().is_inside(p):
            return False
        else:
            if self.open_raster is None:
                self.open_dataset_in_memory()
            view = self.camera.get_sensed_locations_from_point(p)
            values = np.array(list(self.open_raster.sample(view[:,:2],self.channel)))
            mask = np.isin(values,self.nodata_values)
            
            inside = not mask.all()
            return inside

    def get_meshgrid(self, grid_spacing):
        t_X = super().get_meshgrid(grid_spacing)
        out = []
        for p in t_X:
            if self.is_inside(p):
                out.append(p)
        return np.array(out)

    def get_point_inside(self, rs=None):
        p = super().get_point_inside(rs)
        while not self.is_inside(p):
            p = super().get_point_inside(rs)
        return p

    #The following are to allow pickling because you can't pickle an open raster
    def __getstate__(self):
        if self.open_raster is not None:
            self.open_raster.close()
            self.open_raster = None
        if self.open_filehandle is not None:
            self.open_filehandle.close()
            self.open_filehandle = None
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__ = state



class RectangularPrismWorkspace(Workspace):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def is_rectangular(self):
        return True

    def dimensions(self) -> typing.SupportsInt:
        return 3

    def get_bounds(self) -> np.ndarray:
        return np.array([[self.xmin, self.xmax], [self.ymin, self.ymax],
            [self.zmin, self.zmax]])


    def get_point_inside(self, rs=None):
        # TODO uniform is on a closed interval, while workspace is on open interval, make sure it actually lies inside
        if rs is None:
            rs = np.random.RandomState()
        return rs.uniform(self.xmin, self.xmax), \
               rs.uniform(self.ymin, self.ymax), \
               rs.uniform(self.zmin, self.zmax)

    def get_meshgrid(self, grid_spacing):
        if isinstance(grid_spacing, tuple) or isinstance(grid_spacing,list):
            grid_spacing_x, grid_spacing_y, grid_spacing_z = grid_spacing
        else:
            grid_spacing_x = grid_spacing
            grid_spacing_y = grid_spacing
            grid_spacing_z = grid_spacing
        test_x_range = np.linspace(self.xmin, self.xmax, num=grid_spacing_x)
        test_y_range = np.linspace(self.ymin, self.ymax, num=grid_spacing_y)
        test_z_range = np.linspace(self.zmin, self.zmax, num=grid_spacing_z)

        test_x, test_y, test_z = np.meshgrid(test_x_range, test_y_range, test_z_range)
        test_x = test_x.flatten()
        test_y = test_y.flatten()
        test_z = test_z.flatten()

        t_X = np.stack((test_x, test_y, test_z), axis=-1)
        return t_X
    def get_meshgrid_with_resolution(self, resolution):
        if isinstance(resolution,tuple) or isinstance(resolution,list):
            x_resolution = resolution[0]
            y_resolution = resolution[1]
            z_resolution = resolution[2]
        else:
            x_resolution = resolution
            y_resolution = resolution
            z_resolution = resolution
        num_x_points = int((self.xmax - self.xmin) / x_resolution)
        num_y_points = int((self.ymax - self.ymin) / y_resolution)
        num_z_points = int((self.zmax - self.zmin) / z_resolution)
        return self.get_meshgrid((num_x_points, num_y_points, num_z_points))

    def is_inside(self, p):
        return self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax and self.zmin <= p[2] <= self.zmax


class FieldWorkspace3d(RectangularPrismWorkspace):
    def __init__(self,easting1,northing1,easting2,northing2):
        super().__init__(easting1,easting2,northing1,northing2,0,1)
    def is_rectangular(self):
        return True

    def is_inside(self, p):
            return super().is_inside(p) and p[2] == 0.0

class RasterWorkspace3d(RectangularPrismWorkspace):
    def __init__(self, base_workspace: RasterWorkspace):
        super().__init__(base_workspace.xmin, base_workspace.xmax, base_workspace.ymin, base_workspace.ymax,0,1)
        self.base_workspace = base_workspace

    def is_rectangular(self):
        return False

    def is_inside(self, p):
        return self.base_workspace.is_inside(p) and p[2] == 0.0

    def get_meshgrid(self, grid_spacing):
        t_X = super().get_meshgrid(grid_spacing)
        out = []
        for p in t_X:
            if self.is_inside(p):
                out.append(p)
        return np.array(out)

    def get_point_inside(self, rs=None):
        p = super().get_point_inside(rs)
        while not self.is_inside(p):
            p = super().get_point_inside(rs)
        return p

class SDFWorksapce2d(RectangularPlaneWorkspace):
    def __init__(self, sdf_fn, xmin, xmax, ymin, ymax):
        self.sdf_fn = sdf_fn
        super().__init__(xmin, xmax, ymin, ymax)

    def is_rectangular(self):
        return True

    def is_inside(self, p):
        if isinstance(p,tuple) or (isinstance(p,np.ndarray) and len(p.shape) == 1):
            p_sdf = np.array([p])

        #TODO set eps more intelligently
        return super().is_inside(p) and self.sdf_fn(p_sdf) > 10**-5

class SDFWorksapce2d(RectangularPlaneWorkspace):
    def __init__(self, sdf_fn, xmin, xmax, ymin, ymax):
        self.sdf_fn = sdf_fn
        super().__init__(xmin, xmax, ymin, ymax)

    def is_rectangular(self):
        return True

    def is_inside(self, p):
        if isinstance(p,tuple) or (isinstance(p,np.ndarray) and len(p.shape) == 1):
            p_sdf = np.array([p])

        #TODO set eps more intelligently
        return super().is_inside(p) and self.sdf_fn(p_sdf) > 10**-5

class FieldSDFWorksapce2d(RectangularPlaneWorkspace):
    def __init__(self, sdf_fn, xmin, xmax, ymin, ymax):
        self.sdf_fn = sdf_fn
        super().__init__(xmin, xmax, ymin, ymax)

    def is_rectangular(self):
        return True

    def is_inside(self, p):
        if isinstance(p,tuple) or (isinstance(p,np.ndarray) and len(p.shape) == 1):
            p_sdf = np.array([p])

        #TODO set eps more intelligently
        return super().is_inside(p) and self.sdf_fn(p_sdf) > OBSTACLE_BUFFER_m