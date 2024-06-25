import numpy as np
from joblib import Memory
from tqdm import tqdm
from scipy.spatial import KDTree
from sample_sim.action.grid import FinitePlanningGrid
from scipy.interpolate import interp1d


from sample_sim.motion_models.DubinsAirplane.DubinsWrapper import euc_dist

memory = Memory("cache", verbose=0)

@memory.cache
def cached_quantize_to_sensed_locations(points, sensed_locations):
    return quantize_to_sensed_locations(points,sensed_locations)

def quantize_to_sensed_locations(points, sensed_locations):
    #TODO should figure out which is larger and use that as the query since it's O(tree nodes log (search nodes))
    print(sensed_locations.shape)
    tree = KDTree(sensed_locations)
    print(points.shape)
    distance, indexes = tree.query(points)
    return tree.data[indexes,:]
    


def cross_survey(num_sample_points,grid: FinitePlanningGrid,debug=True):
    #https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points
    locations = grid.get_sensed_locations()
    xmin = np.min(locations[:,0])
    xmax = np.max(locations[:,0])
    ymin = np.min(locations[:,1])
    ymax = np.max(locations[:,1])

    x = [xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymax, ymin, ymax, ymin]
    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, x ), interp1d( distance, y )

    alpha = np.linspace(0, 1, int(num_sample_points))
    x_regular, y_regular = fx(alpha), fy(alpha)
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x, y, 'o-');
        plt.plot(x_regular, y_regular, 'or');
        plt.axis('equal');

    return np.stack((x_regular,y_regular),axis=1)