import numpy as np

import math
from func_timeout import func_set_timeout, FunctionTimedOut
import warnings

from sample_sim.motion_models.DubinsAirplane.DubinsAirplaneFunctions import MinTurnRadius_DubinsAirplane, DubinsAirplanePath, \
    ExtractDubinsAirplanePath


class DubinsWrapper():
    def __init__(self, auv):
        self.speed = auv.max_velocity
        self.max_bank = auv.max_bank
        self.max_gamma = auv.max_gamma
        self.R_min = MinTurnRadius_DubinsAirplane( self.speed,self.max_bank)

    def check_feasible(self,p1, p2):
        return ( np.linalg.norm(p1[0:2] - p2[0:2],ord=2) > 6*self.R_min ) #TODO check this 6 makes esnse

    @func_set_timeout(60)
    def get_solution(self, p1, p2):
        DubinsAirplaneSolution = DubinsAirplanePath(p1, p2, self.R_min,self.max_gamma)
        ps =  ExtractDubinsAirplanePath(DubinsAirplaneSolution, step=1).T
        thetas = []
        for i,p in enumerate(ps[:-1]):
            next_p = ps[i+1]
            thetas.append(math.atan2(next_p[1]-p[1],next_p[0]-p[0]))
        thetas.append(p2[3])
        thetas = np.array(thetas)
        out_points =  np.hstack((ps,thetas.reshape(-1,1)))
        return out_points

class DubinsException(RuntimeError):
    pass

class DubinsTimeout(DubinsException):
    pass

def dubins_path(points, auv,include_auv_as_start=True):
    max_velocity = 10 #TOOD I THINK THIS IS CONSTANT VELOCITY
    d = DubinsWrapper(auv) #TODO check these parameters
    path = []
    if include_auv_as_start:
        last_point = np.array([auv.x, auv.y, auv.z, auv.theta, max_velocity])
        starting_idx = 0
        if (points[0] == last_point):
            warnings.warn("Points array contains AUV current position, don't do this")
            starting_idx = 1
    else:
        starting_idx = 1
        last_point = np.array([points[0][0], points[0][1], points[0][2], auv.theta, max_velocity])
    for i in range(starting_idx, len(points)):
        if i +1 < len(points):
            desired_theta = math.atan2(points[i+1][1] - points[i][1], points[i+1][0] - points[i][0])
        else:
            desired_theta = 0 #TODO figure out what orientation to put it
        next_point = np.array([points[i][0], points[i][1], points[i][2], desired_theta, max_velocity])
        if d.check_feasible(last_point,next_point):
            try:
                solution = d.get_solution(last_point,next_point).tolist()
            except FunctionTimedOut:
                raise DubinsTimeout("No Dubins path, timeout")
        else:
            raise DubinsException("No Dubins path, infeasible")
        path += solution
        last_point = next_point
    return path

def dubins_length(dubins_path):
    distance = 0
    last_point = dubins_path[0]
    for point in dubins_path[1:]:
        distance += euc_dist(last_point[:3],point[:3])
        last_point = point
    return distance

def euc_dist(p1,p2):
    return np.linalg.norm(p2-p1,ord=2)



if __name__ == "__main__":
    p0 = np.array([1,1,1,0,1])
    p1 = np.array([10,10,10,0,1])

    points = [[161, 110, 17],
              [0,0,0],
              [322,220,17],
              [0,220,0],
              [322,0,17]]

    auv = AUV(0,0,0)
    dubins = DubinsWrapper(auv)


    path_dubins_airplane = np.array(dubins_path(points,auv))

    plot3(path_dubins_airplane[:, 0], path_dubins_airplane[:, 1], path_dubins_airplane[:, 2], 'o', 'g')
