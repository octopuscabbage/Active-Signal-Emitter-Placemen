import math

import numpy as np
from scipy.spatial import ConvexHull


class RotateSpatialDataToAxesAlignedBox():
    '''
    This class takes a dataset of x,y points, finds a minimum bounding box around them then rotates the points so the minumum bounding box is axes aligned
    '''

    def process_points(self, points):
        corner_points = self.minimum_bounding_rectangle(points)

        # find minium axes aligning rotation
        min_angle = np.inf
        best_origin = None
        for origin, destination in zip(range(4), range(1,4)):
            current_angle = self.angle_between_points(corner_points[origin,:], corner_points[destination,:])
            if current_angle < min_angle:
                min_angle = current_angle
                best_origin = corner_points[origin,:]

        return self.rotate(best_origin, points, min_angle)

    def angle_between_points(self, origin, point):
        dx = point[0] - origin[0]
        dy = point[1] - origin[1]
        return math.atan2(dy, dx)

    def minimum_bounding_rectangle(self, points):
        """
        Find the smallest bounding rectangle for a set of points.
        Returns a set of points representing the corners of the bounding box.

        :param points: an nx2 matrix of coordinates
        :rval: an nx2 matrix of coordinates
        """
        pi2 = np.pi / 2.

        # get the convex hull for the points
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points) - 1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        # XXX both work
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)]).T
        #     rotations = np.vstack([
        #         np.cos(angles),
        #         -np.sin(angles),
        #         np.sin(angles),
        #         np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval

    def rotate(self, origin, points, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px = points[:, 0]
        py = points[:, 1]

        out = np.zeros(points.shape)
        out[:, 0] = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        out[:, 1] = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return out
