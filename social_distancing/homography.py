"""
"""
import cv2
import numpy as np

class Homography:
    def __init__(self, camera_points, map_points):
        self.H = Homography._compute_homography(camera_points, map_points)

    @staticmethod
    def _compute_homography(camera_points, map_points):
        """
        Returns the homography matrix given 4 points for the image plane and 
        4 points for the 2D map plane. The result is a 3x3 matrix which must
        be applied to every point detected in the image plane.
        """
        h, _ = cv2.findHomography(camera_points, map_points)
        return h

    def project(self, camera_point):
        """
        Projects the given camera point onto the map plane using the previously
        computed homography.
        """
        camera_point = np.array(camera_point)
        assert camera_point.shape == (2,)
        camera_point = np.resize(np.array(camera_point), 3).reshape(3, 1)        
        camera_point[2][0] = 1
        point = np.matmul(self.H, camera_point)
        return np.array([point[0][0] / point[2][0], point[1][0] / point[2][0]])
