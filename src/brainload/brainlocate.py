"""
Find vertices closest to a given coordinate in a brain surface mesh. Some of these functions require scipy, which is an optional dependency and needs to be installed manually.
"""
import numpy as np
import brainload as bl


class BrainLocate:

    def __init__(self, vert_coords, faces):
        self.vert_coords = vert_coords
        self.faces = faces

    def get_closest_vertex(self, query_coords):
        """
        Find the vertex closest to each of the given coordinates. Requires scipy.

        Parameters
        ----------
        query_coords: numpy 2D array
            The coordinates, given as a numeric array with shape (n, 3) for n coords.

        Returns
        -------
        numpy 1D array
            Array with shape (n, ). Each value represents the index of the vertex closest to the respective query coordinate.
        """
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(self.vert_coords, query_coords, metric='euclidean')
        return np.argmin(dist_matrix, axis=0)


    def get_closest_vertex_and_distance(self, query_coords):
        """
        Find the vertex closest to each of the given coordinates, and the respective distance. Requires scipy.

        Parameters
        ----------
        query_coords: numpy 2D array
            The coordinates, given as a numeric array with shape (n, 3) for n coords.

        Returns
        -------
        numpy 1D array
            Array with shape (n, 2). Each row represents the index of the vertex closest to the respective query coordinate (at index 0) and the respective distance (at index 1).
        """
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(self.vert_coords, query_coords, metric='euclidean')
        vert_indices = np.argmin(dist_matrix, axis=0)
        vert_distances = np.min(dist_matrix, axis=0)
        return np.column_stack((vert_indices, vert_distances))
