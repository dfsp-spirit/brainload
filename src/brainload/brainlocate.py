"""
Find vertices closest to a given coordindate in a brain surface mesh. Some of these functions require scipy, which is an optional dependency and needs to be installed manually.
"""
import numpy as np
import brainload as bl


class BrainLocate:

    def __init__(self, vert_coords, faces):
        self.vert_coords = vert_coords
        self.faces = faces

    def init_from_subject_surface(subject_id, subjects_dir, surf, hemi):
        self.vert_coords, self.faces, _ = bl.subject_mesh(subject_id, subjects_dir, surf=surf, hemi=hemi)

    def get_closest_vertices(self, query_coords, k=5):
        """
        Find the k closest vertices to each of the given coordinates.
        """
        return self._get_closest_vertices_bruteforce(self, query_coords, k)


    def get_closest_vertex(self, query_coords):
        """
        Find the vertex closest to each of the given coordinates. Requires scipy.

        Returns
        -------
        The index of the closest vertex.
        """
        from scipy.spatial.distance import cdist
        return np.argmin(cdist(self.vert_coords, query_coords), axis=0)

    def get_closest_vertex_and_distance(self, query_coords):
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(self.vert_coords, query_coords)
        vert_indices = np.argmin(dist_matrix, axis=0)
        vert_distances = np.min(dist_matrix, axis=0)
        return np.column_stack((vert_indices, vert_distances))


    def _get_closest_vertices_bruteforce(self, query_coords, k):
        """
        Find the k closest vertices to each of the given coordinates using brute force.
        """
        pass


    def _get_closest_vertices_kdtree(self, query_coords, k):
        """
        Find the closest k vertices using a kdtree.
        """
