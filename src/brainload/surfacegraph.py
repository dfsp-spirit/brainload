# -*- coding: utf-8 -*-
"""
Turn a surface mesh into a networkx graph. Useful for asking questions that can be answered using graph algorithms. An example would be to find, for a given source vertex, all vertices which are connected to it by a certain number of hops. Requires networkx.
"""

import numpy as np
import brainload as bl
import brainload.freesurferdata as blfsd
import networkx as nx

class SurfaceGraph:
    """
    A graph representing the vertices and edges of a brain surface mesh.
    """

    def __init__(self, verts, faces):
        """
        Init the graph from vertices and faces.

        Init the graph from vertices and faces. Note that the faces are not stored directly in the graph, they are turned into edges and the information which edges form a face is lost (this is a graph, not a mesh representation). Also note that the verts and faces parameters which are required here fit the return values of any brainload function which loads a brain mesh, e.g., the ```subject_mesh``` function. When this constructor has finished, a networkx graph of the mesh is available at ```surface_graph_instance.graph```.

        Parameters
        ----------
        verts: numpy 2D float array
            The vertices in an array with shape (n, 3) for n vertices. The 3 floats are the x, y and z coords of the vertex. The vertex is identified by its index in the array.

        faces: numpy 2D int array
            The faces in an array with shape (m, 3). Each of the m faces is identified by the indices of the 3 vertices that form it.
        """
        self.graph = nx.Graph()
        for v_idx, v in enumerate(verts):
            self.graph.add_node(v_idx, x=v[0], y=v[1], z=v[2])
        for f_idx, f in enumerate(faces):
            self.graph.add_edges_from([(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])


    def get_neighbors_up_to_dist(self, source_vert, dist):
        """
        Get all neighbors up to graph distance dist away.

        Get all neighbors up to graph distance dist away. Note that dist is the number of edges to traverse in the graph to get from the source to the vertex. (It is NOT Euclidian distance.)

        Parameters
        ----------
        source_vert: int
            Index of the source vertex.

        dist: The distance up until which neighbors should be returned. This is the number of edges to traverse in the graph (along a shortest path of length dist from source to the respective neighbor).

        Returns
        -------
        neighbors: list
            The indices of all vertices which lie within distance dist of the source_vert.
        """
        dist_dict = nx.single_source_shortest_path_length(self.graph, source_vert, cutoff=dist)
        return dist_dict.keys()
