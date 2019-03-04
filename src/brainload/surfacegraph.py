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
    A graph representing a brain surface mesh.
    """

    def __init__(self, verts, faces):
        self.graph = nx.Graph()
        for v_idx, v in enumerate(verts):
            self.graph.add_node(v_idx, x=v[0], y=v[1], z=v[2])
        for f_idx, f in enumerate(faces):
            self.graph.add_edges_from([(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])


    def get_neighbors_up_to_dist(self, source_vert, dist):
        dist_dict = nx.single_source_shortest_path_length(self.graph, source_vert, cutoff=dist)
        return dist_dict.keys()
