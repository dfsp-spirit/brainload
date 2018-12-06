import brainload as bl
import os
import matplotlib
import brainload.meshexport as me

subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'tim_only')

vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('tim', subjects_dir=subjects_dir, measure='curv', fwhm='10')
vertex_colors = me.scalars_to_colors_matplotlib(morphometry_data, 'viridis')
ply_string = me.mesh_to_ply(vert_coords, faces, vertex_colors=vertex_colors)
print(ply_string)
