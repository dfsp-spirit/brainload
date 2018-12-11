import brainload as bl
import os
import matplotlib
import brainload.meshexport as me

subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'tim_only')

vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('tim', subjects_dir=subjects_dir, measure='area', fwhm='20')
#vertex_colors= me.scalars_to_colors(morphometry_data, cmap)
#ply_string = me.mesh_to_ply(vert_coords, faces, vertex_colors=vertex_colors)
obj_string = me.mesh_to_obj(vert_coords, faces)
#print ply_string
print obj_string
