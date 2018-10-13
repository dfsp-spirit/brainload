import mayavi.mlab as mlab
import numpy as np
from brainload.spatial_transform import *

def print_mlab_view(view):
    '''See http://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html#view for details.'''
    print "[View] azimuth=%f, elevation=%f, distance=%f" % (view[0], view[1], view[2])

def plot_brain_data_on_mesh(vert_coords, faces, morphology_data, meta_data, do_display_interactively=True, export_image_file=None, draw_mesh_only=False, use_multi_view=True):
    '''Plots the morphology data onto the brain mesh defined by vert_coords and faces.
    '''
    x = vert_coords[:,0]
    y = vert_coords[:,1]
    z = vert_coords[:,2]

    mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 600))
    mayavi_mesh = mlab.triangular_mesh(x, y, z, faces, scalars=morphology_data)
    print_mlab_view(mlab.view())
    if use_multi_view:
        print "---Using multi-view---"
        x1, y1, z1 = rotate_3D_coordinates_around_axes(x, y, z, deg2rad(90), 0, 0);
        mayavi_mesh_m1 = mlab.triangular_mesh(x1, y1, z1, faces, scalars=morphology_data, color=(1, 0, 0))
        print_mlab_view(mlab.view())

        x2, y2, z2 = rotate_3D_coordinates_around_axes(x, y, z, deg2rad(90), 0, 0);
        x2, y2, z2 = scale_3D_coordinates(x2, y2, z2, 1.5)
        # = rotate_3D_coordinates_around_axes(x, y, z, rotx, roty, rotz)
        # = scale_3D_coordinates(x, y, z, x_scale_factor, y_scale_factor=None, z_scale_factor=None)
        # = mirror_3D_coordinates_at_axis(x, y, z, axis, mirror_at_axis_coordinate=None)
        # = point_mirror_3D_coordinates(x, y, z, point_x, point_y, point_z):
        x2, y2, z2 = translate_3D_coordinates_along_axes(x, y, z, 200, 0, 0)
        mayavi_mesh_m2 = mlab.triangular_mesh(x2, y2, z2, faces, scalars=morphology_data, color=(0, 0, 1))
        print_mlab_view(mlab.view())

    # we could transform/rotate the x, y, z coords above and create a second mesh to be able to see the brain from different angles.
    # See https://www.lfd.uci.edu/~gohlke/code/transformations.py.html for a python implementation of rotations etc.
    #print mayavi_mesh.position
    fig_handle = mlab.gcf()
    if not draw_mesh_only:
        #mlab.axes().label_text_property.font_size = 12
        mlab.colorbar()
        mlab.title('Brain of subject ' + meta_data.get('subject_id', '?'), size=0.4)
        #mlab.text(0.1, 0.5, meta_data.get('surf', ''), color=(1.0, 0.0, 0.0), width=0.05) # width should be scaled by the number of characters
        #mlab.text(0.1, 0.55, meta_data.get('measure', ''), color=(1.0, 0.0, 0.0), width=0.05)
        #mlab.text(0.1, 0.6, meta_data.get('space', ''), color=(1.0, 0.0, 0.0), width=0.05)

    #mlab.view(49, 31.5, 52.8, (4.2, 37.3, 20.6))
    if not export_image_file is None:
        print "Exporting scene to file '%s'." % export_image_file
        #mlab.savefig(export_image_file, figure=fig_handle)   # save scene: this could also be saved in a 3D file format.

    if do_display_interactively:
        mlab.show()
