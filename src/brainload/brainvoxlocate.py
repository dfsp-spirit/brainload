"""
Given a voxel in a brain volume, find the FreeSurfer region it lies in (or the closest one if it is not in any region).
"""
import numpy as np
import brainload as bl
import brainload.freesurferdata as blfsd

class BrainVoxLocate:

    def __init__:(self, volume_file, lookup_file):
        """
        Load data from files.

        Parameters
        ----------
        volume_file: string
            Path to the mgh or mgz volume file containing the segmentation. The value assigned to each voxel represents the tissue type it was assigned. This is a segmentation result file, like aseg.mgz of a subject.

        lookup_file: string
            Path to the lookup file that maps the voxel values to names, i.e., tissue types. Typically this is FreeSurferColorLUT.txt that can be found in FREESURFER_HOME.
        """
        mgh_data, mgh_meta_data = blfsd.read_mgh_file(mgh_file)
        self.volume = mgh_data
        self.lookup_table = blfsd.read_lookup_file(lookup_file)


    def get_closest_voxel(self, query_coords):
        """
        Find the vertex closest to each of the given coordinates. Requires scipy.

        Parameters
        ----------
        query_coords: numpy 2D float array
            The 3D coordinates, given as a numeric array with shape (n, 3) for n coords.

        Returns
        -------
        numpy 2D int array
            Array with shape (n, 3) representing the voxel indices in the volume file.
        """
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(self.vert_coords, query_coords, metric='euclidean')
        return np.argmin(dist_matrix, axis=0)


    def get_voxel_segmentation_labels(self, query_voxels):
        """
        Find the exact labels for the given voxels.

        Find the exact labels for the given voxels. If a voxel has no label, returns -1 for it.
        """
