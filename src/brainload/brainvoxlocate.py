"""
Given a voxel in a brain volume, find the FreeSurfer region it lies in (or the closest one if it is not in any region).
"""
import numpy as np
import brainload as bl
import brainload.freesurferdata as blfsd
import brainload.spatial as blsp

class BrainVoxLocate:

    def __init__(self, volume_file, lookup_file):
        """
        Load data from files.

        Parameters
        ----------
        volume_file: string
            Path to the mgh or mgz volume file containing the segmentation. The value assigned to each voxel represents the tissue type it was assigned. This is a segmentation result file, like aseg.mgz of a subject.

        lookup_file: string
            Path to the lookup file that maps the voxel values to names, i.e., tissue types. Typically this is FreeSurferColorLUT.txt that can be found in FREESURFER_HOME.
        """
        mgh_data, mgh_meta_data = blfsd.read_mgh_file(volume_file)
        self.volume_file = volume_file
        self.lookup_file = lookup_file
        self.volume = mgh_data
        self.lookup_table = blfsd.read_lookup_file(lookup_file)
        self.ras2vox, self.vox2ras, self.vox2ras_tkr = blfsd.read_mgh_header_matrices(self.volume_file)


    def get_voxel_crs_at_ras_coords(self, query_coords):
        """
        Find the voxel closest to each of the given coordinates.

        Find the voxel closest to each of the given coordinates. A voxel is identified by its indices along the 3 axes, also knows as CRS (column, row, slice). The computation is based on the ras2vox matrix in the file header.

        Parameters
        ----------
        query_coords: numpy 2D float array
            The 3D coordinates, given as a numeric array with shape (n, 3) for n coords.

        Returns
        -------
        numpy 2D int array
            Array with shape (n, 3) representing the voxel indices in the volume file.
        """
        voxel_index = blsp.apply_affine_3D(query_coords, self.ras2vox)
        voxel_index = np.rint(voxel_index).astype(int)
        return voxel_index


    def get_ras_coords_at_voxel_crs(self, query_crs_coords):
        """
        Find the RAS coord of each voxel.

        Find the RAS coord of each voxel. A voxel is identified by its indices along the 3 axes, also knows as CRS (column, row, slice). The computation is based on the ras2vox matrix in the file header.

        Parameters
        ----------
        query_crs_coords: numpy 2D int array
            The 3D row, column, slice indices for each voxel, given as a numeric array with shape (n, 3) for n voxels.

        Returns
        -------
        numpy 2D float array
            Array with shape (n, 3) representing the RAS coordinates in the volume file (x,y,z).
        """
        return blsp.apply_affine_3D(query_crs_coords, self.vox2ras)


    def get_voxel_segmentation_labels(self, query_voxels_crs):
        """
        Find the exact labels for the given voxels.

        Find the exact labels for the given voxels. All voxels will have a label, but label 0 means 'Unknown'.

        Parameters
        ----------
        query_vox_crs: numpy 2D array of int
            The query voxels, each given by its CRS indices. So the shape is (n, 3) for n query voxels.

        Returns
        -------
        voxel_seg_code: numpy 1D int array
            The voxel segmentation codes from the lookup file. All voxels will have a label, but label 0 means 'Unknown'.

        voxel_seg_name: numpy 1D str array
            The voxel segmentation names from the lookup file. All voxels will have a label, but label 0 means 'Unknown'.
        """
        voxel_seg_code = [self.volume[crs[0], crs[1], crs[2]] for crs in query_voxels_crs]
        voxel_seg_code = np.array(voxel_seg_code).astype(int)
        voxel_seg_code_str = np.array(voxel_seg_code).astype(self.lookup_table.dtype)
        voxel_seg_name = np.empty((voxel_seg_code.shape[0],), dtype=self.lookup_table.dtype)
        for idx, code in enumerate(voxel_seg_code_str):
            lut_row = self.lookup_table[self.lookup_table[:,0] == code]
            voxel_seg_name[idx] = lut_row[0][1]
        return voxel_seg_code, voxel_seg_name


    def get_closest_not_unknown(self, query_voxels_crs):
        """
        Determine the closest voxels which have a non-empty label.

        Determine the closest voxels which have a non-empty label, their labels, and the respective distance.

        Parameters
        ----------
        query_voxels_crs: numpy 2D array of int
            The query voxels, each given by its CRS indices. So the shape is (n, 3) for n query voxels.

        Returns
        -------
        voxels: numpy 2D int array
            The result voxels, one for each query voxel. Each voxel is given by its CRS indices. So the shape is (n, 3) for n query voxels.

        codes: numpy 1D int array
            The label codes for the result voxels.

        distances: numpy 1D float array
            The distances from the respective query voxel to the result voxel. These are determined from the RAS coordinates of the voxel pair, using the ras2vox and vox2ras matrices in the volume file header. (The distance is 0.0 if the query voxel itself has a nonzero label.)
        """
        codes, _ = self.get_voxel_segmentation_labels(query_voxels_crs)
        voxels = np.zeros(query_voxels_crs.shape, dtype=int) - 1
        distances = np.zeros((query_voxels_crs.shape[0], ), dtype=float)
        for idx, query_vox_code in enumerate(codes):
            if query_vox_code == 0:
                # The voxel itself has an 'Unknown' label, so find the closest one which has a different label
                voxels[idx] = np.array([-1, -1, -1], dtype=int)
                codes[idx] = -1
                distances[idx] = 1.0
            else:
                voxels[idx,:] = query_voxels_crs[idx,:]
                # The code fits already, and the distance is 0.0, which is also correct.
        return voxels, codes, distances
