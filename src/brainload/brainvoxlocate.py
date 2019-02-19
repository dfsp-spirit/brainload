"""
Given a voxel in a brain volume, find the FreeSurfer region it lies in (or the closest one if it is not in any region).
"""
import numpy as np
import brainload as bl
import brainload.freesurferdata as blfsd
import brainload.spatial as blsp

class BrainVoxLocate:
    """
    Voxel segmentation label locator. This class allows you to determine the label (both label code/number and label name) from a lookup table file (usually FreeSurferColorLUT.txt) that is assigned to a voxel in a segmentation output volume (like aseg.mgz or aparc+asgeg.mgz).

    Examples
    --------
    Initialize a locator based on a volume (a segmentation output, i.e., the value assigned to each voxel in the volume represents the tissue class it has been assigned to by the segmentation) and a lookup table file:

    >>> volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
    >>> lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
    >>> locator = vloc.BrainVoxLocate(volume_file, lookup_file)

    Now define some voxels we are interested in. Voxels are given by their column, row, slice (RCS) indices in the volume. These always start at 0 and range from 0 to d-1, where d is the length of the volume along the respective axis. So if your volume file has shape (128, 128, 128), the 3 indices along the 3 axes all range from 0 to 127.

    >>> query_vox_crs = np.array([[24, 28, 20], [64, 64, 45], [90, 90, 90], [95, 127, 45]], dtype=int)

    Now get the classes of exactly these voxels:

    >>> seg_code, seg_name = locator.get_voxel_segmentation_labels(query_vox_crs)

    Now print the code and the string for the first query voxel, (24, 28, 20):

    >>> print("Voxel has label code %d, which encodes label string '%s'." % (seg_code[0], seg_name[0]))

    Note that some voxels may not be assigned any label. These will show the label code 0, which means label name 'Unknown'. Sometimes, you may want to know the label of the closest voxel which has a valid label for these. For example, because you want to know the brain structure that is closest to this voxel, even if the point does not lie within that structure. You can do this and allow a certain neighborhood to be searched with the ```get_closest_not_unknown``` function:

    >>> voxels, codes, distances, closest_voxels_ras_coords = locator.get_closest_not_unknown(query_vox_crs)

    See the documentation for that function for details on the return values.
    """

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

        Find the exact labels for the given voxels. All voxels will have a label, but label 0 means 'Unknown'. (Sometimes, you may want to know the label of the closest voxel which has a valid label for these. For example, because you want to know the brain structure that is closest to this voxel, even if the point does not lie within that structure. See the function ```get_closest_not_unknown``` for that use case.)

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

        Examples
        --------
        Initialize a locator based on a volume (a segmentation output, i.e., the value assigned to each voxel in the volume represents the tissue class it has been assigned to by the segmentation) and a lookup table file:

        >>> volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
        >>> lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
        >>> locator = vloc.BrainVoxLocate(volume_file, lookup_file)

        Now define some voxels we are interested in. Voxels are given by their column, row, slice (RCS) indices in the volume. These always start at 0 and range from 0 to d-1, where d is the length of the volume along the respective axis. So if your volume file has shape (128, 128, 128), the 3 indices along the 3 axes all range from 0 to 127.

        >>> query_vox_crs = np.array([[24, 28, 20], [64, 64, 45], [90, 90, 90], [95, 127, 45]], dtype=int)

        Now get the classes of exactly these voxels:

        >>> seg_code, seg_name = locator.get_voxel_segmentation_labels(query_vox_crs)


        See also
        --------
        ```get_closest_not_unknown``` can find the brain structure closest to a voxel, even if the voxel does not lie within that structure.
        """
        voxel_seg_code = [self.volume[crs[0], crs[1], crs[2]] for crs in query_voxels_crs]
        voxel_seg_code = np.array(voxel_seg_code).astype(int)
        voxel_seg_code_str = np.array(voxel_seg_code).astype(self.lookup_table.dtype)
        voxel_seg_name = np.empty((voxel_seg_code.shape[0],), dtype=self.lookup_table.dtype)
        for idx, code in enumerate(voxel_seg_code_str):
            lut_row = self.lookup_table[self.lookup_table[:,0] == code]
            voxel_seg_name[idx] = lut_row[0][1]
        return voxel_seg_code, voxel_seg_name


    def get_closest_not_unknown(self, query_voxels_crs, unknown_label=0, neighborhood_size=10):
        """
        Determine the closest voxels which have a non-empty label.

        Determine the closest voxels which have a non-empty label, their labels, and the respective distance. Requires scipy. This allow you to determine the brain structure closest to a voxel (within a distance threshold), even if the voxel idoes not lie directly within the brain structure. This function uses Euclidian distance to determine which voxel is closest (but it only checks voxels with the given neighborhood, of course). If one of the query voxels lies directly within a brains structure (i.e., the voxel itself has a valid label), that label will be returned and the distance will be 0.0, of course.

        Parameters
        ----------
        query_voxels_crs: numpy 2D array of int
            The query voxels, each given by its CRS indices. So the shape is (n, 3) for n query voxels.

        neighborhood_size: int, optional
            Distance threshold in voxels along each direction of each axis, must be a positive integer or zero. Only the neighborhood of each query voxel will be searched. Example: If you pass 0, only the voxel itself is searched. If you pass 1, up to 3x3 = 27 voxels around it will be searched. If you pass 3, up to 7x7x7 = 343 voxels will be searched. The 'up to' refers to the case where the query voxel is at the border of the volume. In that case, some of the voxels do not exist (and thus are not checked). Defaults to 10. Note that if you set this to a very large value, a pairwise distance matrix of considerable size has to be computed, which may take some time. E.g., if you set it to 100, the distances between 201x201x201=8,120,601 voxels will be computed. This means that 8120601 ^ 2 / 2 - 8120601 = 32,972,072,179,999 distances need to be computed. No matter what you put, only valid voxel indices within the volume will be used for computation.

        unknown_label: int, optional
            The segmentation value that represents the 'Unknown' class. Defaults to 0, which is suitable for the FreeSurferColorLUT.txt file.

        Returns
        -------
        voxels: numpy 2D int array
            The result voxels, one for each query voxel. Each voxel is given by its CRS indices. So the shape is (n, 3) for n query voxels. If no suitable voxel with non-empty label code was found for a query voxel, the coordinates are [-1, -1, -1].

        codes: numpy 1D int array
            The label codes for the result voxels. If no suitable voxel with non-empty label code was found for a query voxel, the code is -1.

        distances: numpy 1D float array
            The distances from the respective query voxel to the result voxel. These are determined from the RAS coordinates of the voxel pair, using the ras2vox and vox2ras matrices in the volume file header. (The distance is 0.0 if the query voxel itself has a nonzero label.) If no suitable voxel with non-empty label code was found for a query voxel, the distance is -1.0f.

        closest_voxels_ras_coords: numpy 2D float array
            The RAS coordinates of the chosen voxel. If the query voxel has a none-empty label, this is its own coordinate. Otherwise, it is the coordinate of the closest voxel with valid label. If no such voxel was found, the coordinates are [-1.0, -1.0, -1.0]. Note that this could theoretically be a valid RAS coordinate, so you should not use this field to determine whether a valid voxel was found. Use the ```voxels``` return value for that.

        Examples
        --------
        Initialize a locator based on a volume (a segmentation output, i.e., the value assigned to each voxel in the volume represents the tissue class it has been assigned to by the segmentation) and a lookup table file:

        >>> volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
        >>> lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
        >>> locator = vloc.BrainVoxLocate(volume_file, lookup_file)

        Now define some voxels we are interested in. Voxels are given by their column, row, slice (RCS) indices in the volume. These always start at 0 and range from 0 to d-1, where d is the length of the volume along the respective axis. So if your volume file has shape (128, 128, 128), the 3 indices along the 3 axes all range from 0 to 127.

        >>> query_vox_crs = np.array([[24, 28, 20], [64, 64, 45], [90, 90, 90], [95, 127, 45]], dtype=int)

        Now get the classes of these voxels. If the voxel itself has no label (i.e., it has the 'Unknown' label), search with a square neighborhood of 5 voxels along each direction of each axis for the closest valid label.

        >>> voxels, codes, distances, closest_voxels_ras_coords = locator.get_closest_not_unknown(query_vox_crs, neighborhood_size=5)
        """
        from scipy.spatial.distance import cdist
        query_voxels_ras_coords = self.get_ras_coords_at_voxel_crs(query_voxels_crs)
        codes, _ = self.get_voxel_segmentation_labels(query_voxels_crs)
        voxels = np.zeros(query_voxels_crs.shape, dtype=int) - 1
        closest_voxels_ras_coords = np.zeros(query_voxels_crs.shape) - 1.0
        distances = np.zeros((query_voxels_crs.shape[0], ), dtype=float)
        for idx, query_vox_code in enumerate(codes):
            if query_vox_code == unknown_label:
                # The voxel itself has an 'Unknown' label, so find the closest one which has a different label
                voxels[idx] = np.array([-1, -1, -1], dtype=int)
                codes[idx] = -1
                distances[idx] = -1.0

                neighborhood_voxel_indices_tuple = blsp.get_n_neighborhood_indices_3D(self.volume.shape, query_voxels_crs[idx,:], neighborhood_size)
                neighborhood_voxel_indices = np.zeros((len(neighborhood_voxel_indices_tuple[0]), 3), dtype=int)
                neighborhood_voxel_indices[:,0] = neighborhood_voxel_indices_tuple[0]
                neighborhood_voxel_indices[:,1] = neighborhood_voxel_indices_tuple[1]
                neighborhood_voxel_indices[:,2] = neighborhood_voxel_indices_tuple[2]
                neighborhood_ras_coords = self.get_ras_coords_at_voxel_crs(neighborhood_voxel_indices)
                neighborhood_codes, _ = self.get_voxel_segmentation_labels(neighborhood_voxel_indices)
                query_voxel_ras_coords = query_voxels_ras_coords[idx:idx+1,:]

                dist_matrix = cdist(query_voxel_ras_coords, neighborhood_ras_coords)
                neighborhood_indices_sorted_by_dist = np.argsort(dist_matrix[0])
                print(neighborhood_indices_sorted_by_dist)
                neighborhood_sorted_by_dist = neighborhood_ras_coords[neighborhood_indices_sorted_by_dist]

                num_neighborhood_voxels = len(dist_matrix[0])
                k_closest = 0       # We first select the k-closest voxel with k=0. Then, we increase k until we find a voxel that has a non-empty label in the following loop.
                for k_closest in range(num_neighborhood_voxels):
                    closest_vox_index = neighborhood_indices_sorted_by_dist[k_closest]
                    if neighborhood_codes[closest_vox_index] != unknown_label:
                        voxels[idx] = np.array([neighborhood_voxel_indices[closest_vox_index][0], neighborhood_voxel_indices[closest_vox_index][1], neighborhood_voxel_indices[closest_vox_index][2]], dtype=int)
                        codes[idx] = neighborhood_codes[closest_vox_index]
                        distances[idx] = dist_matrix[0][closest_vox_index]
                        closest_voxels_ras_coords[idx] = np.array([neighborhood_sorted_by_dist[k_closest][0], neighborhood_sorted_by_dist[k_closest][1], neighborhood_sorted_by_dist[k_closest][2]])
                        break
            else:
                voxels[idx,:] = query_voxels_crs[idx,:]
                closest_voxels_ras_coords[idx] = query_voxels_ras_coords[idx]   # the coord is the coord of the voxel itself
                # The code fits already, and the distance is 0.0, which is also correct.
        return voxels, codes, distances, closest_voxels_ras_coords
