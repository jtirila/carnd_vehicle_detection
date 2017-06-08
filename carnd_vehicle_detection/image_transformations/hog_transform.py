from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """Get the hog features from an image. For instructions on the params, see hog documentation. However, 
    note that I make some simplifying assumptions about the pix_per_cell and cell_per_block parameters 
    (assuming the regions are square) so the parameters for this wrapper function are a bit simpler than for the 
    actual hog."""

    # TODO: this is just plain copy-paste from the lab at this point. Don't know if it needs to be modified
    # TODO: for the project. Also see note below.

    # TODO: in the proposed solution, feature_vector is set to False when vis == True. Not sure why this is the case.
    # Also, the return values are manually handled in the solution, even though I guess we can just implicitly return
    # either the single value or the two-tuple the hog function returns.
    return hog(img, orient, (pix_per_cell, ) * 2 , (cell_per_block, ) * 2, visualise=vis, feature_vector=feature_vec)
