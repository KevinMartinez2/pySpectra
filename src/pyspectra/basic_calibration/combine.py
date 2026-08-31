from astropy.io import fits
from astropy.stats import sigma_clip

import numpy as np

from pathlib import Path

def combine(files, align = False, sigma = None, method = 'median', result_img = None):
    """
    Combine multiple FITS images into a single image using a statistical estimator.

    This function optionally aligns the images along the axis that is perpendicular
    to the dispersion axis before combining them. The alignment is performed by 
    locating the maximum of the spatial profile obtained by collapsing the image
    along the dispersion axis.

    Parameters
    ----------
    files : list of str
        List of paths to the FITS files to be combined.
        All images must have identical dimensions.

    align : bool, optional
        If True, images are aligned prior to combination.
        The alignment is performed using integer pixel shifts
        determined from the brightest pixel in each image.
        Default is False.

    sigma : float, optional
        Sigma threshold for pixel-wise sigma clipping applied before
        combination. If None, no sigma clipping is performed.

    method : {'median', 'mean', 'min', 'max'}, optional
        Statistical estimator used to combine the images:

        - 'median' : Pixel-wise median (robust against outliers)
        - 'mean'   : Pixel-wise arithmetic mean
        - 'min'    : Pixel-wise minimum
        - 'max'    : Pixel-wise maximum

        Default is 'median'.

    result_img : str or None, optional
        Output filename for the combined FITS image.
        If None, the output filename is generated from the
        first input file by appending '_combined.fits'.

    Returns
    -------
    None
        The combined image is written to disk.

    Notes
    -----
    - Alignment uses integer pixel shifts only.
      No interpolation is performed.
    - The alignment assumes that the spatial 
      profile is dominated by a single source and 
      that no significant tilt is present.
    - All images must have identical shape.


    Raises
    ------
    ValueError
        If an invalid combination method is provided.
    """

    images = []

    # Get reference data info
    with fits.open(files[0]) as hdul:
        ref_data = hdul[0].data
        ref_hdr = hdul[0].header
    
    # Alignment    
    if align:
        # Determine reference dispersion axis
        disp_axis = ref_hdr.get('DISPAXIS', None)

        if disp_axis not in (1, 2):
            raise ValueError("DISPAXIS not defined or invalid in header")

        # Define spatial collapse axis
        collapse_axis = 1 if disp_axis == 1 else 0

        # Compute reference position
        ref_profile = np.sum(ref_data, axis=collapse_axis)
        ref_pix = np.argmax(ref_profile)

        images.append(ref_data.copy())

        # Process remaining files
        for f in files[1:]:

            with fits.open(f) as hdul:
                data = hdul[0].data
                hdr = hdul[0].header

            if hdr.get('DISPAXIS') != disp_axis:
                raise ValueError("DISPAXIS mismatch between files")

            profile = np.sum(data, axis=collapse_axis)
            pix = np.argmax(profile)

            shift = ref_pix - pix

            shifted = np.zeros_like(data)

            if shift != 0:
                if disp_axis == 2:  # shift in x (columns)
                    if shift > 0:
                        shifted[:, shift:] = data[:, :-shift]
                    else:
                        shifted[:, :shift] = data[:, -shift:]
                else:  # shift in y (rows)
                    if shift > 0:
                        shifted[shift:, :] = data[:-shift, :]
                    else:
                        shifted[:shift, :] = data[-shift:, :]
            else:
                shifted = data.copy()

            images.append(shifted)

    # No alignment needed
    else:
        for f in files:
            with fits.open(f) as hdul:
                images.append(hdul[0].data)

    # Verify images lenght
    shapes = [img.shape for img in images]

    if len(set(shapes)) != 1:
        raise ValueError(f"Input images do not have identical shapes: {shapes}")

    # Create a data cube to compute the result image
    cube = np.array(images)

    # Apply sigma clipping along image axis
    if sigma is not None:
        cube = sigma_clip(
            cube,
            cenfunc = 'mean',
            sigma = sigma,
            maxiters = 2,
            axis = 0,
            masked = False
        )
    
    # Create the combined image
    if method == 'median':
        combined = np.nanmedian(cube, axis=0)
    elif method == 'mean':
        combined = np.nanmean(cube, axis=0)
    elif method == 'min':
        combined = np.nanmin(cube, axis=0)
    elif method == 'max':
        combined = np.nanmax(cube, axis=0)
    else:
        raise ValueError("Invalid combination method")

    # Result image handling
    if result_img is None:
        result_img = files[0].replace('.fits', '_combined.fits')

    else:
        result_img = Path(result_img)

    # Create output directory if it does not exist
    result_img.parent.mkdir(parents=True, exist_ok=True)

    new_header = ref_hdr.copy()

    new_header.add_history('--- COMBINATION ---')
    new_header.add_history('Images combined using custom combine() function')
    new_header.add_history(f'Method: {method}')
    new_header.add_history(f'Aligned: {align}')
    new_header.add_history(f'Input files: {", ".join(files)}')
    
    new_header['NCOMBINE'] = (len(files), 'Number of combined images')
    new_header['COMBMETH'] = (method, 'Combination method')
    new_header['ALIGN'] = (align, 'Images aligned before combination')

    if sigma is not None:
        new_header['SIGCLIP'] = (True, 'Sigma clipping applied')
        new_header['SIGMA'] = (sigma, 'Sigma threshold for clipping')
        new_header['MAXITER'] = (2, 'Max iterations for sigma clipping')
    else:
        new_header['SIGCLIP'] = (False, 'Sigma clipping applied')
    
    # Write output
    fits.PrimaryHDU(data=combined, header=new_header).writeto(
        result_img, overwrite=True
    )
    
    print(f"Combined image saved as {result_img}")
