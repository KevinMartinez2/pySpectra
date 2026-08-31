from astropy.io import fits

from pathlib import Path

import numpy as np

from basic_calibration.combine import combine

def bias(bias_files, files, method = 'median', 
         view_stats = True, subtract = True,
         output_dir = None, overwrite = False):
    
    """
    This function is used to compute the bias level to a set of images.
    Optionally, can plot the statistics of the bias_master and also subtract
    this image to a set of files.

    Parameters
    ----------
    bias_files : list
        List of paths to the bias files to be combined
        for create the bias_master file.
        All images must have identical dimensions.
        
    files : list of str
        List of paths to the FITS files to be corrected
        by the bias_master.

    method : {'median', 'mean', 'min', 'max'}, optional
        Statistical estimator used to combine the bias images:

        - 'median' : Pixel-wise median (robust against outliers)
        - 'mean'   : Pixel-wise arithmetic mean
        - 'min'    : Pixel-wise minimum
        - 'max'    : Pixel-wise maximum

        Default is 'median'.

    view_stats : Boolean, optional
        If True, plot basic statistics (mean, median, std, etc.)
        of the bias_master.
        Default is True.

    subtract : Boolean, optional
        If True, subtract the bias_master file from each file in files.
        Default is True.

    output_dir : str, optional
        Output directory where the bias subtracted files must be saved.
        If None, a directory called stage1 is created and the files are 
        saved there.

    overwrite : Boolean, optional
        If True, existing output files are overwritten.

        If False, original files are never overwritten. The bias-corrected
        images are written to output_dir using the suffix "_b.fits".
        If a file with that name already exists, this is overwritten.
    
        Default is False.

    Returns
    -------
    None
        This function creates a master bias image from bias_files.
        Depending on the input parameters, it may:
    
        - Subtract the master bias from the images in files
          (subtract=True).
        - Overwrite existing files if overwrite=True.
    
        No objects are returned; all operations are performed on disk.

    Notes
    -----
    - Bias files must have the same size.
    - Each file in files must have the same size as the bias_master


    Raises
    ------
    ValueError
        If bias images do not share identical dimensions.
        If any image in 'files' does not match the bias_master dimensions.
        If 'method' is not one of {'median', 'mean', 'min', 'max'}.
    
    """

    # Check the path of output_dir
    output_dir = Path(output_dir) if output_dir is not None else Path(bias_files[0]).parent / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the bias_master FITS file
    bias_master_path = output_dir / "bias_master.fits"
    
    combine(bias_files, align = False, sigma = 3, method = method, result_img = str(bias_master_path))

    # Plot the statistics of bias_master.fits
    if view_stats:
        with fits.open(str(bias_master_path)) as hdul:
            bias_master_data = hdul[0].data
    
        print("Master Bias Statistics")
        print("----------------------")
        print(f"Npix        : {bias_master_data.size}")
        print(f"Mean        : {np.nanmean(bias_master_data):8.2f}")
        print(f"Median      : {np.nanmedian(bias_master_data):8.2f}")
        print(f"Std         : {np.nanstd(bias_master_data):8.2f}")
        print(f"Min         : {np.nanmin(bias_master_data):8.2f}")
        print(f"Max         : {np.nanmax(bias_master_data):8.2f}")

    # Subtract bias level for each file 
    if subtract:
        with fits.open(bias_master_path) as hdul:
            bias_master_data = hdul[0].data.astype(float)

        print('Bias correction applied to:')
    
        for f in files:
            with fits.open(f) as hdul:
                data = hdul[0].data.astype(float)
                hdr = hdul[0].header

            if data.shape != bias_master_data.shape:
                raise ValueError(
                    f"Image {f} has shape {data.shape}, "
                    f"but master bias has shape {bias_master_data.shape}.")

            data = data - bias_master_data

            hdr.add_history('--- Bias correction ---')
            hdr.add_history('Bias subtraction have been made using')
            hdr.add_history(str(bias_master_path))
            hdr.add_history(f'Mean bias level: {np.nanmean(bias_master_data):8.2f}')
            hdr.add_history(f'Median bias level: {np.nanmedian(bias_master_data):8.2f}')

            if overwrite == True:
                out_file = f
            else:
                p = Path(f)
                out_file = output_dir / (p.stem + "_b" + p.suffix)

            fits.writeto(out_file, data, hdr, overwrite=overwrite)
            print(f"File : {out_file}")