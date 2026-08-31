from astropy.io import fits
from astropy.visualization import ZScaleInterval

from scipy.ndimage import gaussian_filter1d

from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 28,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 28
})

plt.rcParams["figure.dpi"] = 200

def crop(path_files, ref_file = '', ref_file_ext = 0, view_crop = False, view_FITS_crop = False, apply_crop = False, new_files = None):
    """
    Performs a crop for a set of FITS files, based on the overscan region of a flat-field file.

    Parameters
    ----------
    path_files : str
        Directory containing the FITS files.
    ref_file : str,
        Path to the flat-field file to use as the reference to make the cropping.
    ref_file_ext : int or str, optional
        Extension containing the reference data.
        Default is 0.
    view_crop: bool, optional
        If True, plots the central file and column of the reference file and
        the range where the crop will be make. Otherwise, dont plot anything.
        Default is False.
    view_FITS_crop : bool, optional
        If True, plots the ref_file with rthe ange where the crop will be make. 
        Otherwise, dont plot anything.
        Default is False.
    apply_crop : bool,
        If False, the cropping is not made it. Otherwise, the files are cropping.
        Default is False.
    new_files : list of str, optional
        Output filenames. If None, files are overwritten.

    Returns
    -------
    None
    """
    
    # Reading the reference file
    with fits.open(ref_file) as hdul:
        data = hdul[ref_file_ext].data
        hdr = hdul[ref_file_ext].header

    # Getting the info for make the crop
    disp_axis = hdr.get('DISPAXIS')
    if disp_axis == 2:
        n_disp_pixs, n_spat_pixs = data.shape
        central_disp_pix = n_disp_pixs // 2
        central_spat_pix = n_spat_pixs // 2
    
        spat_data = data[central_disp_pix, :]
        disp_data = data[:, central_spat_pix]

    elif disp_axis == 1:
        n_spat_pixs, n_disp_pixs = data.shape
        central_disp_pix = n_disp_pixs // 2
        central_spat_pix = n_spat_pixs // 2
    
        spat_data = data[:, central_disp_pix]
        disp_data = data[central_spat_pix, :]

    else:
        raise ValueError(f"DISPAXIS of the ref_file in extension {ref_file_ext} must be 1 or 2")

    # Gaussian convolution to smooth the data and easily find the limits to make the crop
    spat_data_G1d = gaussian_filter1d(spat_data, sigma=10.0)
    disp_data_G1d = gaussian_filter1d(disp_data, sigma=10.0)

    # Finding limits for the spatial axis
    low_idx = 0
    up_idx = n_spat_pixs
    
    for i in range(central_spat_pix):
        low_data = spat_data_G1d[central_spat_pix-1-i:central_spat_pix+1]
        if np.abs(spat_data_G1d[central_spat_pix-1-i] - np.mean(low_data)) > 5 * np.std(low_data):
            low_idx = central_spat_pix-i # Lower limit
            break
    
    for i in range(n_spat_pixs - central_spat_pix-1):
        up_data = spat_data_G1d[central_spat_pix:central_spat_pix+2+i]
        if np.abs(spat_data_G1d[central_spat_pix+1+i] - np.mean(up_data)) > 5 * np.std(up_data):
            up_idx = central_spat_pix+i # Upper limit
            break

    spat_axis_lims = [low_idx, up_idx]

    # Finding limits for the dispersion axis
    low_data = disp_data_G1d[n_disp_pixs//10:0:-1]
    grad_l = np.gradient(low_data)
    
    idx_l_list = np.where(np.abs(grad_l-np.median(grad_l)) > 3 * np.std(grad_l))
    if len(idx_l_list[0]) > 1:
        l_idx = n_disp_pixs//10 - idx_l_list[0][-2]
    elif len(idx_l_list[0]) == 1:
        l_idx = n_disp_pixs//10 - idx_l_list[0][0]
    elif len(idx_l_list[0]) == 0:
        l_idx = 0

    up_data = disp_data_G1d[9*(n_disp_pixs//10):-1]
    grad_u = np.gradient(up_data)
    
    idx_u_list = np.where(np.abs(grad_u-np.median(grad_u)) > 3 * np.std(grad_u))
    if len(idx_u_list[0]) > 0:
        u_idx = 9*(n_disp_pixs//10) + idx_u_list[0][0]
    elif len(idx_u_list[0]) == 0:
        u_idx = n_disp_pixs

    disp_axis_lims = [l_idx, u_idx]

    # Plotting the x and y limits
    if view_crop:

        # spatial axis limits
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(spat_data, lw = 2, color = 'blue', label = 'Data')
        ax.plot(spat_data_G1d, lw = 2, color = 'red', label = 'Convolved data')
        ax.set_ylim(0, 1.5 * np.max(spat_data_G1d))
        ax.axvline(spat_axis_lims[0], linestyle = '--', lw = 2, color = 'black', label = 'Spatial axis limits')
        ax.axvline(spat_axis_lims[1], linestyle = '--', lw = 2, color = 'black')
        ax.grid()
        ax.set_title('Limits on spatial axis')
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Counts')
        ax.legend()
        plt.show()

        # dispersion axis limits
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(disp_data, lw = 2, color = 'blue', label = 'Data')
        ax.plot(disp_data_G1d, lw = 2, color = 'red', label = 'Convolved data')
        ax.set_xlim(-10, len(disp_data)+10)
        ax.axvline(disp_axis_lims[0], linestyle = '--', lw = 2, color = 'black', label = 'Dispersion axis limits')
        ax.axvline(disp_axis_lims[1], linestyle = '--', lw = 2, color = 'black')
        ax.grid()
        ax.set_title('Limits on dispersion axis')
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Counts')
        ax.legend()
        plt.show()

    if view_FITS_crop:

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)

        # FITS image with limits
        fig, ax = plt.subplots(figsize=(6, 6), dpi = 200)
        ax.imshow(data, origin='lower', cmap='Grays', vmin=vmin, vmax=vmax)
        if disp_axis == 2:
            ax.axvline(spat_axis_lims[0], linestyle = '--', lw = 1, color = 'red', label = 'Spatial axis limits')
            ax.axvline(spat_axis_lims[1], linestyle = '--', lw = 1, color = 'red')
            ax.axhline(disp_axis_lims[0], linestyle = '--', lw = 1, color = 'red', label = 'Dispersion axis limits')
            ax.axhline(disp_axis_lims[1], linestyle = '--', lw = 1, color = 'red')
        elif disp_axis == 1:
            ax.axvline(disp_axis_lims[0], linestyle = '--', lw = 1, color = 'red', label = 'Dispersion axis limits')
            ax.axvline(disp_axis_lims[1], linestyle = '--', lw = 1, color = 'red')
            ax.axhline(spat_axis_lims[0], linestyle = '--', lw = 1, color = 'red', label = 'Spatial axis limits')
            ax.axhline(spat_axis_lims[1], linestyle = '--', lw = 1, color = 'red')
        ax.set_xlabel("X (pix)", fontsize = 12)
        ax.set_ylabel("Y (pix)", fontsize = 12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title("Reference FITS file", fontsize = 16)
        plt.show()

    # Making the crop
    path_files = Path(path_files)
    files = list(path_files.glob('*.fits'))

    # Validate new_files parameter
    if new_files is not None:
        if isinstance(new_files, str):
            new_files = [new_files]
    
        if len(new_files) != len(files):
            raise ValueError("new_files must have the same length as the number of FITS files")

    today = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    if apply_crop:
        for i, f in enumerate(files):
            with fits.open(str(f)) as hdul:
                data = hdul[0].data.copy()
                header = hdul[0].header.copy()
                
                header.add_history('========= Begin Reduction Process =========')
                header.add_history(f"Reduction started: {today}")
                header.add_history('--- CROPPING ---')

                if disp_axis == 2:
                    cropped_data = data[disp_axis_lims[0]:disp_axis_lims[1], spat_axis_lims[0]:spat_axis_lims[1]]
                    header.add_history(f'Cropped region: [{spat_axis_lims[0]}:{spat_axis_lims[1]}, {disp_axis_lims[0]}:{disp_axis_lims[1]}]')
                    header['CROP_X1'] = (spat_axis_lims[0], 'Left pixel of crop region')
                    header['CROP_X2'] = (spat_axis_lims[1], 'Right pixel of crop region')
                    header['CROP_Y1'] = (disp_axis_lims[0], 'Lower pixel of crop region')
                    header['CROP_Y2'] = (disp_axis_lims[1], 'Upper pixel of crop region')

                elif disp_axis == 1:
                    cropped_data = data[spat_axis_lims[0]:spat_axis_lims[1], disp_axis_lims[0]:disp_axis_lims[1]]
                    header.add_history(f'Cropped region: [{disp_axis_lims[0]}:{disp_axis_lims[1]}, {spat_axis_lims[0]}:{spat_axis_lims[1]}]')
                    header['CROP_X1'] = (disp_axis_lims[0], 'Left pixel of crop region')
                    header['CROP_X2'] = (disp_axis_lims[1], 'Right pixel of crop region')
                    header['CROP_Y1'] = (spat_axis_lims[0], 'Lower pixel of crop region')
                    header['CROP_Y2'] = (spat_axis_lims[1], 'Upper pixel of crop region')
                
                hdu = fits.PrimaryHDU(data=cropped_data, header=header)
            
                if new_files is not None:
                    hdu.writeto(new_files[i], overwrite=True)
                else:
                    hdu.writeto(str(f), overwrite=True)



# def custom_crop(path_files, view_crop = False, view_FITS_crop = False, apply_crop = False, new_files = None):