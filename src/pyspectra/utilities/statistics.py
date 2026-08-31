import numpy as np
from astropy.io import fits


def imstats(
    fits_file,
    params={
        'mean': True,
        'median': True,
        'std': True,
        'min': True,
        'max': True,
        'sum': True,
        'mad': True,
        'p01': True,
        'p05': True,
        'p25': True,
        'p75': True,
        'p95': True,
        'p99': True,
    },
    print_stats = True
):
    
    """
    Calculate statistical properties of an image stored in a FITS file.

    Parameters
    ----------
    fits_file : str or pathlib.Path
        Path to the FITS file containing the image.

    params : dict, optional
        Dictionary specifying which statistics should be calculated.
        Each statistic is enabled by setting its value to ``True`` and
        disabled by setting it to ``False``.

        Available statistics are:

        "mean"
            Arithmetic mean of the pixel values.

        "median"
            Median pixel value.

        "std"
            Standard deviation of the pixel values.

        "min"
            Minimum pixel value.

        "max"
            Maximum pixel value.

        "sum"
            Sum of all pixel values.

        "mad"
            Median Absolute Deviation (MAD), defined as::
                MAD = median(|x - median(x)|)

        "p01"
            1st percentile.

        "p05"
            5th percentile.

        "p25"
            25th percentile.

        "p75"
            75th percentile.

        "p95"
            95th percentile.

        "p99"
            99th percentile.

    print_stats: bool, optional
        If True, print the values computed on the console.
        Default is True.
        
    Returns
    -------
    stats : dict
        Dictionary containing the requested statistics. The dictionary
        keys correspond to the names specified in ``params``.

    Notes
    -----
    NaN values are ignored when calculating the statistics.

    """

    ######################################
    # Read FITS image
    ######################################
    with fits.open(fits_file) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError(f"No image data found in {fits_file}")

    # Convert to NumPy array and remove NaN values
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError(f"No finite pixel values found in {fits_file}")

    ######################################
    # Calculate requested statistics
    ######################################

    stats = {}

    if params.get('mean', False):
        stats['mean'] = np.mean(data)

    if params.get('median', False):
        stats['median'] = np.median(data)

    if params.get('std', False):
        stats['std'] = np.std(data)

    if params.get('min', False):
        stats['min'] = np.min(data)

    if params.get('max', False):
        stats['max'] = np.max(data)

    if params.get('sum', False):
        stats['sum'] = np.sum(data)

    if params.get('mad', False):
        median = np.median(data)
        stats['mad'] = np.median(np.abs(data - median))

    # Percentiles
    percentile_map = {
        'p01': 1,
        'p05': 5,
        'p25': 25,
        'p75': 75,
        'p95': 95,
        'p99': 99,
    }

    for name, percentile in percentile_map.items():
        if params.get(name, False):
            stats[name] = np.percentile(data, percentile)

    if print_stats:
        for name, value in stats.items():
            print(f'{name} = {value}')
        
    return stats