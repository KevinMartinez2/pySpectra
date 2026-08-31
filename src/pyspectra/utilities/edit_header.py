from astropy.io import fits

def edit_header(files, ext=0, keyword='DATE',
                new_val=None, comment=None,
                new_files=None):
    
    """
    Edit the header keyword of one or more FITS files.

    Parameters
    ----------
    files : str or list of str
        Paths to FITS files to edit.

    ext : int or str, optional
        FITS extension index or name.
        Default is 0.

    keyword : str, optional
        FITS keyword to edit.

    new_val : str, int, float, bool, or None, optional
        New value for the keyword.

    comment : str, optional
        Comment for the FITS keyword.

    new_files : str or list of str, optional
        Output filenames.
        If None, files are overwritten.

    Returns
    -------
    None
    """

    # Convert strings to lists
    if isinstance(files, str):
        files = [files]

    if new_files is not None and isinstance(new_files, str):
        new_files = [new_files]

    # Validate output length
    if new_files is not None and len(new_files) != len(files):
        raise ValueError("new_files must have the same length as files")

    # Validate new value type
    valid_types = (str, int, float, bool)

    if new_val is not None and not isinstance(new_val, valid_types):
        raise TypeError("new_val must be a string, integer, float, or bool")

    # Process files
    for i, f in enumerate(files):
        mode = 'update' if new_files is None else 'readonly'

        with fits.open(f, mode=mode) as hdul:
            try:
                header = hdul[ext].header
            except (KeyError, IndexError):
                raise KeyError(f'Extension {ext} not found in {f}')

            # Edit keyword
            if comment is None:
                header[keyword] = new_val
            else:
                header[keyword] = (new_val, comment)

            # Save
            if new_files is None:
                hdul.flush()
            else:
                hdul.writeto(
                    new_files[i],
                    overwrite=True
                )