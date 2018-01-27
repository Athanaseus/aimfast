import json
import argparse
import numpy as np
from scipy import stats
from functools import partial
from astLib.astWCS import WCS
from astropy.io import fits as fitsio


def json_dump(data_dict, root='.'):
    """Dumps the computed dictionary into a json file

    Parameters
    ----------
    data_dict: dict
        dictionary with output results to save
    root: str
        directory to save output json file (default is current directory)

    Note
    ----
    If the fidelity_results.json file exists, it will be append, and only
    repeated image assessments will be replaced.

    """
    filename = 'fidelity_results.json'
    try:
        # Extract data from the json data file
        with open(filename) as data_file:
            data_existing = json.load(data_file)
            data = dict(data_existing.items() + data_dict.items())
    except IOError:
        data = data_dict
    if data:
        with open('%s/%s' % (root, filename), 'w') as f:
            json.dump(data, f)


def fitsInfo(fitsname=None):
    """Get fits header info

    Parameters
    ----------
    fitsname: fits file
        restored image (cube)

    Returns
    -------
    fitsinfo: dict
        dictionary of fits information
        e.g. {'wcs': wcs, 'ra': ra, 'dec': dec,
        'dra': dra, 'ddec': ddec, 'raPix': raPix,
        'decPix': decPix,  'b_scale': beam_scale}

    """
    hdu = fitsio.open(fitsname)
    hdr = hdu[0].header
    ra = hdr['CRVAL1']
    dra = abs(hdr['CDELT1'])
    raPix = hdr['CRPIX1']
    dec = hdr['CRVAL2']
    ddec = abs(hdr['CDELT2'])
    decPix = hdr['CRPIX2']
    wcs = WCS(hdr, mode='pyfits')
    beam_size = (hdr['BMAJ'], hdr['BMIN'], hdr['BMIN'], hdr['BPA'])
    fitsinfo = {'wcs': wcs, 'ra': ra, 'dec': dec,
                'dra': dra, 'ddec': ddec, 'raPix': raPix,
                'decPix': decPix,  'b_size': beam_size}
    return fitsinfo


def residual_image_stats(fitsname):
    """Gets statistcal properties of a residual image

    Parameters
    ----------
    fitsname: fits file
        residual image (cube)

    Returns
    -------
    stats_props: dict
        dictionary of stats props
        e.g. {'MEAN': 0.0,
        'STDDev': 0.1,
        'SKEW': 0.2,
        'KURT': 0.3}

    """
    stats_props = dict()
    # Open the residual image
    residual_hdu = fitsio.open(fitsname)
    # Get the header data unit for the residual rms
    residual_data = residual_hdu[0].data
    # Get the mean value
    stats_props['MEAN'] = round(abs(residual_data.mean()), 10)
    # Get the sigma value
    stats_props['STDDev'] = float("{0:.6f}".format(residual_data.std()))
    # Flatten image
    res_data = residual_data.flatten()
    # Compute the skewness of the residual
    stats_props['SKEW'] = float("{0:.6f}".format(stats.skew(res_data)))
    # Compute the kurtosis of the residual
    stats_props['KURT'] = float("{0:.6f}".format(stats.kurtosis(res_data, fisher=False)))
    return stats_props


def dynamic_range(fitsname, area_factor=6):
    """Gets the dynamic range in a restored image

    Parameters
    ----------
    fitsname: fits file
        restored image (cube)
    area_factor: int
        Factor to multiply the beam area

    Returns
    -------
    DR: float
        dynamic range value

    """
    fits_info = fitsInfo(fitsname)
    beam_deg = fits_info['b_size']
    # Open the restored image
    restored_hdu = fitsio.open(fitsname)
    # Get the header data unit for the residual rms
    restored_data = restored_hdu[0].data
    # Get the max value
    peak_flux = abs(restored_data.max())
    # Get pixel coordinates of the peak flux
    pix_coord = np.argwhere(restored_data == peak_flux)[0]
    nchan = (restored_data.shape[1] if restored_data.shape[0] == 1
             else restored_data.shape[0])
    # Compute number of pixel in beam and extend by factor area_factor
    ra_num_pix = round((beam_deg[0]*area_factor)/fits_info['dra'])
    dec_num_pix = round((beam_deg[1]*area_factor)/fits_info['ddec'])
    # Create target image slice
    imslice = np.array([pix_coord[2]-ra_num_pix/2, pix_coord[2]+ra_num_pix/2,
                        pix_coord[3]-dec_num_pix/2, pix_coord[3]+dec_num_pix/2])
    imslice = np.array(map(int, imslice))
    # If image is cube then average along freq axis
    min_flux = 0.0
    for frq_ax in range(nchan):
        # In the case where the 0th and 1st axis of the image are not in order
        # i.e. (0, nchan, x_pix, y_pix
        if restored_data.shape[0] == 1:
            target_area = restored_data[0, frq_ax, :, :][imslice]
        else:
            target_area = restored_data[frq_ax, 0, :, :][imslice]
        min_flux += target_area.min()
        if frq_ax == nchan - 1:
            min_flux = min_flux/float(nchan)
    # Compute dynamic range
    DR = peak_flux/abs(min_flux)
    return DR


def get_argparser():
    "Get argument parser"
    parser = argparse.ArgumentParser(
                 description="Examine radio image fidelity by obtaining: \n"
                             "- The four (4) moments of a residual image \n"
                             "- The Dynamic range in restored image")
    argument = partial(parser.add_argument)
    argument('--restored-image',  dest='restored',
             help='Name of the restored image fits file')
    argument('--residual-image',  dest='residual',
             help='Name of the residual image fits file')
    argument('-af', '--area-factor', dest='factor', type=float, default=6,
             help='Factor to multiply the beam area to get target peak area')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()
    output_dict = dict()
    if args.residual:
        stats = residual_image_stats(args.residual)
        output_dict[args.residual] = stats
    if args.restored:
        if args.factor:
            DR = dynamic_range(args.restored, area_factor=args.factor)
        else:
            DR = dynamic_range(args.restored)
        output_dict[args.restored] = {'DR': DR}
    if not args.residual and not args.restored:
        R = '\033[31m'  # red
        W = '\033[0m'   # white (normal)
        print("%sPlease provide fits file name(s)."
              "\nOr\naimfast -h for arguments%s" % (R, W))
    else:
        json_dump(output_dict)
        print(output_dict)
