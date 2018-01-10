import json
import pyfits
import argparse
import numpy as np
from scipy import stats
from functools import partial
from astLib.astWCS import WCS
from astropy.io import fits as fitsio


def json_dump(data_dict, root):
    """Dumps the computed dictionary into a json file"""
    with open('%s/results.json' % root, 'w') as f:
        json.dump(data_dict, f)


def fitsInfo(fitsname=None):
    """Get fits header info"""
    hdu = pyfits.open(fitsname)
    hdr = hdu[0].header
    ra = hdr['CRVAL1']
    dra = abs(hdr['CDELT1'])
    raPix = hdr['CRPIX1']
    dec = hdr['CRVAL2']
    ddec = abs(hdr['CDELT2'])
    decPix = hdr['CRPIX2']
    wcs = WCS(hdr, mode='pyfits')
    return {'wcs': wcs, 'ra': ra, 'dec': dec, 'dra': dra,
            'ddec': ddec, 'raPix': raPix, 'decPix': decPix}


def residual_image_stats(residual_image):
    """Gets statistcal properties of a residual image

    Parameters
    ----------
    image: fits file
        residual image (cube)

    Returns
    -------
    stats_props: dict
        dictionary of stats props
        e.g. {'MEAN': 0.0,
              'STDD': 0.1,
              'SKEW': 0.2,
              'KURT': 0.3}
    """
    stats_props = dict()
    # Open the residual image
    residual_hdu = fitsio.open(residual_image)
    # Get the header data unit for the residual rms
    residual_data = residual_hdu[0].data
    # Get the mean value
    stats_props['MEAN'] = round(abs(residual_data.mean()), 10)
    # Get the sigma value
    stats_props['STDD'] = float("{0:.6f}".format(residual_data.std()))
    # Flatten image
    res_data = residual_data.flatten()
    # Compute the skewness of the residual
    stats_props['SKEW'] = float("{0:.6f}".format(stats.skew(res_data)))
    # Compute the kurtosis of the residual
    stats_props['KURT'] = float("{0:.6f}".format(stats.kurtosis(res_data, fisher=False)))
    return stats_props


def dynamic_range(restored_image, beam_size=6, area_beams=5):
    """Gets the dynamic range in a restored image

    Parameters
    ----------
    image: fits file
        residual image (cube)

    Returns
    -------
    DR: float
        dynamic range value
    """
    beam_deg = (beam_size/3600.0, beam_size/3600.0, 0)
    # Open the restored image
    restored_hdu = fitsio.open(restored_image)
    # Get the header data unit for the residual rms
    restored_data = restored_hdu[0].data
    # Get the max value
    peak_flux = abs(restored_data.max())
    # Get pixel coordinates of the peak flux
    pix_coord = np.argwhere(restored_data == peak_flux)[0]
    nchan = (restored_data.shape[1] if restored_data.shape[0] == 1
             else restored_data.shape[0])
    fits_info = fitsInfo(restored_image)
    # Compute number of pixel in beam and extend by factor area_beams
    ra_num_pix = round((beam_deg[0]*area_beams)/fits_info['dra'])
    dec_num_pix = round((beam_deg[1]*area_beams)/fits_info['ddec'])
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
    required_argument = partial(parser.add_argument, required=True)
    optional_argument = partial(parser.add_argument, required=False)
    required_argument('--mode',  dest='mode', help='Choose examination mode',
                      choices=['restored', 'residual'])
    required_argument('--fitsname',  dest='fitsname', help='Name of the image fits file')
    optional_argument('--beam_size', dest='beam', type=float,
                      help='Beam size in units of arcsec')
    optional_argument('--area_factor', dest='factor', type=float,
                      help='Factor to multiply the beam area to get target peak area')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()
    output_dict = dict()
    if args.mode == 'residual':
        stats = residual_image_stats(args.fitsname)
        output_dict[args.fitsname] = stats
    else:
        DR = dynamic_range(args.fitsname, beam_size=args.beam, area_beams=args.factor)
        output_dict[args.fitsname] = {'DR': DR}
    print output_dict
    json_dump(output_dict, '.')

if __name__ == "__main__":
    main()
