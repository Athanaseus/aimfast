import os
import sys
import json
import Tigger
import random
import string
import logging
import aimfast
import argparse
import tempfile
import numpy as np

from regions import Regions
from functools import partial
from collections import OrderedDict

import scipy
from scipy import stats
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.ndimage import measurements as measure

from bokeh.io import export_svgs

from bokeh.transform import transform

from bokeh.models.widgets import Div, PreText
from bokeh.models.widgets import DataTable, TableColumn

from bokeh.models import Circle
from bokeh.models import CheckboxGroup, CustomJS
from bokeh.models import HoverTool, LinearAxis, Range1d
from bokeh.models import ColorBar, ColumnDataSource, ColorBar
from bokeh.models import LogColorMapper, LogTicker, LinearColorMapper

from bokeh.layouts import row, column, gridplot, grid
from bokeh.plotting import figure, output_file, show, save, ColumnDataSource

from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table
from astropy.io import fits as fitsio
from astropy.coordinates import Angle, SkyCoord

from Tigger.Models import SkyModel, ModelClasses
from Tigger.Coordinates import angular_dist_pos_angle
from sklearn.metrics import mean_squared_error, r2_score

from aimfast.auxiliary import deg2arcsec, deg2arcsec, rad2arcsec
from aimfast.auxiliary import dec2deg, ra2deg, rad2deg, deg2rad, unwrap
from aimfast.auxiliary import aegean, bdsf, get_subimage, get_online_catalog

from aimfast.auxiliary import deg2dec, deg2ra

# Get version
from pkg_resources import get_distribution
_version = get_distribution('aimfast').version

# Unit multipleirs for plotting
FLUX_UNIT_SCALER = {
                    'jansky': [1e0, 'Jy'],
                    'milli': [1e3, 'mJy'],
                    'micro': [1e6, u'\u03bcJy'],
                    'nano': [1e9, 'nJy'],
                   }


POSITION_UNIT_SCALER = {
                        'deg': [1e0, 'deg'],
                        'arcmin': [60.0, u'`'],
                        'arcsec': [3600.0, u'``'],
                       }

# Backgound color for plots
BG_COLOR = 'rgb(229,229,229)'
# Highlighters
R = '\033[31m'  # red
W = '\033[0m'   # white (normal)
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# Decimal places
DECIMALS = 2


def create_logger():
    """Create a console logger"""
    log = logging.getLogger(__name__)
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)
    log.addHandler(console)
    return log


LOGGER = create_logger()

def generate_default_config(configfile):
    "Generate default config file for running source finders"
    from shutil import copyfile
    LOGGER.info(f"Getting parameter file: {configfile}")
    aim_path = os.path.dirname(os.path.dirname(os.path.abspath(aimfast.__file__)))
    copyfile(f"{aim_path}/aimfast/source_finder.yml", configfile)


def get_aimfast_data(filename='fidelity_results.json', dir='.'):
    "Extracts data from the json data file"
    filepath = f"{dir}/{filename}"
    LOGGER.info('Extracting data from the json data file')
    with open(filepath) as f:
        data = json.load(f)
        return data


def json_dump(data_dict, filename='fidelity_results.json'):
    """Dumps the computed dictionary results into a json file.

    Parameters
    ----------
    data_dict : dict
        Dictionary with output results to save.
    filename : str
        Name of file json file where fidelity results will be dumped.
        Default is 'fidelity_results.json' in the current directory.

    Note1
    ----
    If the fidelity_results.json file exists, it will be append, and only
    repeated image assessments will be replaced.

    """
    LOGGER.info(f"Dumping results into the '{filename}' file")
    try:
        # Extract data from the json data file
        with open(filename) as data_file:
            data_existing = json.load(data_file)
            data_existing.update(data_dict)
            data = data_existing
    except IOError:
        data = data_dict
    if data:
        with open(filename, 'w') as f:
            json.dump(data, f)


def fitsInfo(fitsname=None):
    """Get fits header info.

    Parameters
    ----------
    fitsname : fits file
        Restored image (cube)

    Returns
    -------
    fitsinfo : dict
        Dictionary of fits information
        e.g. {'wcs': wcs, 'ra': ra, 'dec': dec,
        'dra': dra, 'ddec': ddec, 'raPix': raPix,
        'decPix': decPix,  'b_size': beam_size,
        'numPix': numPix, 'centre': centre,
        'skyArea': skyArea}

    """
    hdu = fitsio.open(fitsname)
    hdr = hdu[0].header
    ra = hdr['CRVAL1']
    dra = abs(hdr['CDELT1'])
    raPix = hdr['CRPIX1']
    dec = hdr['CRVAL2']
    ddec = abs(hdr['CDELT2'])
    decPix = hdr['CRPIX2']
    wcs = WCS(hdr)
    numPix = hdr['NAXIS1']
    try:
        beam_size = (hdr['BMAJ'], hdr['BMIN'], hdr['BPA'])
    except:
        beam_size = None
    try:
        centre = (hdr['CRVAL1'], hdr['CRVAL2'])
    except:
        centre = None
    try:
        freq0=None
        for i in range(1, hdr['NAXIS']+1):
            if hdr['CTYPE{0:d}'.format(i)].startswith('FREQ'):
                freq0 = hdr['CRVAL{0:d}'.format(i)]
    except:
        freq0=None

    skyArea = (numPix * ddec) ** 2
    fitsinfo = {'wcs': wcs, 'ra': ra, 'dec': dec,
                'dra': dra, 'ddec': ddec, 'raPix': raPix,
                'decPix': decPix, 'b_size': beam_size,
                'numPix': numPix, 'centre': centre,
                'skyArea': skyArea, 'freq0': freq0}
    return fitsinfo


def measure_psf(psffile, arcsec_size=20):
    """Measure point spread function after deconvolution.

    Parameters
    ----------
    psfile : fits file
        Point spread function file.
    arcsec_size : float
        Cross section size

    Returns
    -------
    r0 : float
        Average psf size.

    """
    with fitsio.open(psffile) as hdu:
        pp = hdu[0].data.T[:, :, 0, 0]
        secpix = abs(hdu[0].header['CDELT1'] * 3600)
    # Get midpoint and size of cross-sections
    xmid, ymid = measure.maximum_position(pp)
    sz = int(arcsec_size / secpix)
    xsec = pp[xmid - sz: xmid + sz, ymid]
    ysec = pp[xmid, ymid - sz: ymid + sz]

    def fwhm(tsec):
        """Determine the full width half maximum"""
        tmid = len(tsec) / 2.0
        # First minima off the peak, and flatten cross-section outside them
        xmin = measure.minimum_position(tsec[:tmid])[0]
        tsec[:xmin] = tsec[xmin]
        xmin = measure.minimum_position(tsec[tmid:])[0]
        tsec[tmid + xmin:] = tsec[tmid + xmin]
        if tsec[0] > 0.5 or tsec[-1] > 0.5:
            LOGGER.info(f"PSF FWHM over {arcsec_size * 2:.2f} arcsec")
            return arcsec_size, arcsec_size
        x1 = interp1d(tsec[:tmid], range(tmid))(0.5)
        x2 = interp1d(1 - tsec[tmid:], range(tmid, len(tsec)))(0.5)
        return x1, x2

    ix0, ix1 = fwhm(xsec)
    iy0, iy1 = fwhm(ysec)
    rx, ry = (ix1 - ix0) * secpix, (iy1 - iy0) * secpix
    r0 = (rx + ry) / 2.0
    return r0


def get_box(wcs, radec, w):
    """Get box of width w around source coordinates radec.

    Parameters
    ----------
    radec : tuple
        RA and DEC in degrees.
    w : int
        Width of box.
    wcs : atropy.wcs instance
        World Coordinate System.

    Returns
    -------
    box : tuple
        A box centred at radec.

    """
    radec_pix = SkyCoord(*radec,unit='deg').to_pixel(wcs)
    raPix, decPix = radec_pix[0] , radec_pix[1]
    raPix = int(raPix)
    decPix = int(decPix)
    box = (slice(decPix - int(w / 2), decPix + int(w / 2)),
           slice(raPix - int(w / 2), raPix + int(w / 2)))
    return box


def noise_sigma(noise_image):
    """Determines the noise sigma level in a dirty image with no source

    Parameters
    ----------
    noise_image : file
        Noise image (cube).

    Returns
    -------
    noise_std : float
        Noise image standard deviation

    """
    # Read the simulated noise image
    dirty_noise_hdu = fitsio.open(noise_image)
    # Get the header data unit for the simulated noise
    dirty_noise_data = dirty_noise_hdu[0].data
    # Get the noise sigma
    noise_std = dirty_noise_data.std()
    return noise_std


def _get_ra_dec_range(area, phase_centre):
    """Get RA and DEC range from area of observations and phase centre"""
    ra = phase_centre[0]
    dec =  phase_centre[1]
    d_ra = np.sqrt(area) / 2.0
    d_dec = np.sqrt(area) / 2.0
    ra_range = [ra - d_ra, ra + d_ra]
    dec_range = [dec - d_dec, dec + d_dec]
    return ra_range, dec_range


def _source_angular_dist_pos_angle(src1, src2):
    """Computes the angular distance between the two points on a sphere, and
    the position angle (North through East) of the direction from 1 to 2.""";
    ra1, dec1 = src1.pos.ra, src1.pos.dec
    ra2, dec2 = src2.pos.ra, src2.pos.dec
    return angular_dist_pos_angle(ra1, dec1, ra2, dec2)


def _get_phase_centre(model):
    """Compute the phase centre of observation"""
    # Get all sources in the model
    model_sources = model.sources
    # Get source Ra and Dec coordinates
    RA = [rad2deg(src.pos.ra) for src in model_sources]
    DEC = [rad2deg(src.pos.dec) for src in model_sources]
    xc = np.sum(RA)/len(RA)
    yc = np.sum(DEC)/len(DEC)
    return (xc ,yc)


def _get_random_pixel_coord(num, sky_area, phase_centre=[0.0, -30.0]):
    """Provides random pixel coordinates

    Parameters
    ----------
    num: int
        Number of data points
    sky: float
        Sky area to extract random points
        Phase tracking centre of the telescope during observation [ra0,dec0]
    phase_centre: list
        Phase centre in degrees

    Returns
    -------
    COORDs: list
        List of coordinates
    """
    ra_range, dec_range = _get_ra_dec_range(sky_area, phase_centre)
    COORDs = []
    for i in range(num):
        current = []
        # add another number to the current list
        current.append(random.uniform(ra_range[0], ra_range[1]))
        current.append(random.uniform(dec_range[0], dec_range[1]))
        # convert current list into a tuple and add to resulting list
        COORDs.append(tuple(current))
    random.shuffle(COORDs)
    return COORDs


def get_image_products(images, mask):
    """Get a product of images with a mask

    Parameters
    ----------
    images: list
        List of fits images to get product
    mask: str
        Mask to multiply the images

    Returns
    -------
    prod_images: list
        List of resulting fits images
    """
    LOGGER.info("Computing product...")
    prod_images = []
    msk = fitsio.open(mask)
    msk_data = msk[0].data
    for img in images:
        outname = img.replace('.fits', '.prod.fits')
        im = fitsio.open(img)
        img_data = im[0].data
        p_img_data = img_data * msk_data
        LOGGER.info("Writing output images")
        if os.path.exists(outname):
            LOGGER.warning("Output image exists")
            sys.exit(1)
        else:
            im[0].data = p_img_data
            im.writeto(outname)
            LOGGER.info(f"New image: {outname}")
        prod_images.append(outname)
    return prod_images

def residual_image_stats(fitsname, test_normality=None, data_range=None,
                         threshold=None, chans=None, mask=None):
    """Gets statistcal properties of a residual image.

    Parameters
    ----------
    fitsname : file
        Residual image (cube).
    test_normality : str
        Perform normality testing using either `shapiro` or `normaltest`.
    data_range : int, optional
        Range of data to perform normality testing.
    threshold : float, optional
        Cut-off threshold to select channels in a cube
    chans : str, optional
        Channels to compute stats (e.g. 1;0~50;100~200)
    mask : file
        Fits mask to get stats in image

    Returns
    -------
    props : dict
        Dictionary of stats properties.
        e.g. {'MEAN': 0.0, 'STDDev': 0.1, 'RMS': 0.1,
              'SKEW': 0.2, 'KURT': 0.3, 'MAD': 0.4,
              'MAX': 0.7, 'SUM_NEG': -0.1}

    Notes
    -----
    If normality_test=True, dictionary of stats props becomes \
    e.g. {'MEAN': 0.0, 'STDDev': 0.1, 'SKEW': 0.2, 'KURT': 0.3, \
          'MAD': 0.4, 'RMS': 0.5, 'SUM_NEG': -0.1, 'MAX': 0.7, \
          'NORM': (123.3,0.012)} \
    whereby the first element is the statistics (or average if data_range specified) \
    of the datasets and second element is the p-value.

    """
    # Open the residual image
    residual_hdu = fitsio.open(fitsname)
    # Get the header data unit for the residual rms
    residual_data = residual_hdu[0].data
    # Get residual data
    # In case the first two axes are swapped
    data = (residual_data[0]
            if residual_data.shape[0] == 1
            else residual_data[1])
    if chans:
        nchans = []
        chan_ranges = chans.split(';')
        for cr in chan_ranges:
            if '~' in cr:
                c = cr.split('~')
                nchans.extend(range(int(c[0]), int(c[1]) + 1))
            else:
                nchans.append(int(cr))
        residual_data = data[nchans]
        data = residual_data
    if threshold:
        nchans = []
        for i in range(data.shape[0]):
            d = data[i][data[i] > float(threshold)]
            if d.shape[0] > 0:
                nchans.append(i)
        residual_data = data[nchans]
        data = residual_data
    if mask:
        import numpy.ma as ma
        mask_hdu = fitsio.open(mask)
        mask_data = mask_hdu[0].data
        residual_data = ma.masked_array(data, mask=mask_data)
        data = residual_data
    residual_data = data

    props = image_stats(residual_data, test_normality=test_normality)

    return props


def image_stats(image_data, test_normality=None, data_range=None):

    img_stats = dict()
    # Get the min value
    LOGGER.info("Computing min ...")
    img_stats['MIN'] = float("{0:.6}".format(image_data.min()))
    LOGGER.info("MIN = {}".format(img_stats['MIN']))
    # Get the max value
    LOGGER.info("Computing max ...")
    img_stats['MAX'] = float("{0:.6}".format(image_data.max()))
    LOGGER.info("MAX = {}".format(img_stats['MAX']))
    # Get the mean value
    LOGGER.info("Computing mean ...")
    img_stats['MEAN'] = float("{0:.6}".format(image_data.mean()))
    LOGGER.info("MEAN = {}".format(img_stats['MEAN']))
    # Get the rms value
    LOGGER.info("Computing root mean square ...")
    img_stats['RMS'] = float("{0:.6f}".format(np.sqrt(np.mean(np.square(image_data)))))
    LOGGER.info("RMS = {}".format(img_stats['RMS']))
    # Get the sigma value
    LOGGER.info("Computing standard deviation ...")
    img_stats['STDDev'] = float("{0:.6f}".format(image_data.std()))
    LOGGER.info("STDDev = {}".format(img_stats['STDDev']))
    # Flatten image
    img_data = image_data.flatten()
    # Get the maximum absolute deviation
    LOGGER.info("Computing median absolute deviation ...")
    img_stats['MAD'] = float("{0:.6f}".format(stats.median_abs_deviation(img_data)))
    LOGGER.info("MAD = {}".format(img_stats['MAD']))
    # Compute the skewness of the residual
    LOGGER.info("Computing skewness ...")
    img_stats['SKEW'] = float("{0:.6f}".format(stats.skew(img_data)))
    LOGGER.info("SKEW = {}".format(img_stats['SKEW']))
    # Compute the kurtosis of the residual
    LOGGER.info("Computing kurtosis ...")
    img_stats['KURT'] = float("{0:.6f}".format(stats.kurtosis(img_data, fisher=False)))
    LOGGER.info("KURT = {}".format(img_stats['KURT']))
    # Compute the sum of Negative pixels
    LOGGER.info("Computing sum of negative pixels ...")
    img_stats['SUM_NEG'] = float("{0:.6f}".format(np.sum(img_data[np.where(img_data<0.0)])))
    LOGGER.info("SUM_NEG = {}".format(img_stats['SUM_NEG']))
    # Perform normality testing
    if test_normality:
        LOGGER.info("Performing normality test ...")
        norm_props = normality_testing(img_data, test_normality, data_range)
        img_stats.update(norm_props)
        LOGGER.info("NORM = {}".format(img_stats['NORM']))
    # Return dictionary of results
    return img_stats


def fix_wcs_fits(wcs, dropaxis=2):
    """This removes the degenerated dimensions in APLpy 2.X...
    The input must be the object returned by aplpy.FITSFigure().
    `dropaxis` is the index where to start dropping the axis (by default it assumes the 3rd,4th place).
    """
    temp_wcs = wcs.dropaxis(dropaxis)
    temp_wcs = temp_wcs.dropaxis(dropaxis)
    return temp_wcs


def get_region_stats(fitsname, regions_file):
    """Extract flux densities measurements within the provided region"""
    regions_stats = dict()
    LOGGER.info(f'Reading region file: {regions_file}')
    regions_list = Regions.read(regions_file, format='ds9')
    LOGGER.info(f'Number of regions: {len(regions_list)}')
    image = fitsio.open(fitsname)
    image_data = image[0].data
    fitsinfo = fitsInfo(fitsname)
    wcs = fitsinfo['wcs']
    beam =  fitsinfo['b_size']
    dra = fitsinfo['dra']
    beam_area = (beam[0]*beam[1])/(dra*dra)
    for i, input_region in enumerate(regions_list):
        if hasattr(input_region, 'to_pixel'):
            input_region = input_region.to_pixel(fix_wcs_fits(wcs))
        mask = input_region.to_mask().to_image(image_data.shape[-2:])
        data = mask * image_data[0][0]
        #nndata=nndata[~np.isnan(data)]
        nndata = np.flip(data, axis=0)
        nndata = nndata[~np.isnan(nndata)]
        nndata = nndata[nndata != -0.0]
        stats = image_stats(nndata)
        regions_stats[f'region-{i}'] = stats
    return regions_stats


def normality_testing(data, test_normality='normaltest', data_range=None):
    """Performs a normality test on the image data.

    Parameters
    ----------
    data : numpy.array
        Residual residual array. i.e. fitsio.open(fitsname)[0].data
    test_normality : str
        Perform normality testing using either `shapiro` or `normaltest`.
    data_range : int
        Range of data to perform normality testing.

    Returns
    -------
    normality : dict
        dictionary of stats props.
        e.g. {'NORM': (123.3,  0.012)}
        whereby the first element is the statistics
        (or average if data_range specified) of the
        datasets and second element is the p-value.

    """
    norm_res = []
    normality = dict()
    # Get residual image data
    res_data = data
    # Shuffle the data
    random.shuffle(res_data)
    # Normality test
    counter = 0
    # Check size of image data
    if len(res_data) == 0:
        raise ValueError(f"{R}No data to compute stats."
                         "\nEither threshold too high "
                         "or all data is masked.{W}")
    if data_range:
        for dataset in range(len(res_data) / int(data_range)):
            i = counter
            counter += data_range
            norm_res.append(getattr(stats, test_normality)(res_data[i: counter]))
        # Compute sum of pvalue
        if test_normality == 'normaltest':
            sum_statistics = sum([norm.statistic for norm in norm_res])
            sum_pvalues = sum([norm.pvalue for norm in norm_res])
        elif test_normality == 'shapiro':
            sum_statistics = sum([norm[0] for norm in norm_res])
            sum_pvalues = sum([norm[1] for norm in norm_res])
        normality['NORM'] = (sum_statistics / dataset, sum_pvalues / dataset)
    else:
        norm_res = getattr(stats, test_normality)(res_data)
        if test_normality == 'normaltest':
            statistic = float(norm_res.statistic)
            pvalue = float(norm_res.pvalue)
            normality['NORM'] = (statistic, pvalue)
        elif test_normality == 'shapiro':
            normality['NORM'] = norm_res
    return normality


def model_dynamic_range(lsmname, fitsname, beam_size=5, area_factor=2):
    """Gets the dynamic range using model lsm and residual fits.

    Parameters
    ----------
    fitsname : fits file
        Residual image (cube).
    lsmname : lsm.html or .txt file
        Model .lsm.html from pybdsm (or .txt converted tigger file).
    beam_size : float
        Average beam size in arcsec.
    area_factor : float
        Factor to multiply the beam area.

    Returns
    -------
    DR : dict
        DRs - dynamic range values.

    """
    # Open the residual image
    residual_hdu = fitsio.open(fitsname)
    residual_data = residual_hdu[0].data
    # Load model file
    model_lsm = Tigger.load(lsmname)
    # Get detected sources
    model_sources = model_lsm.sources
    # Obtain peak flux source
    peak_flux = None
    try:
        sources_flux = dict([(model_source, model_source.getTag('I_peak'))
                            for model_source in model_sources])
        peak_source_flux = [(_model_source, flux)
                            for _model_source, flux in sources_flux.items()
                            if flux == max(list(sources_flux.values()))][0][0]
        peak_flux = peak_source_flux.getTag('I_peak')
    except TypeError:
        pass
    if not peak_flux:
        # In case no I_peak is not found use the integrated flux
        sources_flux = dict([(model_source, model_source.flux.I)
                            for model_source in model_sources])
        peak_source_flux = [(_model_source, flux)
                            for _model_source, flux in sources_flux.items()
                            if flux == max(list(sources_flux.values()))][0][0]
        peak_flux = peak_source_flux.flux.I
    # Get astrometry of the source in degrees
    RA = rad2deg(peak_source_flux.pos.ra)
    DEC = rad2deg(peak_source_flux.pos.dec)
    # Get source region and slice
    width = int(beam_size * area_factor)
    imslice = get_box(fitsInfo(fitsname)['wcs'], (RA, DEC), width)
    source_res_area = np.array(residual_data[0, 0, :, :][imslice])
    min_flux = source_res_area.min()
    local_std = source_res_area.std()
    global_std = residual_data[0, 0, ...].std()
    # Compute dynamic range
    DR = {
        "deepest_negative"  : peak_flux/abs(min_flux)*1e0,
        "local_rms"         : peak_flux/local_std*1e0,
        "global_rms"        : peak_flux/global_std*1e0,
    }
    return DR


def image_dynamic_range(fitsname, residual, area_factor=6):
    """Gets the dynamic range in a restored image.

    Parameters
    ----------
    fitsname : fits file
        Restored image (cube).
    residual : fits file
        Residual image (cube).
    area_factor: int
        Factor to multiply the beam area.

    Returns
    -------
    DR : dict
        DRs - dynamic range values.

    """
    fits_info = fitsInfo(fitsname)
    # Get beam size otherwise use default (~6``).
    beam_default = (0.00151582804885738, 0.00128031965017612, 20.0197348935424)
    beam_deg = fits_info['b_size'] if fits_info['b_size'] else beam_default
    # Open the restored and residual images
    restored_hdu = fitsio.open(fitsname)
    residual_hdu = fitsio.open(residual)
    # Get the header data unit for the peak and residual rms
    restored_data = restored_hdu[0].data
    residual_data = residual_hdu[0].data
    # Get the max value
    peak_flux = abs(restored_data.max())
    # Get pixel coordinates of the peak flux
    pix_coord = np.argwhere(restored_data == peak_flux)[0]
    nchan = (restored_data.shape[1] if restored_data.shape[0] == 1
             else restored_data.shape[0])
    # Compute number of pixel in beam and extend by factor area_factor
    ra_num_pix = round((beam_deg[0] * area_factor) / fits_info['dra'])
    dec_num_pix = round((beam_deg[1] * area_factor) / fits_info['ddec'])
    # Create target image slice
    imslice = np.array([pix_coord[2]-ra_num_pix/2, pix_coord[2]+ra_num_pix/2,
                        pix_coord[3]-dec_num_pix/2, pix_coord[3]+dec_num_pix/2])
    imslice = np.array(list(map(int, imslice)))
    # If image is cube then average along freq axis
    min_flux = 0.0
    for frq_ax in range(nchan):
        # In the case where the 0th and 1st axis of the image are not in order
        # i.e. (0, nchan, x_pix, y_pix)
        if residual_data.shape[0] == 1:
            target_area = residual_data[0, frq_ax, :, :][imslice]
        else:
            target_area = residual_data[frq_ax, 0, :, :][imslice]
        min_flux += target_area.min()
        if frq_ax == nchan - 1:
            min_flux = min_flux/float(nchan)
    # Compute dynamic range
    local_std = target_area.std()
    global_std = residual_data[0, 0, ...].std()
    # Compute dynamic range
    DR = {
        "deepest_negative"  : peak_flux / abs(min_flux) * 1e0,
        "local_rms"         : peak_flux / local_std * 1e0,
        "global_rms"        : peak_flux / global_std * 1e0,
    }
    return DR


def get_src_scale(source_shape):
    """Get scale measure of the source in arcsec.

    Parameters
    ----------
    source_shape : lsm object
        Source shape object from model

    Returns
    -------
    (scale_out_arc_sec, scale_out_err_arc_sec) : tuple
        Output source scale with error value

    """
    if source_shape:
        shape_out = source_shape.getShape()
        shape_out_err = source_shape.getShapeErr()
        minx = shape_out[0]
        majx = shape_out[1]
        minx_err = shape_out_err[0]
        majx_err = shape_out_err[1]
        if minx > 0 and majx > 0:
            scale_out = np.sqrt(minx*majx)
            scale_out_err = np.sqrt(minx_err*minx_err + majx_err*majx_err)
        elif minx > 0:
            scale_out = minx
            scale_out_err = minx_err
        elif majx > 0:
            scale_out = majx
            scale_out_err = majx_err
        else:
            scale_out = 0
            scale_out_err = 0
    else:
        scale_out = 0
        scale_out_err = 0
    scale_out_arc_sec = rad2arcsec(scale_out)
    scale_out_err_arc_sec = rad2arcsec(scale_out_err)
    return scale_out_arc_sec, scale_out_err_arc_sec


def get_model(catalog):
    """Get model model object from file catalog"""

    def tigger_src_ascii(src, idx):
        """Get ascii catalog source as a tigger source """

        name = "SRC%d" % idx
        flux = ModelClasses.Polarization(float(src["int_flux"]), 0, 0, 0,
                                         I_err=float(src["err_int_flux"]))
        ra, ra_err = map(np.deg2rad, (float(src["ra"]), float(src["err_ra"])))
        dec, dec_err = map(np.deg2rad, (float(src["dec"]),
                                        float(src["err_dec"])))
        pos = ModelClasses.Position(ra, dec, ra_err=ra_err, dec_err=dec_err)
        ex, ex_err = map(np.deg2rad, (float(src["a"]), float(src["err_a"])))
        ey, ey_err = map(np.deg2rad, (float(src["b"]), float(src["err_b"])))
        pa, pa_err = map(np.deg2rad, (float(src["pa"]), float(src["err_pa"])))
        if ex and ey:
            shape = ModelClasses.Gaussian(ex, ey, pa, ex_err=ex_err,
                                          ey_err=ey_err, pa_err=pa_err)
        else:
            shape = None
        source = SkyModel.Source(name, pos, flux, shape=shape)
        # Adding source peak flux (error) as extra flux attributes for sources,
        # and to avoid null values for point sources I_peak = src["Total_flux"]
        if shape:
            source.setAttribute("I_peak", float(src["peak_flux"]))
            source.setAttribute("I_peak_err", float(src["err_peak_flux"]))
        else:
            source.setAttribute("I_peak", float(src["int_flux"]))
            source.setAttribute("I_peak_err", float(src["err_int_flux"]))
        return source

    def tigger_src_nvss(src, idx):
        """Get ascii catalog source as a tigger source """

        name = "SRC%d" % idx
        flux = ModelClasses.Polarization(float(src["S1.4"]/1000.), 0, 0, 0,
                                         I_err=float(src["e_S1.4"]/1000.))
        ra, ra_err = map(np.deg2rad, (float(ra2deg(src["RAJ2000"])),
                                      float(src["e_RAJ2000"]/3600.)))
        dec, dec_err = map(np.deg2rad, (float(dec2deg(src["DEJ2000"])),
                                        float(src["e_DEJ2000"]/3600.)))
        pos = ModelClasses.Position(ra, dec, ra_err=ra_err, dec_err=dec_err)
        ex, ex_err = map(np.deg2rad, (float(src['MajAxis']), float(0.00)))
        ey, ey_err = map(np.deg2rad, (float(src['MinAxis']), float(0.00)))
        pa, pa_err = map(np.deg2rad, (float(0.00), float(0.00)))
        if ex and ey:
            shape = ModelClasses.Gaussian(ex, ey, pa, ex_err=ex_err,
                                          ey_err=ey_err, pa_err=pa_err)
        else:
            shape = None
        source = SkyModel.Source(name, pos, flux, shape=shape)
        # Adding source peak flux (error) as extra flux attributes for sources,
        # and to avoid null values for point sources I_peak = src["Total_flux"]
        source.setAttribute("I_peak", float(src["S1.4"]/1000.))
        source.setAttribute("I_peak_err", float(src["e_S1.4"]/1000.))
        return source

    def tigger_src_sumss(src, idx):
        """Get ascii catalog source as a tigger source """

        name = "SRC%d" % idx
        flux = ModelClasses.Polarization(float(src["St"]/1000.), 0, 0, 0,
                                         I_err=float(src["e_St1.4"]/1000.))
        ra, ra_err = map(np.deg2rad, (float(ra2deg(src["RAJ2000"])),
                                      float(src["e_RAJ2000"]/3600.)))
        dec, dec_err = map(np.deg2rad, (float(dec2deg(src["DEJ2000"])),
                                        float(src["e_DEJ2000"]/3600.)))
        pos = ModelClasses.Position(ra, dec, ra_err=ra_err, dec_err=dec_err)
        ex, ex_err = map(np.deg2rad, (float(src['MajAxis']), float(0.00)))
        ey, ey_err = map(np.deg2rad, (float(src['MinAxis']), float(0.00)))
        pa, pa_err = map(np.deg2rad, (float(src['PA']), float(0.00)))
        if ex and ey:
            shape = ModelClasses.Gaussian(ex, ey, pa, ex_err=ex_err,
                                          ey_err=ey_err, pa_err=pa_err)
        else:
            shape = None
        source = SkyModel.Source(name, pos, flux, shape=shape)
        # Adding source peak flux (error) as extra flux attributes for sources,
        # and to avoid null values for point sources I_peak = src["Total_flux"]
        source.setAttribute("I_peak", float(src["Sp"]/1000.))
        source.setAttribute("I_peak_err", float(src["e_Sp"]/1000.))
        return source

    def tigger_src_fits(src, idx, freq0=None):
        """Get fits catalog source as a tigger source """

        name = "SRC%d" % idx
        flux = ModelClasses.Polarization(float(src["Total_flux"]), 0, 0, 0,
                                         I_err=float(src["E_Total_flux"]))
        ra, ra_err = map(np.deg2rad, (float(src["RA"]), float(src["E_RA"])))
        dec, dec_err = map(np.deg2rad, (float(src["DEC"]), float(src["E_DEC"])))
        pos = ModelClasses.Position(ra, dec, ra_err=ra_err, dec_err=dec_err)
        ex, ex_err = map(np.deg2rad, (float(src["DC_Maj"]), float(src["E_DC_Maj"])))
        ey, ey_err = map(np.deg2rad, (float(src["DC_Min"]), float(src["E_DC_Min"])))
        pa, pa_err = map(np.deg2rad, (float(src["PA"]), float(src["E_PA"])))
        # Try to get spectral index

        if ex and ey:
            shape = ModelClasses.Gaussian(ex, ey, pa, ex_err=ex_err,
                                          ey_err=ey_err, pa_err=pa_err)
        else:
            shape = None
        source = SkyModel.Source(name, pos, flux, shape=shape)
        # Adding source peak flux (error) as extra flux attributes for sources,
        # and to avoid null values for point sources I_peak = src["Total_flux"]
        if shape:
            pass  # TODO: Check for other models what peak is
            #source.setAttribute("I_peak", src["Peak_flux"])
            #source.setAttribute("I_peak_err", src["E_peak_flux"])
        else:
            source.setAttribute("I_peak", src["Total_flux"])
            source.setAttribute("I_peak_err", src["E_Total_flux"])
        if freq0:
            try:
                spi, spi_err = (src['Spec_Indx'], src['E_Spec_Indx'])
                source.spectrum = ModelClasses.SpectralIndex(spi, freq0)
                source.setAttribute('spi_error', spi_err)
            except:
                pass
        return source

    def tigger_src_wsclean(src, idx):
        """Get ascii catalog source as a tigger source """

        name = src['col1']
        flux = ModelClasses.Polarization(float(src["col5"]), 0, 0, 0,
                                         I_err=float(0.00))
        ra, ra_err = map(np.deg2rad, (float(ra2deg(src["col3"])),
                                      float(0.00)))
        dec, dec_err = map(np.deg2rad, (float(dec2deg(src["col4"])),
                                        float(0.00)))
        pos = ModelClasses.Position(ra, dec, ra_err=ra_err, dec_err=dec_err)
        ex, ex_err = map(np.deg2rad, (float(src["col9"])
                                      if type(src["col9"]) is not
                                      np.ma.core.MaskedConstant else
                                      0.00, float(0.00)))
        ey, ey_err = map(np.deg2rad, (float(src["col10"])
                                      if type(src["col10"]) is not
                                      np.ma.core.MaskedConstant else
                                      0.00, float(0.00)))
        pa, pa_err = map(np.deg2rad, (float(0.00), float(0.00)))
        if ex and ey:
            shape = ModelClasses.Gaussian(ex, ey, pa, ex_err=ex_err,
                                          ey_err=ey_err, pa_err=pa_err)
        else:
            shape = None
        source = SkyModel.Source(name, pos, flux, shape=shape)
        # Adding source peak flux (error) as extra flux attributes for sources,
        # and to avoid null values for point sources I_peak = src["Total_flux"]
        source.setAttribute("I_peak", float(src['col5']))
        source.setAttribute("I_peak_err", float(0.00))
        return source

    tfile = tempfile.NamedTemporaryFile(suffix='.txt')
    tfile.flush()
    with open(tfile.name, "w") as stdw:
        stdw.write("#format:name ra_d dec_d i emaj_s emin_s pa_d\n")
    model = Tigger.load(tfile.name)
    tfile.close()
    ext = os.path.splitext(catalog)[-1]
    if ext in ['.html', '.txt']:
        if 'catalog_table' in catalog and not catalog.endswith('.html'):
            data = Table.read(catalog, format='ascii')
            for i, src in enumerate(data):
                # Check which online catalog the source belongs to
                # Prefix is in the name by default when created
                if 'nvss' in catalog and not catalog.endswith('.html'):
                    model.sources.append(tigger_src_nvss(src, i))
                if 'sumss' in catalog and not catalog.endswith('.html'):
                    model.sources.append(tigger_src_sumss(src, i))
            centre = _get_phase_centre(model)
            model.ra0, model.dec0 = map(np.deg2rad, centre)
            model.save(catalog[:-4]+".lsm.html")
        elif 'sources.txt' in catalog:
            data = Table.read(catalog, format='ascii')
            for i, src in enumerate(data):
                if i:
                    model.sources.append(tigger_src_wsclean(src, i))
            centre = _get_phase_centre(model)
            model.ra0, model.dec0 = map(np.deg2rad, centre)
            model.save(catalog[:-4]+".lsm.html")
        else:
            model = Tigger.load(catalog)
    if ext in ['.tab', '.csv']:
        data = Table.read(catalog, format='ascii')
        if ext == '.tab':
            fits_file = catalog.replace('_comp.tab', '.fits')
        else:
            fits_file = catalog.replace('_comp.csv', '.fits')
        fitsinfo = fitsInfo(fits_file)
        for i, src in enumerate(data):
            model.sources.append(tigger_src_ascii(src, i))
        centre = fitsinfo['centre'] or _get_phase_centre(model)
        model.ra0, model.dec0 = map(np.deg2rad, centre)
        model.save(catalog[:-4]+".lsm.html")
    if ext in ['.fits']:
        data = Table.read(catalog, format='fits')
        fits_file = catalog.replace('-pybdsf', '')
        fitsinfo = fitsInfo(fits_file)
        freq0 = fitsinfo['freq0']
        for i, src in enumerate(data):
            model.sources.append(tigger_src_fits(src, i, freq0))
        centre = fitsinfo['centre'] or _get_phase_centre(model)
        model.ra0, model.dec0 = map(np.deg2rad, centre)
        model.save(catalog[:-5]+".lsm.html")
    return model


def get_detected_sources_properties(model_1, model_2, tolerance, shape_limit=6.0,
                                    all_sources=False, closest_only=False, off_axis=None):
    """Extracts the output simulation sources properties.

    Parameters
    ----------
    models_1 : file
        Tigger formatted or txt model 1 file
    models_2 : file
        Compare all sources in the catalog (else only sources with maj<shape_limit)
    tolerance : float
        Tolerace to cross-match sources
    shape_limit: float
        Cross match only sources with maj-axis less than this value
    closest_only: bool
        Returns the closest source only as the matching source
    off_axis: float
        Cross-match only sources within this distance from the centre

    Returns
    -------
    (targets_flux, targets_scale, targets_position) : tuple
        Tuple of target flux, morphology and astrometry information

    """
    model_lsm1 = get_model(model_1)
    model_lsm2 = get_model(model_2)
    # Sources from the input model
    model1_sources = model_lsm1.sources
    # {"source_name": [I_out, I_out_err, I_in, source_name]}
    targets_flux = dict()       # recovered sources flux
    # {"source_name": [delta_pos_angle_arc_sec, ra_offset, dec_offset,
    #                  delta_phase_centre_arc_sec, I_in, source_name]
    targets_position = dict()   # recovered sources position
    # {"source_name: [shape_out=(maj, min, angle), shape_out_err=, shape_in=,
    #                 scale_out, scale_out_err, I_in, source_name]
    targets_scale = dict()         # recovered sources scale
    deci = DECIMALS  # round off to this decimal places
    tolerance *= (np.pi / (3600.0 * 180))  # Convert to radians
    names = dict()
    closest_only = True
    for model1_source in model1_sources:
        I_out = 0.0
        I_out_err = 0.0
        source1_name = model1_source.name
        ra1 = model1_source.pos.ra
        dec1 = model1_source.pos.dec
        ra_err1 = model1_source.pos.ra_err
        dec_err1 = model1_source.pos.dec_err
        I_in = model1_source.flux.I
        I_in_err = model1_source.flux.I_err if model1_source.flux.I_err else 0.0
        model2_sources = model_lsm2.getSourcesNear(ra1, dec1, tolerance)
        if not model2_sources:
            continue
        # More than one source detected, thus we sum up all the detected sources with
        # a radius equal to the beam size in radians around the true target coordinate
        # Or use the closest source only
        if closest_only:
            if len(model2_sources) > 1:
                rdist = np.array([_source_angular_dist_pos_angle(model1_source,
                                                                 model2_source)[0]
                                  for model2_source in model2_sources])
                model2_sources = [model2_sources[np.argmin(rdist)]]

        I_out_err_list = []
        I_out_list = []
        for target in model2_sources:
            I_out_list.append(target.flux.I)
            I_out_err_list.append(target.flux.I_err * target.flux.I_err)

        if I_out_list[0] > 0.0:
            model2_source = model2_sources[0]
            try:
                shape_in = tuple(map(rad2arcsec, model1_source.shape.getShape()))
                shape_in_err = tuple(map(rad2arcsec, model1_source.shape.getShapeErr()))
            except AttributeError:
                shape_in = (0, 0, 0)
                shape_in_err = (0, 0, 0)
            if model2_source.shape:
                shape_out = tuple(map(rad2arcsec, model2_source.shape.getShape()))
                shape_out_err = tuple(map(rad2arcsec, model2_source.shape.getShapeErr()))
            else:
                shape_out = (0, 0, 0)
                shape_out_err = (0, 0, 0)
            if not all_sources:
                if shape_out[0] > shape_limit:
                    continue

            if closest_only:
                I_out = model2_source.flux.I
                I_out_err = model2_source.flux.I_err
                ra2 = model2_source.pos.ra
                dec2 = model2_source.pos.dec
                ra_err2 = model2_source.pos.ra_err
                dec_err2 = model2_source.pos.dec_err
            else:
                # weighting with the flux error appears to be dangerous thing as
                # these values are very small taking their reciprocal
                # leads to very high weights
                # Also if the model has no errors this will raise
                # a div by zero exception (ZeroDivisionError)
                try:
                    I_out = sum([val / err for val, err in zip(I_out_list, I_out_err_list)])
                    I_out_err = sum([1.0 / I_out_error for I_out_error
                                     in I_out_err_list])
                    I_out_var_err = np.sqrt(1.0 / I_out_err)
                    I_out /= I_out_err
                    I_out_err = I_out_var_err
                    ra2 = (np.sum([src.pos.ra * src.flux.I for src in model2_sources]) /
                          np.sum([src.flux.I for src in model2_sources]))
                    dec2 = (np.sum([src.pos.dec * src.flux.I for src in model2_sources]) /
                          np.sum([src.flux.I for src in model2_sources]))
                   # Get position weighted error
                   # _err_a = np.sqrt(np.sum([np.sqrt((src.flux.I_err/src.flux.I)**2 +
                   #                          (src.pos.ra_err/src.pos.ra)**2)*np.abs(src.flux.I*src.pos.ra)
                   #                  for src in model2_sources]))
                   # _a = np.sum([src.flux.I*src.pos.ra for src in model2_sources])
                   # _err_b = np.sqrt(np.sum([src.flux.I_err**2 for src in model2_sources]))
                   # _b = np.sum([src.flux.I for src in model2_sources])
                   # ra_err = np.abs(ra) * (np.sqrt((_err_a / _a)**2 + (_err_b / _b)**2))
                   # _err_a = np.sqrt(np.sum([np.sqrt((src.flux.I_err/src.flux.I)**2 +
                   #                          (src.pos.ra_err/src.pos.dec)**2)*abs(src.flux.I*src.pos.dec)
                   #                  for src in model2_sources]))
                   # _a = np.sum([src.flux.I*src.pos.dec for src in model2_sources])
                   # _err_b = np.sqrt(np.sum([src.flux.I_err**2 for src in model2_sources]))
                   # _b = np.sum([src.flux.I for src in model2_sources])
                   # dec_err = np.abs(dec) * (np.sqrt((_err_a / _a)**2 + (_err_b / _b)**2))
                    ra_err2 = sorted(model2_sources, key=lambda x: x.flux.I, reverse=True)[0].pos.ra_err
                    dec_err2 = sorted(model2_sources, key=lambda x: x.flux.I, reverse=True)[0].pos.dec_err
                except ZeroDivisionError:
                    if len(model2_sources) > 1:
                        LOGGER.warn('Position ({}, {}): Since more than one source is detected'
                                    ' at the matched position,'
                                    'only the closest to the matched position will be considered.'
                                    'NB: This is because model2 does not have photometric errors.'
                                    'otherwise a weighted average source would be returned'.format(
                                           rad2deg(ra1), rad2deg(dec1)))
                        rdist = np.array([_source_angular_dist_pos_angle(model2_source, model1_source)[0]
                                         for model2_source in model2_sources])
                        model2_sources = [model2_sources[np.argmin(rdist)]]
                    model2_source = model2_sources[0]
                    I_out = model2_source.flux.I
                    I_out_err = model2_source.flux.I_err
                    ra2 = model2_source.pos.ra
                    dec2 = model2_source.pos.dec
                    ra_err2 = model2_source.pos.ra_err
                    dec_err2 = model2_source.pos.dec_err

            RA0, DEC0 = model_lsm1.ra0, model_lsm1.dec0
            source2_name = model2_source.name

            if ra2 > np.pi:
                ra2 -= 2.0*np.pi
            if ra1 > np.pi:
                ra1 -= 2.0*np.pi
            delta_pos_angle_arc_sec = angular_dist_pos_angle(
                rad2arcsec(ra1), rad2arcsec(dec1),
                rad2arcsec(ra2), rad2arcsec(dec2))[0]
            delta_pos_angle_arc_sec = float('{0:.7f}'.format(delta_pos_angle_arc_sec))
            if RA0 or DEC0:
                delta_phase_centre = angular_dist_pos_angle(RA0, DEC0, ra2, dec2)
                delta_phase_centre_arc_sec = rad2arcsec(delta_phase_centre[0])
            else:
                delta_phase_centre_arc_sec = None

            src_scale = get_src_scale(model2_source.shape)

            if not off_axis:
                off_axis = 360.0
            if delta_phase_centre_arc_sec <= deg2arcsec(off_axis):
                targets_flux[source2_name] = [I_out, I_out_err,
                                              I_in, I_in_err,
                                              (source1_name, source2_name)]

                targets_position[source1_name] = [delta_pos_angle_arc_sec,
                                                  rad2arcsec(ra2 - ra1),
                                                  rad2arcsec(dec2 - dec1),
                                                  delta_phase_centre_arc_sec, I_in,
                                                  rad2arcsec(ra_err2),
                                                  rad2arcsec(dec_err2),
                                                  (round(rad2deg(ra1), deci),
                                                  round(rad2deg(dec1), deci)),
                                                  (source1_name, source2_name)]
                targets_scale[source1_name] = [shape_out, shape_out_err, shape_in,
                                               src_scale[0], src_scale[1], I_in,
                                               source2_name]
                names[source1_name] = source2_name
            else:
                LOGGER.info(f"Source {source1_name} will be unmatched "
                             "due to greater distance from phase centre")



    sources1 = model_lsm1.sources
    sources2 = model_lsm2.sources
    targets_not_matching_a, targets_not_matching_b = targets_not_matching(sources1,
                                                                          sources2,
                                                                          names)
    sources_overlay = get_source_overlay(sources1, sources2)
    num_of_sources = len(targets_flux)
    LOGGER.info(f"Number of sources matched: {num_of_sources}")
    return (targets_flux, targets_scale, targets_position,
            targets_not_matching_a, targets_not_matching_b,
            sources_overlay)


def compare_models(models, tolerance=0.2, plot=True, all_sources=False, shape_limit=6.0,
                   off_axis=None, closest_only=False, prefix=None, flux_plot='log',
                   fxlabels=None, fylabels=None, ftitles=None, svg=False,
                   title_size='16pt', x_label_size='12pt', y_label_size='12pt',
                   legend_size='10pt', xmajor_size='8pt', ymajor_size='8pt',
                   bar_size='12pt', bar_major_size='8pt'):
    """Plot model1 source properties against that of model2

    Parameters
    ----------
    models : dict
        Tigger formatted model files e.g {model1: model2}.
    tolerance : float
        Tolerace in detecting source from model 2 (in arcsec).
    plot : bool
        Output html plot from which a png can be obtained.
    all_source: bool
        Compare all sources in the catalog (else only point-like source)
    shape_limit: float
        Cross match only sources with maj-axis less than this value
    closest_only: bool
        Returns the closest source only as the matching source
    flux_plot: str
        The type of output flux comparison plot (options:log,snr,inout)
    prefix : str
        Prefix for output htmls
    fxlabels : str[]
        X-axis labels for the flux comparison plots
    fylabels : str[]
        Y-axis labels for the flux comparison plots
    fylabels : str[]
        Title labels for the flux comparison plots

    Returns
    -------
    results : dict
        Dictionary of source properties from each model.

    """
    results = dict()
    for _models in models:
        input_model = _models[0]
        output_model = _models[1]
        heading = input_model["label"]
        results[heading] = {'models': [input_model["path"], output_model["path"]]}
        results[heading]['flux'] = []
        results[heading]['shape'] = []
        results[heading]['position'] = []
        # No matching source
        results[heading]['no_match1'] = []
        results[heading]['no_match2'] = []
        results[heading]['overlay'] = []
        props = get_detected_sources_properties('{}'.format(input_model["path"]),
                                                '{}'.format(output_model["path"]),
                                                all_sources=all_sources,
                                                tolerance=tolerance,
                                                closest_only=closest_only,
                                                off_axis=off_axis)
        for i in range(len(props[0])):
            flux_prop = list(props[0].items())
            results[heading]['flux'].append(flux_prop[i][-1])
        for i in range(len(props[1])):
            shape_prop = list(props[1].items())
            results[heading]['shape'].append(shape_prop[i][-1])
        for i in range(len(props[2])):
            pos_prop = list(props[2].items())
            results[heading]['position'].append(pos_prop[i][-1])
        for i in range(len(props[3])):
            no_match_prop1 = list(props[3].items())
            results[heading]['no_match1'].append(no_match_prop1[i][-1])
        for i in range(len(props[4])):
            no_match_prop2 = list(props[4].items())
            results[heading]['no_match2'].append(no_match_prop2[i][-1])
        for i in range(len(props[5])):
            no_match_prop2 = list(props[5].items())
            results[heading]['overlay'].append(no_match_prop2[i][-1])
        results[heading]['tolerance'] = tolerance
    if plot:
        _source_flux_plotter(results, models, prefix=prefix, plot_type=flux_plot,
                             titles=ftitles, xlabels=fxlabels, ylabels=fylabels,
                             svg=svg, title_size=title_size, x_label_size=x_label_size,
                             y_label_size=y_label_size, legend_size=legend_size,
                             xmajor_size=xmajor_size, ymajor_size=ymajor_size,
                             bar_size=bar_size, bar_major_size=bar_major_size)
        _source_astrometry_plotter(results, models, prefix=prefix, svg=svg,
                                   title_size=title_size, x_label_size=x_label_size,
                                   y_label_size=y_label_size, legend_size=legend_size,
                                   xmajor_size=xmajor_size, ymajor_size=ymajor_size,
                                   bar_size=bar_size, bar_major_size=bar_major_size)
    return results


def compare_residuals(residuals, skymodel=None, points=None,
                      inline=False, area_factor=None,
                      prefix=None, fov_factor=None,
                      units='micro',
                      title_size='14pt',
                      xmajor_size='6pt',
                      ymajor_size='6pt',
                      legend_size='10pt',
                      x_label_size='12pt',
                      y_label_size='12pt'):
    if skymodel:
        res = _source_residual_results(residuals, skymodel, area_factor)
    else:
        res = _random_residual_results(residuals, points,
                                       fov_factor, area_factor)
    _residual_plotter(residuals, results=res, points=points,
                      inline=inline, prefix=prefix, units=units,
                      title_size=title_size,
                      legend_size=legend_size,
                      xmajor_size=xmajor_size,
                      ymajor_size=ymajor_size,
                      x_label_size=x_label_size,
                      y_label_size=y_label_size)
    return res

def targets_not_matching(sources1, sources2, matched_names, flux_units='milli'):
    """Plot model-model fluxes from lsm.html/txt models

    Parameters
    ----------
    sources1: list
        List of sources from model 1
    sources2: list
        List of sources Sources from model 2
    matched_names: dict
        Dict of names from model 2 that matched that of model 1
    flux_units: str
        Units of flux density for tabulated values

    Returns
    -------
    target_no_match1: dict
        Sources from model 1 that have no match in model 2
    target_no_match2: dict
        Sources from model 2 that have no match in model 1

    """
    deci = DECIMALS  # round off to this decimal places
    units = flux_units
    targets_not_matching_a = dict()
    targets_not_matching_b = dict()
    for s1 in sources1:
        if s1.name not in matched_names.keys():
            props1 = [s1.name,
                      round(s1.flux.I*FLUX_UNIT_SCALER[units][0], deci),
                      round(s1.flux.I_err*FLUX_UNIT_SCALER[units][0], deci)
                            if s1.flux.I_err else None,
                      unwrap(round(rad2deg(s1.pos.ra), deci)),
                      f'{rad2deg(s1.pos.ra_err):.{deci}e}'
                          if s1.pos.ra_err else None,
                      round(rad2deg(s1.pos.dec), deci),
                      f'{rad2deg(s1.pos.dec_err):.{deci}e}'
                          if s1.pos.dec_err else None]
            targets_not_matching_a[s1.name] = props1
    for s2 in sources2:
        if s2.name not in matched_names.values():
            props2 = [s2.name,
                      round(s2.flux.I*FLUX_UNIT_SCALER[units][0], deci),
                      round(s2.flux.I_err*FLUX_UNIT_SCALER[units][0], deci)
                            if s2.flux.I_err else None,
                      unwrap(round(rad2deg(s2.pos.ra), deci)),
                      f'{rad2deg(s2.pos.ra_err):.{deci}e}'
                          if s2.pos.ra_err else None,
                      round(rad2deg(s2.pos.dec), deci),
                      f'{rad2deg(s2.pos.dec_err):.{deci}e}'
                          if s2.pos.dec_err else None]
            targets_not_matching_b[s2.name] = props2
    return targets_not_matching_a, targets_not_matching_b


def get_source_overlay(sources1, sources2):
    """Get source from models compare for overlay"""
    sources = dict()
    for s1 in sources1:
        props1 = [s1.name,
                  s1.flux.I, s1.flux.I_err,
                  s1.pos.ra, s1.pos.ra_err,
                  s1.pos.dec, s1.pos.dec_err,
                  1]
        sources[s1.name+'-1'] = props1
    LOGGER.info("Model 1 source: {}".format(len(sources1)))
    for s2 in sources2:
        props2 = [s2.name,
                  s2.flux.I, s2.flux.I_err,
                  s2.pos.ra, s2.pos.ra_err,
                  s2.pos.dec, s2.pos.dec_err,
                  2]
        sources[s2.name+'-2'] = props2
    LOGGER.info("Model 2 source: {}".format(len(sources2)))
    return sources


def plot_photometry(models, label=None, tolerance=0.2, phase_centre=None,
                    all_sources=False, flux_plot='log', off_axis=None,
                    shape_limit=6.0):
    """Plot model-model fluxes from lsm.html/txt models

    Parameters
    ----------
    models : dict
        Tigger/text formatted model files e.g {model1: model2}.
    label : str
        Use this label instead of the FITS image path when saving data.
    tolerance: float
        Radius around the source to be cross matched (in arcsec).
    phase_centre : str
        Phase centre of catalog (if not already embeded)
    all_source: bool
        Compare all sources in the catalog (else only point-like source)

    """
    _models = []
    i = 0
    for model1, model2 in models.items():
        _models.append([dict(label="{}-model_a_{}".format(label, i), path=model1),
                        dict(label="{}-model_b_{}".format(label, i), path=model2)])
        i += 1
    results = compare_models(_models, tolerance, False, phase_centre, all_sources, off_axis)
    _source_flux_plotter(results, _models, inline=True, plot_type=flux_plot)


def plot_astrometry(models, label=None, tolerance=0.2, phase_centre=None,
                    all_sources=False, off_axis=None):
    """Plot model-model positions from lsm.html/txt models

    Parameters
    ----------
    models : dict
        Tigger/text formatted model files e.g {model1: model2}.
    label : str
        Use this label instead of the FITS image path when saving data.
    tolerance: float
        Radius around the source to be cross matched.
    phase_centre : str
        Phase centre of catalog (if not already embeded)
    all_source: bool
        Compare all sources in the catalog (else only point-like source)

    """
    _models = []
    i = 0
    for model1, model2 in models.items():
        _models.append([dict(label="{}-model_a_{}".format(label, i), path=model1),
                        dict(label="{}-model_b_{}".format(label, i), path=model2)])
        i += 1
    results = compare_models(_models, tolerance, False, phase_centre, all_sources, off_axis)
    _source_astrometry_plotter(results, _models, inline=True)


def plot_residuals_noise(res_noise_images, skymodel=None, label=None,
                         area_factor=2.0, points=100):
    """Plot residual-residual or noise data

    Parameters
    ----------
    res_noise_images: dict
        Dictionary of residual images to plot {res1.fits: res2.fits}.
    skymodel: file
        Skymodel file to locate on source residuals (lsm.html/txt)
    label : str
        Use this label instead of the FITS image path when saving data.
    area_factor : float
        Factor to multiply the beam area.
    points: int
        Number of data point to generate in case of random residuals.

    """
    _residual_images = []
    i = 0
    for res1, res2 in res_noise_images.items():
        _residual_images.append([dict(label="{}-res_a_{}".format(label, i), path=res1),
                                 dict(label="{}-res_b_{}".format(label, i), path=res2)])
        i += 1
    compare_residuals(_residual_images, skymodel, points, True, area_factor)


def _source_flux_plotter(results, all_models, inline=False, units='milli',
                         prefix=None, plot_type='log', titles=None, svg=False,
                         xlabels=None, ylabels=None, title_size='16pt',
                         x_label_size='12pt', y_label_size='12pt',
                         legend_size='10pt', xmajor_size='8pt',
                         ymajor_size='8pt', bar_size='12pt',
                         bar_major_size='8pt'):
    """Plot flux results and save output as html file.

    Parameters
    ----------
    results : dict
        Structured output results.
    models : list
        Tigger/text formatted model files.
        e.g. [[{'label': 'model_a_1', 'path': 'point_skymodel1.txt'},
               {'label': 'model_b_1', 'path': 'point_skymodel1.lsm.html'}]]
    inline : bool
        Allow inline plotting inside a notebook.
    units : str
        Data points and axis label units
    plot_type: str
        The type of output flux comparison plot (options:log,snr,inout)
    prefix : str
        Prefix for output htmls
    fxlabels : str[]
        X-axis labels for the flux comparison plots
    fylabels : str[]
        Y-axis labels for the flux comparison plots
    fylabels : str[]
        Title labels for the flux comparison plots
    title_size : str
        Title label size for the flux comparison plots
    x_label_size : str
        X-axis  label size for the flux comparison plots
    y_label_size : str
        Y-axis label size for the flux comparison plots
    legend_size : str
        Legend label size for the flux comparison plots
    xmajor_size : str
        X-axis major label size for the flux comparison plots
    ymajor_size : str
        Y-axis major label size for the flux comparison plots
    bar_size : str
        Colorbar text font size
    bar_major_size : str
        Colorbar major axis text font size
    svg : bool
        Whether to save svg plots in addition to the standard html
    """
    if prefix:
        outfile = f'{prefix}-FluxOffset.html'
    else:
        outfile = 'FluxOffset.html'
    output_file(outfile)
    flux_plot_list = []
    for pair, model_pair in enumerate(all_models):
        heading = model_pair[0]['label']
        name_labels = []
        flux_in_data = []
        flux_out_data = []
        source_scale = []
        positions_in_out = []
        flux_in_err_data = []
        flux_out_err_data = []
        phase_centre_dist = []
        no_match1 = results[heading]['no_match1']
        no_match2 = results[heading]['no_match2']
        for n in range(len(results[heading]['flux'])):
            flux_out_data.append(results[heading]['flux'][n][0])
            flux_out_err_data.append(results[heading]['flux'][n][1])
            flux_in_data.append(results[heading]['flux'][n][2])
            flux_in_err_data.append(results[heading]['flux'][n][3])
            name_labels.append(results[heading]['flux'][n][4])
            phase_centre_dist.append(results[heading]['position'][n][3])
            positions_in_out.append(results[heading]['position'][n][7])
            source_scale.append(results[heading]['shape'][n][3])
        if len(flux_in_data) > 1:
            # Error lists
            err_xs1 = []
            err_ys1 = []
            err_xs2 = []
            err_ys2 = []
            model_1_name = model_pair[0]['path'].split('/')[-1].split('.')[0]
            model_2_name = model_pair[1]['path'].split('/')[-1].split('.')[0]
            # Format data points value to a readable units
            # and select type of comparison plot
            x = np.array(flux_in_data) * FLUX_UNIT_SCALER[units][0]
            y = np.array(flux_out_data) * FLUX_UNIT_SCALER[units][0]
            xerr = np.array(flux_in_err_data) * FLUX_UNIT_SCALER[units][0]
            yerr = np.array(flux_out_err_data) * FLUX_UNIT_SCALER[units][0]
            if plot_type == 'inout':
                x1 = x
                y1 = y
                xerr1 = xerr
                yerr1 = yerr
                axis_labels = [f"{model_1_name} S1 ({FLUX_UNIT_SCALER[units][1]})"
                               if not xlabels else xlabels[pair],
                               f"{model_2_name} S2 ({FLUX_UNIT_SCALER[units][1]})"
                               if not ylabels else ylabels[pair]]
            elif plot_type == 'log':
                x1 = np.log(x)
                y1 = np.log(y)
                xerr1 = np.log(xerr)
                yerr1 = np.log(yerr)
                axis_labels = [f"log S1: {model_1_name}" if not xlabels else xlabels[pair],
                               f"log S2: {model_2_name}" if not ylabels else ylabels[pair]]
            elif plot_type == 'snr':
                x1 = np.log(x)
                y1 = (x/y)
                xerr1 = xerr
                yerr1 = yerr
                axis_labels = ['log S1' if not xlabels else xlabels[pair],
                               'S1/S2' if not ylabels else ylabels[pair]]
            # RA and Dec with a cross-match in deg:arcmin:arcsec
            position_ra_dec = [(deg2ra(ra), deg2dec(dec)) for (ra, dec) in positions_in_out]
            # Phase centre distance in degree
            z = np.array(phase_centre_dist)/3600.
            # Compute some fit stats of the two models being compared
            if plot_type in ['log', 'inout']:
                flux_MSE = mean_squared_error(x1, y1)
                reg1 = linregress(x1, y1)
                flux_R_score = reg1.rvalue
            elif plot_type in ['snr']:
                reg1 = linregress(x1, y1)
                mean_val = np.mean(y1)
                median = np.median(y1)
                std_val = np.std(y1)
                mad_val = scipy.stats.median_abs_deviation(y1)
                max_val = y1.max()
                min_val = y1.min()
            # Table with stats data
            deci = DECIMALS  # Round off to this decimal places
            cols = ["Stats", "Value"]
            if plot_type in ['log', 'inout']:
                stats = {"Stats": ["Slope",
                                   f"Intercept ({FLUX_UNIT_SCALER[units][1]})",
                                   f"RMS_Error ({FLUX_UNIT_SCALER[units][1]})",
                                   "R2"],
                         "Value": [f"{reg1.slope:.{deci}f}",
                                   f"{reg1.intercept:.{deci}f}",
                                   f"{np.sqrt(flux_MSE):.{deci}e}",
                                   f"{flux_R_score:.{deci}f}"]}
            elif plot_type in ['snr']:
                stats = {"Stats": ["MAX", "MIN", "MEAN", "MAD", "MEDIAN", "STD"],
                        "Value": [f"{max_val:.{deci}f}",
                                  f"{min_val:.{deci}f}",
                                  f"{mean_val:.{deci}f}",
                                  f"{mad_val:.{deci}f}",
                                  f"{median:.{deci}f}",
                                  f"{std_val:.{deci}f}"]}
            # Create additional feature on the plot such as hover, display text
            TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,hover,save"
            source = ColumnDataSource(
                        data=dict(flux_1=x, flux_2=y,
                                  plot_flux_1=x1, plot_flux_2=y1,
                                  flux_1_err=xerr1, flux_2_err=yerr1,
                                  phase_centre_dist=z,
                                  ra_dec=position_ra_dec,
                                  label=name_labels))
            text = "Flux Offset" if not titles else titles[pair]
            # Create a plot object
            plot_flux = figure(title=text,
                               x_axis_label=axis_labels[0],
                               y_axis_label=axis_labels[1],
                               tools=TOOLS)
            # Plot title font sizes
            plot_flux.title.text_font_size = title_size
            plot_flux.xaxis.axis_label_text_font_size = x_label_size
            plot_flux.yaxis.axis_label_text_font_size = y_label_size
            plot_flux.xaxis.major_label_text_font_size = xmajor_size
            plot_flux.yaxis.major_label_text_font_size = ymajor_size
            # Create a color bar and size objects
            color_bar_height = 100
            mapper_opts = dict(palette="Plasma11",
                               low=min(z),
                               high=max(z))
            flux_mapper = LinearColorMapper(**mapper_opts)
            color_bar = ColorBar(color_mapper=flux_mapper,
                                 ticker=plot_flux.xaxis.ticker,
                                 formatter=plot_flux.xaxis.formatter,
                                 title='Distance off-axis (deg)',
                                 title_text_font_size=bar_size,
                                 title_text_align='center',
                                 major_label_text_font_size=bar_major_size,
                                 orientation='horizontal')
            #color_bar_plot = figure(title="Distance off-axis (deg)",
                                    #title_location="below",
                                    #height=color_bar_height,
                                    #toolbar_location=None,
                                    #outline_line_color='red',
                                    #min_border=0)
            #color_bar_plot.title.text_font_size = '20pt'
            # Get errors from the input/output fluxes
            for xval, yval, xerr, yerr in zip(x1, y1,
                                  np.array(flux_in_err_data) * FLUX_UNIT_SCALER[units][0],
                                  np.array(flux_out_err_data) * FLUX_UNIT_SCALER[units][0]):
                err_xs1.append((xval - xerr, xval + xerr))
                err_ys2.append((yval - yerr, yval + yerr))
                err_ys1.append((yval, yval))
                err_xs2.append((xval, xval))
            # Create S2plot object for errors
            error1_plot = plot_flux.multi_line(err_xs1, err_ys1,
                                                   legend_label="Errors",
                                                   color="red")
            error2_plot = plot_flux.multi_line(err_xs2, err_ys2,
                                                   legend_label="Errors",
                                                   color="red")
            # Create a plot object for a Fit
            if plot_type == 'inout':
                fit_points = 100
                slope = reg1.slope
                intercept = reg1.intercept
                fit_xs = np.linspace(0 if 0 < min(x1) else min(x1), max(x1), fit_points)
                fit_ys = slope * fit_xs + intercept
                # Regression fit plot
                fit = plot_flux.line(fit_xs, fit_ys,
                                     legend_label="Fit",
                                     color="blue")
                # Create a plot object for I_out = I_in line .i.e. Perfect match
                min_val = min(x1) if min(x1) < min(y1) else min(y1)
                max_val = max(y1) if max(y1) > max(x1) else max(x1)
                equal = plot_flux.line(np.array([0 if 0 < min_val else min_val, max_val]),
                                       np.array([0 if 0 < min_val else min_val, max_val]),
                                       legend_label=u"S1=S2",
                                       line_dash="dashed",
                                       color="gray")
            elif plot_type == 'snr':
                fit_points = 100
                # Regression fit plot
                min_val = min(x1) if max(x1) < 0 else 0
                max_val = max(x1)
                # Create a plot object for I_out = I_in line .i.e. Perfect match
                equal = plot_flux.line(np.array([min_val, max_val]),
                                       np.array([1, 1]),
                                       legend_label=u"S1/S2=1",
                                       line_dash="dashed",
                                       color="gray")
            elif plot_type == 'log':
                fit_points = 100
                slope = reg1.slope
                intercept = reg1.intercept
                min_val = min(x1) if min(x1) < min(y1) else min(y1)
                max_val = max(y1) if max(y1) > max(x1) else max(x1)
                # Regression fit plot
                fit_xs = np.linspace(min_val, max_val, fit_points)
                fit_ys = slope * fit_xs + intercept
                fit = plot_flux.line(fit_xs, fit_ys,
                                     legend_label="Fit",
                                     color="blue")
                # Create a plot object for I_out = I_in line .i.e. Perfect match
                equal = plot_flux.line(np.array([0 if 0 < min_val else min_val, max_val]),
                                       np.array([0 if 0 < min_val else min_val, max_val]),
                                       legend_label=u"log(S1)=log(S2)",
                                       line_dash="dashed",
                                       color="gray")
            # Create a plot object for the data points
            data = plot_flux.circle('plot_flux_1', 'plot_flux_2',
                                    name='data',
                                    legend_label="Data",
                                    source=source,
                                    line_color=None,
                                    fill_color={"field": "phase_centre_dist",
                                               "transform": flux_mapper})
            source = ColumnDataSource(data=stats)
            columns = [TableColumn(field=x, title=x.capitalize()) for x in cols]
            dtab = DataTable(source=source, columns=columns,
                             width=500, max_width=550,
                             height=100, max_height=150,
                             sizing_mode='stretch_both')
            table_title = Div(text="Cross Matching Statistics")
            table_title.align = "center"
            stats_table = column([table_title, dtab])
            # Table with no match data1
            _fu = FLUX_UNIT_SCALER[units][1]
            cols1 = ["Source", "Flux [%s]"%_fu, "Flux_err [%s]"%_fu, "RA",
                     "RA_err ['']", "DEC", "DEC_err ['']"]
            stats1 = {"Source": [s[0] for s in no_match1],
                      "Flux [%s]"%_fu: [s[1] for s in no_match1],
                      "Flux_err [%s]"%_fu: [s[2] for s in no_match1],
                      "RA": [deg2ra(s[3], deci) for s in no_match1],
                      "RA_err ['']": [round(deg2arcsec(s[4] if s[4] else 0),
                                      deci) for s in no_match1],
                      "DEC": [deg2dec(s[5], deci) for s in no_match1],
                      "DEC_err ['']": [round(deg2arcsec(s[6] if s[6] else 0),
                                       deci) for s in no_match1]}
            source1 = ColumnDataSource(data=stats1)
            columns1 = [TableColumn(field=x, title=x.capitalize()) for x in cols1]
            dtab1 = DataTable(source=source1, columns=columns1,
                              width=500, max_width=550,
                              height=150, max_height=200,
                              sizing_mode='stretch_both')
            table_title1 = Div(text=f"Non-matching sources from {model_1_name}")
            table_title1.align = "center"
            stats_table1 = column([table_title1, dtab1])
            # Table with no match data1
            cols2 = ["Source", "Flux [%s]"%_fu, "Flux_err [%s]"%_fu, "RA",
                     "RA_err ['']", "DEC", "DEC_err ['']"]
            stats2 = {"Source": [s[0] for s in no_match2],
                      "Flux [%s]"%_fu: [s[1] for s in no_match2],
                      "Flux_err [%s]"%_fu: [s[2] for s in no_match2],
                      "RA": [deg2ra(s[3], deci) for s in no_match2],
                      "RA_err ['']": [round(deg2arcsec(s[4] if s[4] else 0),
                                      deci) for s in no_match2],
                      "DEC": [deg2dec(s[5], deci) for s in no_match2],
                      "DEC_err ['']": [round(deg2arcsec(s[6]),
                                       deci) for s in no_match2]}
            source2 = ColumnDataSource(data=stats2)
            columns2 = [TableColumn(field=x, title=x.capitalize()) for x in cols2]
            dtab2 = DataTable(source=source2, columns=columns2,
                              width=500, max_width=550,
                              height=150, max_height=200,
                              sizing_mode='stretch_both')
            table_title2 = Div(text=f"Non-matching sources from {model_2_name}")
            table_title2.align = "center"
            stats_table2 = column([table_title2, dtab2])
            # Attaching the hover object with labels
            hover = plot_flux.select(dict(type=HoverTool))
            hover.names = ['data']
            hover.tooltips = OrderedDict([
                ("source", "(@label)"),
                ("(S1,S2)", "(@flux_1, @flux_2)"),
                ("(S_err1, S_err2)"," (@flux_1_err, @flux_2_err)"),
                ("(RA,DEC)", "@ra_dec"),
                ("Distance off-axis", "@phase_centre_dist")])
            # Legend position, size and title align
            plot_flux.legend.location = "top_left"
            plot_flux.legend.label_text_font_size = legend_size
            plot_flux.title.align = "center"
            plot_flux.legend.click_policy = "hide"
            # Colorbar position
            plot_flux.add_layout(color_bar, "below")
            #color_bar_plot.add_layout(color_bar, "below")
            #color_bar_plot.title.align = "center"
            # Append all plots
            flux_plot_list.append(column(row(plot_flux,
                                             column(stats_table,
                                                    stats_table1,
                                                    stats_table2))))
        else:
            LOGGER.warn('No photometric plot created for {}'.format(model_pair[1]["path"]))
    if flux_plot_list:
        # Make the plots in a column layout
        flux_plots = column(flux_plot_list)
        if svg:
            plot_flux.output_backend='svg'
            prefix = '.'.join(outfile.split('.')[:-1])
            export_svgs(flux_plots, filename=f"{prefix}.svg")
        # Save the plot (html)
        save(flux_plots, title=outfile)
        LOGGER.info('Saving photometry comparisons in {}'.format(outfile))


def _source_astrometry_plotter(results, all_models, inline=False, units='',
                               prefix=None, svg=False, title_size='16pt',
                               x_label_size='12pt', y_label_size='12pt',
                               legend_size='10pt', xmajor_size='6pt',
                               ymajor_size='6pt', bar_size='8pt',
                               bar_major_size='8pt'):
    """Plot astrometry results and save output as html file.

    Parameters
    ----------
    results: dict
        Structured output results.
    models : list
        Tigger/text formatted model files.
        e.g. [[{'label': 'model_a_1', 'path': 'point_skymodel1.txt'},
               {'label': 'model_b_1', 'path': 'point_skymodel1.lsm.html'}]]
    inline : bool
        Allow inline plotting inside a notebook.
    units : str
        Data points and axis label units
    prefix : str
        Prefix for output htmls
    svg : bool
        Whether to save svg plots in addition to the standard html
    title_size : str
        Title label size for the flux comparison plots
    x_label_size : str
        X-axis  label size for the flux comparison plots
    y_label_size : str
        Y-axis label size for the flux comparison plots
    legend_size : str
        Legend label size for the flux comparison plots
    xmajor_size : str
        X-axis major label size for the flux comparison plots
    ymajor_size : str
        Y-axis major label size for the flux comparison plots
    bar_size : str
        Colorbar text font size
    bar_major_size : str
        Colorbar major axis text font size

    """
    if prefix:
        outfile = f'{prefix}-PositionOffset.html'
    else:
        outfile = 'PositionOffset.html'
    output_file(outfile)
    position_plot_list = []
    for model_pair in all_models:
        RA_offset = []
        RA_err = []
        DEC_offset = []
        DEC_err = []
        source_labels = []
        flux_in_data = []
        flux_out_data = []
        delta_pos_data = []
        positions_in_out = []
        phase_centre_dist = []
        heading = model_pair[0]['label']
        overlays = results[heading]['overlay']
        tolerance = results[heading]['tolerance']
        for n in range(len(results[heading]['flux'])):
            flux_out_data.append(results[heading]['flux'][n][0])
            delta_pos_data.append(results[heading]['position'][n][0])
            RA_offset.append(results[heading]['position'][n][1])
            DEC_offset.append(results[heading]['position'][n][2])
            phase_centre_dist.append(results[heading]['position'][n][3])
            flux_in_data.append(results[heading]['position'][n][4])
            RA_err.append(results[heading]['position'][n][5])
            DEC_err.append(results[heading]['position'][n][6])
            positions_in_out.append(results[heading]['position'][n][7])
            source_labels.append(results[heading]['position'][n][8])
        # Compute some stats of the two models being compared
        if len(flux_in_data) > 1:
            model_1_name = model_pair[0]['path'].split('/')[-1].split('.')[0]
            model_2_name = model_pair[1]['path'].split('/')[-1].split('.')[0]
            RA_mean = np.mean(RA_offset)
            DEC_mean = np.mean(DEC_offset)
            r1, r2 = np.array(RA_offset).std(), np.array(DEC_offset).std()
            # Generate data for a sigma circle around data points
            fit_points = 100
            pi, cos, sin = np.pi, np.cos, np.sin
            theta = np.linspace(0, 2.0 * pi, fit_points)
            x1 = RA_mean+(r1 * cos(theta))
            y1 = DEC_mean+(r2 * sin(theta))
            # Get the number of sources recovered and within 1 sigma
            recovered_sources = len(DEC_offset)
            one_sigma_sources = len([
                (ra_off, dec_off) for ra_off, dec_off in zip(RA_offset, DEC_offset)
                if abs(ra_off) <= max(abs(x1)) and abs(dec_off) <= max(abs(y1))])
            # Format data list into numpy arrays
            x_ra = np.array(RA_offset)
            y_dec = np.array(DEC_offset)
            x_ra_err = np.array(RA_err)
            y_dec_err = np.array(DEC_err)
            # TODO: Use flux as a radius dimension
            flux_in_mjy = np.array(flux_in_data) * FLUX_UNIT_SCALER['milli'][0]
            flux_out_mjy = np.array(flux_out_data) * FLUX_UNIT_SCALER['milli'][0]
            z = np.array(phase_centre_dist)/3600. # For color
            # RA and Dec with a cross-match in deg:arcmin:arcsec
            position_ra_dec = [(deg2ra(ra), deg2dec(dec)) for (ra, dec) in positions_in_out]
            # Create additional feature on the plot such as hover, display text
            TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,hover,save"
            source = ColumnDataSource(
                        data=dict(ra_offset=x_ra,
                                  ra_err=x_ra_err,
                                  dec_offset=y_dec,
                                  dec_err=y_dec_err,
                                  ra_dec=position_ra_dec,
                                  phase_centre_dist=z,
                                  flux_s1=flux_in_mjy,
                                  flux_s2=flux_out_mjy,
                                  label=source_labels))
            # Create a plot object
            plot_position = figure(title="Position Offset",
                                   x_axis_label='RA offset ({:s})'.format(
                                       POSITION_UNIT_SCALER['arcsec'][1]),
                                   y_axis_label='DEC offset ({:s})'.format(
                                       POSITION_UNIT_SCALER['arcsec'][1]),
                                   tools=TOOLS)
            plot_position.title.text_font_size = title_size
            plot_position.xaxis.axis_label_text_font_size = x_label_size
            plot_position.yaxis.axis_label_text_font_size = y_label_size
            # Create an image overlay
            s1_ra_rad = [src[3] for src in overlays if src[-1] == 1]
            s1_ra_deg = [unwrap(rad2deg(s_ra)) for s_ra in s1_ra_rad]
            s1_dec_rad = [src[5] for src in overlays if src[-1] == 1]
            s1_dec_deg = [rad2deg(s_dec) for s_dec in s1_dec_rad]
            s1_ra_err = [rad2deg(src[4]*3600.) for src in overlays if src[-1] == 1]
            s1_dec_err = [rad2deg(src[6]*3600.) for src in overlays if src[-1] == 1]
            s1_labels = [src[0] for src in overlays if src[-1] == 1]
            s1_flux = [src[1] for src in overlays if src[-1] == 1]
            s2_ra_rad = [src[3] for src in overlays if src[-1] == 2]
            s2_ra_deg = [unwrap(rad2deg(s_ra)) for s_ra in s2_ra_rad]
            s2_dec_rad = [src[5] for src in overlays if src[-1] == 2]
            s2_dec_deg = [rad2deg(s_dec) for s_dec in s2_dec_rad]
            s2_ra_err = [rad2deg(src[4]*3600.) for src in overlays if src[-1] == 2]
            s2_dec_err = [rad2deg(src[6]*3600.) for src in overlays if src[-1] == 2]
            s2_labels = [src[0] for src in overlays if src[-1] == 2]
            s2_flux = [src[1] for src in overlays if src[-1] == 2]
            overlay_source1 = ColumnDataSource(
                        data=dict(ra1=s1_ra_deg, dec1=s1_dec_deg,
                                  str_ra1=[deg2ra(_s1_radeg) for _s1_radeg in s1_ra_deg],
                                  str_dec1=[deg2dec(_s1_decdeg) for _s1_decdeg in s1_dec_deg],
                                  ra_err1=s1_ra_err, dec_err1=s1_dec_err,
                                  label1=s1_labels, flux1=s1_flux))
            overlay_source2 = ColumnDataSource(
                        data=dict(ra2=s2_ra_deg, dec2=s2_dec_deg,
                                  str_ra2=[deg2ra(_s2_radeg) for _s2_radeg in s2_ra_deg],
                                  str_dec2=[deg2dec(_s2_decdeg) for _s2_decdeg in s2_dec_deg],
                                  ra_err2=s2_ra_err, dec_err2=s2_dec_err,
                                  label2=s2_labels, flux2=s2_flux))
            plot_overlay = figure(title="Catalogs Overlay",
                                  x_axis_label='RA ({:s})'.format(
                                      POSITION_UNIT_SCALER['deg'][1]),
                                  y_axis_label='DEC ({:s})'.format(
                                      POSITION_UNIT_SCALER['deg'][1]),
                                  match_aspect=True,
                                  tools=("crosshair,pan,wheel_zoom,"
                                         "box_zoom,reset,save"))
            plot_overlay.ellipse('ra1', 'dec1',
                                 source=overlay_source1,
                                 width=tolerance/3600.,
                                 height=tolerance/3600.,
                                 line_color=None,
                                 color='#CAB2D6')
            plot_overlay_1 = plot_overlay.circle('ra1', 'dec1',
                                                 name='model1',
                                                 legend_label=model_1_name,
                                                 source=overlay_source1,
                                                 #line_color=None,
                                                 color='blue')
            plot_overlay_2 = plot_overlay.circle('ra2', 'dec2',
                                                 name='model2',
                                                 legend_label=model_2_name,
                                                 source=overlay_source2,
                                                 #line_color=None,
                                                 color='red')
            plot_position.title.text_font_size = title_size
            plot_position.xaxis.axis_label_text_font_size = x_label_size
            plot_position.yaxis.axis_label_text_font_size = y_label_size
            plot_position.xaxis.major_label_text_font_size = xmajor_size
            plot_position.yaxis.major_label_text_font_size = ymajor_size
            plot_position.axis.axis_label_text_font_style = 'normal'
            plot_overlay.title.text_font_size = title_size
            plot_overlay.xaxis.axis_label_text_font_size = x_label_size
            plot_overlay.yaxis.axis_label_text_font_size = y_label_size
            plot_overlay.legend.label_text_font_size = legend_size
            plot_overlay.xaxis.major_label_text_font_size = xmajor_size
            plot_overlay.yaxis.major_label_text_font_size = ymajor_size
            plot_overlay.axis.axis_label_text_font_style = 'normal'
            plot_overlay.title.align = "center"
            plot_overlay.legend.location = "top_left"
            plot_overlay.legend.click_policy = "hide"
            color_bar_height = 100
            plot_overlay.x_range.flipped = True
            # Colorbar Mapper
            mapper_opts = dict(palette="Plasma11",
                               low=min(z),
                               high=max(z))
            position_mapper = LinearColorMapper(**mapper_opts)
            color_bar = ColorBar(color_mapper=position_mapper,
                                 ticker=plot_position.xaxis.ticker,
                                 formatter=plot_position.xaxis.formatter,
                                 location=(0,0),
                                 title='Distance off-axis (deg)',
                                 title_text_font_size=bar_size,
                                 title_text_align='center',
                                 major_label_text_font_size=bar_major_size,
                                 orientation='horizontal')

#            color_bar_plot = figure(title="Distance off-axis (deg)",
#                                    title_location="below",
#                                    height=color_bar_height,
#                                    toolbar_location=None,
#                                    outline_line_color=None,
#                                    min_border=0)
#            color_bar_plot.title.text_font_size = '10pt'
            # Get errors from the output positions
            err_xs1 = []
            err_ys1 = []
            err_xs2 = []
            err_ys2 = []
            for x, y, xerr, yerr in zip(x_ra, y_dec, np.array(RA_err), np.array(DEC_err)):
                err_xs1.append((x - xerr, x + xerr))
                err_ys1.append((y, y))
                err_xs2.append((x, x))
                err_ys2.append((y - yerr, y + yerr))
            # Create a plot object for errors
            error1_plot = plot_position.multi_line(err_xs1, err_ys1,
                                                   legend_label="Errors",
                                                   color="red")
            error2_plot = plot_position.multi_line(err_xs2, err_ys2,
                                                   legend_label="Errors",
                                                   color="red")
            # Creat an sigma circle plot object
            sigma_plot = plot_position.line(np.array(x1), np.array(y1),
                                            legend_label='Sigma')
            # Create position data points plot object
            plot_position.circle('ra_offset', 'dec_offset',
                                 name='data',
                                 source=source,
                                 line_color=None,
                                 legend_label='Data',
                                 fill_color={"field": "phase_centre_dist",
                                             "transform": position_mapper})
            # Table with stats data
            deci = DECIMALS  # round off to this decimal places
            cols = ["Stats", "Value"]
            stats = {"Stats": ["Total sources",
                               "(RA, DEC) mean offset ['']",
                               "Sigma sources",
                               "(RA, DEC) sigma offset ['']"],
                     "Value": [recovered_sources,
                               f"({round(deg2arcsec(RA_mean), deci)},"
                               f"{round(deg2arcsec(DEC_mean), deci)})",
                               one_sigma_sources,
                               f"({round(deg2arcsec(r1), deci)},"
                               f"{round(deg2arcsec(r2), deci)})"]}
            source = ColumnDataSource(data=stats)
            columns = [TableColumn(field=x, title=x.capitalize()) for x in cols]
            dtab = DataTable(source=source, columns=columns,
                             width=450, max_width=500,
                             height=100, max_height=150,
                             sizing_mode='stretch_both')
            table_title = Div(text="Cross Matching Statistics")
            table_title.align = "center"
            stats_table = column([table_title, dtab])
            # Attaching the hover object with labels
            hover = plot_position.select(dict(type=HoverTool))
            hover.names = ['data']
            hover.tooltips = OrderedDict([
                ("source", "(@label)"),
                ("(S1,S2) [mJy]",
                 "(@flux_s1, @flux_s2)"),
                ("(RA,DEC)", "(@ra_dec)"),
                ("(RA_err,DEC_err)",
                 "(@ra_err, @dec_err)"),
                ("(RA_offset,DEC_offset)",
                 "(@ra_offset, @dec_offset)"),
                ("Distance off-axis",
                 "@phase_centre_dist")])
            plot_overlay.add_tools(
                HoverTool(renderers=[plot_overlay_1],
                          tooltips=OrderedDict([
                              ("source1", "@label1"),
                              ("Flux (mJy)", "@flux1"),
                              ("(RA,DEC)", "(@str_ra1, @str_dec1)"),
                              ("(RA_err,DEC_err)",
                               "(@ra_err1, @dec_err1)")])))
            plot_overlay.add_tools(
                HoverTool(renderers=[plot_overlay_2],
                          tooltips=OrderedDict([
                              ("source2", "@label2"),
                              ("Flux (mJy)", "@flux2"),
                              ("(RA,DEC)", "(@str_ra2, @str_dec2)"),
                              ("(RA_err,DEC_err)",
                               "(@ra_err2, @dec_err2)")])))
            # Legend position and title align
            plot_position.legend.location = "top_left"
            plot_position.legend.click_policy = "hide"
            plot_position.title.align = "center"
            # Colorbar position
            plot_position.add_layout(color_bar, "below")
            plot_position.legend.label_text_font_size = legend_size
  #          color_bar_plot.add_layout(color_bar, "below")
  #          color_bar_plot.title.align = "center"
            if svg:
                plot_overlay.output_backend = "svg"
                plot_position.output_backend = "svg"
                prefix = '.'.join(outfile.split('.')[:-1])
                export_svgs(column(plot_overlay), filename=f"{prefix}_1.svg")
                export_svgs(column(plot_position), filename=f"{prefix}_2.svg")
            # Append object to plot list
            position_plot_list.append(column(row(plot_position, plot_overlay,
                                                 column(stats_table))))

        else:
            LOGGER.warn('No plot astrometric created for {}'.format(model_pair[1]["path"]))
    if position_plot_list:
        # Make the plots in a column layout
        position_plots = column(position_plot_list)
        # Save the plot (html)
        save(position_plots, title=outfile)
        LOGGER.info('Saving astrometry comparisons in {}'.format(outfile))


def _residual_plotter(res_noise_images, points=None, results=None,
                      inline=False, prefix=None, title_size='16pt',
                      x_label_size='12pt', y_label_size='12pt',
                      legend_size='10pt', xmajor_size='6pt',
                      ymajor_size='6pt', units='micro', svg=False):
    """Plot ratios of random residuals and noise

    Parameters
    ----------
    res_noise_images: dict
        Structured input images with labels.
    points: int
        Number of data point to generate in case of random residuals
    results: dict
        Structured output results.
    inline : bool
        Allow inline plotting inside a notebook.
    prefix : str
        Prefix for output htmls

    """
    if points:
        title = "Random Residual Noise"
        if prefix:
            outfile = f'{prefix}-RandomResidualNoiseRatio.html'
        else:
            outfile = 'RandomResidualNoiseRatio.html'
    else:
        title = "Source Residual Noise"
        if prefix:
            outfile = f'{prefix}-SourceResidualNoiseRatio.html'
        else:
            outfile = 'SourceResidualNoiseRatio.html'
    output_file(outfile)
    residual_plot_list = []
    for residual_pair in res_noise_images:
        residuals1 = []
        residuals2 = []
        name_labels = []
        phase_centre_dist = []
        res_noise_ratio = []
        res_image = residual_pair[0]['label']
        for res_src in results[res_image]:
            residuals1.append(res_src[0])
            residuals2.append(res_src[1])
            res_noise_ratio.append(res_src[2])
            phase_centre_dist.append(res_src[3])
            name_labels.append(res_src[4])
        if len(name_labels) > 1:
            # Get sigma value of residuals
            res1 = np.array(residuals1) * FLUX_UNIT_SCALER[units][0]
            res2 = np.array(residuals2) * FLUX_UNIT_SCALER[units][0]
            # Get ratio data
            y1 = np.array(res_noise_ratio)
            x1 = np.array(range(len(res_noise_ratio)))
            # Create additional feature on the plot such as hover, display text
            TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,hover,save"
            source = ColumnDataSource(
                        data=dict(x=x1, y=y1, res1=res1, res2=res2, label=name_labels))
            text1 = residual_pair[0]["path"].split("/")[-1].split('.fits')[0]
            text2 = residual_pair[1]["path"].split("/")[-1].split('.fits')[0]
            # Get y2 label and range
            y2_label = "Flux density ({})".format(FLUX_UNIT_SCALER[units][1])
            y_max = max(res1) if max(res1) > max(res2) else max(res2)
            y_min = min(res1) if min(res1) < min(res2) else min(res2)
            # Create a plot objects and set axis limits
            plot_residual = figure(title=title,
                                   x_axis_label='Sources',
                                   y_axis_label='Res1-to-Res2',
                                   #sizing_mode='stretch_both',
                                   plot_width=1200, plot_height=800,
                                   tools=TOOLS)
            plot_residual.y_range = Range1d(start=min(y1) - .01, end=max(y1) + .01)
            plot_residual.extra_y_ranges = {y2_label: Range1d(start=y_min - .01 * abs(y_min),
                                                              end=y_max + .01 * abs(y_max))}
            plot_residual.add_layout(LinearAxis(y_range_name=y2_label,
                                                axis_label=y2_label),
                                     'right')
            plot_residual.axis.axis_label_text_font_style = 'bold'
            res1_object = plot_residual.line(x1, res1,
                                             color='red',
                                             legend_label=f'res1: {text1}',
                                             y_range_name=y2_label)
            res2_object = plot_residual.line(x1, res2,
                                             color='blue',
                                             legend_label=f'res2: {text2}',
                                             y_range_name=y2_label)
            res_ratio_object = plot_residual.line('x', 'y',
                                                  name='ratios',
                                                  source=source,
                                                  color='green',
                                                  legend_label='res1-to-res2')
            plot_residual.title.text_font_size = title_size
            plot_residual.xaxis.axis_label_text_font_size = x_label_size
            plot_residual.yaxis.axis_label_text_font_size = y_label_size
            plot_residual.legend.label_text_font_size = legend_size
            plot_residual.xaxis.major_label_text_font_size = xmajor_size
            plot_residual.yaxis.major_label_text_font_size = ymajor_size
            # Table with stats data
            cols = ["Stats", "Value"]
            stats = {"Stats": [f"{text1} ({FLUX_UNIT_SCALER[units][1]})",
                               f"{text2} ({FLUX_UNIT_SCALER[units][1]})",
                               "Res1-to-Res2"],
                     "Value": [np.mean(residuals1) * FLUX_UNIT_SCALER[units][0],
                               np.mean(residuals2) * FLUX_UNIT_SCALER[units][0],
                               np.mean(residuals2) / np.mean(residuals1)]}
            source = ColumnDataSource(data=stats)
            columns = [TableColumn(field=x, title=x.capitalize()) for x in cols]
            dtab = DataTable(source=source, columns=columns,
                             width=550, max_width=800,
                             height=100, max_height=150,
                             sizing_mode='stretch_both')
            table_title = Div(text="Cross Match Stats")
            table_title.align = "center"
            stats_table = column([table_title, dtab])
            # Attaching the hover object with labels
            hover = plot_residual.select(dict(type=HoverTool))
            hover.names = ['ratios']
            hover.tooltips = OrderedDict([
                ("ratio", "@y"),
                ("(Res1,Res2)", "(@res1,@res2)"),
                ("source", "@label")])
            # Position of legend and title align
            plot_residual.legend.location = "top_left"
            plot_residual.title.align = "center"
            # Add object to plot list
            residual_plot_list.append(row(plot_residual, column(stats_table)))
            if svg:
                plot_residual.output_backend = "svg"
                prefix = '.'.join(outfile.split('.')[:-1])
                export_svgs(plot_residual, filename=f"{prefix}.svg")
        else:
            LOGGER.warn('No plot created. Found 0 or 1 data point in {}'.format(res_image))

    if residual_plot_list:
        # Make the plots in a column layout
        residual_plots = column(residual_plot_list)
        # Save the plot (html)
        save(residual_plots, title=outfile)
        LOGGER.info('Saving residual comparision plots {}'.format(outfile))


def _random_residual_results(res_noise_images, data_points=None,
                             fov_factor=None, area_factor=None):
    """Plot ratios of random residuals and noise

    Parameters
    ----------
    res_noise_images: list
        List of dictionaries with residual images
    data_points: int
        Number of data points to extract
    area_factor : float
        Factor to multiply the beam area
    fov_factor : float
        Factor to multiply the field of view for random points

    Returns
    -------
    results : dict
        Dictionary of source residual properties from each residual image.

    """
    LOGGER.info("Plotting ratios of random residuals and noise")
    # dictinary to store results
    results = dict()
    # Get beam size otherwise use default (~6``).
    beam_default = (0.00151582804885738, 0.00128031965017612, 20.0197348935424)
    for images in res_noise_images:
        # Source counter
        i = 0
        # Get label
        res_label1 = images[0]['label']
        # Get residual image names
        res_image1 = images[0]['path']
        res_image2 = images[1]['path']
        # Data structure for residuals compared
        results[res_label1] = []
        # Get fits info
        fits_info = fitsInfo(res_image1)
        # Get beam size otherwise use default (~6``).
        beam_deg = fits_info['b_size'] if fits_info['b_size'] else beam_default
        # In case the images was not deconvloved aslo use default beam
        if beam_deg == (0.0, 0.0, 0.0):
            beam_deg = beam_default
        # Open residual images header
        res_hdu1 = fitsio.open(res_image1)
        res_hdu2 = fitsio.open(res_image2)
        # Get data from residual images
        res_data1 = res_hdu1[0].data
        res_data2 = res_hdu2[0].data
        # Get random pixel coordinates
        pix_coord_deg = _get_random_pixel_coord(
                            data_points,
                            phase_centre=fits_info['centre'],
                            sky_area=fits_info['skyArea'] * fov_factor)
        # Get the number of frequency channels
        nchan1 = (res_data1.shape[1]
                  if res_data1.shape[0] == 1
                  else res_data1.shape[0])
        nchan2 = (res_data2.shape[1]
                  if res_data2.shape[0] == 1
                  else res_data2.shape[0])
        for RA, DEC in pix_coord_deg:
            i += 1
            # Get width of box around source
            width = int(deg2arcsec(beam_deg[0]) * area_factor)
            # Get a image slice around source
            imslice = get_box(fits_info["wcs"], (RA, DEC), width)
            # Get noise rms in the box around the point coordinate
            res1_area = res_data1[0, 0, :, :][imslice]
            res2_area = res_data2[0, 0, :, :][imslice]
            # Ignore empty arrays due to points at the edge
            if not res1_area.size or not res2_area.size:
                continue
            res1_rms = res1_area.std()
            res2_rms = res2_area.std()
            # if image is cube then average along freq axis
            if nchan1 > 1:
                flux_rms1 = 0.0
                for frq_ax in range(nchan1):
                    # In case the first two axes are swapped
                    if res_data1.shape[0] == 1:
                        target_area1 = res_data1[0, frq_ax, :, :][imslice]
                    else:
                        target_area1 = res_data1[frq_ax, 0, :, :][imslice]
                    # Sum of all the fluxes
                    flux_rms1 += target_area1.std()
                # Get the average std and mean along all frequency channels
                res1_rms = flux_rms1/float(nchan1)
            if nchan2 > 1:
                flux_rms2 = 0.0
                for frq_ax in range(nchan2):
                    # In case the first two axes are swapped
                    if res_data2.shape[0] == 1:
                        target_area2 = res_data2[0, frq_ax, :, :][imslice]
                    else:
                        target_area2 = res_data2[frq_ax, 0, :, :][imslice]
                    # Sum of all the fluxes
                    flux_rms2 += target_area2.std()
                # Get the average std and mean along all frequency channels
                res2_rms = flux_rms2/float(nchan2)
            # Get phase centre and determine phase centre distance
            RA0 = fits_info['centre'][0]
            DEC0 = fits_info['centre'][1]
            phase_centre_dist= (np.sqrt((RA-RA0)**2 + (DEC-DEC0)**2))
            # Store all outputs in the results data structure
            results[res_label1].append([res1_rms*1e0,
                                       res2_rms*1e0,
                                       res1_rms/res2_rms*1e0,
                                       phase_centre_dist,
                                       'source{0}'.format(i)])
    return results


def _source_residual_results(res_noise_images, skymodel, area_factor=None):
    """Plot ratios of source residuals and noise

    Parameters
    ----------
    res_noise: list
        List of dictionaries with residual images
    skymodel: file
        Tigger skymodel file to locate on source residuals
    area_factor : float
        Factor to multiply the beam area.

    Returns
    -------
    results : dict
        Dictionary of source residual properties from each residual image.

    """
    LOGGER.info("Plotting ratios of source residuals and noise")
    # Dictinary to store results
    results = dict()
    # Get beam size otherwise use default (6``).
    beam_default = (0.00151582804885738, 0.00128031965017612, 20.0197348935424)
    for images in res_noise_images:
        # Get label
        res_label1 = images[0]['label']
        # Get residual image names
        res_image1 = images[0]['path']
        res_image2 = images[1]['path']
        # Data structure for residuals compared
        results[res_label1] = []
        # Get fits info
        fits_info = fitsInfo(res_image1)
        # Get beam size otherwise use default (~6``).
        beam_deg = fits_info['b_size'] if fits_info['b_size'] else beam_default
        # In case the images was not deconvloved also use default beam
        if beam_deg == (0.0, 0.0, 0.0):
            beam_deg = beam_default
        # Open residual images header
        res_hdu1 = fitsio.open(res_image1)
        res_hdu2 = fitsio.open(res_image2)
        # Get data from residual images
        res_data1 = res_hdu1[0].data
        res_data2 = res_hdu2[0].data
        # Load skymodel to get source positions
        model_lsm = Tigger.load(skymodel)
        # Get all sources in the model
        model_sources = model_lsm.sources
        # Data structure for each residuals to compare
        results[res_label1] = []
        # Get the number of frequency channels
        nchan1 = (res_data1.shape[1]
                  if res_data1.shape[0] == 1
                  else res_data1.shape[0])
        nchan2 = (res_data2.shape[1]
                  if res_data2.shape[0] == 1
                  else res_data2.shape[0])
        for model_source in model_sources:
            # Get phase centre Ra and Dec coordinates
            RA0 = model_lsm.ra0
            DEC0 = model_lsm.dec0
            # Get source Ra and Dec coordinates
            ra = model_source.pos.ra
            dec = model_source.pos.dec
            # Convert to degrees
            RA = rad2deg(ra)
            DEC = rad2deg(dec)
            # Remove any wraps
            if ra > np.pi:
                ra -= 2.0*np.pi
            # Get distance from phase centre
            delta_phase_centre = angular_dist_pos_angle(RA0, DEC0, ra, dec)
            phase_dist_arcsec = rad2arcsec(delta_phase_centre[0])
            phase_centre_dist = phase_dist_arcsec/3600.
            # Get width of box around source
            width = int(deg2arcsec(beam_deg[0]) * area_factor)
            # Get a image slice around source
            imslice = get_box(fits_info["wcs"], (RA, DEC), width)
            # Get noise rms in the box around the point coordinate
            res1_area = res_data1[0, 0, :, :][imslice]
            res2_area = res_data2[0, 0, :, :][imslice]
            # Ignore empty arrays due to sources at the edge
            if not res1_area.size or not res2_area.size:
                continue
            res1_rms = res1_area.std()
            res2_rms = res2_area.std()
            # if image is cube then average along freq axis
            if nchan1 > 1:
                flux_rms1 = 0.0
                for frq_ax in range(nchan1):
                    # In case the first two axes are swapped
                    if res_data1.shape[0] == 1:
                        target_area1 = res_data1[0, frq_ax, :, :][imslice]
                    else:
                        target_area1 = res_data1[frq_ax, 0, :, :][imslice]
                    # Sum of all the fluxes
                    flux_rms1 += target_area1.std()
                # Get the average std and mean along all frequency channels
                res1_rms = flux_rms1/float(nchan1)
            if nchan2 > 1:
                flux_rms2 = 0.0
                for frq_ax in range(nchan2):
                    # In case the first two axes are swapped
                    if res_data2.shape[0] == 1:
                        target_area2 = res_data2[0, frq_ax, :, :][imslice]
                    else:
                        target_area2 = res_data2[frq_ax, 0, :, :][imslice]
                    # Sum of all the fluxes
                    flux_rms2 += target_area2.std()
                # Get the average std and mean along all frequency channels
                res2_rms = flux_rms2/float(nchan2)
            # Store all outputs in the results data structure
            results[res_label1].append([res1_rms * 1e0,
                                       res2_rms * 1e0,
                                       res1_rms / res2_rms * 1e0,
                                       phase_centre_dist,
                                       model_source.name,
                                       model_source.flux.I])
    return results


def plot_aimfast_stats(fidelity_results_file, units='micro', prefix=''):
    """Plot stats results if more that one residual images where assessed"""

    with open(fidelity_results_file) as f:
        data = json.load(f)
    res_stats = dict()
    dr_stats = dict()
    for par, val in data.items():
        val_copy = val.copy()
        if '.fits' not in par and 'models' not in val and type(val) is not list:
            for p, v in val.items():
                if type(v) is dict:
                    dr_stats[p] = v
                    val_copy.pop(p)
            res_stats[par] = val_copy
            res_stats[par]['NORM'] = res_stats[par]['NORM'][0]

    res_stats = dict(sorted(res_stats.items()))
    dr_stats = dict(sorted(dr_stats.items()))

    im_keys = []
    rms_values = []
    stddev_values = []
    mad_values = []
    max_values = []
    skew_values = []
    kurt_values = []
    norm_values = []
    for res_stat in res_stats:
        im_keys.append(res_stat.replace('-residual', ''))
        rms_values.append(res_stats[res_stat]['RMS'])
        stddev_values.append(res_stats[res_stat]['STDDev'])
        mad_values.append(res_stats[res_stat]['MAD'])
        max_values.append(res_stats[res_stat]['MAX'])
        skew_values.append(res_stats[res_stat]['SKEW'])
        kurt_values.append(res_stats[res_stat]['KURT'])
        norm_values.append(res_stats[res_stat]['NORM'])

    width = 400
    height = 300
    multiplier = FLUX_UNIT_SCALER[units][0]

    # Vriance plots
    variance_plotter = figure(x_range=im_keys, x_axis_label="Image", y_axis_label="Flux density (µJy)",
                              plot_width=width, plot_height=height, title='Residual Variance')
    variance_plotter.line(im_keys, np.array(stddev_values)*multiplier, legend_label='std', color='blue')
    variance_plotter.line(im_keys, np.array(mad_values)*multiplier, legend_label='mad', color='red')
    variance_plotter.line(im_keys, np.array(max_values)*multiplier, legend_label='max', color='green')
    variance_plotter.title.align = 'center'

    # Moment 3 & 4 plots
    mom34_plotter = figure(x_range=im_keys, x_axis_label="Image", y_axis_label="Value",
                           plot_width=width, plot_height=height, title='Skewness & Kurtosis')
    mom34_plotter.line(im_keys, skew_values, legend_label='Skewness', color='blue')
    mom34_plotter.line(im_keys, kurt_values, legend_label='kurtosis', color='red')
    mom34_plotter.title.align = 'center'

    # Normality test plot
    normalised = np.array(norm_values)/norm_values[0]
    norm_plotter = figure(x_range=im_keys, x_axis_label="Image", y_axis_label="Value",
                          plot_width=width, plot_height=height, title='Normality Tests')
    norm_plotter.vbar(x=im_keys, top=normalised, width=0.9)
    #norm_plotter.y_range.start = 0
    norm_plotter.title.align = 'center'

    # Dynamic Range plot
    dr_keys = []
    dr_values = []
    for dr_stat in dr_stats:
        dr_keys.append(dr_stat.replace('-model', ''))
        dr_values.append(dr_stats[dr_stat]['DR'])
    dr_plotter = figure(x_range=dr_keys, x_axis_label="Image", y_axis_label="Value",
                        plot_width=width, plot_height=height, title='Dynamic Range')
    dr_plotter.vbar(x=dr_keys, top=dr_values, width=0.9)
    #dr_plotter.y_range.start = 0
    dr_plotter.title.align = 'center'
    outfile = '{}-stats-plot.html'.format(prefix or 'aimfast')
    output_file(outfile)
    save(column(row(variance_plotter, mom34_plotter),
                    row(norm_plotter, dr_plotter)), title=outfile)


def plot_subimage_stats(fitsnames, centre_coords, sizes, htmlprefix='default',
                        title_size='12pt', x_label_size='10pt', y_label_size='10pt',
                        bar_label_size='15pt', units='micro', svg=False):
    """Plot subimages and stats"""
    output_dict = {}
    subplot_list = []
    plot_height = 300
    plot_width = 400
    for im in range(len(centre_coords)):
        im_subplot_list = []
        LOGGER.info(f"Making Subimage with centre pixels ({centre_coords[im]})")
        size = sizes[im]
        centre_coord = centre_coords[im]
        rx, ry = centre_coord[0], centre_coord[1]
        rx_0, ry_0 = int(rx-size/2), int(ry-size/2)
        for n, fitsname in enumerate(fitsnames):
            fitsinfo = fitsInfo(fitsname)
            subimage_data = get_subimage(fitsname, centre_coord, size)
            subimg_stats = image_stats(subimage_data, test_normality='normaltest')
            centre_str = ','.join([str(cc) for cc in centre_coord])
            sub_stats ={"RMS": round(subimg_stats['RMS'] * FLUX_UNIT_SCALER[units][0],
                                     DECIMALS),
                        "STDDev": round(subimg_stats['STDDev'] * FLUX_UNIT_SCALER[units][0],
                                        DECIMALS),
                        "MAD": round(subimg_stats['MAD'] * FLUX_UNIT_SCALER[units][0],
                                     DECIMALS),
                        "MIN": round(subimg_stats['MIN'] * FLUX_UNIT_SCALER[units][0],
                                     DECIMALS),
                        "MAX": round(subimg_stats['MAX'] * FLUX_UNIT_SCALER[units][0],
                                     DECIMALS),
                        "SUM_NEG": round(subimg_stats['SUM_NEG'] * FLUX_UNIT_SCALER[units][0],
                                     DECIMALS),
                        "SKEW": round(subimg_stats['SKEW'], DECIMALS),
                        "KURT": round(subimg_stats['KURT'], DECIMALS),
                        "NORM": round(subimg_stats['NORM'][0], DECIMALS),
                        "UNITS": units}
            output_dict[f"centre-{centre_str}-{n}"] = {fitsname: sub_stats}
            cols = ["Stats", f"Value ({FLUX_UNIT_SCALER[units][1]})"]
            stats = {"Stats": ["RMS", "STDDev", "MAD", "MIN", "SUM_NEG",
                               "MAX", "*SKEW", "*KURT", "*NORM"],
                     f"Value ({FLUX_UNIT_SCALER[units][1]})":
                       [sub_stats['RMS'], sub_stats['STDDev'],
                        sub_stats['MAD'], sub_stats['MIN'],
                        sub_stats['SUM_NEG'], sub_stats['MAX'],
                        sub_stats['SKEW'],
                        sub_stats['KURT'],
                        sub_stats['NORM']]}

            source = ColumnDataSource(data=stats)
            columns = [TableColumn(field=x, title=x.capitalize()) for x in cols]
            dtab = DataTable(source=source, columns=columns,
                             width=250, max_width=350,
                             height=200, max_height=250,
                             sizing_mode='stretch_both')
            table_title = Div(text="Sub-image Statistics")
            table_title.align = "center"
            stats_table = column([table_title, dtab])

            plot_title = f"{fitsname.split('/')[-1].split('.')[0]} sub-image: {im+1}"
            if len(im_subplot_list) > 0:
                s1 = im_subplot_list[0]
                subplot = figure(title=plot_title,
                                 x_axis_label='Right Ascension (deg)',
                                 y_axis_label='Declination (deg)',
                                 width=plot_width, height=plot_height,
                                 x_range=s1.x_range, y_range=s1.y_range,
                                 tooltips=[("(x, y)", "($x, $y)"),
                                           (f"value ({FLUX_UNIT_SCALER[units][1]})",
                                           "@image")])
            else:
                # Initial column 1 plot
                subplot = figure(title=plot_title,
                                 x_axis_label='Right Ascension (deg)',
                                 y_axis_label='Declination (deg)',
                                 width=plot_width, height=plot_height,
                                 tooltips=[("(x, y)", "($x, $y)"),
                                           (f"value ({FLUX_UNIT_SCALER[units][1]})",
                                           "@image")])

            # must give a vector of images
            subimage = subimage_data[0,0,:,:]
            if svg:
                # Save subimages as svg
                try:
                    import matplotlib.pyplot as plt
                    wcs = fitsinfo['wcs']
                    ax = plt.subplot(111, projection=wcs, slices=('x','y',0,0))
                    shw = plt.imshow(subimage *  FLUX_UNIT_SCALER[units][0],
                                     extent=[rx_0, rx_0+size, ry_0, ry_0+size],
                                     vmin=-0.1, vmax=1)
                    outname = fitsname.split('.fits')[0]
                    bar = plt.colorbar(shw)
                    plt.xlabel('Right Ascension (hours)',
                               fontsize=float(x_label_size.split('pt')[0]))
                    plt.ylabel('Declination (deg)',
                               fontsize=float(y_label_size.split('pt')[0]))
                    bar.set_label(f"Flux density ({FLUX_UNIT_SCALER[units][1]})",
                                  fontsize=float(bar_label_size.split('pt')[0]))
                    plt.savefig(f"{outname}.svg")
                    print(f"{outname}.svg")
                except ImportError:
                    LOGGER.warn("SVGs are requested but matplotlib is not installed")
                    LOGGER.warn("RUN: pip install aimfast[svg_images]")

            subplot.image(image=[subimage * FLUX_UNIT_SCALER[units][0]],
                          x=rx_0, y=ry_0, dw=size, dh=size,
                          palette="Plasma11", level="image")
            color_mapper = LinearColorMapper(palette="Plasma11",
                                             low=subimage.min() * FLUX_UNIT_SCALER[units][0],
                                             high=subimage.max() * FLUX_UNIT_SCALER[units][0])
            color_bar = ColorBar(color_mapper=color_mapper, width=80, label_standoff=4,
                                 location=(0, 0), orientation='vertical')
            color_bar_plot = figure(title=f"Flux Density ({FLUX_UNIT_SCALER[units][1]})",
                                    title_location="right",
                                    height=plot_height, width=8,
                                    toolbar_location=None, min_border=0,
                                    outline_line_color=None)
            color_bar_plot.add_layout(color_bar, 'right')
            color_bar_plot.title.align="center"
            color_bar_plot.title.text_font_size = '10pt'
            im_subplot_list.append(subplot)
            im_subplot_list.append(stats_table)
        subplot_list.append(column(row(im_subplot_list)))
    output_file(f"{htmlprefix}_subimage_stats.html", title="subimage plots and stats")
    save(column(subplot_list))
    return output_dict


def get_source_properties_from_catalog(catalog_file):
    model = get_model(catalog_file)
    sources = model.sources
    source_properties = {}
    for source in sources:
        if 'name' not in source_properties.keys():
            source_properties['name'] = [source.name]
        else:
            source_properties['name'].append(source.name)
        fluxes = source.get_attr('flux').strAttributes().split(',')
        for flux in fluxes:
            prop = flux.split('=')[0]
            val = float(flux.split('=')[1])
            if prop not in source_properties.keys():
                source_properties[prop] = [val]
            else:
                source_properties[prop].append(val)
        positions = source.get_attr('pos').strAttributes().split(',')
        for pos in positions:
            prop = pos.split('=')[0]
            if prop not in source_properties.keys():
                source_properties[prop] = [rad2deg(getattr(source.pos, prop))]
            else:
                source_properties[prop].append(rad2deg(getattr(source.pos, prop)))
        try:
            shapes = source.get_attr('shape').strAttributes().split(',')
        except AttributeError:
            shapes = ['ex=0', 'ey=0', 'pa=0',
                        'ex_err=0', 'ey_err=0', 'pa_err=0']
        for shape in shapes:
            prop = shape.split('=')[0]
            val = float(shape.split('=')[1])
            if prop not in source_properties.keys():
                source_properties[prop] = [val]
            else:
                source_properties[prop].append(val)
        try:
            spectrums = source.get_attr('spectrum').strAttributes().split(',')
            spectrums.append(f"spi_err={source.spi_error}")
        except AttributeError:
            spectrums = ['spi=999', 'spi_err=999', 'freq0=999']
        for spectrum in spectrums:
            prop = spectrum.split('=')[0]
            val = float(spectrum.split('=')[1])
            if prop not in source_properties.keys():
                source_properties[prop] = [val]
            else:
                source_properties[prop].append(val)
    return source_properties


def plot_model_columns(catalog_file, x, y, x_err=None, y_err=None, svg=False,
                       x_label=None, y_label=None, title=None, html_prefix=None,
                       title_size='16pt', x_label_size='12pt', y_label_size='12pt',
                       legend_size='10pt', xmajor_size='6pt', ymajor_size='6pt',
                       units='micro'):
    """Plot catalog columns including their uncertainties"""
    width, height = 800, 800
    if 'lsm.html' in catalog_file:
        source_properties = get_source_properties_from_catalog(catalog_file)
    else:
        data = Table.read(catalog_file)
        print('Model not yet supported')
    bokeh_source = ColumnDataSource(data=source_properties)
    x_y_plotter = figure(x_axis_label=x if not x_label else x_label,
                         y_axis_label=y if not y_label else y_label,
                         plot_width=width, plot_height=height,
                         #tools=TOOLS,
                         title=f"{catalog_file.split('.')[0]} {x.upper()} vs {y.upper()}"
                               if not title else title)
    x_y_plotter.scatter(x, y, source=bokeh_source,
                             name='x_y_data')
    x_y_plotter.title.align = 'center'
    x_y_plotter.title.text_font_size = title_size
    x_y_plotter.xaxis.axis_label_text_font_size = x_label_size
    x_y_plotter.yaxis.axis_label_text_font_size = y_label_size
    #x_y_plotter.legend.label_text_font_size = legend_size
    x_y_plotter.xaxis.major_label_text_font_size = xmajor_size
    x_y_plotter.yaxis.major_label_text_font_size = ymajor_size
    x_y_plotter.axis.axis_label_text_font_style = 'normal'
    if x in ['RA', 'ra']:
        x_y_plotter.x_range.flipped = True
    elif y in ['RA', 'ra']:
        x_y_plotter.y_range.flipped = True
    # Attaching the hover object with labels
    tool_values = []
    for p in source_properties.keys():
        tool_values.append((p, f'@{p}'))
    x_y_plotter.add_tools(HoverTool(tooltips=tool_values))
    # create the coordinates for the errorbars
    err_xs = []
    err_ys = []
    xs = source_properties[x]
    ys = source_properties[y]
    if x_err:
        xerrs = source_properties[x_err]
        for x, y, xerr in zip(xs, ys, xerrs):
            err_xs.append((x - xerr, x + xerr))
            err_ys.append((y, y))
        x_y_plotter.multi_line(err_xs, err_ys, color='red')
    if y_err:
        yerrs = source_properties[y_err]
        for x, y, yerr in zip(xs, ys, yerrs):
            err_xs.append((x, x))
            err_ys.append((y - yerr, y + yerr))
        x_y_plotter.multi_line(err_xs, err_ys, color='red')
    column_list = source_properties.keys()
    bokeh_source_table = ColumnDataSource(data=source_properties)
    columns = [TableColumn(field=col, title=col) for col in column_list]
    dtab = DataTable(source=bokeh_source_table, columns=columns,
                             width=width, max_width=width + 50,
                             height=height, max_height=width + 50,
                             sizing_mode='stretch_both')
    table_title = Div(text="Source Table")
    table_title.align = "center"
    source_table = column([table_title, dtab])

    LOGGER.info(f"Total number of sources: {len(source_properties['name'])}")
    if not html_prefix:
        output_file_name = f"{catalog_file.split('.')[0]}_column_properties.html"
    else:
        output_file_name = f"{html_prefix}.html"
    LOGGER.info(f"Saving results in {output_file_name}")
    output_file(output_file_name)
    save(row(source_table, x_y_plotter))
    svg = True
    if svg:
        x_y_plotter.output_backend = "svg"
        prefix = '.'.join(output_file_name.split('.')[:-1])
        export_svgs(x_y_plotter, filename=f"{prefix}.svg")


def plot_model_data(catalog_file, html_prefix=''):
    """Plotting catalog table"""
    width, height = 1000, 2000
    if 'lsm.html' in catalog_file:
        source_properties = get_source_properties_from_catalog(catalog_file)
    else:
        data = Table.read(catalog_file)
        print('Model not yet supported')
    column_list = source_properties.keys()
    bokeh_source_table = ColumnDataSource(data=source_properties)
    columns = [TableColumn(field=col, title=col) for col in column_list]
    dtab = DataTable(source=bokeh_source_table, columns=columns,
                             width=width, max_width=width + 50,
                             height=height, max_height=width + 50,
                             sizing_mode='stretch_both')
    table_title = Div(text="Source Table")
    table_title.align = "center"
    source_table = column([table_title, dtab])

    LOGGER.info(f"Total number of sources: {len(source_properties['name'])}")
    print(html_prefix)
    if not html_prefix:
        output_file_name = f"{catalog_file.split('.')[0]}_column_properties.html"
    else:
        output_file_name = f"{html_prefix}.html"
    LOGGER.info(f"Saving results in {output_file_name}")
    output_file(output_file_name)
    save(source_table)


def get_sf_params(configfile):
    import yaml
    with open(r'{}'.format(configfile)) as file:
        sf_parameters = yaml.load(file, Loader=yaml.FullLoader)
    return sf_parameters


def source_finding(sf_params, sf=None):
    outfile = None
    aegean_sf = sf_params.pop('aegean')
    pybd_sf = sf_params.pop('pybdsf')
    enable_aegean = aegean_sf.pop('enable')
    enable_pybdsf = pybd_sf.pop('enable')
    if enable_pybdsf or sf in ['pybdsf']:
        filename = pybd_sf['filename']
        LOGGER.info(f"Running pybdsf source finder on image: {filename}")
        outfile = bdsf(filename, pybd_sf, LOGGER)
    elif enable_aegean or sf in ['aegean']:
        filename = aegean_sf['filename']
        LOGGER.info(f"Running aegean source finder on image: {filename}")
        outfile = aegean(filename, aegean_sf, LOGGER)
    else:
        LOGGER.warn(f"{WARNING}No source finder selected.{ENDC}")
    return outfile


def get_argparser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description=("Examine radio image fidelity and source recovery by obtaining: \n"
                     "- The four (4) moments of a residual image \n"
                     "- The Dynamic range in restored image \n"
                     "- Comparing the fits images by running source finder \n"
                     "- Comparing the tigger models and online catalogs (NVSS, SUMSS) \n"
                     "- Comparing the on source/random residuals to noise \n"
                     "- Comparing residual stats from sub-images"))
    subparser = parser.add_subparsers(dest='subcommand')
    sf = subparser.add_parser('source-finder')
    sf.add_argument('-c', '--config', dest='config',
                    help='Config file to run source finder of choice (YAML format)')
    sf.add_argument('-gc', '--generate-config', dest='generate',
                    help='Genrate config file to run source finder of choice')
    argument = partial(parser.add_argument)
    argument('-v', "--version", action='version',
             version='{0:s} version {1:s}'.format(parser.prog, _version))
    # Inputs to analyse
    argument('--compare-models', dest='models', nargs=2, action='append',
             help='List of tigger model (text/lsm.html) files to compare \n'
                  'e.g. --compare-models model1.lsm.html model2.lsm.html')
    argument('--compare-images', dest='images', nargs=2, action='append',
             help='List of restored image (fits) files to compare. \n'
                  'Note that this will initially run a source finder. \n'
                  'e.g. --compare-images image1.fits image2.fits')
    argument('--compare-online', dest='online', nargs=1, action='append',
             help='List of catalog models (html/ascii, fits) restored image (fits)'
                  ' files to compare with online catalog. \n'
                  'e.g. --compare-online image1.fit')
    argument('--compare-residuals', dest='noise', nargs=2, action='append',
             help='List of noise-like (fits) files to compare \n'
                  'e.g. --compare-residuals residual1.fits residual2.fits')
    argument('--compare-residual-subimages', dest='subimage_noise',
             nargs='+', action='append',
             help='List of noise-like (fits) files to compare \n'
                  'e.g. --compare-residuals residual1.fits residual2.fits')
    argument('-catalog', '--tigger-model', dest='model',
             help='Name of the tigger model lsm.html file or any supported catalog')
    argument('--restored-image', dest='restored',
             help='Name of the restored image fits file')
    argument('-psf', '--psf-image', dest='psf',
             help='Name of the point spread function file or psf size in arcsec')
    argument('--residual-image', dest='residual',
             help='Name of the residual image fits file')
    argument('--mask-image', dest='mask',
             help='Name of the mask image fits file')
    argument('-fdr', '--fidelity-results', dest='json',
             help='aimfast fidelity results file (JSON format)')
    argument('-reg', '--input-regions', dest='reg',
             help='Region file with regions to generate stats)')
    # Source finding
    argument('-c', '--config', dest='config',
             help='Config file to run source finder of choice (YAML format)')
    argument('-sf', '--source-finder', dest='sourcery',
             choices=('aegean', 'pybdsf'), default='pybdsf',
             help='Source finder to run if comparing restored images')
    # Online catalog query
    argument("--online-catalog-name", dest='catalog_name',
             help='Prefix of output catalog file name')
    argument('-oc', '--online-catalog', dest='online_catalog',
             choices=('sumss', 'nvss'), default='nvss',
             help='Online catalog to compare local image/model.')
    argument('-ptc', '--centre_coord', dest='centre_coord',
             default="0:0:0, -30:0:0",
             help='Centre of online catalog to compare local image/model \n'
                  'in "RA hh:mm:ss, Dec deg:min:sec".')
    argument('-w', '--width', dest='width',
             help='Field of view width to querry online catalog in degrees.'
                   'e.g. -w 3.0d')
    # Image stats parameters
    argument('--normality-test', dest='test_normality',
             choices=('shapiro', 'normaltest'),
             help='Name of model to use for normality testing. \n'
                  'options: [shapiro, normaltest] \n'
                  'NB: normaltest is the D`Agostino')
    argument('-dr', '--data-range', dest='data_range',
             help='Data range to perform normality testing')
    argument('-thresh', '--threshold', dest='thresh',
             help='Get stats of channels with pixel flux above thresh in Jy/Beam. \n'
                  'Also this can be used to filter out sources from online catalog')
    argument('-chans', '--channels', dest='channels',
             help='Get stats of specified channels e.g. "10~20;100~1000"')
    argument('-cps', '--centre-pixels-size', dest='centre_pix_size',
             nargs='+', action='append',
             help='List of subimage centre pixels and their sizes to compute stats. \n'
                  'e.g. 500,500,20 200,10,5')
    # Formatting
    argument('-dp', '--data-points', dest='points',
             help='Data points to sample the residual/noise image')
    argument('-fp', '--flux-plot', dest='fluxplot', default='log',
             choices=('log', 'snr', 'inout'),
             help='Type of plot for flux comparison of the two catalogs')
    argument('-units', '--units', dest='units', default="jansky",
             choices=('jansky', 'milli', 'micro', 'nano'),
             help='Units to represent the results')
    argument('-deci', '--decimals', dest='deci', default=2,
             help='Number of decimal places to round off results')
    argument('-oa', '--only-off-axis', dest='off_axis', default=None,
             help='Plot only cross-matched sources with distance from the phase centre'
                  ' less than this value')
    argument('-af', '--area-factor', dest='factor', type=float, default=2,
             help='Factor to multiply the beam area to get target peak area')
    argument('-fov', '--fov-factor', dest='fov_factor', type=float, default=0.9,
             help='Factor to multiply the field of view for random points. i.e. 0.0-1.0')
    argument('-tol', '--tolerance', dest='tolerance', type=float, default=0.2,
             help='Tolerance to cross-match sources in arcsec')
    argument('-as', '--all-source', dest='all', default=False, action='store_true',
             help='Compare all sources irrespective of shape, otherwise only '
                  'point-like sources are compared')
    argument('-closest', '--closest', dest='closest_only', default=False, action='store_true',
             help='Use the closest source only when cross matching sources')
    argument('-sl', '--shape-limit', dest='shape_limit', default=6.0,
             help='Cross-match only sources with a maj-axis equal or less than this value')
    argument("--label",
             help='Use this label instead of the FITS image path when saving '
                  'data as JSON file')
    # Plot labelling for basic catalog plotting
    argument('-x', '--x-col-data', dest='x_col',
             help='Catalog column name to plot on the x-axis')
    argument('-y', '--y-col-data', dest='y_col',
             help='Catalog column name to plot on the y-axis')
    argument('-x-err', '--x-col-err-data', dest='x_col_err',
             help='Catalog column name to plot error data on the x-axis')
    argument('-y-err', '--y-col-err-data', dest='y_col_err',
             help='Catalog column name to plot error data on the y-axis')
    argument('-x-label', '--x-label', dest='x_label',
             help='x-axis labels for the plot')
    argument('-y-label', '--y-label', dest='y_label',
             help='y-axis labels for the plots')
    argument('-title', '--plot-title', dest='title',
             help="Title label for the basic catalog plot")
    # Plot labelling for the flux comparison plotting
    argument('-fx', '--flux-xlabels', dest='fxlabels', nargs='+',
             help="x-axis labels for the Flux plots")
    argument('-fy', '--flux-ylabels', dest='fylabels', nargs='+',
             help="y-axis labels for the Flux plots")
    argument('-ftitle', '--flux-plot-title', dest='ftitles', nargs='+',
             help="Title labels for the Flux plots")
    # Plot labelling for the position (comparison & overlay) plotting
    argument('-px1', '--position-xlabels1', dest='pxlabels1', nargs='+',
             help="x-axis labels for the position plots")
    argument('-py1', '--position-ylabels1', dest='pylabels1', nargs='+',
             help="y-axis labels for the comparison position plots")
    argument('-ptitle1', '--position-plot-title1', dest='ptitles1', nargs='+',
             help="Title labels for the comparison position  plots")
    argument('-px2', '--position-xlabels2', dest='pxlabels2', nargs='+',
             help="x-axis labels for the overlay position plots")
    argument('-py2', '--position-ylabels2', dest='pylabels2', nargs='+',
             help="y-axis labels for the overlay position plots")
    argument('-ptitle2', '--position-plot-title2', dest='ptitles2', nargs='+',
             help="Title labels for the overlay position plots")
    # Plot labelling sizes for all plots
    argument('-bar-major-size', '--colorbar-major-labels-size', dest='bar_major_size', default='6pt',
             help="x-axis label size for plots")
    argument('-bar-size', '--colorbar-labels-size', dest='barsize', default='14pt',
             help="x-axis label size for plots")
    argument('-x-size', '--xlabels-size', dest='xsize', default='14pt',
             help="x-axis label size for plots")
    argument('-y-size', '--ylabels-size', dest='ysize', default='14pt',
             help="y-axis label size for plots")
    argument('-x-maj-size', '--x-major-labels-size', dest='xmaj_size', default='6pt',
             help="x-axis major label size for plots")
    argument('-y-maj-size', '--y-mojar-labels-size', dest='ymaj_size', default='6pt',
             help="y-axis major label size for plots")
    argument('-legend-size', '--legend-font-size', dest='legsize', default='14pt',
             help="Label size for legends on the plots")
    argument('-title-size', '--plot-title-size', dest='tsize', default='18pt',
             help="Title label size for plots")
    # Outputs
    argument("--html-prefix", dest='htmlprefix',
             help='Prefix of output html files. Default: None.')
    argument("--outfile",
             help='Name of output file name. Default: fidelity_results.json')
    argument('-svg', '--save-svg', dest='svg', default=False, action='store_true',
             help='Save plots in SVG format.')
    return parser


def main():
    """Main function."""
    LOGGER.info("Welcome to AIMfast")
    LOGGER.info(f"Version: {_version}")
    _command = ' '.join(sys.argv)
    LOGGER.info(f"Command: {_command}")
    output_dict = dict()
    parser = get_argparser()
    args = parser.parse_args()
    # Print default args
    LOGGER.info(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    DECIMALS = args.deci
    svg = args.svg
    if args.subcommand:
        if args.config:
            source_finding(get_sf_params(args.config))
        if args.generate:
            generate_default_config(args.generate)
    elif args.json:
       plot_aimfast_stats(args.json, prefix=args.htmlprefix)
    elif not args.residual and not args.restored and not args.model \
            and not args.models and not args.noise and not args.images \
            and not args.subimage_noise and not args.online and not args.json:
        LOGGER.warn(f"{R}No arguments file(s) provided.{W}")
        LOGGER.warn(f"{R}Or 'aimfast -h' for arguments.{W}")

    if args.label:
        residual_label = "{0:s}-residual".format(args.label)
        restored_label = "{0:s}-restored".format(args.label)
        model_label = "{0:s}-model".format(args.label)
    else:
        residual_label = args.residual
        restored_label = args.restored
        model_label = args.model

    if args.model and args.x_col and args.y_col:
        plot_model_columns(args.model, args.x_col, args.y_col,
                           args.x_col_err, args.y_col_err,
                           x_label=args.x_label,
                           y_label=args.y_label,
                           title=args.title,
                           title_size=args.tsize,
                           x_label_size=args.xsize,
                           y_label_size=args.ysize,
                           legend_size=args.legsize,
                           xmajor_size=args.xmaj_size,
                           ymajor_size=args.ymaj_size,
                           units=args.units,
                           html_prefix=args.htmlprefix)

    if args.model and not args.noise and args.residual:
        if not args.residual:
            raise RuntimeError(f"{R}Please provide residual fits file{W}")

        if args.psf:
            psf_val = args.psf.replace(".", "", 1)
            if psf_val.isdigit():
                psf_size = float(args.psf)
            else:
                psf_size = measure_psf(args.psf)
        else:
            psf_size = 6
            LOGGER.warning(f"{R}Please provide psf fits file or psf size.\n"
                           "Otherwise a default beam size of six (~6``) asec "
                           f"is used{W}")

        if args.factor:
            DR = model_dynamic_range(args.model, args.residual, psf_size,
                                     area_factor=args.factor)
        else:
            DR = model_dynamic_range(args.model, args.residual, psf_size)

        if args.test_normality in ['shapiro', 'normaltest']:
            stats = residual_image_stats(args.residual,
                                         args.test_normality,
                                         args.data_range,
                                         args.thresh,
                                         args.channels,
                                         args.mask)
        else:
            if not args.test_normality:
                stats = residual_image_stats(args.residual,
                                             args.test_normality,
                                             args.data_range,
                                             args.thresh,
                                             args.channels,
                                             args.mask)
            else:
                LOGGER.error(f"{R}Please provide correct normality model{W}")
        stats.update({model_label: {
            'DR'                    : DR["global_rms"],
            'DR_deepest_negative'   : DR["deepest_negative"],
            'DR_global_rms'         : DR['global_rms'],
            'DR_local_rms'          : DR['local_rms']}})
        output_dict[residual_label] = stats
    elif args.residual and not args.reg:
        if args.residual not in output_dict.keys():
            if args.test_normality in ['shapiro', 'normaltest']:
                stats = residual_image_stats(args.residual,
                                             args.test_normality,
                                             args.data_range,
                                             args.thresh,
                                             args.channels,
                                             args.mask)
            else:
                if not args.test_normality:
                    stats = residual_image_stats(args.residual,
                                                 args.test_normality,
                                                 args.data_range,
                                                 args.thresh,
                                                 args.channels,
                                                 args.mask)
                else:
                    LOGGER.error(f"{R}Please provide correct normality model{W}")
            output_dict[residual_label] = stats

    if args.restored and args.residual:
        if args.factor:
            DR = image_dynamic_range(args.restored, args.residual,
                                     area_factor=args.factor)
        else:
            DR = image_dynamic_range(args.restored, args.residual)
        output_dict[restored_label] = {
            'DR'                  : DR["global_rms"],
            'DR_deepest_negative' : DR["deepest_negative"],
            'DR_global_rms'       : DR['global_rms'],
            'DR_local_rms'        : DR['local_rms']}

    if args.models:
        models = args.models
        LOGGER.info(f"Number of model pair(s) to compare: {len(models)}")
        if len(models) < 1:
            LOGGER.warn(f"{R}Can only compare two models at a time.{W}")
        else:
            models_list = []
            for i, comp_mod in enumerate(models):
                model1, model2 = comp_mod[0], comp_mod[1]
                models_list.append(
                    [dict(label="{}-model_a_{}".format(args.label, i),
                          path=model1),
                     dict(label="{}-model_b_{}".format(args.label, i),
                          path=model2)],
                )
            output_dict = compare_models(models_list,
                                         tolerance=args.tolerance,
                                         off_axis=args.off_axis,
                                         all_sources=args.all,
                                         shape_limit=args.shape_limit,
                                         closest_only=args.closest_only,
                                         prefix=args.htmlprefix,
                                         flux_plot=args.fluxplot,
                                         ftitles=args.ftitles,
                                         fxlabels=args.fxlabels,
                                         fylabels=args.fylabels,
                                         title_size=args.tsize,
                                         x_label_size=args.xsize,
                                         y_label_size=args.ysize,
                                         legend_size=args.legsize,
                                         xmajor_size=args.xmaj_size,
                                         ymajor_size=args.ymaj_size,
                                         bar_size=args.barsize,
                                         bar_major_size=args.bar_major_size,
                                         svg=svg)

    if args.noise:
        residuals = args.noise
        LOGGER.info(f"Number of residual pairs to compare: {len(residuals)}")
        if len(residuals) < 1:
            LOGGER.error(f"{R}Can only compare atleast one residual pair.{W}")
        else:
            residuals_list = []
            for i, comp_res in enumerate(residuals):
                res1, res2 = comp_res[0], comp_res[1]
                residuals_list.append(
                    [dict(label="{}-res_a_{}".format(args.label, i),
                          path=res1),
                     dict(label="{}-res_b_{}".format(args.label, i),
                          path=res2)],
                )
            if args.model:
                output_dict = compare_residuals(residuals_list,
                                                args.model,
                                                units=args.units,
                                                title_size=args.tsize,
                                                legend_size=args.legsize,
                                                xmajor_size=args.xmaj_size,
                                                ymajor_size=args.ymaj_size,
                                                x_label_size=args.xsize,
                                                y_label_size=args.ysize,
                                                area_factor=args.factor,
                                                prefix=args.htmlprefix)
            else:
                output_dict = compare_residuals(
                    residuals_list,
                    area_factor=args.factor,
                    fov_factor=args.fov_factor,
                    units=args.units,
                    title_size=args.tsize,
                    xmajor_size=args.xmaj_size,
                    ymajor_size=args.ymaj_size,
                    legend_size=args.legsize,
                    x_label_size=args.xsize,
                    y_label_size=args.ysize,
                    prefix=args.htmlprefix,
                    points=int(args.points) if args.points else 100)

    if args.images:
        configfile = args.config
        if not configfile:
            configfile = 'default_sf_config.yml'
            generate_default_config(configfile)
        images = args.images
        sourcery = args.sourcery
        images_list = []
        for i, comp_ims in enumerate(images):
            if args.mask:
                image1, image2 = get_image_products(comp_ims, args.mask)
            else:
                image1, image2 = comp_ims[0], comp_ims[1]
            sf_params1 = get_sf_params(configfile)
            sf_params1[sourcery]['filename'] = image1
            out1 = source_finding(sf_params1, sourcery)
            sf_params2 = get_sf_params(configfile)
            sf_params2[sourcery]['filename'] = image2
            out2 = source_finding(sf_params2, sourcery)
            images_list.append(
                [dict(label="{}-model_a_{}".format(args.label, i),
                      path=out1),
                 dict(label="{}-model_b_{}".format(args.label, i),
                      path=out2)])
        output_dict = compare_models(images_list,
                                     tolerance=args.tolerance,
                                     off_axis=args.off_axis,
                                     shape_limit=args.shape_limit,
                                     all_sources=args.all,
                                     closest_only=args.closest_only,
                                     prefix=args.htmlprefix,
                                     flux_plot=args.fluxplot,
                                     ftitles=args.ftitles,
                                     fxlabels=args.fxlabels,
                                     fylabels=args.fylabels,
                                     title_size=args.tsize,
                                     x_label_size=args.xsize,
                                     y_label_size=args.ysize,
                                     legend_size=args.legsize,
                                     xmajor_size=args.xmaj_size,
                                     ymajor_size=args.ymaj_size,
                                     svg=svg)

    if args.online:
        models = args.online
        sourcery = args.sourcery
        threshold = args.thresh
        width = args.width or '5.0d'
        LOGGER.info(f'Using sky width of {width}')
        catalog_prefix = args.catalog_name or 'default'
        online_catalog = args.online_catalog
        catalog_name = f"{catalog_prefix}_{online_catalog}_catalog_table.txt"
        images_list = []

        LOGGER.info(f'Extracting phase centre coordinates form {models[0][0]}')
        if models[0][0].endswith('.html'):
            Tigger_model = Tigger.load(models[0][0])
            centre_ra_deg, centre_dec_deg = _get_phase_centre(Tigger_model)
            centre_coord =  deg2ra(centre_ra_deg) + ',' + deg2dec(centre_dec_deg)
            centre_coord = centre_coord.split(',')
        elif models[0][0].endswith('.fits'):
            centre_ra_deg, centre_dec_deg = fitsInfo(models[0][0])['centre']
            centre_coord =  deg2ra(centre_ra_deg) + ',' + deg2dec(centre_dec_deg)
            centre_coord = centre_coord.split(',')
        else:
            if args.centre_coord:
                centre_coord = args.centre_coord.split(',')
            else:
                LOGGER.error('Please supply central coordinates using -ptc. See --help')

        LOGGER.info(f'Quering the {online_catalog} catalog with width of {width} at {centre_coord}')
        table = get_online_catalog(catalog=online_catalog.upper(), centre_coord=centre_coord,
                                   width='5.0d', thresh=threshold, catalog_table=catalog_name)


        if table:
            for i, ims in enumerate(models):
                image1 = ims[0]
                if image1.endswith('.fits'):
                    configfile = 'default_sf_config.yml'
                    generate_default_config(configfile)
                    sf_params1 = get_sf_params(configfile)
                    sf_params1[sourcery]['filename'] = image1
                    out1 = source_finding(sf_params1, sourcery)
                    image1 = out1

                images_list.append(
                    [dict(label="{}-model_a_{}".format(args.label, i),
                          path=image1),
                     dict(label="{}-model_b_{}".format(args.label, i),
                          path=catalog_name)])

            output_dict = compare_models(images_list,
                                         tolerance=args.tolerance,
                                         shape_limit=args.shape_limit,
                                         off_axis=args.off_axis,
                                         all_sources=args.all,
                                         closest_only=args.closest_only,
                                         prefix=args.htmlprefix,
                                         flux_plot=args.fluxplot,
                                         ftitles=args.ftitles,
                                         fxlabels=args.fxlabels,
                                         fylabels=args.fylabels,
                                         title_size=args.tsize,
                                         x_label_size=args.xsize,
                                         y_label_size=args.ysize,
                                         legend_size=args.legsize,
                                         xmajor_size=args.xmaj_size,
                                         ymajor_size=args.ymaj_size,
                                         svg=svg)
        else:
            LOGGER.warn(f'No object found around (ICRS) position {centre_coord}')

    if args.subimage_noise:
        centre_coords = []
        output_dict = {}
        sizes = []
        if args.centre_pix_size:
            for cps in args.centre_pix_size[0]:
                centre_pix = (int(cps.split(',')[0]), int(cps.split(',')[1]))
                centre_coords.append(centre_pix)
                sizes.append(int(cps.split(',')[-1]))
            print(args.svg)
            output_dict = plot_subimage_stats(args.subimage_noise[0],
                                              centre_coords, sizes,
                                              units=args.units,
                                              svg=args.svg,
                                              title_size=args.tsize,
                                              x_label_size=args.xsize,
                                              y_label_size=args.ysize,
                                              bar_label_size=args.barsize,
                                              htmlprefix=(args.htmlprefix
                                              if args.htmlprefix else 'default'))
        else:
            LOGGER.error(f"{R}Provide Centre coordinates in pixels "
                         f"and size of subimage(s).{W}")

    if args.reg:
        centre_coords = []
        stats = get_region_stats(args.residual, args.reg)
        output_dict[residual_label] = stats

    if output_dict:
        if args.outfile:
            json_dump(output_dict, filename=args.outfile)
        else:
            json_dump(output_dict)
