import os
import json
import Tigger
import random
import string
import logging
import argparse
import tempfile
import numpy as np

from functools import partial
from collections import OrderedDict

from scipy import stats
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.ndimage import measurements as measure

from bokeh.models import CheckboxGroup, CustomJS
from bokeh.models import HoverTool, LinearAxis, Range1d
from bokeh.layouts import row, column, gridplot, widgetbox, grid
from bokeh.plotting import figure, output_file, show, save, ColumnDataSource

from astLib.astWCS import WCS
from astropy.table import Table
from astropy.io import fits as fitsio

from Tigger.Models import SkyModel, ModelClasses
from Tigger.Coordinates import angular_dist_pos_angle
from sklearn.metrics import mean_squared_error, r2_score


# Unit multipleirs for plotting
FLUX_UNIT_SCALER = {'jansky': [1e0, 'Jy'],
                    'milli': [1e3, 'mJy'],
                    'micro': [1e6, u'\u03bcJy'],
                    'nano': [1e9, 'nJy'],
}


POSITION_UNIT_SCALER = {'deg': [1e0, 'deg'],
                        'arcmin': [60.0, u'`'],
                        'arcsec': [3600.0, u'``'],
}


# Backgound color for plots
BG_COLOR = 'rgb(229,229,229)'


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


def get_aimfast_data(filename='fidelity_results.json', dir='.'):
    "Extracts data from the json data file"
    file = '{:s}/{:s}'.format(dir, filename)
    LOGGER.info('Extracting data from the json data file')
    with open(file) as f:
        data = json.load(f)
        return data


def deg2arcsec(x):
    """Converts 'x' from degrees to arcseconds."""
    result = float(x) * 3600.00
    return result


def rad2deg(x):
    """Converts 'x' from radian to degrees."""
    result = float(x) * (180 / np.pi)
    return result


def rad2arcsec(x):
    """Converts `x` from radians to arcseconds."""
    result = float(x) * (3600.0 * 180.0 / np.pi)
    return result


def json_dump(data_dict, root='.'):
    """Dumps the computed dictionary results into a json file.

    Parameters
    ----------
    data_dict : dict
        Dictionary with output results to save.
    root : str
        Directory to save output json file (default is current directory).

    Note1
    ----
    If the fidelity_results.json file exists, it will be append, and only
    repeated image assessments will be replaced.

    """
    filename = ('{:s}/fidelity_results.json'.format(root))
    LOGGER.info("Dumping results into the '{}' file".format(filename))
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
    wcs = WCS(hdr, mode='pyfits')
    numPix = hdr['NAXIS1']
    try:
        beam_size = (hdr['BMAJ'], hdr['BMIN'], hdr['BPA'])
    except:
        beam_size = None
    try:
        centre = '{0},{1},{2}'.format('J' + str(hdr['EQUINOX']),
                                      str(hdr['CRVAL1']) + hdr['CUNIT1'],
                                      str(hdr['CRVAL2']) + hdr['CUNIT2'])
    except:
        centre = 'J2000.0,0.0deg,0.0deg'
    skyArea = (numPix * ddec)**2
    fitsinfo = {'wcs': wcs, 'ra': ra, 'dec': dec,
                'dra': dra, 'ddec': ddec, 'raPix': raPix,
                'decPix': decPix, 'b_size': beam_size,
                'numPix': numPix, 'centre': centre,
                'skyArea': skyArea}
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
            LOGGER.info("PSF FWHM over {:.2f} arcsec".format(arcsec_size * 2))
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
    wcs : astLib.astWCS.WCS instance
        World Coordinate System.

    Returns
    -------
    box : tuple
        A box centered at radec.

    """
    raPix, decPix = wcs.wcs2pix(*radec)
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


def _get_ra_dec_range(area, phase_centre="J2000,0deg,-30deg"):
    """Get RA and DEC range from area of observations and phase centre"""
    ra = float(phase_centre.split(',')[1].split('deg')[0])
    dec = float(phase_centre.split(',')[2].split('deg')[0])
    d_ra = np.sqrt(area) / 2.0
    d_dec = np.sqrt(area) / 2.0
    ra_range = [ra - d_ra, ra + d_ra]
    dec_range = [dec - d_dec, dec + d_dec]
    return ra_range, dec_range


def _get_random_pixel_coord(num, sky_area, phase_centre="J2000,0deg,-30deg"):
    """Provides random pixel coordinates

    Parameters
    ----------
    num: int
        Number of data points
    sky: float
        Sky area to extract random points
    phase_centre: str
        Phase tracking centre of the telescope during observation

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


def residual_image_stats(fitsname, test_normality=None, data_range=None,
                         threshold=None, chans=None, mask=None,
                         step_size=None, window_size=None):
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
        Channels to compute stats (e.g. 0~50;100~200)
    mask : file
        Fits mask to get stats in image
    window_size : int
        Window size to compute rms
    step_size : int
        Step size of sliding window

    Returns
    -------
    props : dict
        Dictionary of stats properties.
        e.g. {'MEAN': 0.0, 'STDDev': 0.1, 'RMS': 0.1,
              'SKEW': 0.2, 'KURT': 0.3, 'MAD': 0.4,
              'SLIDING_STDDev': 0.5}.

    Notes
    -----
    If normality_test=True, dictionary of stats props becomes \
    e.g. {'MEAN': 0.0, 'STDDev': 0.1, 'SKEW': 0.2, 'KURT': 0.3, \
          'MAD': 0.4, 'RMS': 0.5, 'SLIDING_STDDev': 0.6,
          'NORM': (123.3,0.012)} \
    whereby the first element is the statistics (or average if data_range specified) \
    of the datasets and second element is the p-value.

    """
    res_props = dict()
    # Open the residual image
    residual_hdu = fitsio.open(fitsname)
    # Get the header data unit for the residual rms
    residual_data = residual_hdu[0].data

    # Get residual data
    data = residual_data[0]
    if chans:
        nchans = []
        chan_ranges = chans.split(';')
        for cr in chan_ranges:
            c = cr.split('~')
            nchans.extend(range(int(c[0]), int(c[1])))
            residual_data = data[nchans]
    if threshold:
        nchans = []
        for i in range(data.shape[0]):
            d = data[i][data[i] > float(threshold)]
            if d.shape[0] > 0:
                nchans.append(i)
        import IPython; IPython.embed()
        residual_data = data[nchans]
    if mask:
        import numpy.ma as ma
        mask_hdu = fitsio.open(mask)
        mask_data = mask_hdu[0].data
        residual_data = ma.masked_array(residual_data, mask=mask_data)

    # Get the mean value
    LOGGER.info("Computing mean ...")
    res_props['MEAN'] = float("{0:.6}".format(residual_data.mean()))
    LOGGER.info("MEAN = {}".format(res_props['MEAN']))
    # Get the rms value
    LOGGER.info("Computing root mean square ...")
    res_props['RMS'] = float("{0:.6f}".format(np.sqrt(np.mean(np.square(residual_data)))))
    LOGGER.info("RMS = {}".format(res_props['RMS']))
    # Get the sigma value
    LOGGER.info("Computing standard deviation ...")
    res_props['STDDev'] = float("{0:.6f}".format(residual_data.std()))
    LOGGER.info("STDDev = {}".format(res_props['STDDev']))
    # Flatten image
    res_data = residual_data.flatten()
    # Get the maximum absolute deviation
    LOGGER.info("Computing median absolute deviation ...")
    res_props['MAD'] = float("{0:.6f}".format(stats.median_absolute_deviation(res_data)))
    LOGGER.info("MAD = {}".format(res_props['MAD']))
    # Compute the skewness of the residual
    LOGGER.info("Computing skewness ...")
    res_props['SKEW'] = float("{0:.6f}".format(stats.skew(res_data)))
    LOGGER.info("SKEW = {}".format(res_props['SKEW']))
    # Compute the kurtosis of the residual
    LOGGER.info("Computing kurtosis ...")
    res_props['KURT'] = float("{0:.6f}".format(stats.kurtosis(res_data, fisher=False)))
    LOGGER.info("KURT = {}".format(res_props['KURT']))
    # Compute sliding window sigma
    LOGGER.info("Computing sliding window standard deviation ...")
    res_props['SLIDING_STDDev'] = float("{0:.6f}".format(sliding_window_std(
                                                         residual_data,
                                                         window_size,
                                                         step_size)))
    LOGGER.info("SLIDING_STDDev = {}".format(res_props['SLIDING_STDDev']))
    # Perform normality testing
    if test_normality:
        LOGGER.info("Performing normality test ...")
        norm_props = normality_testing(res_data, test_normality, data_range)
        res_props.update(norm_props)
        LOGGER.info("NORM = {}".format(res_props['NORM']))
    props = res_props
    # Return dictionary of results
    return props


def sliding_window_std(data, window_size=20, step_size=1):
    """Gets the standard deviation of the sliding window boxes pixel values

    Parameters
    ----------
    data : numpy.array
        Residual residual array. i.e. fitsio.open(fitsname)[0].data
    window_size : int
        Window size to compute rms
    step_size : int
        Step size of sliding window

    Returns
    -------
    w_std : float
        Standard deviation of the windows.

    """
    windows_avg = []
    # Get residual image data
    residual_data = data
    # Get the number of frequency channels
    nchan = (residual_data.shape[1]
             if residual_data.shape[0] == 1
             else residual_data.shape[0])
    # Define a nxn window
    (w_width, w_height) = (int(window_size), int(window_size))
    # Check if window is less than image size
    image_size = residual_data.shape[-1]
    if int(window_size) > image_size:
        raise Exception("Window size of {} should be less than image size {}".format(
                         window_size, image_size))
    for frq_ax in range(nchan):
        # In case the first two axes are swapped
        if residual_data.shape[0] == 1:
            image = residual_data[0, frq_ax, :, :]
        else:
            image = residual_data[frq_ax, 0, :, :]
        for x in range(0, image.shape[1] - w_width + 1, int(step_size)):
            for y in range(0, image.shape[0] - w_height + 1, int(step_size)):
                window = image[x:x + w_width, y:y + w_height]
                windows_avg.append(np.array(window).mean())
    return np.array(windows_avg).std()


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
        raise ValueError("{:s}No data to compute stats."
                         "\nEither threshold too high "
                         "or all data is masked.{:s}".format(R, W))
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
    wcs = WCS(residual_hdu[0].header, mode="pyfits")
    width = int(beam_size * area_factor)
    imslice = get_box(wcs, (RA, DEC), width)
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

    def tigger_src_fits(src, idx):
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

        if ex and ey:
            shape = ModelClasses.Gaussian(ex, ey, pa, ex_err=ex_err,
                                          ey_err=ey_err, pa_err=pa_err)
        else:
            shape = None
        source = SkyModel.Source(name, pos, flux, shape=shape)
        # Adding source peak flux (error) as extra flux attributes for sources,
        # and to avoid null values for point sources I_peak = src["Total_flux"]
        if shape:
            source.setAttribute("I_peak", src["Peak_flux"])
            source.setAttribute("I_peak_err", src["E_peak_flux"])
        else:
            source.setAttribute("I_peak", src["Total_flux"])
            source.setAttribute("I_peak_err", src["E_Total_flux"])

        return source

    tfile = tempfile.NamedTemporaryFile(suffix='.txt')
    tfile.flush()
    with open(tfile.name, "w") as stdw:
        stdw.write("#format:name ra_d dec_d i emaj_s emin_s pa_d\n")
    model = Tigger.load(tfile.name)
    tfile.close()
    ext = os.path.splitext(catalog)[-1]
    if ext in ['.html', '.txt']:
        model = Tigger.load(catalog)
    if ext in ['.tab', '.csv']:
        data = Table.read(catalog, format='ascii')
        for i, src in enumerate(data):
            model.sources.append(tigger_src_ascii(src, i))
    if ext in ['.fits']:
        data = Table.read(catalog, format='fits')
        for i, src in enumerate(data):
            model.sources.append(tigger_src_fits(src, i))
    return model


def get_detected_sources_properties(model_1, model_2, area_factor,
                                    phase_centre=None, all_sources=False):
    """Extracts the output simulation sources properties.

    Parameters
    ----------
    models_1 : file
        Tigger formatted or txt model 1 file.
    models_2 : file
        Tigger formatted or txt model 2 file.
    area_factor : float
        Area factor to multiply the psf size around source.
    phase_centre : str
        Phase centre of catalog (if not already embeded)
    all_source: bool
        Compare all sources in the catalog (else only point-like source)

    Returns
    -------
    (targets_flux, targets_scale, targets_position) : tuple
        Tuple of target flux, morphology and astrometry information

    """
    model_lsm = get_model(model_1)
    pybdsm_lsm = get_model(model_2)
    # Sources from the input model
    model_sources = model_lsm.sources
    # {"source_name": [I_out, I_out_err, I_in, source_name]}
    targets_flux = dict()       # recovered sources flux
    # {"source_name": [delta_pos_angle_arc_sec, ra_offset, dec_offset,
    #                  delta_phase_centre_arc_sec, I_in, source_name]
    targets_position = dict()   # recovered sources position
    # {"source_name: [shape_out=(maj, min, angle), shape_out_err=, shape_in=,
    #                 scale_out, scale_out_err, I_in, source_name]
    targets_scale = dict()         # recovered sources scale
    for model_source in model_sources:
        I_out = 0.0
        I_out_err = 0.0
        name = model_source.name
        RA = model_source.pos.ra
        DEC = model_source.pos.dec
        I_in = model_source.flux.I
        sources = pybdsm_lsm.getSourcesNear(RA, DEC, area_factor)
        # More than one source detected, thus we sum up all the detected sources
        # within a radius equal to the beam size in radians around the true target
        # coordinate
        I_out_err_list = []
        I_out_list = []
        for target in sources:
            I_out_list.append(target.flux.I)
            I_out_err_list.append(target.flux.I_err * target.flux.I_err)
        I_out = sum([val / err for val, err in zip(I_out_list, I_out_err_list)])
        if I_out != 0.0:
            source = sources[0]
            try:
                shape_in = model_source.shape.getShape()
            except AttributeError:
                shape_in = (0, 0, 0)
            if source.shape:
                shape_out = tuple(map(rad2arcsec, source.shape.getShape()))
                shape_out_err = tuple(map(rad2arcsec, source.shape.getShapeErr()))
            else:
                shape_out = (0, 0, 0)
                shape_out_err = (0, 0, 0)
            if not all_sources:
                if shape_out[0] > 2.0:
                    continue
            I_out_err = sum([1.0 / I_out_error for I_out_error
                            in I_out_err_list])
            I_out_var_err = np.sqrt(1.0 / I_out_err)
            I_out = I_out / I_out_err
            I_out_err = I_out_var_err
            RA0 = pybdsm_lsm.ra0
            DEC0 = pybdsm_lsm.dec0
            if phase_centre:
                RA0 = np.deg2rad(float(phase_centre.split(',')[1].split('deg')[0]))
                DEC0 = np.deg2rad(float(phase_centre.split(',')[-1].split('deg')[0]))
            ra = source.pos.ra
            dec = source.pos.dec
            ra_err = source.pos.ra_err
            dec_err = source.pos.dec_err
            source_name = source.name
            targets_flux[name] = [I_out, I_out_err, I_in, source_name]
            if ra > np.pi:
                ra -= 2.0*np.pi
            if RA > np.pi:
                RA -= 2.0*np.pi
            delta_pos_angle_arc_sec = angular_dist_pos_angle(
                rad2arcsec(RA), rad2arcsec(DEC),
                rad2arcsec(ra), rad2arcsec(dec))[0]
            delta_pos_angle_arc_sec = float('{0:.7f}'.format(delta_pos_angle_arc_sec))
            if RA0 or DEC0:
                delta_phase_centre = angular_dist_pos_angle(RA0, DEC0, ra, dec)
                delta_phase_centre_arc_sec = rad2arcsec(delta_phase_centre[0])
            else:
                delta_phase_centre_arc_sec = None
            targets_position[name] = [delta_pos_angle_arc_sec,
                                      rad2arcsec(ra - RA),
                                      rad2arcsec(dec - DEC),
                                      delta_phase_centre_arc_sec, I_in,
                                      rad2arcsec(ra_err),
                                      rad2arcsec(dec_err),
                                      source_name]
            src_scale = get_src_scale(source.shape)
            targets_scale[name] = [shape_out, shape_out_err, shape_in,
                                   src_scale[0], src_scale[1], I_in,
                                   source_name]
    LOGGER.info("Number of sources recovered: {:d}".format(len(targets_scale)))
    return targets_flux, targets_scale, targets_position


def compare_models(models, tolerance=0.000001, plot=True, phase_centre=None,
                   all_sources=False):
    """Plot model1 source properties against that of model2

    Parameters
    ----------
    models : dict
        Tigger formatted model files e.g {model1: model2}.
    tolerance : float
        Tolerace in detecting source from model 2.
    plot : bool
        Output html plot from which a png can be obtained.
    phase_centre : str
        Phase centre of catalog (if not already embeded)
    all_source: bool
        Compare all sources in the catalog (else only point-like source)

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
        props = get_detected_sources_properties('{}'.format(input_model["path"]),
                                                '{}'.format(output_model["path"]),
                                                tolerance, phase_centre,
                                                all_sources)
        for i in range(len(props[0])):
            flux_prop = list(props[0].items())
            results[heading]['flux'].append(flux_prop[i][-1])
        for i in range(len(props[1])):
            shape_prop = list(props[1].items())
            results[heading]['shape'].append(shape_prop[i][-1])
        for i in range(len(props[2])):
            pos_prop = list(props[2].items())
            results[heading]['position'].append(pos_prop[i][-1])
    if plot:
        _source_flux_plotter(results, models)
        _source_astrometry_plotter(results, models)
    return results


def compare_residuals(residuals, skymodel=None, points=None,
                      inline=False, area_factor=2.0):
    if skymodel:
        res = _source_residual_results(residuals, skymodel, area_factor)
    else:
        res = _random_residual_results(residuals, points)
    _residual_plotter(residuals, results=res, points=points, inline=inline)
    return res


def plot_photometry(models, label=None, tolerance=0.00001, phase_centre=None,
                    all_sources=False):
    """Plot model-model fluxes from lsm.html/txt models

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
    results = compare_models(_models, tolerance, False, phase_centre, all_sources)
    _source_flux_plotter(results, _models, inline=True)


def plot_astrometry(models, label=None, tolerance=0.00001, phase_centre=None,
                    all_sources=False):
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
        _models.append([dict(label="{0:s}-model_a_{1:d}".format(label, i), path=model1),
                        dict(label="{0:s}-model_b_{1:d}".format(label, i), path=model2)])
        i += 1
    results = compare_models(_models, tolerance, False, phase_centre, all_sources)
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
        _residual_images.append([dict(label="{0:s}-res_a_{1:d}".format(label, i), path=res1),
                                 dict(label="{0:s}-res_b_{1:d}".format(label, i), path=res2)])
        i += 1
    compare_residuals(_residual_images, skymodel, points, True, area_factor)


def _source_flux_plotter(results, all_models, inline=False, units='milli'):
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

    """
    outfile = 'InputOutputFluxDensity.html'
    output_file(outfile)
    flux_plot_list = []
    for model_pair in  all_models:
        name_labels = []
        flux_in_data = []
        flux_out_data = []
        source_scale = []
        phase_center_dist = []
        flux_out_err_data = []
        heading = model_pair[0]['label']
        for n in range(len(results[heading]['flux'])):
            flux_out_data.append(results[heading]['flux'][n][0])
            flux_out_err_data.append(results[heading]['flux'][n][1])
            flux_in_data.append(results[heading]['flux'][n][2])
            name_labels.append(results[heading]['flux'][n][3])
            phase_center_dist.append(results[heading]['position'][n][3])
            source_scale.append(results[heading]['shape'][n][3])
        # Compute some fit stats of the two models being compared
        flux_MSE = mean_squared_error(flux_in_data, flux_out_data)
        reg = linregress(flux_in_data, flux_out_data)
        flux_R_score = reg.rvalue
        # Format data points value to a readable units
        x = np.array(flux_in_data)*FLUX_UNIT_SCALER[units][0]
        y = np.array(flux_out_data)*FLUX_UNIT_SCALER[units][0]
        # Create additional feature on the plot such as hover, display text
        TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
        source = ColumnDataSource(
                    data=dict(x=x, y=y,
                              label=["%s X %s" % (x_, y_) for x_, y_ in zip(x, y)]))
        text="Slope: {:.4f} | Intercept: {:.4f} | RMS Error: {:.4f} | R2: {:.4f} ".format(
            reg.slope, reg.intercept * FLUX_UNIT_SCALER[units][0],
            np.sqrt(flux_MSE) * FLUX_UNIT_SCALER[units][0], flux_R_score)
        # Create a plot object
        plot_flux = figure(title='[ {} ]'.format(text),
                           x_axis_label='Input flux ({:s})'.format(
                               FLUX_UNIT_SCALER[units][1]),
                           y_axis_label='Output flux ({:s})'.format(
                               FLUX_UNIT_SCALER[units][1]),
                           tools=TOOLS)
        # Get errors from the output fluxes
        err_xs = []
        err_ys = []
        for x, y, yerr in zip(np.array(flux_in_data)*FLUX_UNIT_SCALER[units][0],
                              np.array(flux_out_data)*FLUX_UNIT_SCALER[units][0],
                              np.array(flux_out_err_data)*FLUX_UNIT_SCALER[units][0]):
            err_xs.append((x, x)) # TODO: Also if the main model has error plot them (+)
            err_ys.append((y - yerr, y + yerr))
        # Create a plot object for errors
        errors = plot_flux.multi_line(err_xs, err_ys, color='red')
        # Create a plot object for I_out = I_in line .i.e. Perfect match
        equal = plot_flux.line(np.array([flux_in_data[0],
                                 flux_in_data[-1]])*FLUX_UNIT_SCALER[units][0],
                       np.array([flux_in_data[0],
                                 flux_in_data[-1]])*FLUX_UNIT_SCALER[units][0])
        # Create a plot object for the data points
        data = plot_flux.circle('x', 'y', source=source)
        # Create checkboxes to hide and display error and fits
        checkbox = CheckboxGroup(labels=["Errors", "I_out=I_in", "Fit"],
                                 active=[0, 1, 2], width=100)
        checkbox.callback = CustomJS(args=dict(errors=errors,
                                               equal=equal,
                                               data=data,
                                               checkbox=checkbox),
                                     code=""" 
                                          if (cb_obj.active.includes(0)) {
                                            errors.visible = true;
                                          } else {
                                            errors.visible = false;
                                          }
                                          if (cb_obj.active.includes(1)) {
                                            equal.visible = true;
                                          } else {
                                            equal.visible = false;
                                          }
                                          if (cb_obj.active.includes(2)) {
                                            data.visible = true;
                                          } else {
                                            data.visible = false;
                                          }
                                          """)
        # Attaching the hover object with labels
        hover = plot_flux.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("index", "$index"),
            ("(x,y)", "(@x, @y)"),
            ("label", "@label")])
        # Append all plots and checkboxes
        flux_plot_list.append(row(plot_flux, widgetbox(checkbox)))
    # Make the plots in a column layout
    flux_plots = column(flux_plot_list)
    # Save the plot (html)
    save(flux_plots, title=outfile)
    LOGGER.info('Saving photometry comparisons in {}'.format(outfile))


def _source_astrometry_plotter(results, all_models, inline=False, units=''):
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

    """
    outfile = 'InputOutputPosition.html'
    output_file(outfile)
    position_plot_list = []
    for model_pair in  all_models:
        RA_offset = []
        RA_err = []
        DEC_offset = []
        DEC_err = []
        DELTA_PHASE0 = []
        source_labels = []
        flux_in_data = []
        flux_out_data = []
        delta_pos_data = []
        heading = model_pair[0]['label']
        for n in range(len(results[heading]['flux'])):
            flux_out_data.append(results[heading]['flux'][n][0])
            delta_pos_data.append(results[heading]['position'][n][0])
            RA_offset.append(results[heading]['position'][n][1])
            DEC_offset.append(results[heading]['position'][n][2])
            DELTA_PHASE0.append(results[heading]['position'][n][3])
            flux_in_data.append(results[heading]['position'][n][4])
            RA_err.append(results[heading]['position'][n][5])
            DEC_err.append(results[heading]['position'][n][6])
            source_labels.append(results[heading]['position'][n][7])
        # Compute some stats of the two models being compared
        RA_mean = np.mean(RA_offset)
        DEC_mean = np.mean(DEC_offset)
        r1, r2 = np.array(RA_offset).std(), np.array(DEC_offset).std()
        # Generate data for a sigma circle around data points 
        pi, cos, sin = np.pi, np.cos, np.sin
        theta = np.linspace(0, 2.0 * pi, len(DEC_offset))
        x1 = RA_mean+(r1 * cos(theta))
        y1 = DEC_mean+(r2 * sin(theta))
        # Get the number of sources recovered and within 1 sigma
        recovered_sources = len(DEC_offset)
        one_sigma_sources = len([
            (ra_off, dec_off) for ra_off, dec_off in zip(RA_offset, DEC_offset)
            if abs(ra_off) <= max(abs(x1)) and abs(dec_off) <= max(abs(y1))])
        # Format data list into numpy arrays
        x = np.array(RA_offset)
        y = np.array(DEC_offset)
        flux_in =  np.array(flux_in_data)*FLUX_UNIT_SCALER['milli'][0] # For color
        phase_centre_distance = DELTA_PHASE0 # For color
        # Create additional feature on the plot such as hover, display text
        TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
        source = ColumnDataSource(
                    data=dict(x=x, y=y,
                              label=["%s X %s" % (x_, y_) for x_, y_ in zip(x, y)]))
        text="Total sources: {:d} | (RA, DEC) mean: ({:.4f}, {:.4f})".format(
                 recovered_sources, RA_mean, DEC_mean)
        # Create a plot object
        plot_position = figure(title='[ {} ]'.format(text),
                           x_axis_label='RA offset ({:s})'.format(
                               POSITION_UNIT_SCALER['arcsec'][1]),
                           y_axis_label='DEC offset ({:s})'.format(
                               POSITION_UNIT_SCALER['arcsec'][1]),
                           tools=TOOLS)
        # Get errors from the output positions
        err_xs = []
        err_ys = []
        for x, y, xerr, yerr in zip(x, y, np.array(RA_err), np.array(DEC_err)):
            err_xs.append((x - xerr, x + xerr))
            err_ys.append((y - yerr, y + yerr))

        # Creat an sigma circle plot object
        plot_position.line(np.array(x1), np.array(y1))
        # Create position data points plot object
        plot_position.circle('x', 'y', source=source)
        # Create a plot object for errors
        plot_position.multi_line(err_xs, err_ys, color='red')
        # Attaching the hover object with labels
        hover = plot_position.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("index", "$index"),
            ("(x,y)", "(@x, @y)"),
            ("label", "@label")])
        position_plot_list.append(plot_position)
    # Make the plots in a column layout
    position_plots = column(position_plot_list)
    # Save the plot (html)
    save(position_plots, title=outfile)
    LOGGER.info('Saving astrometry comparisons in {}'.format(outfile))


def _residual_plotter(res_noise_images, points=None, results=None, inline=False):
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

    """
    if points:
        outfile = 'RandomResidualNoiseRatio.html'
    else:
        outfile = 'SourceResidualNoiseRatio.html'
    output_file(outfile)
    residual_plot_list = []
    for residual_pair in res_noise_images:
        residuals1 = []
        residuals2 = []
        name_labels = []
        dist_from_phase = []
        res_noise_ratio = []
        res_image = residual_pair[0]['label']
        for res_src in results[res_image]:
            residuals1.append(res_src[0])
            residuals2.append(res_src[1])
            res_noise_ratio.append(res_src[2])
            dist_from_phase.append(res_src[3])
            name_labels.append(res_src[4])
        # Get sigma value of residuals
        res1 = np.array(residuals1) * FLUX_UNIT_SCALER['micro'][0]
        res2 = np.array(residuals2) * FLUX_UNIT_SCALER['micro'][0]
        # Get ratio data
        y1 = np.array(res_noise_ratio)
        x1 = np.array(range(len(res_noise_ratio)))
        # Create additional feature on the plot such as hover, display text
        TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
        source = ColumnDataSource(
                    data=dict(x=x1, y=y1,
                              label=["%s X %s" % (x_, y_) for x_, y_ in zip(x1, y1)]))
        text="residuals1: {:.4f} | residuals2: {:.4f} | res1-to-res2: {:.4f}".format(
                 np.mean(residuals1) * FLUX_UNIT_SCALER['micro'][0],
                 np.mean(residuals2) * FLUX_UNIT_SCALER['micro'][0],
                 np.mean(residuals2) / np.mean(residuals1))
        # Get y2 label and range
        y2_label = "Flux ({})".format(FLUX_UNIT_SCALER['micro'][1])
        y_max = max(res1) if max(res1) > max(res2) else max(res2)
        y_min = min(res1) if min(res1) < min(res2) else min(res2)
        # Create a plot objects and set axis limits
        plot_residual = figure(title='[ {} ]'.format(text),
                               x_axis_label='Sources',
                               y_axis_label='Res1-to-Res2',
                               tools=TOOLS)
        plot_residual.y_range = Range1d(start=min(y1) - .01, end=max(y1) + .01)
        plot_residual.extra_y_ranges = {y2_label: Range1d(start=y_min - .01 * abs(y_min),
                                                          end=y_max + .01 * abs(y_max))}
        plot_residual.add_layout(LinearAxis(y_range_name=y2_label,
                                            axis_label=y2_label),
                                 'right')
        res_ratio_object = plot_residual.line(x1, y1, color='green')
        res1_object = plot_residual.line(x1, res1, color='red', y_range_name=y2_label)
        res2_object = plot_residual.line(x1, res2, color='blue', y_range_name=y2_label)
        # Create checkboxes to hide and display error and fits
        checkbox = CheckboxGroup(labels=["res1-to-res2", "residual1", "residual2"],
                                 active=[0, 1, 2], width=100)
        checkbox.callback = CustomJS(args=dict(res_ratio_object=res_ratio_object,
                                               res1_object=res1_object,
                                               res2_object=res2_object,
                                               checkbox=checkbox),
                                     code=""" 
                                          if (cb_obj.active.includes(0)) {
                                            res_ratio_object.visible = true;
                                          } else {
                                            res_ratio_object.visible = false;
                                          }
                                          if (cb_obj.active.includes(1)) {
                                            res1_object.visible = true;
                                          } else {
                                            res1_object.visible = false;
                                          }
                                          if (cb_obj.active.includes(2)) {
                                            res2_object.visible = true;
                                          } else {
                                            res2_object.visible = false;
                                          }
                                          """)
        # Attaching the hover object with labels
        hover = plot_residual.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("index", "$index"),
            ("(xx,yy)", "(@x, @y)"),
            ("label", "@label")])
        residual_plot_list.append(row(plot_residual, checkbox))
    # Make the plots in a column layout
    residual_plots = column(residual_plot_list)
    # Save the plot (html)
    save(residual_plots, title=outfile)
    LOGGER.info('Saving residual comparision plots {}'.format(outfile))


def _random_residual_results(res_noise_images, data_points=100, area_factor=2):
    """Plot ratios of random residuals and noise

    Parameters
    ----------
    res_noise_images: list
        List of dictionaries with residual images
    data_points: int
        Number of data points to extract
    area_factor : float
        Factor to multiply the beam area.

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
        pix_coord_deg = _get_random_pixel_coord(data_points,
                                                sky_area=fits_info['skyArea'] * 0.9,
                                                phase_centre=fits_info['centre'])
        # Get the number of frequency channels
        nchan = (res_data1.shape[1]
                 if res_data1.shape[0] == 1
                 else res_data1.shape[0])
        for RA, DEC in pix_coord_deg:
            i += 1
            # Get width of box around source
            width = int(deg2arcsec(beam_deg[0]) * area_factor)
            # Get a image slice around source
            imslice = get_box(fits_info["wcs"], (RA, DEC), width)
            # Get noise rms in the box around the point coordinate
            res1_area = res_data1[0, 0, :, :][imslice]
            res1_rms = res1_area.std()
            res2_area = res_data1[0, 0, :, :][imslice]
            res2_rms = res2_area.std()
            # if image is cube then average along freq axis
            if nchan > 1:
                flux_rms1 = 0.0
                flux_rms2 = 0.0
                for frq_ax in range(nchan):
                    # In case the first two axes are swapped
                    if res_data1.shape[0] == 1:
                        target_area1 = res_data1[0, frq_ax, :, :][imslice]
                    else:
                        target_area1 = res_data1[frq_ax, 0, :, :][imslice]
                    if res_data2.shape[0] == 1:
                        target_area2 = res_data2[0, frq_ax, :, :][imslice]
                    else:
                        target_area2 = res_data2[frq_ax, 0, :, :][imslice]
                    # Sum of all the fluxes
                    flux_rms1 += target_area1.std()
                    flux_rms2 += target_area2.std()
                # Get the average std and mean along all frequency channels
                res1_rms = flux_rms1/float(nchan)
                res2_rms = flux_rms2/float(nchan)
            # Get phase centre and determine phase centre distance
            RA0 = float(fits_info['centre'].split(',')[1].split('deg')[0])
            DEC0 = float(fits_info['centre'].split(',')[-1].split('deg')[0])
            phase_dist_arcsec = deg2arcsec(np.sqrt((RA-RA0)**2 + (DEC-DEC0)**2))
            # Store all outputs in the results data structure
            results[res_label1].append([res1_rms*1e0,
                                       res2_rms*1e0,
                                       res2_rms/res1_rms*1e0,
                                       phase_dist_arcsec,
                                       'source{0}'.format(i)])
    return results


def _source_residual_results(res_noise_images, skymodel, area_factor=2):
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
    # Get beam size otherwise use default (5``).
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
        # In case the images was not deconvloved aslo use default beam
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
        nchan = (res_data1.shape[1]
                 if res_data1.shape[0] == 1
                 else res_data1.shape[0])
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
            # Get width of box around source
            width = int(deg2arcsec(beam_deg[0]) * area_factor)
            # Get a image slice around source
            imslice = get_box(fits_info["wcs"], (RA, DEC), width)
            # Get noise rms in the box around the point coordinate
            res1_area = res_data1[0, 0, :, :][imslice]
            res1_rms = res1_area.std()
            res2_area = res_data1[0, 0, :, :][imslice]
            res2_rms = res2_area.std()
            # if image is cube then average along freq axis
            if nchan > 1:
                flux_rms1 = 0.0
                flux_rms2 = 0.0
                for frq_ax in range(nchan):
                    # In case the first two axes are swapped
                    if res_data1.shape[0] == 1:
                        target_area1 = res_data1[0, frq_ax, :, :][imslice]
                    else:
                        target_area1 = res_data1[frq_ax, 0, :, :][imslice]
                    if res_data2.shape[0] == 1:
                        target_area2 = res_data2[0, frq_ax, :, :][imslice]
                    else:
                        target_area2 = res_data2[frq_ax, 0, :, :][imslice]
                    # Sum of all the fluxes
                    flux_rms1 += target_area1.std()
                    flux_rms2 += target_area2.std()
                # Get the average std and mean along all frequency channels
                res1_rms = flux_rms1/float(nchan)
                res2_rms = flux_rms2/float(nchan)
            # Store all outputs in the results data structure
            results[res_label1].append([res1_rms * 1e0,
                                       res2_rms * 1e0,
                                       res2_rms / res1_rms * 1e0,
                                       phase_dist_arcsec,
                                       model_source.name,
                                       model_source.flux.I])
    return results


def get_argparser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description=("Examine radio image fidelity by obtaining: \n"
                     "- The four (4) moments of a residual image \n"
                     "- The Dynamic range in restored image \n"
                     "- Comparing the tigger input and output model sources \n"
                     "- Comparing the on source/random residuals to noise"))
    argument = partial(parser.add_argument)
    argument('--tigger-model', dest='model',
             help='Name of the tigger model lsm.html file')
    argument('--restored-image', dest='restored',
             help='Name of the restored image fits file')
    argument('-psf', '--psf-image', dest='psf',
             help='Name of the point spread function file or psf size in arcsec')
    argument('--residual-image', dest='residual',
             help='Name of the residual image fits file')
    argument('--mask-image', dest='mask',
             help='Name of the mask image fits file')
    argument('--normality-test', dest='test_normality',
             choices=('shapiro', 'normaltest'),
             help='Name of model to use for normality testing. \n'
                  'options: [shapiro, normaltest] \n'
                  'NB: normaltest is the D`Agostino')
    argument('-dr', '--data-range', dest='data_range',
             help='Data range to perform normality testing')
    argument('-af', '--area-factor', dest='factor', type=float, default=6,
             help='Factor to multiply the beam area to get target peak area')
    argument('-as', '--all-source', dest='all', default=False, action='store_true',
             help='Compare all sources irrespective of shape, otherwise only '
                  'point-like sources are compared')
    argument('--compare-models', dest='models', nargs="+", type=str,
             help='List of tigger model (text/lsm.html) files to compare \n'
                  'e.g. --compare-models model1.lsm.html model2.lsm.html')
    argument('--compare-residuals', dest='noise', nargs="+", type=str,
             help='List of noise-like (fits) files to compare \n'
                  'e.g. --compare-residuals residuals.fits noise.fits')
    argument('-dp', '--data-points', dest='points',
             help='Data points to sample the residual/noise image')
    argument('-ptc', '--phase-centre', dest='phase',
             help='Phase tracking centre of the catalogs e.g. "J2000.0,0.0deg,-30.0"')
    argument('-thresh', '--threshold', dest='thresh',
             help='Get stats of channels with pixel flux above thresh in Jy/Beam')
    argument('-chans', '--channels', dest='channels',
             help='Get stats of specified channels e.g. "10~20;100~1000"')
    argument('-ws', '--window-size', dest='window', default=20,
             help='Window size to compute rms')
    argument('-ss', '--step-size', dest='step', default=1,
             help='Step size of sliding window')
    argument("--label",
             help='Use this label instead of the FITS image path when saving '
                  'data as JSON file')
    argument("--outfile",
             help='Name of output file name. Default: fidelity_results.json')
    return parser


def main():
    """Main function."""
    LOGGER.info("Welcome to AIMfast")
    parser = get_argparser()
    args = parser.parse_args()
    output_dict = dict()
    R = '\033[31m'  # red
    W = '\033[0m'   # white (normal)
    if not args.residual and not args.restored and not args.model \
            and not args.models and not args.noise:
        print("{:s}Please provide lsm.html/fits file name(s)."
              "\nOr\naimfast -h for arguments{:s}".format(R, W))

    if args.label:
        residual_label = "{0:s}-residual".format(args.label)
        restored_label = "{0:s}-restored".format(args.label)
        model_label = "{0:s}-model".format(args.label)
    else:
        residual_label = args.residual
        restored_label = args.restored
        model_label = args.model

    if args.model and not args.noise:
        if not args.residual:
            raise RuntimeError("{:s}Please provide residual fits file{:s}".format(R, W))

        if args.psf:
            if isinstance(args.psf, (str, unicode)):
                psf_size = measure_psf(args.psf)
            else:
                psf_size = int(args.psf)
        else:
            psf_size = 5

        if args.factor:
            DR = model_dynamic_range(args.model, args.residual, psf_size,
                                     area_factor=args.factor)
        else:
            DR = model_dynamic_range(args.model, args.residual, psf_size)
            print("{:s}Please provide psf fits file or psf size.\n"
                  "Otherwise a default beam size of six (~6``) asec "
                  "is used{:s}".format(R, W))

        if args.test_normality in ['shapiro', 'normaltest']:
            stats = residual_image_stats(args.residual,
                                         args.test_normality,
                                         args.data_range,
                                         args.thresh,
                                         args.channels,
                                         args.mask,
                                         args.step,
                                         args.window)
        else:
            if not args.test_normality:
                stats = residual_image_stats(args.residual,
                                             args.test_normality,
                                             args.data_range,
                                             args.thresh,
                                             args.channels,
                                             args.mask,
                                             args.step,
                                             args.window)
            else:
                print("{:s}Please provide correct normality "
                      "model{:s}".format(R, W))
        stats.update({model_label: {
            'DR'                    : DR["global_rms"],
            'DR_deepest_negative'   : DR["deepest_negative"],
            'DR_global_rms'         : DR['global_rms'],
            'DR_local_rms'          : DR['local_rms']}})
        output_dict[residual_label] = stats
    elif args.residual:
        if args.residual not in output_dict.keys():
            if args.test_normality in ['shapiro', 'normaltest']:
                stats = residual_image_stats(args.residual,
                                             args.test_normality,
                                             args.data_range,
                                             args.thresh,
                                             args.channels,
                                             args.mask,
                                             args.step,
                                             args.window)
            else:
                if not args.test_normality:
                    stats = residual_image_stats(args.residual,
                                                 args.test_normality,
                                                 args.data_range,
                                                 args.thresh,
                                                 args.channels,
                                                 args.mask,
                                                 args.step,
                                                 args.window)
                else:
                    print("{:s}Please provide correct normality "
                          "model{:s}".format(R, W))
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

    LOGGER.info(output_dict)
    if args.models:
        models = args.models
        print("Number of model files: {:d}".format(len(models)))
        if len(models) < 1:
            print("{:s}Can only compare two models at a time.{:s}".format(R, W))
        else:
            models_list = []
            for i, comp_mod in enumerate(models):
                model1, model2 = comp_mod.split(':')
                models_list.append(
                    [dict(label="{0}-model_a_{1}".format(args.label, i),
                          path=model1),
                     dict(label="{0}-model_b_{1}".format(args.label, i),
                          path=model2)],
                )
            output_dict = compare_models(models_list, phase_centre=args.phase,
                                         all_sources=args.all)

    if args.noise:
        residuals = args.noise
        LOGGER.info("Number of residual pairs to compare: {:d}".format(len(residuals)))
        if len(residuals) < 1:
            print("{:s}Can only compare atleast one residual pair.{:s}".format(R, W))
        else:
            residuals_list = []
            for i, comp_res in enumerate(residuals):
                res1, res2 = comp_res.split(':')
                residuals_list.append(
                    [dict(label="{0}-res_a_{1}".format(args.label, i),
                          path=res1),
                     dict(label="{0}-res_b_{1}".format(args.label, i),
                          path=res2)],
                )
            if args.model:
                output_dict = compare_residuals(residuals_list, args.model)
            else:
                output_dict = compare_residuals(
                    residuals_list,
                    points=int(args.points) if args.points else 100)

    if output_dict:
        json_dump(output_dict)
