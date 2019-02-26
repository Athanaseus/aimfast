import os
import json
import Tigger
import random
import logging
import argparse
import tempfile
import numpy as np

from functools import partial

from scipy import stats
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.ndimage import measurements as measure

from plotly import tools
from plotly import offline as py
from plotly import graph_objs as go
from plotly.graph_objs import XAxis, YAxis

from astLib.astWCS import WCS
from astropy.table import Table
from astropy.io import fits as fitsio

from sklearn.metrics import mean_squared_error
from Tigger.Models import SkyModel, ModelClasses
from Tigger.Coordinates import angular_dist_pos_angle


PLOT_NUM_FLUX = {'format':
                 {  # num of plots: [colorbar spacing, colorbar y, colorbar len,
                    #                plot height, plot width]
                     1: [0.90, 0.45, 0.80, 700, 700],
                     2: [0.59, 0.78, 0.40, 1000, 600],
                     3: [0.38, 0.88, 0.30, 1900, 700],
                     4: [0.27, 0.91, 0.18, 2800, 700],
                     5: [0.207, 0.91, 0.15, 2000, 600],
                     6: [0.177, 0.94, 0.15, 2500, 600],
                     7: [0.15, 0.95, 0.13, 2800, 600]},
                 'plots':
                 {  # num of plots: [vertical spacing, horizontal spacing]
                     1: [0.06, 0.16],
                     2: [0.15, 0.16],
                     3: [0.1, 0.16],
                     4: [0.06, 0.16],
                     5: [0.04, 0.16],
                     6: [0.06, 0.16],
                     7: [0.04, 0.16]},
                 }

PLOT_NUM_POS = {'format':
                {  # num of plots: [colorbar spacing, colorbar y, colorbar len,
                   #                plot height, plot width]
                    1: [0.90, 0.45, 0.80, 470, 940],
                    2: [0.59, 0.78, 0.40, 1000, 1000],
                    3: [0.37, 0.87, 0.30, 1700, 1200],
                    4: [0.27, 0.90, 0.20, 1800, 1000],
                    5: [0.207, 0.91, 0.15, 2000, 1000],
                    6: [0.177, 0.94, 0.13, 3000, 1000],
                    7: [0.15, 0.94, 0.12, 3000, 1000]},
                'plots':
                {  # num of plots: [vertical spacing, horizontal spacing]
                    1: [0.1, 0.2],
                    2: [0.14, 0.21],
                    3: [0.10, 0.22],
                    4: [0.06, 0.23],
                    5: [0.04, 0.24],
                    6: [0.05, 0.25],
                    7: [0.08, 0.26]},
                }

PLOT_NUM_RES = {'format':
                {  # num of plots: [colorbar spacing, colorbar y, colorbar len,
                   #                plot height, plot width]
                    1: [0.90, 0.45, 0.8, 470, 940],
                    2: [0.59, 0.78, 0.4, 1000, 1000],
                    3: [0.37, 0.87, 0.3, 1700, 1200],
                    4: [0.27, 0.90, 0.20, 1800, 1000],
                    5: [0.207, 0.91, 0.15, 2000, 1000],
                    6: [0.177, 0.94, 0.13, 2200, 1000],
                    7: [0.15, 0.95, 0.15, 3000, 1000]},
                'plots':
                {  # num of plots: [vertical spacing, horizontal spacing]
                    1: [0.1, 0.16],
                    2: [0.14, 0.21],
                    3: [0.10, 0.22],
                    4: [0.06, 0.23],
                    5: [0.02, 0.16],
                    6: [0.03, 0.25],
                    7: [0.08, 0.26]},
                'legend':
                {  # num of plots: [x pos, y pos]
                    1: [0.48, 1.08],
                    2: [0.48, 1.00],
                    3: [0.48, 1.00],
                    4: [0.48, 1.00],
                    5: [0.48, 1.00],
                    6: [0.48, 1.00],
                    7: [0.48, 1.00]},
                }

# Unit multipleirs for plotting
FLUX_UNIT_SCALER = {'jansky': [1e0, 'Jy'],
                    'milli': [1e3, 'mJy'],
                    'micro': [1e6, u'\u03bcJy'],
                    'nano': [1e9, 'nJy'],
                    }


# Backgound color for plots
BG_COLOR = 'rgb(229,229,229)'


def creat_logger():
    """Create a console logger"""
    log = logging.getLogger(__name__)
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)
    log.addHandler(console)
    return log


def get_aimfast_data(filename='fidelity_results.json', dir='.'):
    "Extracts data from the json data file"
    file = '{:s}/{:s}'.format(dir, filename)
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
    """Dumps the computed dictionary into a json file.

    Parameters
    ----------
    data_dict : dict
        Dictionary with output results to save.
    root : str
        Directory to save output json file (default is current directory).

    Note
    ----
    If the fidelity_results.json file exists, it will be append, and only
    repeated image assessments will be replaced.

    """
    filename = ('{:s}/fidelity_results.json'.format(root))
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
        centre = 'J2000.0,0.0deg,-30.0deg'
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
            print("PSF FWHM over {:.2f} arcsec".format(arcsec_size * 2))
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
    noise_image: file
        Noise image (cube).

    Returns
    -------
    noise_std: float
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


def residual_image_stats(fitsname, test_normality=None, data_range=None):
    """Gets statistcal properties of a residual image.

    Parameters
    ----------
    fitsname : file
        Residual image (cube).
    test_normality : str
        Perform normality testing using either `shapiro` or `normaltest`.
    data_range : int, optional
        Range of data to perform normality testing.

    Returns
    -------
    props : dict
        Dictionary of stats properties.
        e.g. {'MEAN': 0.0, 'STDDev': 0.1, 'SKEW': 0.2, 'KURT': 0.3}.

    Notes
    -----
    If normality_test=True, dictionary of stats props becomes \
    e.g. {'MEAN': 0.0, 'STDDev': 0.1, 'SKEW': 0.2, 'KURT': 0.3, 'NORM': (123.3,0.012)} \
    whereby the first element is the statistics (or average if data_range specified) \
    of the datasets and second element is the p-value.

    """
    res_props = dict()
    # Open the residual image
    residual_hdu = fitsio.open(fitsname)
    # Get the header data unit for the residual rms
    residual_data = residual_hdu[0].data
    # Get the mean value
    res_props['MEAN'] = float("{0:.6}".format(residual_data.mean()))
    # Get the sigma value
    res_props['STDDev'] = float("{0:.6f}".format(residual_data.std()))
    # Flatten image
    res_data = residual_data.flatten()
    # Compute the skewness of the residual
    res_props['SKEW'] = float("{0:.6f}".format(stats.skew(res_data)))
    # Compute the kurtosis of the residual
    res_props['KURT'] = float("{0:.6f}".format(stats.kurtosis(res_data, fisher=False)))
    # Perform normality testing
    if test_normality:
        norm_props = normality_testing(fitsname, test_normality, data_range)
        res_props.update(norm_props)
    props = res_props
    print(props)
    # Return dictionary of results
    return props


def normality_testing(fitsname, test_normality='normaltest', data_range=None):
    """Performs a normality test on the image.

    Parameters
    ----------
    fitsname : file
        Residual image (cube).
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
    normality = dict()
    # Open the residual image
    residual_hdu = fitsio.open(fitsname)
    # Get the header data unit for the residual rms
    residual_data = residual_hdu[0].data
    # Flatten image
    res_data = residual_data.flatten()
    # Shuffle the data
    random.shuffle(res_data)
    # Normality test
    norm_res = []
    counter = 0
    if type(data_range) is int:
        for dataset in range(len(res_data) / data_range):
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
    # Get beam size otherwise use default (5``).
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
    """Get model"""

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
                                    phase_centre=None):
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
            source = sources[0]
            ra = source.pos.ra
            dec = source.pos.dec
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
                                      rad2arcsec(abs(ra - RA)),
                                      rad2arcsec(abs(dec - DEC)),
                                      delta_phase_centre_arc_sec, I_in,
                                      source_name]
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
            src_scale = get_src_scale(source.shape)
            targets_scale[name] = [shape_out, shape_out_err, shape_in,
                                   src_scale[0], src_scale[1], I_in,
                                   source_name]
    print("Number of sources recovered: {:d}".format(len(targets_scale)))
    return targets_flux, targets_scale, targets_position


def compare_models(models, tolerance=0.000001, plot=True, phase_centre=None):
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
                                                tolerance, phase_centre)
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


def plot_photometry(models, label=None, tolerance=0.00001, phase_centre=None):
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

    """
    _models = []
    i = 0
    for model1, model2 in models.items():
        _models.append([dict(label="{}-model_a_{}".format(label, i), path=model1),
                        dict(label="{}-model_b_{}".format(label, i), path=model2)])
        i += 1
    results = compare_models(_models, tolerance, False, phase_centre)
    _source_flux_plotter(results, _models, inline=True)


def plot_astrometry(models, label=None, tolerance=0.00001, phase_centre=None):
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

    """
    _models = []
    i = 0
    for model1, model2 in models.items():
        _models.append([dict(label="{0:s}-model_a_{1:d}".format(label, i), path=model1),
                        dict(label="{0:s}-model_b_{1:d}".format(label, i), path=model2)])
        i += 1
    results = compare_models(_models, tolerance, False, phase_centre)
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


def _source_flux_plotter(results, all_models, inline=False):
    """Plot flux results and save output as html file.

    Parameters
    ----------
    results : dict
        Structured output results.
    models : list
        Tigger/text formatted model files e.g [model1, model2].
    inline : bool
        Allow inline plotting inside a notebook.

    """
    im_titles = []
    models_compare = dict()
    for models in all_models:
        output_model = models[-1]['path']
        input_model = models[0]['path']
        if 'html' in output_model:
            header = output_model.split('/')[-1][:-9]
        else:
            header = output_model.split('/')[-1][:-4]
        models_compare[input_model] = output_model
        im_titles.append('<b>{:s} flux density</b>'.format(header.upper()))

    PLOTS = len(models_compare.keys())
    fig = tools.make_subplots(rows=PLOTS, cols=1, shared_yaxes=False,
                              print_grid=False,
                              vertical_spacing=PLOT_NUM_FLUX['plots'][PLOTS][0],
                              subplot_titles=sorted(im_titles))
    j = 0
    i = -1
    counter = 0
    annotate = []
    for input_model, output_model in sorted(models_compare.items()):
        i += 1
        counter += 1
        name_labels = []
        flux_in_data = []
        flux_out_data = []
        source_scale = []
        phase_center_dist = []
        flux_out_err_data = []
        heading = all_models[i][0]['label']
        for n in range(len(results[heading]['flux'])):
            flux_out_data.append(results[heading]['flux'][n][0])
            flux_out_err_data.append(results[heading]['flux'][n][1])
            flux_in_data.append(results[heading]['flux'][n][2])
            name_labels.append(results[heading]['flux'][n][3])
            phase_center_dist.append(results[heading]['position'][n][-3])
            source_scale.append(results[heading]['shape'][n][3])
        zipped_props = zip(flux_out_data, flux_out_err_data, flux_in_data,
                           name_labels, phase_center_dist, source_scale)
        (flux_out_data, flux_out_err_data, flux_in_data, name_labels,
            phase_center_dist, source_scale) = zip(*sorted(zipped_props, key=lambda x: x[0]))

        flux_MSE = mean_squared_error(flux_in_data, flux_out_data)
        reg = linregress(flux_in_data, flux_out_data)
        flux_R_score = reg.rvalue

        annotate.append(
            go.Annotation(
                x=(sorted(flux_in_data)[-1]/2.0 - sorted(flux_in_data)[0]/2.0
                   + sorted(flux_in_data)[0]) * FLUX_UNIT_SCALER['milli'][0],
                y=flux_in_data[-1]*FLUX_UNIT_SCALER['milli'][0] + 0.0005*FLUX_UNIT_SCALER['milli'][0],
                xref='x{:d}'.format(counter),
                yref='y{:d}'.format(counter),
                text="Slope: {:.4f} | Intercept: {:.4f} | RMS Error: {:.4f} | R2: {:.4f} ".format(
                    reg.slope, reg.intercept * FLUX_UNIT_SCALER['milli'][0],
                    np.sqrt(flux_MSE) * FLUX_UNIT_SCALER['milli'][0], flux_R_score),
                ax=0,
                ay=-10,
                showarrow=False,
                bordercolor='#c7c7c7',
                borderwidth=2,
                font=dict(color="black", size=12)))
        fig.append_trace(
            go.Scatter(
                x=np.array([flux_in_data[0], flux_in_data[-1]])*FLUX_UNIT_SCALER['milli'][0],
                showlegend=False,
                marker=dict(color='rgb(0,0,255)'),
                y=np.array([flux_in_data[0], flux_in_data[-1]])*FLUX_UNIT_SCALER['milli'][0],
                mode='lines'), i+1, 1)
        fig.append_trace(
            go.Scatter(
                x=np.array(flux_in_data)*FLUX_UNIT_SCALER['milli'][0],
                y=np.array(flux_out_data)*FLUX_UNIT_SCALER['milli'][0],
                mode='markers', showlegend=False,
                text=name_labels, name='{:s} flux_ratio'.format(heading),
                marker=dict(color=phase_center_dist, showscale=True, colorscale='Jet',
                            reversescale=False, colorbar=dict(
                                title='Distance from phase center (arcsec)',
                                titleside='right',
                                titlefont=dict(size=16),
                                len=PLOT_NUM_FLUX['format'][PLOTS][2],
                                y=PLOT_NUM_FLUX['format'][PLOTS][1]-j)) if
                phase_center_dist[-1] else dict(),
                error_y=dict(type='data',
                             array=np.array(flux_out_err_data)*FLUX_UNIT_SCALER['milli'][0],
                             color='rgb(158, 63, 221)',
                             visible=True)), i+1, 1)
        fig['layout'].update(title='', height=PLOT_NUM_FLUX['format'][PLOTS][3],
                             width=PLOT_NUM_FLUX['format'][PLOTS][4],
                             paper_bgcolor='rgb(255,255,255)',
                             plot_bgcolor=BG_COLOR,
                             legend=dict(x=0.8, y=1.0),)
        fig['layout'].update(
            {'yaxis{}'.format(counter): YAxis(
                title='Output flux({:s})'.format(FLUX_UNIT_SCALER['milli'][1]),
                gridcolor='rgb(255,255,255)',
                tickfont=dict(size=15),
                titlefont=dict(size=17),
                showgrid=True,
                showline=False,
                showticklabels=True,
                tickcolor='rgb(51,153,225)',
                ticks='outside',
                zeroline=False)})
        fig['layout'].update(
            {'xaxis{}'.format(counter+i): XAxis(
                title='Input Flux ({:s})'.format(FLUX_UNIT_SCALER['milli'][1]),
                position=0.0, titlefont=dict(size=17), overlaying='x')})
        if counter == PLOTS:
            fig['layout']['annotations'] = annotate
        j += PLOT_NUM_FLUX['format'][PLOTS][0]
    outfile = 'InputOutputFluxDensity.html'
    if inline:
        py.init_notebook_mode(connected=True)
        py.iplot(fig, filename=outfile)
    else:
        py.plot(fig, filename=outfile, auto_open=False)


def _source_astrometry_plotter(results, all_models, inline=False):
    """Plot astrometry results and save output as html file.

    Parameters
    ----------
    results: dict
        Structured output results.
    models: list
        Tigger/text formatted model files e.g [model1, model2].
    inline : bool
        Allow inline plotting inside a notebook.

    """
    im_titles = []
    models_compare = dict()
    for models in all_models:
        output_model = models[-1]['path']
        input_model = models[0]['path']
        if 'html' in output_model:
            header = output_model.split('/')[-1][:-9]
        else:
            header = output_model.split('/')[-1][:-4]
        models_compare[input_model] = output_model
        im_titles.append('<b>{:s} Position Offset</b>'.format(header.upper()))
        im_titles.append('<b>{:s} Delta Position</b>'.format(header.upper()))

    PLOTS = len(models_compare.keys())
    fig = tools.make_subplots(rows=PLOTS, cols=2,
                              shared_yaxes=False,
                              print_grid=False,
                              vertical_spacing=PLOT_NUM_POS['plots'][PLOTS][0],
                              horizontal_spacing=PLOT_NUM_POS['plots'][PLOTS][1],
                              subplot_titles=im_titles)
    j = 0
    i = -1
    counter = 0
    annotate = []
    for input_model, output_model in sorted(models_compare.items()):
        i += 1
        counter += 1
        RA_offset = []
        DEC_offset = []
        DELTA_PHASE0 = []
        source_labels = []
        flux_in_data = []
        flux_out_data = []
        delta_pos_data = []
        heading = all_models[i][0]['label']
        for n in range(len(results[heading]['flux'])):
            flux_out_data.append(results[heading]['flux'][n][0])
            delta_pos_data.append(results[heading]['position'][n][0])
            RA_offset.append(results[heading]['position'][n][1])
            DEC_offset.append(results[heading]['position'][n][2])
            DELTA_PHASE0.append(results[heading]['position'][n][3])
            flux_in_data.append(results[heading]['position'][n][4])
            source_labels.append(results[heading]['position'][n][5])
        zipped_props = zip(delta_pos_data, RA_offset, DEC_offset,
                           DELTA_PHASE0, flux_in_data, source_labels)
        (delta_pos_data, RA_offset, DEC_offset, DELTA_PHASE0,
            flux_in_data, source_labels) = zip(
            *sorted(zipped_props, key=lambda x: x[-2]))
        fig.append_trace(
            go.Scatter(
                x=np.array(flux_in_data) * FLUX_UNIT_SCALER['milli'][0],
                y=np.array(delta_pos_data),
                mode='markers', showlegend=False,
                text=source_labels, name='{:s} flux_ratio'.format(header),
                marker=dict(color=DELTA_PHASE0, showscale=True,
                            colorscale='Jet', reversescale=True,
                            colorbar=dict(title='Distance from phase center (arcsec)',
                                          titleside='right',
                                          len=PLOT_NUM_POS['format'][PLOTS][2],
                                          y=PLOT_NUM_POS['format'][PLOTS][1]-j))
                if DELTA_PHASE0[-1] else dict()),
            i+1, 2)
        fig.append_trace(
            go.Scatter(
                x=np.array(RA_offset), y=np.array(DEC_offset),
                mode='markers', showlegend=False,
                text=source_labels, name='{:s} flux_ratio'.format(heading),
                marker=dict(color=np.array(flux_out_data) * FLUX_UNIT_SCALER['milli'][0],
                            showscale=True,
                            colorscale='Viridis',
                            reversescale=True,
                            colorbar=dict(title='Output flux (mJy)',
                                          titleside='right',
                                          len=PLOT_NUM_POS['format'][PLOTS][2],
                                          y=PLOT_NUM_POS['format'][PLOTS][1]-j,
                                          x=0.4))),
            i+1, 1)
        RA_mean = np.mean(RA_offset)
        DEC_mean = np.mean(DEC_offset)
        r1, r2 = np.array(RA_offset).std(), np.array(DEC_offset).std()
        pi, cos, sin = np.pi, np.cos, np.sin
        theta = np.linspace(0, 2.0 * pi, len(DEC_offset))
        x1 = RA_mean+(r1 * cos(theta))
        y1 = DEC_mean+(r2 * sin(theta))
        recovered_sources = len(DEC_offset)
        one_sigma_sources = len([
            (ra_off, dec_off) for ra_off, dec_off in zip(RA_offset, DEC_offset)
            if abs(ra_off) <= max(abs(x1)) and abs(dec_off) <= max(abs(y1))])
        annotate.append(
            go.Annotation(
                x=0,
                y=max(DEC_offset) + 0.18,
                xref='x{:d}'.format(counter+i),
                yref='y{:d}'.format(counter+i),
                text=("Total sources: {:d} | (RA, DEC) mean: ({:.4f}, {:.4f})".format(
                      recovered_sources, RA_mean, DEC_mean)),
                ax=0,
                ay=-40,
                showarrow=False,
                font=dict(color="black", size=10)))
        annotate.append(
            go.Annotation(
                x=0,
                y=max(DEC_offset) + 0.19 + 0.055,
                xref='x{:d}'.format(counter+i),
                yref='y{:d}'.format(counter+i),
                text=("Sigma sources: {:d} | (RA, DEC) sigma: ({:.4f}, {:.4f})".format(
                      one_sigma_sources, r1, r2)),
                ax=0,
                ay=-40,
                showarrow=False,
                font=dict(color="black", size=10)))
        fig.append_trace(go.Scatter(x=x1, y=y1,
                                    mode='lines', showlegend=False,
                                    name=r'1 sigma',
                                    text=r'1 sigma ~ {:f}'.format(np.sqrt(r1*r2)),
                                    marker=dict(color='rgb(0, 0, 255)')), i+1, 1)
        fig['layout'].update(title='', height=PLOT_NUM_POS['format'][PLOTS][3],
                             width=PLOT_NUM_POS['format'][PLOTS][4],
                             paper_bgcolor='rgb(255,255,255)', plot_bgcolor=BG_COLOR,
                             legend=dict(xanchor='auto', x=1.2, y=1))
        fig['layout'].update(
            {'yaxis{}'.format(counter+i): YAxis(
                title=u'Dec offset (")',
                gridcolor='rgb(255,255,255)',
                color='rgb(0,0,0)',
                tickfont=dict(size=14, color='rgb(0,0,0)'),
                titlefont=dict(size=15),
                showgrid=True,
                showline=True,
                showticklabels=True,
                tickcolor='rgb(51,153,225)',
                ticks='outside',
                zeroline=True)})
        fig['layout'].update(
            {'yaxis{}'.format(counter+i+1): YAxis(
                title='Delta position (")',
                gridcolor='rgb(255,255,255)',
                color='rgb(0,0,0)',
                tickfont=dict(size=10, color='rgb(0,0,0)'),
                titlefont=dict(size=17),
                showgrid=True,
                showline=True,
                showticklabels=True,
                tickcolor='rgb(51,153,225)',
                ticks='outside',
                zeroline=True)})
        fig['layout'].update(
            {'xaxis{}'.format(counter+i): XAxis(
                title=u'RA offset (")',
                titlefont=dict(size=17),
                zeroline=False,
                position=1.0,
                overlaying='x',)})
        fig['layout'].update(
            {'xaxis{}'.format(counter+i+1): XAxis(
                title='Input Flux ({:s})'.format(FLUX_UNIT_SCALER['milli'][1]),
                position=0.0,
                overlaying='x',
                titlefont=dict(size=17),
                zeroline=True)})

        if counter == PLOTS:
            fig['layout']['annotations'] = annotate
        j += PLOT_NUM_POS['format'][PLOTS][0]

    outfile = 'InputOutputPosition.html'
    if inline:
        py.init_notebook_mode(connected=True)
        py.iplot(fig, filename=outfile)
    else:
        py.plot(fig, filename=outfile, auto_open=False)


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

    # Plot titles list
    im_titles = []
    residuals_compare = dict()
    for res_ims in res_noise_images:
        # Get residual image names
        output_model = res_ims[-1]['path']
        input_model = res_ims[0]['path']
        header = output_model.split('/')[-1][:-5]
        residuals_compare[input_model] = output_model
        # Assign plot titles
        im_titles.append('<b>{:s} RMS</b>'.format(header.upper()))
        im_titles.append('<b>{:s} residual-noise</b>'.format(header.upper()))

    PLOTS = len(residuals_compare.keys())
    fig = tools.make_subplots(rows=PLOTS, cols=2, shared_yaxes=False,
                              print_grid=False,
                              horizontal_spacing=0.2,
                              vertical_spacing=0.08,
                              subplot_titles=im_titles)

    i = 0
    j = 0
    counter = 0
    annotate = []
    for res_image, noise in residuals_compare.items():
        counter += 1
        rmss = []
        residuals = []
        name_labels = []
        dist_from_phase = []
        res_noise_ratio = []
        for res_src in results[res_image]:
            rmss.append(res_src[0])
            residuals.append(res_src[1])
            res_noise_ratio.append(res_src[2])
            dist_from_phase.append(res_src[3])
            name_labels.append(res_src[4])
        fig.append_trace(
            go.Scatter(
                x=np.array(range(len(rmss))),
                y=np.array(rmss) * FLUX_UNIT_SCALER['micro'][0],
                mode='lines',
                showlegend=True if i == 0 else False,
                name='residual 1',
                text=name_labels,
                marker=dict(color='rgb(255,0,0)'),
                error_y=dict(type='data', color='rgb(158, 63, 221)',
                             visible=True)),
            i+1, 1)
        fig.append_trace(
            go.Scatter(
                x=np.array(range(len(rmss))),
                y=np.array(residuals) * FLUX_UNIT_SCALER['micro'][0],
                mode='lines', showlegend=True if i == 0 else False,
                name='residual 2',
                text=name_labels,
                marker=dict(color='rgb(0,0,255)'),
                error_y=dict(type='data', color='rgb(158, 63, 221)',
                             visible=True)),
            i+1, 1)
        fig.append_trace(
            go.Scatter(
                x=np.array(range(len(rmss))), y=np.array(res_noise_ratio),
                mode='markers', showlegend=False,
                text=name_labels,
                marker=dict(color=dist_from_phase,
                            showscale=True,
                            colorscale='Jet',
                            colorbar=dict(
                                title='Phase center dist (arcsec)',
                                titleside='right',
                                len=PLOT_NUM_FLUX['format'][PLOTS][2],
                                y=PLOT_NUM_FLUX['format'][PLOTS][1]-j)),
                error_y=dict(type='data', color='rgb(158, 63, 221)',
                             visible=True)),
            i+1, 2)
        fig.append_trace(
            go.Scatter(
                x=[np.array(range(len(rmss)))[0],
                   np.array(range(len(rmss)))[-1]],
                y=[np.mean(residuals) / np.mean(rmss),
                   np.mean(residuals) / np.mean(rmss)],
                mode='lines', showlegend=False,
                marker=dict(color='rgb(0,300,0)'),
                text=name_labels),
            i+1, 2)
        annotate.append(
            go.Annotation(
                x=0.00005 * FLUX_UNIT_SCALER['micro'][0],
                y=7.8 + max(residuals) * FLUX_UNIT_SCALER['micro'][0],
                xref='x{:d}'.format(counter+i),
                yref='y{:d}'.format(counter+i),
                text="res1: {:.2f} | res2: {:.2f} | res1-res2: {:.2f}".format(
                     np.mean(residuals) * FLUX_UNIT_SCALER['micro'][0],
                     np.mean(rmss) * FLUX_UNIT_SCALER['micro'][0],
                     np.mean(residuals) / np.mean(rmss)),
                ax=0,
                ay=-10,
                showarrow=False,
                bordercolor='#c7c7c7',
                borderwidth=2,
                font=dict(color="black", size=12)))
        fig['layout'].update(title='', height=PLOT_NUM_RES['format'][PLOTS][3],
                             width=PLOT_NUM_RES['format'][PLOTS][4],
                             paper_bgcolor='rgb(255,255,255)',
                             plot_bgcolor=BG_COLOR,
                             legend=dict(xanchor='auto',
                                         x=PLOT_NUM_RES['legend'][PLOTS][0],
                                         y=PLOT_NUM_RES['legend'][PLOTS][1]))
        fig['layout'].update(
            {'yaxis{}'.format(counter+i): YAxis(
                title=u'rms [\u03BCJy/beam]',
                gridcolor='rgb(255,255,255)',
                color='rgb(0,0,0)',
                tickfont=dict(size=14, color='rgb(0,0,0)'),
                titlefont=dict(size=17),
                showgrid=True,
                showline=True,
                showticklabels=True,
                tickcolor='rgb(51,153,225)',
                ticks='outside',
                zeroline=False)})
        fig['layout'].update(
            {'yaxis{}'.format(counter+i+1): YAxis(
                title='Res1-to-Res2',
                gridcolor='rgb(255,255,255)',
                color='rgb(0,0,0)',
                tickfont=dict(size=10, color='rgb(0,0,0)'),
                titlefont=dict(size=15),
                showgrid=True,
                showline=True,
                showticklabels=True,
                tickcolor='rgb(51,153,225)',
                ticks='outside',
                zeroline=False)})
        fig['layout'].update(
            {'xaxis{}'.format(counter+i): XAxis(
                title='Sources',
                titlefont=dict(size=17),
                showline=True,
                zeroline=False,
                position=0.0,
                overlaying='x')})
        fig['layout'].update(
            {'xaxis{}'.format(counter+i+1): XAxis(
                title='Sources',
                titlefont=dict(size=17),
                showline=True,
                zeroline=False)})
        i += 1
        j += PLOT_NUM_RES['format'][PLOTS][0]
        if counter == PLOTS:
            fig['layout']['annotations'] = annotate
    if points:
        outfile = 'RandomResidualNoiseRatio.html'
    else:
        outfile = 'SourceResidualNoiseRatio.html'
    if inline:
        py.init_notebook_mode(connected=True)
        py.iplot(fig, filename=outfile)
    else:
        py.plot(fig, filename=outfile, auto_open=False)


def _random_residual_results(res_noise_images, data_points=100, area_factor=2.0):
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
    # dictinary to store results
    results = dict()
    # Get beam size otherwise use default (5``).
    beam_default = (0.00151582804885738, 0.00128031965017612, 20.0197348935424)
    # Source counter
    i = 0
    for images in res_noise_images:
        # Get residual image names
        res_image = images[0]['path']
        noise_image = images[-1]['path']
        # Get fits info
        fits_info = fitsInfo(res_image)
        # Get beam size otherwise use default (5``).
        beam_deg = fits_info['b_size'] if fits_info['b_size'] else beam_default
        # Open noise header
        noise_hdu = fitsio.open(noise_image)
        # Get data from noise image
        noise_data = noise_hdu[0].data
        # Data structure for each residuals to compare
        results[res_image] = []
        residual_hdu = fitsio.open(res_image)
        # Get the header data unit for the residual rms
        residual_data = residual_hdu[0].data
        # Get random pixel coordinates
        pix_coord_deg = _get_random_pixel_coord(data_points,
                                                sky_area=fits_info['skyArea'] * 0.9,
                                                phase_centre=fits_info['centre'])
        # Get the number of frequency channels
        nchan = (residual_data.shape[1]
                 if residual_data.shape[0] == 1
                 else residual_data.shape[0])
        for RA, DEC in pix_coord_deg:
            i += 1
            # Get width of box around source
            width = int(deg2arcsec(beam_deg[0]) * area_factor)
            # Get a image slice around source
            imslice = get_box(fits_info["wcs"], (RA, DEC), width)
            # Get noise rms in the box around source
            noise_area = noise_data[0, 0, :, :][imslice]
            noise_rms = noise_area.std()
            # if image is cube then average along freq axis
            flux_std = 0.0
            flux_mean = 0.0
            for frq_ax in range(nchan):
                # In case the first two axes are swapped
                if residual_data.shape[0] == 1:
                    target_area = residual_data[0, frq_ax, :, :][imslice]
                else:
                    target_area = residual_data[frq_ax, 0, :, :][imslice]
                # Sum of all the fluxes
                flux_std += target_area.std()
                flux_mean += target_area.mean()
            # Get the average std and mean along all frequency channels
            flux_std = flux_std/float(nchan)
            flux_mean = flux_mean/float(nchan)
            # Get phase centre and determine phase centre distance
            RA0 = float(fits_info['centre'].split(',')[1].split('deg')[0])
            DEC0 = float(fits_info['centre'].split(',')[-1].split('deg')[0])
            phase_dist_arcsec = deg2arcsec(np.sqrt((RA-RA0)**2 + (DEC-DEC0)**2))
            # Store all outputs in the results data structure
            results[res_image].append([noise_rms*1e0,
                                       flux_std*1e0,
                                       flux_std/noise_rms,
                                       phase_dist_arcsec, 'source{0}'.format(i),
                                       flux_mean,
                                       flux_mean/noise_rms])
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
    # Dictinary to store results
    results = dict()
    # Get beam size otherwise use default (5``).
    beam_default = (0.00151582804885738, 0.00128031965017612, 20.0197348935424)
    for images in res_noise_images:
        # Get residual image names
        res_image = images[0]['path']
        noise_image = images[-1]['path']
        # Get fits info
        fits_info = fitsInfo(res_image)
        # Get beam size otherwise use default (5``).
        beam_deg = fits_info['b_size'] if fits_info['b_size'] else beam_default
        # Open noise header
        noise_hdu = fitsio.open(noise_image)
        # Get data from noise image
        noise_data = noise_hdu[0].data
        # Get label
        if 'None' in images[0]['label']:
            label = res_image
        else:
            label = images[0]['label']
        # Load skymodel to get source positions
        model_lsm = Tigger.load(skymodel)
        # Get all sources in the model
        model_sources = model_lsm.sources
        # Get global rms of noise image
        noise_sig = noise_sigma(noise_image)
        noise_hdu = fitsio.open(noise_image)
        noise_data = noise_hdu[0].data
        # Get data from residual image
        residual_hdu = fitsio.open(res_image)
        residual_data = residual_hdu[0].data
        # Data structure for each residuals to compare
        results[label] = []
        # Get the number of frequency channels
        nchan = (residual_data.shape[1]
                 if residual_data.shape[0] == 1
                 else residual_data.shape[0])
        for model_source in model_sources:
            src = model_source
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
            delta_phase_centre_arc_sec = rad2arcsec(delta_phase_centre[0])
            # Get beam size otherwise use default (5``).
            beam_default = (0.00151582804885738, 0.00128031965017612, 20.0197348935424)
            beam_deg = fits_info['b_size'] if fits_info['b_size'] else beam_default
            # Get width of box around source
            width = int(deg2arcsec(beam_deg[0]) * area_factor)
            # Get a image slice around source
            imslice = get_box(fits_info["wcs"], (RA, DEC), width)
            # Get noise rms in the box around source
            noise_area = noise_data[0, 0, :, :][imslice]
            noise_rms = noise_area.std()
            # if image is cube then average along freq axis
            flux_std = 0.0
            flux_mean = 0.0
            for frq_ax in range(nchan):
                # In case the first two axes are swapped
                if residual_data.shape[0] == 1:
                    target_area = residual_data[0, frq_ax, :, :][imslice]
                else:
                    target_area = residual_data[frq_ax, 0, :, :][imslice]
                # Sum of all the fluxes
                flux_std += target_area.std()
                flux_mean += target_area.mean()
            # Get the average std and mean along all frequency channels
            flux_std = flux_std/float(nchan)
            flux_mean = flux_mean/float(nchan)
            # Store all outputs in the results data structure
            results[label].append([noise_rms*1e0, flux_std*1e0,
                                   flux_std/noise_rms,
                                   delta_phase_centre_arc_sec,
                                   model_source.name, src.flux.I,
                                   src.flux.I/flux_std,
                                   src.flux.I/noise_sig, flux_mean,
                                   abs(flux_mean/noise_rms)])
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
    argument('--normality-test', dest='test_normality',
             help='Name of model to use for normality testing. \n'
                  'options: [shapiro, normaltest] \n'
                  'NB: normaltest is the D`Agostino')
    argument('-dr', '--data-range', dest='data_range',
             help='Data range to perform normality testing')
    argument('-af', '--area-factor', dest='factor', type=float, default=6,
             help='Factor to multiply the beam area to get target peak area')
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
    argument("--label",
             help='Use this label instead of the FITS image path when saving'
                  'data as JSON file.')
    return parser


def main():
    """Main function."""
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
                  "Otherwise a default beam size of five (5``) asec "
                  "is used{:s}".format(R, W))
        if args.test_normality in ['shapiro', 'normaltest']:
            if args.data_range:
                stats = residual_image_stats(args.residual,
                                             args.test_normality,
                                             int(args.data_range))
            else:
                stats = residual_image_stats(args.residual, args.test_normality)
        else:
            if not args.test_normality:
                stats = residual_image_stats(args.residual)
            else:
                print("{:s}Please provide correct normality"
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
                if args.data_range:
                    stats = residual_image_stats(args.residual,
                                                 args.test_normality,
                                                 int(args.data_range))
                else:
                    stats = residual_image_stats(args.residual, args.test_normality)
            else:
                if not args.test_normality:
                    stats = residual_image_stats(args.residual)
                else:
                    print("{:s}Please provide correct normality"
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
                          path=model2),
                     dict(label="{0}-model_b_{1}".format(args.label, i),
                          path=model2)],
                )
            output_dict = compare_models(models_list, phase_centre=args.phase)

    if args.noise:
        residuals = args.noise
        print("Number of model files: {:d}".format(len(residuals)))
        if len(residuals) < 1:
            print("{:s}Can only compare two models at a time.{:s}".format(R, W))
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
