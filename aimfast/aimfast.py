import json
import Tigger
import random
import argparse
import numpy as np
from scipy import stats
from plotly import tools
from functools import partial
from astLib.astWCS import WCS
import plotly.graph_objs as go
from plotly import offline as py
from scipy.stats import linregress
from astropy.io import fits as fitsio
from scipy.interpolate import interp1d
from plotly.graph_objs import XAxis, YAxis
import scipy.ndimage.measurements as measure
from sklearn.metrics import mean_squared_error
from Tigger.Coordinates import angular_dist_pos_angle


PLOT_NUM = {'colorbar':
               {   # num of plots: [colorbar spacing, colorbar y, colorbar len]
                1: [0.95, 0.5, 0.95],
                2: [0.59, 0.78, 0.4],
                3: [0.41, 0.81, 0.34],
                4: [0.28, 0.86, 0.31],
                5: [0.22, 0.93, 0.2]
               }
           }


# Unit multipleirs for plotting
UNIT_SCALER = {'milli': 1e3,
               'micro': 1e6,
               'nano' : 1e9}


# Backgound color for plots
BG_COLOR = 'rgb(229,229,229)'


def deg2arcsec(x):
    """Converts 'x' from degrees to arcseconds."""
    return float(x)*3600.00


def rad2deg(x):
    """Converts 'x' from radian to degrees."""
    return float(x)*(180/np.pi)


def rad2arcsec(x):
    """Converts `x` from radians to arcseconds."""
    return float(x)*3600.0*180.0/np.pi


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
    filename = ('fidelity_results.json')
    try:
        # Extract data from the json data file
        with open(filename) as data_file:
            data_existing = json.load(data_file)
            data = dict(data_existing.items() + data_dict.items())
    except IOError:
        data = data_dict
    if data:
        with open('{:s}/{:s}'.format(root, filename), 'w') as f:
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
        centre = '{0},{1},{2}'.format('J'+str(hdr['EQUINOX']),
                                      str(hdr['CRVAL1'])+hdr['CUNIT1'],
                                      str(hdr['CRVAL2'])+hdr['CUNIT2'])
    except:
        centre = 'J2000.0,0.0deg,-30.0deg'
    skyArea = (numPix*ddec)**2
    fitsinfo = {'wcs': wcs, 'ra': ra, 'dec': dec,
                'dra': dra, 'ddec': ddec, 'raPix': raPix,
                'decPix': decPix,  'b_size': beam_size,
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
        secpix = abs(hdu[0].header['CDELT1']*3600)
    # get midpoint and size of cross-sections
    xmid, ymid = measure.maximum_position(pp)
    sz = int(arcsec_size/secpix)
    xsec = pp[xmid-sz:xmid+sz, ymid]
    ysec = pp[xmid, ymid-sz:ymid+sz]

    def fwhm(tsec):
        """Determine the full width half maximum"""
        tmid = len(tsec)/2
        # find first minima off the peak, and flatten cross-section outside them
        xmin = measure.minimum_position(tsec[:tmid])[0]
        tsec[:xmin] = tsec[xmin]
        xmin = measure.minimum_position(tsec[tmid:])[0]
        tsec[tmid+xmin:] = tsec[tmid+xmin]
        if tsec[0] > .5 or tsec[-1] > .5:
            print("PSF FWHM over {:.2f} arcsec".format(arcsec_size*2))
            return arcsec_size, arcsec_size
        x1 = interp1d(tsec[:tmid], range(tmid))(0.5)
        x2 = interp1d(1-tsec[tmid:], range(tmid, len(tsec)))(0.5)
        return x1, x2

    ix0, ix1 = fwhm(xsec)
    iy0, iy1 = fwhm(ysec)
    rx, ry = (ix1-ix0)*secpix, (iy1-iy0)*secpix
    r0 = (rx+ry)/2
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
    box = slice(decPix-w/2, decPix+w/2), slice(raPix-w/2, raPix+w/2)
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
    d_ra = np.sqrt(area)/2
    d_dec = np.sqrt(area)/2
    ra_range = [ra-d_ra, ra+d_ra]
    dec_range = [dec-d_dec, dec+d_dec]
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
    res_props['MEAN'] = round(abs(residual_data.mean()), 10)
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
        props = dict(res_props.items() + norm_props.items())
    else:
        props = res_props
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
        for dataset in range(len(res_data)/data_range):
            i = counter
            counter += data_range
            norm_res.append(getattr(stats, test_normality)(res_data[i:counter]))
        # Compute sum of pvalue
        if test_normality == 'normaltest':
            sum_statistics = sum([norm.statistic for norm in norm_res])
            sum_pvalues = sum([norm.pvalue for norm in norm_res])
        elif test_normality == 'shapiro':
            sum_statistics = sum([norm[0] for norm in norm_res])
            sum_pvalues = sum([norm[1] for norm in norm_res])
        normality['NORM'] = (sum_statistics/dataset, sum_pvalues/dataset)
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
    sources_flux = dict([(model_source, model_source.getTag('I_peak'))
                        for model_source in model_sources])
    peak_source_flux = [(_model_source, flux)
                        for _model_source, flux in sources_flux.items()
                        if flux == max(sources_flux.values())][0][0]
    peak_flux = peak_source_flux.getTag('I_peak')
    # Get astrometry of the source in degrees
    RA = rad2deg(peak_source_flux.pos.ra)
    DEC = rad2deg(peak_source_flux.pos.dec)
    # Get source region and slice
    wcs = WCS(residual_hdu[0].header, mode="pyfits")
    width = int(beam_size*area_factor)
    imslice = get_box(wcs, (RA, DEC), width)
    # TODO please confirm
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


def image_dynamic_range(fitsname, area_factor=6):
    """Gets the dynamic range in a restored image.

    Parameters
    ----------
    fitsname : fits file
        Restored image (cube).
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
        # i.e. (0, nchan, x_pix, y_pix)
        if restored_data.shape[0] == 1:
            target_area = restored_data[0, frq_ax, :, :][imslice]
        else:
            target_area = restored_data[frq_ax, 0, :, :][imslice]
        min_flux += target_area.min()
        if frq_ax == nchan - 1:
            min_flux = min_flux/float(nchan)
    # Compute dynamic range
    local_std = target_area.std()
    global_std = restored_data[0, 0, ...].std()
    # Compute dynamic range
    DR = {
        "deepest_negative"  : peak_flux/abs(min_flux)*1e0,
        "local_rms"         : peak_flux/local_std*1e0,
        "global_rms"        : peak_flux/global_std*1e0,
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


def get_detected_sources_properties(model_1, model_2, area_factor):
    """Extracts the output simulation sources properties.

    Parameters
    ----------
    models_1 : file
        Tigger formatted or txt model 1 file.
    models_2 : file
        Tigger formatted or txt model 2 file.
    area_factor : float
        Area factor to multiply the psf size around source.

    Returns
    -------
    (targets_flux, targets_scale, targets_position) : tuple
        Tuple of target flux, morphology and astrometry information

    """
    model_lsm = Tigger.load(model_1)
    pybdsm_lsm = Tigger.load(model_2)
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
            I_out_err_list.append(target.flux.I_err*target.flux.I_err)
        I_out = sum([val/err for val, err in zip(I_out_list, I_out_err_list)])
        if I_out != 0.0:
            I_out_err = sum([1/I_out_error for I_out_error
                            in I_out_err_list])
            I_out_var_err = np.sqrt(1/I_out_err)
            I_out = I_out/I_out_err
            I_out_err = I_out_var_err
            source = sources[0]
            RA0 = pybdsm_lsm.ra0
            DEC0 = pybdsm_lsm.dec0
            ra = source.pos.ra
            dec = source.pos.dec
            source_name = source.name
            targets_flux[name] = [I_out, I_out_err, I_in, source_name]
            if ra > np.pi:
                ra -= 2.0*np.pi
            delta_pos_angle = angular_dist_pos_angle(RA, DEC, ra, dec)
            delta_pos_angle_arc_sec = rad2arcsec(delta_pos_angle[0])
            delta_phase_centre = angular_dist_pos_angle(RA0, DEC0, ra, dec)
            delta_phase_centre_arc_sec = rad2arcsec(delta_phase_centre[0])
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


def compare_models(models, tolerance=0.00001, plot=True):
    """Plot model1 source properties against that of model2

    Parameters
    ----------
    models : dict
        Tigger formatted model files e.g {model1: model2}.
    tolerance : float
        Tolerace in detecting source from model 2.
    plot : bool
        Output html plot from which a png can be obtained.

    Returns
    -------
    results : dict
        Dictionary of source properties from each model.

    """
    results = dict()
    input_model = models[0]
    output_model = models[1]
    heading = input_model["label"]
    results[heading] = {'models': [input_model["path"], output_model["path"]]}
    results[heading]['flux'] = []
    results[heading]['shape'] = []
    results[heading]['position'] = []
    props = get_detected_sources_properties('{:s}'.format(input_model["path"]),
                                            '{:s}'.format(output_model["path"]),
                                            tolerance)  # TOD0 area to be same as beam
    for i in range(len(props[0])):
        results[heading]['flux'].append(props[0].items()[i][-1])
    for i in range(len(props[1])):
        results[heading]['shape'].append(props[1].items()[i][-1])
    for i in range(len(props[2])):
        results[heading]['position'].append(props[2].items()[i][-1])
    if plot:
        _source_flux_plotter(results, models)
        _source_astrometry_plotter(results, models)
    return results


def compare_residuals(residuals, skymodel=None, points=None, plot=True):
    if skymodel:
        res = _source_residual_results(residuals, skymodel, area_factor=2)
    else:
        res = _random_residual_results(residuals, points)
    if plot:
        _residual_plotter(residuals, results=res, points=points)
    return res


def _source_flux_plotter(results, models):
    """Plot flux results and save output as html file.

    Parameters
    ----------
    results : dict
        Structured output results.
    models : list
        Tigger/text formatted model files e.g [model1, model2].

    """
    im_titles = []
    output_model = models[-1]['path']
    if 'html' in output_model:
        header = output_model[:-9]
    else:
        header = output_model[:-4]
    im_titles.append('<b>{:s} flux density</b>'.format(header.upper()))

    fig = tools.make_subplots(rows=1, cols=1, shared_yaxes=False,
                              print_grid=False, horizontal_spacing=0.005,
                              vertical_spacing=0.15, subplot_titles=im_titles)
    i = 0
    counter = 1
    output_model = models[1]['path']
    annotate = []
    name_labels = []
    flux_in_data = []
    flux_out_data = []
    source_scale = []
    phase_center_dist = []
    flux_out_err_data = []
    heading = models[0]['label']
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
        phase_center_dist, source_scale) = zip(*sorted(
                zipped_props, key=lambda x: x[0]))

    flux_MSE = mean_squared_error(flux_in_data, flux_out_data)
    reg = linregress(flux_in_data, flux_out_data)
    flux_R_score = reg.rvalue
    annotate.append(go.Annotation(
            x=0.0012*UNIT_SCALER['milli'],
            y=flux_in_data[-1]*UNIT_SCALER['milli'] + 0.0005*UNIT_SCALER['milli'],
            xref='x{:d}'.format(counter),
            yref='y{:d}'.format(counter),
            text="Slope: {:.4f} | Intercept: {:.4f} | RMS Error: {:.4f} | R2: {:.4f} ".format(
                    reg.slope, reg.intercept*UNIT_SCALER['milli'],
                    np.sqrt(flux_MSE)*UNIT_SCALER['milli'], flux_R_score),
            ax=0,
            ay=-10,
            showarrow=False,
            bordercolor='#c7c7c7',
            borderwidth=2,
            font=dict(color="black", size=15),
        ))
    fig.append_trace(go.Scatter(x=np.array([flux_in_data[0],
                                            flux_in_data[-1]])*UNIT_SCALER['milli'],
                                showlegend=False,
                                y=np.array([flux_in_data[0],
                                            flux_in_data[-1]])*UNIT_SCALER['milli'],
                                mode='line'), i+1, 1)
    fig.append_trace(go.Scatter(x=np.array(flux_in_data)*UNIT_SCALER['milli'],
                                y=np.array(flux_out_data)*UNIT_SCALER['milli'],
                                mode='markers', showlegend=False,
                                text=name_labels, name='{:s} flux_ratio'.format(heading),
                                marker=dict(color=phase_center_dist,
                                            showscale=True, colorscale='Jet',
                                            reversescale=False,
                                            colorbar=dict(
                                                title='Distance from phase center (arcsec)',
                                                titleside='right',
                                                titlefont=dict(size=16), x=1.0)),
                                error_y=dict(type='data',
                                             array=np.array(flux_out_err_data)*UNIT_SCALER['milli'],
                                             color='rgb(158, 63, 221)',
                                             visible=True)), i+1, 1)
    fig['layout'].update(title='', height=900, width=900,
                         paper_bgcolor='rgb(255,255,255)',
                         plot_bgcolor='rgb(229,229,229)',
                         legend=dict(x=0.8, y=1.0),)
    fig['layout'].update(
        {'yaxis{}'.format(counter): YAxis(title=u'$I_{out}$ (mJy)',
                                          gridcolor='rgb(255,255,255)',
                                          tickfont=dict(size=15),
                                          titlefont=dict(size=17),
                                          showgrid=True,
                                          showline=False,
                                          showticklabels=True,
                                          tickcolor='rgb(51,153,225)',
                                          ticks='outside',
                                          zeroline=False)})
    fig['layout'].update({'xaxis{}'.format(counter+i): XAxis(title=u'$I_{in}$ (mJy)',
                                                             position=0.0,
                                                             titlefont=dict(size=17),
                                                             overlaying='x')})
    fig['layout']['annotations'].update({'font': {'size': 18}})
    fig['layout']['annotations'].extend(annotate)
    outfile = 'InputOutputFluxDensity.html'
    py.plot(fig, filename=outfile, auto_open=False)


def _source_astrometry_plotter(results, models):
    """Plot astrometry results and save output as html file.

    Parameters
    ----------
    results: dict
        Structured output results.
    models: list
        Tigger/text formatted model files e.g [model1, model2].

    """
    PLOTS = 1
    im_titles = []
    output_model = models[-1]['path']
    if 'html' in output_model:
        header = output_model[:-9]
    else:
        header = output_model[:-4]
    im_titles.append('<b>{:s} Position Offset</b>'.format(header.upper()))
    im_titles.append('<b>{:s} Delta Position</b>'.format(header.upper()))

    fig = tools.make_subplots(rows=1, cols=2, shared_yaxes=False, print_grid=False,
                              horizontal_spacing=0.15,
                              vertical_spacing=0.15,
                              subplot_titles=im_titles)

    output_model = models[1]['path']
    i = 0
    counter = 1
    annotate = []
    RA_offset = []
    DEC_offset = []
    DELTA_PHASE0 = []
    source_labels = []
    flux_in_data = []
    flux_out_data = []
    delta_pos_data = []
    heading = models[0]['label']
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
    fig.append_trace(go.Scatter(x=np.array(flux_in_data)*UNIT_SCALER['milli'],
                                y=np.array(delta_pos_data),
                                mode='markers', showlegend=False,
                                text=source_labels, name='{:s} flux_ratio'.format(header),
                                marker=dict(color=DELTA_PHASE0, showscale=True,
                                            colorscale='Jet', reversescale=True,
                                            colorbar=dict(title='Distance from phase center (arcsec)',
                                                          titleside='right',
                                                          len=PLOT_NUM['colorbar'][PLOTS][2],
                                                          y=PLOT_NUM['colorbar'][PLOTS][1])
                                            )), i+1, 2)
    fig.append_trace(go.Scatter(x=np.array(RA_offset), y=np.array(DEC_offset),
                                mode='markers', showlegend=False,
                                text=source_labels, name='{:s} flux_ratio'.format(heading),
                                marker=dict(color=np.array(flux_out_data)*UNIT_SCALER['milli'],
                                            showscale=True,
                                            colorscale='Viridis',
                                            reversescale=True,
                                            colorbar=dict(title='Output flux (mJy)',
                                                          titleside='right',
                                                          len=PLOT_NUM['colorbar'][PLOTS][2],
                                                          y=PLOT_NUM['colorbar'][PLOTS][1],
                                                          x=0.45)
                                            )), i+1, 1)

    RA_mean = np.mean(RA_offset)
    DEC_mean = np.mean(DEC_offset)
    r1, r2 = np.array(RA_offset).std(), np.array(DEC_offset).std()
    pi, cos, sin = np.pi, np.cos, np.sin
    theta = np.linspace(0, 2*pi, len(DEC_offset))
    x1 = RA_mean+(r1*cos(theta))
    y1 = DEC_mean+(r2*sin(theta))
    recovered_sources = len(DEC_offset)
    one_sigma_sources = len([(ra_off, dec_off) for ra_off, dec_off in zip(RA_offset, DEC_offset)
                            if abs(ra_off) <= max(abs(x1)) and abs(dec_off) <= max(abs(y1))])
    annotate.append(go.Annotation(
            x=RA_mean*3,
            y=max(DEC_offset) + 0.05,
            xref='x{:d}'.format(counter),
            yref='y{:d}'.format(counter),
            text="Total sources: {:d} | (RA, DEC) mean: ({:.4f}, {:.4f}) |"
                 "  (RA, DEC) sigma: ({:.4f}, {:.4f}) | sigma sources: {:d}".format(
                    recovered_sources, RA_mean, DEC_mean, r1, r2, one_sigma_sources),
            ax=0,
            ay=-40,
            showarrow=False,
            bordercolor='#c7c7c7',
            borderwidth=2,
            font=dict(color="black", size=10),
        ))
    fig.append_trace(go.Scatter(x=x1, y=y1,
                                mode='lines', showlegend=False,
                                name=r'1 sigma',
                                text=r'1 sigma ~ {:f}'.format(np.sqrt(r1*r2)),
                                marker=dict(color='rgb(0, 0, 255)')), i+1, 1)
    fig['layout'].update(title='', height=800, width=1800,
                         paper_bgcolor='rgb(255,255,255)', plot_bgcolor=BG_COLOR,
                         legend=dict(xanchor=True, x=1.2, y=1))
    fig['layout'].update(
        {'yaxis{}'.format(counter+i): YAxis(title=u'Dec offset [arcsec]',
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
        {'yaxis{}'.format(counter+i+1): YAxis(title='Delta position [arcsec]',
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
    fig['layout'].update({'xaxis{}'.format(counter+i): XAxis(title=u'RA offset [arcsec]',
                                                             titlefont=dict(size=17),
                                                             zeroline=True,
                                                             position=0.0,
                                                             overlaying='x',)})
    fig['layout'].update({'xaxis{}'.format(counter+i+1): XAxis(title=u'$I_{in}$ (mJy)',
                                                               titlefont=dict(size=17),
                                                               zeroline=False)})
    fig['layout']['annotations'].update({'font': {'size': 18}})
    fig['layout']['annotations'].extend(annotate)
    outfile = 'InputOutputPosition.html'
    py.plot(fig, filename=outfile, auto_open=False)


def _residual_plotter(res_noise_images, points=None, results=None):
    """Plot ratios of random residuals and noise

    Parameters
    ----------
    res_noise_images: dict
        Structured input images with labels.
    points: int
        Number of data point to generate in case of random residuals
    results: dict
        Structured output results.

    """
    # Converter
    TO_MICRO = UNIT_SCALER['micro']
    # Plot titles
    im_titles = []
    # Get residual image names
    res_image = res_noise_images[0]['path']
    noise_image = res_noise_images[-1]['path']
    # Get label
    label = res_noise_images[0]['label']
    if 'None' in label:
        label = res_image[:-5]
    # Assign plot titles
    header1 = res_image[:-5]
    header2 = noise_image[:-5]
    im_titles.append('<b>{:s} RMS</b>'.format(header1.upper()))
    im_titles.append('<b>{:s} residual-noise</b>'.format(header2.upper()))
    # Create figure canvas
    fig = tools.make_subplots(rows=1, cols=2,
                              shared_yaxes=False,
                              print_grid=False,
                              horizontal_spacing=0.15,
                              vertical_spacing=0.15,
                              subplot_titles=im_titles)
    i = 0
    counter = 1
    rmss = []
    residuals = []
    name_labels = []
    dist_from_phase = []
    res_noise_ratio = []
    for res_src in results[label]:
        rmss.append(res_src[0])
        residuals.append(res_src[1])
        res_noise_ratio.append(res_src[2])
        dist_from_phase.append(res_src[3])
        name_labels.append(res_src[4])
    fig.append_trace(go.Scatter(x=range(len(rmss)), y=np.array(rmss)*TO_MICRO,
                                mode='lines',
                                showlegend=True if i == 0 else False,
                                name='residual image 1',
                                text=name_labels,
                                marker=dict(color='rgb(255,0,0)'),
                                error_y=dict(type='data',
                                             color='rgb(158, 63, 221)',
                                             visible=True)), i+1, 1)
    fig.append_trace(go.Scatter(x=range(len(rmss)), y=np.array(residuals)*TO_MICRO,
                                mode='lines', showlegend=True if i == 0 else False,
                                name='residual image 2',
                                text=name_labels,
                                marker=dict(color='rgb(0,0,255)'),
                                error_y=dict(type='data',
                                             color='rgb(158, 63, 221)', visible=True)), i+1, 1)
    fig.append_trace(go.Scatter(x=range(len(rmss)), y=np.array(res_noise_ratio),
                                mode='markers', showlegend=False,
                                text=name_labels,
                                marker=dict(color=dist_from_phase, showscale=True, colorscale='Jet',
                                colorbar=dict(title='Phase center dist (arcsec)',
                                              titleside='right')),
                                error_y=dict(type='data',
                                             color='rgb(158, 63, 221)', visible=True)), i+1, 2)
    fig['layout'].update(title='', height=800, width=1500,
                         paper_bgcolor='rgb(255,255,255)', plot_bgcolor='rgb(229,229,229)',
                         legend=dict(xanchor=True, x=1.0, y=1.05))
    fig['layout'].update(
        {'yaxis{}'.format(counter+i): YAxis(title=u'rms [\u03BCJy]',
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
        {'yaxis{}'.format(counter+i+1): YAxis(title=u'${I_{res}/I_{noise}}$',
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
    fig['layout'].update({'xaxis{}'.format(counter+i): XAxis(title='Sources',
                                                             titlefont=dict(size=17),
                                                             showline=True,
                                                             zeroline=False,
                                                             position=0.0,
                                                             overlaying='x')})
    fig['layout'].update({'xaxis{}'.format(counter+i+1): XAxis(title='Sources',
                                                               titlefont=dict(size=17),
                                                               showline=True,
                                                               zeroline=False)})
    if points:
        outfile = 'RandomResidualNoiseRatio.html'
    else:
        outfile = 'SourceResidualNoiseRatio.html'
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
    # Quick converting functions
    rad = lambda a: a*(180/np.pi)  # convert radians to degrees
    deg2arcsec = lambda a: a*3600  # convert degrees to arcsec
    # Dictinary to store results
    results = dict()
    # Get residual image names
    res_image = res_noise_images[0]['path']
    noise_image = res_noise_images[-1]['path']
    # Residual-noise dictionary
    res_noise_image_dict = {res_image: noise_image}
    # Get label
    label = res_noise_images[0]['label']
    if 'None' in label:
        label = res_image[:-5]
    # Get fits info
    fits_info = fitsInfo(res_image)
    # Get beam size otherwise use default (5``).
    beam_default = (0.00151582804885738, 0.00128031965017612, 20.0197348935424)
    beam_deg = fits_info['b_size'] if fits_info['b_size'] else beam_default
    # Get random pixel coordinates
    pix_coord_deg = _get_random_pixel_coord(data_points,
                                            sky_area=fits_info['skyArea']*0.9,
                                            phase_centre=fits_info['centre'])
    # Source counter
    i = 0
    for res_image, noise_image in res_noise_image_dict.items():
        # Open noise header
        noise_hdu = fitsio.open(noise_image)
        # Get data from noise image
        noise_data = noise_hdu[0].data
        # Data structure for each residuals to compare
        results[label] = []
        residual_hdu = fitsio.open(res_image)
        # Get the header data unit for the residual rms
        residual_data = residual_hdu[0].data
        # Get the number of frequency channels
        nchan = (residual_data.shape[1]
                 if residual_data.shape[0] == 1
                 else residual_data.shape[0])
        for RA, DEC in pix_coord_deg:
            i += 1
            # Get width of box around source
            width = int(deg2arcsec(beam_deg[0])*area_factor)
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
            results[label].append([noise_rms,
                                   flux_std,
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
    # Quick converting functions
    rad = lambda a: a*(180/np.pi)  # convert radians to degrees
    deg2arcsec = lambda a: a*3600  # convert degrees to arcsec
    # Dictinary to store results
    results = dict()
    # Get residual image names
    res_image = res_noise_images[0]['path']
    noise_image = res_noise_images[-1]['path']
    # Get label
    label = res_noise_images[0]['label']
    if 'None' in label:
        label = res_image[:-5]
    # Get fits info
    fits_info = fitsInfo(res_image)
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
        RA = rad(ra)
        DEC = rad(dec)
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
        width = int(deg2arcsec(beam_deg[0])*area_factor)
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
        results[label].append([noise_rms, flux_std,
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
                 description="Examine radio image fidelity by obtaining: \n"
                             "- The four (4) moments of a residual image \n"
                             "- The Dynamic range in restored image \n"
                             "- Comparing the tigger input and output model sources \n"
                             "- Comparing the on source/random residuals to noise")
    argument = partial(parser.add_argument)
    argument('--tigger-model',  dest='model',
             help='Name of the tigger model lsm.html file')
    argument('--restored-image',  dest='restored',
             help='Name of the restored image fits file')
    argument('-psf', '--psf-image',  dest='psf',
             help='Name of the point spread function file or psf size in arcsec')
    argument('--residual-image',  dest='residual',
             help='Name of the residual image fits file')
    argument('--normality-test',  dest='test_normality',
             help='Name of model to use for normality testing. \n'
                  'options: [shapiro, normaltest] \n'
                  'NB: normaltest is the D`Agostino')
    argument('-dr', '--data-range',  dest='data_range',
             help='Data range to perform normality testing')
    argument('-af', '--area-factor', dest='factor', type=float, default=6,
             help='Factor to multiply the beam area to get target peak area')
    argument('--compare-models', dest='models', nargs="+", type=str,
             help='List of tigger model (text/lsm.html) files to compare \n'
                  'e.g. --compare-models model1.lsm.html model2.lsm.html')
    argument('--compare-residuals', dest='noise', nargs="+", type=str,
             help='List of noise-like (fits) files to compare \n'
                  'e.g. --compare-residuals2noise residuals.fits noise.fits')
    argument('-dp', '--data-points',  dest='points',
             help='Data points to sample the residual/noise image')
    argument("--label",
             help='Use this label instead of the FITS image path when saving'
                  'data as JSON file')
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
        output_dict[residual_label] = dict(
                    stats.items() + {model_label: {
                        'DR': DR["global_rms"],
                        'DR_deepest_negative'   : DR["deepest_negative"],
                        'DR_global_rms'         : DR['global_rms'],
                        'DR_local_rms'          : DR['local_rms'],
                        }}.items())
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

    if args.restored:
        if args.factor:
            DR = image_dynamic_range(args.restored, area_factor=args.factor)
        else:
            DR = image_dynamic_range(args.restored)
        output_dict[restored_label] = {
                            'DR': DR["global_rms"],
                            'DR_deepest_negative' : DR["deepest_negative"],
                            'DR_global_rms' : DR['global_rms'],
                            'DR_local_rms'  : DR['local_rms'],
                        }

    if args.models:
        models = args.models
        print("Number of model files: {:d}".format(len(models)))
        if len(models) > 2 or len(models) < 2:
            print("{:s}Can only compare two models at a time.{:s}".format(R, W))
        else:
            model1, model2 = models
            output_dict = compare_models(
                    [
                        dict(label="{0:s}-model1".format(args.label), path=model1),
                        dict(label="{0:s}-model2".format(args.label), path=model2),
                    ]
                )

    if args.noise:
        residuals = args.noise
        print("Number of model files: {:d}".format(len(residuals)))
        if len(residuals) > 2 or len(residuals) < 2:
            print("{:s}Can only compare two models at a time.{:s}".format(R, W))
        else:
            noise1, noise2 = residuals
            if args.model:
                output_dict = compare_residuals(
                        [
                            dict(label="{0:s}-noise1".format(args.label), path=noise1),
                            dict(label="{0:s}-noise2".format(args.label), path=noise2),
                        ], args.model
                    )
            else:
                output_dict = compare_residuals(
                        [
                            dict(label="{0:s}-noise1".format(args.label), path=noise1),
                            dict(label="{0:s}-noise2".format(args.label), path=noise2),
                        ], points=int(args.points) if args.points else 100
                    )

    if output_dict and not args.noise and not args.models:
        json_dump(output_dict)
