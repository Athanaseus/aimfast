import json
import Tigger
import argparse
import numpy as np
from scipy import stats
from plotly import tools
from functools import partial
from astLib.astWCS import WCS
import plotly.graph_objs as go
from plotly import offline as py
from astropy.io import fits as fitsio
from scipy.interpolate import interp1d
from plotly.graph_objs import XAxis, YAxis
import scipy.ndimage.measurements as measure
from Tigger.Coordinates import angular_dist_pos_angle


def deg2arcsec(x):
    """Converts 'x' from degrees to arcseconds"""
    return float(x)*3600.00


def rad2deg(x):
    """Converts 'x' from radian to degrees"""
    return float(x)*(180/np.pi)


def rad2arcsec(x):
    """Converts `x` from radians to arcseconds"""
    return float(x)*3600.0*180.0/np.pi


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
    beam_size = (hdr['BMAJ'], hdr['BMIN'], hdr['BPA'])
    fitsinfo = {'wcs': wcs, 'ra': ra, 'dec': dec,
                'dra': dra, 'ddec': ddec, 'raPix': raPix,
                'decPix': decPix,  'b_size': beam_size}
    return fitsinfo


def measure_psf(psffile, arcsec_size=20):
    """Measure point spread function after deconvolution

    Parameters
    ----------
    psfile: fits file
        point spread function file

    Returns
    -------
    r0: float
        Average psf size

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
    """Get box of width w around source coordinates radec

    Parameters
    ----------
    radec: tuple
        RA and DEC in degrees
    w: int
        width of box
    wcs: astLib.astWCS.WCS instance
        World Coordinate System

    Returns
    -------
    box: tuple
        A box centered at radec
    """
    raPix, decPix = wcs.wcs2pix(*radec)
    raPix = int(raPix)
    decPix = int(decPix)
    box = slice(decPix-w/2, decPix+w/2), slice(raPix-w/2, raPix+w/2)
    return box


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


def model_dynamic_range(lsmname, fitsname, beam_size=5, area_factor=2):
    """Gets the dynamic range using model lsm and residual fits

    Parameters
    ----------
    fitsname: fits file
        residual image (cube)
    lsmname: lsm.html or .txt file
        model .lsm.html from pybdsm (or .txt converted tigger file)
    beam_size: float
        Average beam size in arcsec
    area_factor: float
        Factor to multiply the beam area

    Returns
    -------
    (DR, peak_flux, min_flux): tuple
        DR - dynamic range value
        peak_flux - peak flux source in the image
        min_flux - min flux pixel value in the image

    Note
    ----
    DR = Peak source from model / deepest negative around source position in residual

    """
    try:
        fits_info = fitsInfo(fitsname)
        beam_deg = fits_info['b_size']
        beam_size = beam_deg[0]*3600
    except IOError:
        pass
    # Open the residual image
    residual_hdu = fitsio.open(fitsname)
    residual_data = residual_hdu[0].data
    # Load model file
    model_lsm = Tigger.load(lsmname)
    # Get detected sources
    model_sources = model_lsm.sources
    # Obtain peak flux source
    sources_flux = dict([(model_source, model_source.flux.I)
                        for model_source in model_sources])
    peak_source_flux = [(_model_source, flux)
                        for _model_source, flux in sources_flux.items()
                        if flux == max(sources_flux.values())][0][0]
    peak_flux = peak_source_flux.flux.I
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
    # Compute dynamic range
    DR = peak_flux/abs(min_flux)
    return (DR, peak_flux, min_flux)


def image_dynamic_range(fitsname, area_factor=6):
    """Gets the dynamic range in a restored image

    Parameters
    ----------
    fitsname: fits file
        restored image (cube)
    area_factor: int
        Factor to multiply the beam area

    Returns
    -------
    (DR, peak_flux, min_flux): tuple
        DR - dynamic range value
        peak_flux - peak flux source in the image
        min_flux - min flux pixel value in the image

    Note
    ----
    DR = Peak source / deepest negative around source position

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
    return (DR, peak_flux, min_flux)


def get_src_scale(source_shape):
    """Get scale measure of the source in arcsec"""
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


def get_detected_sources_properties(model_lsm_file, pybdsm_lsm_file, area_factor):
    """Extracts the output simulation sources properties"""
    model_lsm = Tigger.load(model_lsm_file)
    pybdsm_lsm = Tigger.load(pybdsm_lsm_file)
    # Sources from the input model
    model_sources = model_lsm.sources
    # {"source_name": [I_out, I_out_err, I_in, source_name]}
    targets_flux = dict()       # recovered sources flux
    # {"source_name": [delta_pos_angle_arc_sec, ra_offset, dec_offset,
    #                  delta_phase_centre_arc_sec, I_in]
    targets_position = dict()   # recovered sources position
    # {"source_name: [(majx_out, minx_out, pos_angle_out),
    #                 (majx_in, min_in, pos_angle_in),
    #                 scale_out, scale_out_err, I_in]
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


def compare_models(models, tolerance=0.0001, plot=True):
    """Plot model1 source properties against that of model2

    Parameters
    ----------
    models: dict
        Tigger formatted model files e.g {model1: model2}
    tolerance: float
        Tolerace in detecting source from model 2
    plot: bool
        Output html plot from which a png can be obtained

    Returns
    -------
    results: dict
        Dictionary of source properties from each model
    """
    results = dict()
    for input_model, output_model in models.items():
        heading = output_model[:-9]
        results[heading] = {'models': [input_model, output_model]}
        results[heading]['flux'] = []
        results[heading]['shape'] = []
        results[heading]['position'] = []
        props = get_detected_sources_properties('{:s}'.format(input_model),
                                                '{:s}'.format(output_model),
                                                tolerance)  # TOD0 area to be same as beam
        for i in range(len(props[0])):
            results[heading]['flux'].append(props[0].items()[i][-1])
        for i in range(len(props[1])):
            results[heading]['shape'].append(props[1].items()[i][-1])
        for i in range(len(props[2])):
            results[heading]['position'].append(props[2].items()[i][-1])
        if plot:
            _source_property_ploter(results, models)
    return results


def _source_property_ploter(results, models):
    """Plot results"""
    im_titles = []
    for input_model, output_model in models.items():
        header = output_model[:-9].split('_')[0]
        im_titles.append('<b>{:s} flux density</b>'.format(header.upper()))

    fig = tools.make_subplots(rows=len(models.keys()), cols=1, shared_yaxes=False,
                              print_grid=False, horizontal_spacing=0.005,
                              vertical_spacing=0.15, subplot_titles=im_titles)
    i = -1
    counter = 0
    for input_model, output_model in models.items():
        i += 1
        counter += 1
        name_labels = []
        flux_in_data = []
        flux_out_data = []
        source_scale = []
        phase_center_dist = []
        flux_out_err_data = []
        heading = output_model[:-9]
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
        fig.append_trace(go.Scatter(x=np.array([flux_in_data[0], flux_in_data[-1]]),
                                    showlegend=False,
                                    y=np.array([flux_in_data[0],
                                                flux_in_data[-1]]),
                                    mode='line'), i+1, 1)
        fig.append_trace(go.Scatter(x=np.array(flux_in_data), y=np.array(flux_out_data),
                                    mode='markers', showlegend=False,
                                    text=name_labels, name='{:s} flux_ratio'.format(heading),
                                    marker=dict(color=phase_center_dist,
                                                showscale=True, colorscale='Jet',
                                                reversescale=False,
                                                colorbar=dict(
                                                    title='Distance from phase center (arcsec)',
                                                    titleside='right',
                                                    titlefont=dict(size=16), x=1.0)
                                               ) if i == 0 else
                                    dict(color=phase_center_dist,
                                         colorscale='Jet',
                                         reversescale=False),
                                    error_y=dict(type='data', array=flux_out_err_data,
                                                 color='rgb(158, 63, 221)',
                                                 visible=True)), i+1, 1)
        fig['layout'].update(title='', height=900, width=900,
                             paper_bgcolor='rgb(255,255,255)',
                             plot_bgcolor='rgb(229,229,229)',
                             legend=dict(x=0.8, y=1.0),)
        fig['layout'].update(
            {'yaxis{}'.format(counter): YAxis(title=u'I_out (Jy)',
                                              gridcolor='rgb(255,255,255)',
                                              tickfont=dict(size=15),
                                              titlefont=dict(size=17),
                                              showgrid=True,
                                              showline=False,
                                              showticklabels=True,
                                              tickcolor='rgb(51,153,225)',
                                              ticks='outside',
                                              zeroline=False)})
        fig['layout'].update({'xaxis{}'.format(counter+i): XAxis(title='I_in (Jy)',
                                                                 position=0.0,
                                                                 titlefont=dict(size=17),
                                                                 overlaying='x')})
    outfile = 'InputOutputFluxDensity'
    py.plot(fig, filename=outfile)


def get_argparser():
    """Get argument parser"""
    parser = argparse.ArgumentParser(
                 description="Examine radio image fidelity by obtaining: \n"
                             "- The four (4) moments of a residual image \n"
                             "- The Dynamic range in restored image \n"
                             "- Comparing the tigger input and output model sources")
    argument = partial(parser.add_argument)
    argument('--tigger-model',  dest='model',
             help='Name of the tigger model lsm.html file')
    argument('--restored-image',  dest='restored',
             help='Name of the restored image fits file')
    argument('-psf', '--psf-image',  dest='psf',
             help='Name of the point spread function file or psf size in arcsec')
    argument('--residual-image',  dest='residual',
             help='Name of the residual image fits file')
    argument('-af', '--area-factor', dest='factor', type=float, default=6,
             help='Factor to multiply the beam area to get target peak area')
    argument('--compare-models',  dest='models', nargs="+", type=str,
             help='List of tigger model (text/lsm.html) files to compare \n'
                  'e.g. --compare-models model1.lsm.html, model2.lsm.html')
    return parser


def main():
    """Main function"""
    parser = get_argparser()
    args = parser.parse_args()
    output_dict = dict()
    R = '\033[31m'  # red
    W = '\033[0m'   # white (normal)
    if not args.residual and not args.restored and not args.model and not args.models:
        print("{:s}Please provide lsm.html/fits file name(s)."
              "\nOr\naimfast -h for arguments{:s}".format(R, W))
    if args.model:
        if not args.residual:
            print("{:s}Please provide residual fits file{:s}".format(R, W))
        else:
            if args.psf:
                if '.fits' in args.psf:
                    psf_size = measure_psf(args.psf)
                else:
                    psf_size = int(args.psf)
            else:
                psf_size = 5
            if args.factor:
                DR = model_dynamic_range(args.model, args.residual, psf_size,
                                         area_factor=args.factor)[0]
            else:
                DR = model_dynamic_range(args.model, args.residual, psf_size)[0]
                print("{:s}Please provide psf fits file or psf size.\n"
                      "Otherwise a default beam size of five (5``) asec "
                      "is used{:s}".format(R, W))
            stats = residual_image_stats(args.residual)
            output_dict[args.model] = {'DR': DR}
            output_dict[args.residual] = stats
    if args.residual:
        if args.residual not in output_dict.keys():
            stats = residual_image_stats(args.residual)
            output_dict[args.residual] = stats
    if args.restored:
        if args.factor:
            DR = image_dynamic_range(args.restored, area_factor=args.factor)[0]
        else:
            DR = image_dynamic_range(args.restored)[0]
        output_dict[args.restored] = {'DR': DR}
    if args.models:
        models = args.models
        print("Number of model files: {:s}".format(len(models)))
        if len(models) > 2 or len(models) < 2:
            print("{:s}Can only compare two models at a time.{:s}".format(R, W))
        else:
            model1, model2 = models
            output_dict = compare_models({model1: model2})
    if output_dict:
        json_dump(output_dict)
        print(output_dict)
