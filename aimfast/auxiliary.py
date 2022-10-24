import os
import sys
import subprocess

import numpy
import numpy as np

from astropy.io import ascii
from astropy import units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from astropy.io import fits as pyfits
from astropy.io import fits as fitsio
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord


def deg2arcsec(x):
    """Converts 'x' from degrees to arcseconds

    Parameters
    ----------
    x : float
        Angle in deg

    Returns
    -------
    result : float
        Angle in arcsec

    """
    result = float(x) * 3600.00
    return result


def rad2deg(x):
    """Converts 'x' from radian to degrees

    Parameters
    ----------
    x : float
        Angle in radians

    Returns
    -------
    result : float
        Angle in degrees

    """
    result = float(x) * (180 / np.pi)
    return result


def deg2rad(x):
    """Converts 'x' from degrees to radians

    Parameters
    ----------
    x : float
        Angle in degree

    Returns
    -------
    result : float
        Angle in radians

    """
    result = float(x) * (np.pi/ 180)
    return result


def rad2arcsec(x):
    """Converts `x` from radians to arcseconds

    Parameters
    ----------
    x : float
        Angle in radians

    Returns
    -------
    result : float
        Angle in arcsec

    """
    result = float(x) * (3600.0 * 180.0 / np.pi)
    return result


def ra2deg(ra_hms):
    """Converts right ascension in hms coordinates to degrees

    Parameters
    ----------
    ra_hms : str
        ra in HH:MM:SS format

    Returns
    -------
    h_m_s : float
        conv_units.radeg: ra in degrees

    """
    ra = ra_hms.split(':')
    hh = float(ra[0]) * 15.0
    mm = (float(ra[1]) / 60.0) * 15.0
    ss = (float(ra[2]) / 3600) * 15.0
    h_m_s = hh + mm + ss
    return h_m_s


def deg2ra(ra_deg, deci=2):
    """Converts right ascension in hms coordinates to degrees

    Parameters
    ----------
    ra_deg : float
    ra in degrees format

    Returns
    -------
    HH:MM:SS : str

    """
    if ra_deg < 0:
       ra_deg = 360 + ra_deg
    HH     = int((ra_deg*24)/360.)
    MM     = int((((ra_deg*24)/360.)-HH)*60)
    SS     = round(((((((ra_deg*24)/360.)-HH)*60)-MM)*60), deci)
    return "%s:%s:%s"%(HH,MM,SS)


def dec2deg(dec_dms):
    """Converts declination in dms coordinates to degrees

    Parameters
    ----------
    dec_hms : str
        dec in DD:MM:SS format

    Returns
    -------
    hms : float
        conv_units.radeg: dec in degrees

    """
    if ':' not in dec_dms:
        # In the case dec is specified as -30.12.40.2 (Not sure why).
        dec_dms = dec_dms.split('.')
        dec_dms = ':'.join(dec_dms[:3])
        dec_dms += f'.{dec_dms[-1]}'
    dec = dec_dms.split(':')
    dd = abs(float(dec[0]))
    mm = float(dec[1]) / 60
    ss = float(dec[2]) / 3600
    if float(dec[0]) >= 0:
        return dd + mm + ss
    else:
        return -(dd + mm + ss)
    d_m_s = dd + mm + ss
    return d_m_s


def deg2dec(dec_deg, deci=2):
    """Converts declination in degrees to dms coordinates

    Parameters
    ----------
    dec_deg : float
      Declination in degrees
    dec: int
      Decimal places in float format

    Returns
    -------
    dms : str
      Declination in degrees:arcmin:arcsec format
    """
    DD          = int(dec_deg)
    dec_deg_abs = np.abs(dec_deg)
    DD_abs      = np.abs(DD)
    MM          = int((dec_deg_abs - DD_abs)*60)
    SS          = round((((dec_deg_abs - DD_abs)*60)-MM), deci)
    return "%s:%s:%s"%(DD,MM,SS)

def unwrap(angle):
    """Unwrap angle greater than 180"""
    if angle > 180:
        angle -= 360
    return angle


def compute_in_out_slice(N, N0, R, R0):
    """Given an input axis of size N, and an output axis of size N0,
    and reference pixels of R and R0 respectively, computes two slice
    objects such that A0[slice_out] = A1[slice_in]
    would do the correct assignment (with I mapping to I0,
    and the overlapping regions transferred)
    """
    i, j = 0, N     # input slice
    i0 = R0 - R
    j0 = i0 + N     # output slice
    if i0 < 0:
        i = -i0
        i0 = 0
    if j0 > N0:
        j = N - (j0 - N0)
        j0 = N0
    if i >= j or i0 >= j0:
        return None, None
    return slice(i0, j0), slice(i, j)


def get_subimage(fitsname, centre_coord, size, padding=1):
    """Get a subimage given coordinates and size for area to extract

    Parameters
    ----------
    fitsname: str
      Fits file name to extract subimage
    centre_coord: list
      List of centre coordinates pixels to be extracted
    padding: float
      Amount of subimage padding
    size: int
      Size of output subimage in pixels (sizexsize)


    Returns
    -------
    subimage: header.data
      Header data unit for the subimage
    """
    # open image and obtain number of pixels
    image = fitsio.open(fitsname)
    data = image[0].data
    nx = data.shape[-1]
    ny = data.shape[-2]
    # make output array of shape size x size
    subdata_shape = list(data.shape)
    subdata_shape[-1] = subdata_shape[-2] = size
    subdata = numpy.zeros(subdata_shape, dtype=data.dtype)
    # make input/output slices
    rx, ry = centre_coord
    rx0 = ry0 = size//2
    xout, xin = compute_in_out_slice(nx, size, rx, rx0)
    yout, yin = compute_in_out_slice(ny, size, ry, ry0)
    subdata[..., yout, xout] = data[..., yin, xin]

    return subdata


def get_online_catalog(catalog='NVSS', width='5.0d', thresh=None,
                       centre_coord=['0:0:0', '-30:0:0'],
                       catalog_table='nvss_catalog_table.txt'):
    """Query an online catalog to compare with local catalog

    Parameters
    ----------
    catalog : str
        Name of online catalog to query
    width : str
        The width of the field iin degrees
    thresh : float
        Flux density threshold to select sources (mJy)
    centre_coord : list
        List of central coordinates of the region of interest [RA, DEC]
    catalog_table : str
        Name of output catalog table with results

    Return
    ------
    table : Table
        Table with online catalog data

    """
    Vizier.ROW_LIMIT = -1
    C = Vizier.query_region(coord.SkyCoord(centre_coord[0], centre_coord[1],
                            unit=(u.hourangle, u.deg), frame='icrs'),
                            width=width, catalog=catalog)
    if C.values():

        table = C[0]
        ra_deg = []
        dec_deg = []

        if catalog in ['NVSS', 'SUMSS']:
            for i in range(0, len(table['RAJ2000'])):
                table['RAJ2000'][i] = ':'.join(table['RAJ2000'][i].split(' '))
                ra_deg.append(ra2deg(table['RAJ2000'][i]))
                table['DEJ2000'][i] = ':'.join(table['DEJ2000'][i].split(' '))
                dec_deg.append(dec2deg(table['DEJ2000'][i]))

            if thresh:
                if catalog in ['NVSS']:
                    above_thresh = table['S1.4'] < thresh
                if catalog in ['SUMSS']:
                    above_thresh = table['St'] < thresh

                for i in range(1, len(table.colnames)):
                    table[table.colnames[i]][above_thresh] = np.nan

        table = Table(table, masked=True)
        ascii.write(table, catalog_table, overwrite=True)
        return table
    else:
        return None


def aegean(image, kwargs, log):
    args = ['aegean']
    outfile = ''
    for name, value in kwargs.items():
        if value is None:
            continue
        elif value is False:
            continue
        if name == 'filename':  # positional argument
            args += ['{0}'.format(value)]
        elif name == 'table':
            outfile = "{}.tab".format(kwargs['filename'][:-5])
            args += ['{0}{1} {2}'.format('--', name, outfile)]
            # Aegean add '_comp' to the file name e.g. im_comp.tab
            outfile = "{}_comp.tab".format(kwargs['filename'][:-5])
        else:
            args += ['{0}{1} {2}'.format('--', name, value)]
    log.info("Running: {}".format(" ".join(args)))
    run = subprocess.run(" ".join(args), shell=True)
    log.info("The exit code was: {}".format(run.returncode))
    return outfile


def bdsf(image, kwargs, log):

    try:
        import bdsf as bdsm
    except (ModuleNotFoundError, ImportError):
        raise ModuleNotFoundError("Source finding module is not "
                                  " installed. Install with "
                                  "`pip install aimfast[bdsf]`")

    img_opts = {}
    write_opts = {'outfile': None}
    freq0 = None
    spi_do = False
    ncores = 4
    write_catalog = ['bbs_patches', 'bbs_patches_mask',
                     'catalog_type', 'clobber',
                     'correct_proj', 'format',
                     'incl_chan', 'incl_empty',
                     'srcroot', 'port2tigger',
                     'outfile']

    for name, value in kwargs.items():

        if value is None:
            continue
        if name in ['multi_chan_beam']:
            multi_chan_beam = value
            continue
        if name in ['ncores']:
            ncores = value
            continue
        if name in write_catalog:
            write_opts[name] = value
        elif name in ['freq0', 'frequency']:
            freq0 = value
        else:
            img_opts[name] = value
            if name == 'spectralindex_do':
                spi_do = value

    img_opts.pop('freq0', None)
    if freq0 is None:
        with pyfits.open(img_opts['filename']) as hdu:
            hdr = hdu[0].header
            for i in range(1, hdr['NAXIS']+1):
                if hdr['CTYPE{0:d}'.format(i)].startswith('FREQ'):
                    freq0 = hdr['CRVAL{0:d}'.format(i)]

    if spi_do and multi_chan_beam:
        with pyfits.open(img_opts['filename']) as hdu:
            hdr = hdu[0].header
        beams = []
        # Get a sequence of BMAJ with digit suffix from the image header keys
        bmaj_ind = filter(lambda a: a.startswith('BMAJ')
                          and a[-1].isdigit(), hdr.keys())
        for bmaj in bmaj_ind:
            ind = bmaj.split('BMAJ')[-1]
            beam = [hdr['{0:s}{1:s}'.format(b, ind)]
                    for b in 'BMAJ BMIN BPA'.split()]
            beams.append(tuple(beam))
        # parse beam info to pybdsm
        img_opts['beam_spectrum'] = beams

    image = img_opts.pop('filename')
    filename = os.path.basename(image)
    outfile = write_opts.pop('outfile') or '{}-pybdsf.fits'.format(image[:-5])
    img = bdsm.process_image(image, **img_opts, ncores=ncores)
    img.write_catalog(outfile=outfile, **write_opts)
    return outfile
