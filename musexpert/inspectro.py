import numpy as np

from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from mpdaf.obj import Image

from .slit import slit


__all__ = ('Line', 'Galaxy', 'VelocityMap')


class Line:
    """
    Represents an emission line. This class is just a collection of data,
    saving the provided parameters in members of the same names, converting
    them to :class:`astropy.units.Quantity` where appropriate.

    Parameters
    ----------
    name : string
        identifier of the line, e.g. Halpha, O_III, etc. in any format as
        convenient
    lbda_obs : float or :class:`~astropy.units.Quantity`
        the observed wavelength of the emission line
    lbda_obs_err : float or :class:`~astropy.units.Quantity`
        error in the observed wavelength
    flux : float or :class:`~astropy.units.Quantity`
        total flux of the line using any convention as convenient
    flux_err : float or :class:`~astropy.units.Quantity`
        error in the total flux
    """

    def __init__(self, name=None, lbda_obs=None, lbda_obs_err=None,
                 flux=None, flux_err=None):
        self.name = name

        self.lbda_obs = u.Quantity(lbda_obs)
        self.lbda_obs_err = u.Quantity(lbda_obs_err)

        self.flux = u.Quantity(flux)
        self.flux_err = u.Quantity(flux_err)

    def __repr__(self):
        return '{} @ {}'.format('Unknown' if self.name is None else self.name,
                                self.lbda_obs)


class Galaxy:
    """
    Represents a source that looks like a galaxy, i.e. has an elliptical shape.

    Attributes
    ----------
    ID : int
        an identifier for the source
    coord : :class:`~astropy.units.Quantity`
        the position of the source on the sky
    x
        x-position on an image from which the sources has been extracted (in
        pixels)
    y
        y-position (in pixels)
    a
        semi-major axis (in pixels)
    b
        semi-minor axis (in pixels)
    th : :class:`~astropy.units.Quantity`
        angle from x-axis of the image to the major axis of the source (the
        direction is from +x to +y to -x to -y)
    z
        redshift of the source
    z_bounds : tuple(z_min, z_max)
        bounds of the possible values for the redshift
    lines : list of :class:`inspectro.Line` objects
        the emission lines associated with the source
    """

    def __init__(self):
        self.ID = None

        self.coord = None

        self.x = None
        self.y = None
        self.a = None
        self.b = None
        self.th = None

        self.z = None
        self.z_bounds = None

        self.lines = None

    @classmethod
    def from_row(cls, row):
        """
        Generate a Galaxy from the first row of a
        :class:`~astropy.table.Table`. This function is useful mainly for
        extract Galaxy objects from `Muselet`_ catalogues.

        The provided argument should have columns:
            - ID
            - Z_EMI, Z_EMI_MIN, Z_EMI_MAX
            - LINE[N], LBDA_OBS[N], LBDA_OBS_ERR[N], FLUX[N], FLUX_ERR[N]
            - X_IMAGE, Y_IMAGE, A_IMAGE, B_IMAGE, THETA_IMAGE

        Parameters
        ----------
        row : :class:`~astropy.table.Table`
            the table to extract the data from. It does not strictly need to
            contain a single row. In this case, the data will be taken from
            the first row.


        .. _Muselet:
            http://mpdaf.readthedocs.io/en/latest/muselet.html
        """
        self = cls()

        self.ID = row['ID'][0][0]
        self.coord = SkyCoord(_get_quantity(row, 'RA'),
                              _get_quantity(row, 'DEC'))
        self.z = row['Z_EMI'][0][0]
        self.z_bounds = (row['Z_EMI_MIN'][0][0], row['Z_EMI_MAX'][0][0])

        self.lines = []
        i = 1
        while True:
            dig = '{:0>3.0f}'.format(i)
            if not ('LINE' + dig in row.columns and bool(
                    row['LINE' + dig])):
                break

            l = Line(name=row['LINE' + dig][0][0],
                     lbda_obs=_get_quantity(row, 'LBDA_OBS' + dig),
                     lbda_obs_err=_get_quantity(row, 'LBDA_OBS_ERR' + dig),
                     flux=_get_quantity(row, 'FLUX' + dig),
                     flux_err=_get_quantity(row, 'FLUX_ERR' + dig))

            self.lines.append(l)

            i += 1

        self.x = row['X_IMAGE'][0][0]
        self.y = row['Y_IMAGE'][0][0]
        self.a = row['A_IMAGE'][0][0]
        self.b = row['B_IMAGE'][0][0]
        self.th = _get_quantity(row, 'THETA_IMAGE')

        return self

    @classmethod
    def from_fits(cls, hdu):
        self = cls()

        self.ID = hdu.header['ID']

        self.coord = SkyCoord(ra=hdu.header['RA'], dec=hdu.header['DEC'],
                              unit=u.deg)

        self.x = hdu.header['X_IMAGE']
        self.y = hdu.header['Y_IMAGE']
        self.a = hdu.header['A_IMAGE']
        self.b = hdu.header['B_IMAGE']
        self.th = u.Quantity(hdu.header['TH_IMAGE'], u.deg)

        self.z = hdu.header['Z']
        self.z_bounds = (hdu.header['Z_MIN'], hdu.header['Z_MAX'])

        self.lines = [Line(row[0],
                           *[u.Quantity(row[i], hdu.columns[i].unit)
                             for i in range(1,5)])
                      for row in hdu.data]
        return self

    @classmethod
    def read(cls, fname):
        """
        Generate a Galaxy from a FITS file. The file should have a "GALAXY"
        extension with the appropriate keywords and a binary table for the
        lines data. The surest way to get such a file is to first save a
        Galaxy instance using :meth:`~inspectro.Galaxy.write`.

        Parameters
        ----------
        fname : file path, file object, file-like object or pathlib.Path object
            file to be read
        """
        return cls.from_fits(fits.open(fname)['GALAXY'])

    def write(self, fname, **kwargs):
        """
        Write the data held by this Galaxy instance to a FITS file.

        Parameters
        ----------
        fname : file path, file object, file-like object or pathlib.Path object
            file to be written
        kwargs
            any additional keyword arguments are passed directly to
            :func:`astropy.io.fits.writeto`
        """
        thdr = fits.table_to_hdu(self.get_table())
        thdr.name = 'GALAXY'
        thdr.writeto(fname, **kwargs)

    def get_spec(self, cube, extent=3, width=0.5):
        """
        Generate a spectrum of the Galaxy along the major axis of the
        observed ellipse. This is achieved by simulating a slit aligned with
        the major axis of the Galaxy and interpolating from the data of the
        supplied cube.

        Parameters
        ----------
        cube : :class:`mpdaf.obj.Cube`
            the cube from which to extract the spectrum. The Galaxy's
            positional parameters (:attr:`x`, :attr:`y`) are used to determine
            its location in the cube
        extent
            length of the pseudo slit as a fraction of the major axis of the
            Galaxy
        width
            width of the slit as a fraction of the minor axis of the Galaxy

        Returns
        -------
        :class:`mpdaf.obj.Image`
            An image with the extracted spectrum. Its first dimension
            represents wavelength, while its second dimension -- distance
            along the slit. For more details about the spectrum see
            :func:`slit.slit`.
        """
        h = extent * self.a * np.sin(self.th)
        w = extent * self.a * np.cos(self.th)

        return slit(cube,
                    begin=np.array((self.x - w, self.y - h)),
                    end=np.array((self.x + w, self.y + h)),
                    width=width*self.b)

    def get_table(self):
        """
        Generate an :class:`astropy.table.Table` which contains the data
        about the Galaxy. The members of the instance are saved as keywords
        in the :attr:`~astropy.table.Table.meta` attribute while the table
        itself contains data about the lines.

        Returns
        -------
        :class:`astropy.table.Table`
            the generated table
        """
        rows = [(l.name, l.lbda_obs.value, l.lbda_obs_err.value,
                 l.flux.value, l.flux_err.value) for l in self.lines]

        t = Table(
            rows=rows,
            meta={
                'ID': self.ID,
                'RA': self.coord.ra.degree,
                'DEC': self.coord.dec.degree,
                'X_IMAGE': self.x,
                'Y_IMAGE': self.y,
                'A_IMAGE': self.a,
                'B_IMAGE': self.b,
                'TH_IMAGE': self.th.to(u.deg).value,
                'Z': self.z,
                'Z_MIN': self.z_bounds[0],
                'Z_MAX': self.z_bounds[1]
            },
            names=('NAME', 'LBDA_OBS', 'LBDA_OBS_ERR', 'FLUX', 'FLUX_ERR'),
            dtype=('U16', float, float, float, float)
        )

        t['LBDA_OBS'].unit = t['LBDA_OBS_ERR'].unit =\
            self.lines[0].lbda_obs.unit
        t['FLUX'].unit = t['FLUX_ERR'].unit = \
            self.lines[0].flux.unit

        return t

    def __repr__(self):
        return 'Galaxy @ [{}], z={}'.format(self.coord.to_string(), self.z)


class VelocityMap:
    """
    This class stores and processes the data associated with a galaxy
    detected in a datacube. Its main purpose is to extract the spectrum of a
    galaxy and generate its rotation curve. This is achieve through the
    :meth:`build` method -- the main entry point.

    Parameters
    ----------
    gal : :class:`Galaxy`
        the galaxy which to process

    Attributes
    ----------
    gal : :class:`Galaxy`
        the galaxy which is processed
    spec : :class:`mpdaf.obj.Image`
        the spectrum of the galaxy, generated by :meth:`Galaxy.get_spec`
    datasets : `list` of `dict`\ s
        the rotation curves from each of the lines of the galaxy. Each dict
        contains two keys:

            - "name", giving the line identifier
            - "data", the rotation curve

    all_data : `ndarray`
        all the datasets combined
    ddata : `ndarray`
        combined and binned data

    Notes
    -----
    The three `data` attributes are two dimensional `ndarray`\ s, shaped like a
    table. In all cases ``data[:, 0]`` gives the positions along the slit in
    pixels, and ``data[:, 1]`` -- the measured velocity in km/s. The following
    column(s) are related to the uncertainties in the velocity.
    In the :attr:`datasets` and in :attr:`all_data` (which is just all the
    datasets combined), only one column is present beyond the second one:
    ``all_data[:, 2]`` holds the **variance** of the measurement.
    In :attr:`ddata` there are two columns containing *standard deviations*:
    ``ddata[:, 2]`` is computed using ``std(v) / sqrt(len(v))`` with ``v``
    the array of velocities, while ``ddata[:, 3]`` holds the standard
    deviation of the mean: ``1 / sqrt(sum(1/var))`` where ``var`` is the
    variance array.
    """
    def __init__(self, gal):
        self.gal = gal

        self.cube = None
        self.wave = None

        self.spec = None
        self.datasets = None
        self.all_data = None
        self.ddata = None

    @classmethod
    def read(cls, fname):
        """
        Create a VelocityMap from a FITS file.

        Parameters
        ----------
        fname : file path, file object, file-like object or pathlib.Path object
            file to be read

        Notes
        -----
        The file must contain extensions:

            - GALAXY - to create the Galaxy
            - SPEC_DATA, SPEC_VAR - to create the spectrum image
            - DATA - for the :attr:`ddata` attribute
            - extensions for the :attr:`datasets`

        The surest way to get such a file is to first save a VelocityMap
        instance using :meth:`write`.

        Loading of "partial" data is also possible. That is, any necessary
        extensions that are not present in the file are ignored and the
        respective attributes are left unset. It is the responsibility of the
        user to check what data has been loaded.
        """

        f = fits.open(fname)

        gal = Galaxy.from_fits(f['GALAXY'])
        self = cls(gal)

        if f[0].header['HASSPEC']:
            self.spec = Image(data=f['SPEC_DATA'].data, var=f['SPEC_VAR'].data)
        if f[0].header['HASDATA']:
            self.ddata = f['DATA'].data

            ids = f.index_of('DATA') + 1
            nds = f[0].header['NDSETS']
            self.datasets = [{'name': f[i].name, 'data': f[i].data}
                             for i in range(ids, ids + nds)]

        return self

    def write(self, fname, **kwargs):
        """
        Write the data of the VelocityMap to a FITS file.

        The structure of the file is as follows:

            - PRIMARY extension:
                contains auxiliary information about whether data is present
                and how many lines there are
            - GALAXY extension:
                contains the Galaxy data. See :meth:`Galaxy.read`
            - SPEC_DATA, SPEC_VAR extensions:
                contain the data and variance arrays of the spectrum image
            - DATA extension:
                contains the :attr:`ddata` array
            - dataset extensions:
                each of the following extensions is named as the identifier
                of a Line and holds the data from analysing this particular
                line.
            - PLOT extension (*optional*):
                contains an RGBA image of a plot of `ddata`. This is generated
                when the VelocityMap is saved and is not loaded by :meth:`read`.

        Parameters
        ----------
        fname : file path, file object, file-like object or pathlib.Path object
            file to be written
        kwargs :
            any additional keyword arguments are passed directly to
            :func:`astropy.io.fits.writeto`

        Notes
        -----
        This function will write only the data that is available (the
        corresponding attribute is not ``None``).
        """

        f = fits.HDUList()

        f.append(fits.PrimaryHDU())

        thdr = fits.table_to_hdu(self.gal.get_table())
        thdr.name = 'GALAXY'
        f.append(thdr)

        if self.spec is not None:
            f[0].header['HASSPEC'] = True
            f.append(fits.ImageHDU(name='SPEC_DATA', data=self.spec.data.data))
            f.append(fits.ImageHDU(name='SPEC_VAR', data=self.spec.var.data))
        else:
            f[0].header['HASSPEC'] = False

        if self.ddata is not None:
            f[0].header['HASDATA'] = True
            f[0].header['NDSETS'] = len(self.datasets)

            f.append(fits.ImageHDU(name='DATA', data=self.ddata))
            for ds in self.datasets:
                f.append(fits.ImageHDU(**ds))

            fig = Figure()
            canvas = Canvas(fig)
            self.plot(fig.gca())
            canvas.draw()

            w, h = canvas.get_width_height()
            plot = np.fromstring(canvas.tostring_argb(), dtype='uint8').reshape(
                (h, w, 4))

            f.append(fits.ImageHDU(name='PLOT', data=plot))
        else:
            f[0].header['HASDATA'] = False

        f.writeto(fname, **kwargs)

    def build_spec(self, extent=3, width=0.5):
        print('Building spectrum.')
        self.spec = self.gal.get_spec(self.cube, extent, width)

    def build(self, cube, extent=3, width=0.5):
        """
        Extract a rotation curve.

        The parameters are passed directly to :attr:`gal`\ 's
        :meth:`Galaxy.get_spec` method.

        After calling `build`, the VelocityMap's :attr:`spec`,
        :attr:`datasets`, :attr:`all_data`, and :attr:`ddata` attributes are
        set.
        """
        self.cube = cube
        self.wave = cube.wave

        if self.spec is None:
            self.build_spec(extent, width)

        model = (models.Gaussian1D + models.Const1D)()
        fitter = fitting.LevMarLSQFitter()

        a2 = (self.gal.z + 1)**2
        ascale = 3e5 * self.wave.get_step() * 4 * a2 / (a2 + 1)**2
        datasets = []
        all_data = []
        for l in self.gal.lines:
            scale = ascale / l.lbda_obs.value
            print(l)
            x = int(self.wave.pixel(l.lbda_obs.to(self.wave.unit).value))

            # TODO: Maybe don't bother with the fucking units!!
            maxv = 2e6
            width = int(
                (np.sqrt((3e8 + maxv) / (3e8 - maxv)) - 1) * l.lbda_obs /
                (self.wave.get_step(l.lbda_obs.unit) * l.lbda_obs.unit))

            subspec = self.spec[:, x - width: x + width]
            xdata = np.arange(-width, width)

            data = []
            for i in range(len(subspec.data)):
                md, fit = fitting.FittingWithOutlierRemoval(fitter, sigma_clip) \
                    (model, x=xdata, y=subspec.data[i],
                     weights=1 / subspec.var[i])

                if fitter.fit_info['param_cov'] is not None:
                    errs = np.abs(np.diag(fitter.fit_info['param_cov']))

                    pos = scale * fit.mean_0.value
                    var = scale ** 2 * errs[1]

                    data.append((i, pos, var))

            if data:
                datasets.append({'name': l.name, 'data': np.array(data)})
                all_data.extend(data)

        self.datasets = datasets
        self.all_data = np.array(all_data)
        self.ddata = _digitize(self.all_data, nbins=len(self.spec.data)//2)

        return self.ddata

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        for ds in self.datasets:
            ax.plot(ds['data'][:,0], ds['data'][:,1], 'o', label=ds['name'])
        ax.errorbar(self.ddata[:,0], self.ddata[:,1], self.ddata[:,2], fmt='k-')
        ax.legend()
        ax.set_xlabel('Position, px')
        ax.set_ylabel('Velocity, km/s')


# Utility functions
# -----------------
def _get_quantity(row, col):
    return row[col].quantity[0][0]


def _digitize(data, nbins=10):
    x = data[:, 0]
    bins = np.linspace(np.min(x), np.max(x), nbins)

    ind = np.digitize(x, bins)

    binsize = bins[1] - bins[0]
    X = bins[0] + binsize / 2
    newdata = []
    for n in range(1,nbins):
        nind = np.argwhere(ind == n)

        if len(nind):
            y = data[nind, 1]
            w = 1 / data[nind, 2]
            newdata.append(
                [X, np.average(y, weights=w),
                 np.std(y) / np.sqrt(len(y)), 1 / np.sqrt(np.sum(w))])
        X += binsize

    return np.array(newdata)
