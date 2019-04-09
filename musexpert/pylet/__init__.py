"""
PyLET is a slightly reworked and heavily optimized version of the `MUSELET
<http://mpdaf.readthedocs.io/en/latest/muselet.html>`_ program by Johan
Richard. It was designed with two goals in mind: improving the performance
without changing the algorithm, and providing a more natural and Pythonic
interface, along with some finer-grained control and options.
"""
from __future__ import print_function, division
import os, sys, subprocess, glob
from time import time
from functools import partial
from multiprocessing import Pool
import numpy as np
from astropy import units as u
from astropy.table import Table
from mpdaf.obj import Cube, Image
from mpdaf.sdetect import Source, SourceList

from .globals import *
from .utils import *
from ..incubator import _pylet_step1


__all__ = ('Pylet',)


class Pylet:
    """
    This class encapsulates all aspects of the PyLET procedure and provides a
    handful of new capabilities: configuring the logger; setting working
    and output directories; running different steps while preserving the
    data.

    Its signature has been intentionally kept compatible with that of the
    original MUSELET, although some arguments are not used (or are yet to
    be implemented), and new arguments have been added at the end. For all
    arguments not documented here, please see `the documentation of the
    original <http://mpdaf.readthedocs.io/en/latest/muselet.html>`_.

    Parameters
    ----------
    cubename
       filename of the cube
    SNcrit
       step 3 cleans lines with S/N lower than SNcrit, or at
       least that is what it is supposed to do
    workdir
       the working directory, where narrow-band, white and
       color images and individual narrow-band catalogues are placed. The
       default is "pylet_work" in the current directory.
    outdir
       the output directory, where the final catalogues are
       written by :func:`Pylet.write`
    logger
       A logger instance to be used instead of printing to stdout. This could
       be any object that has the "logging" methods: logger.info(),
       logger.warning() and logger.error(). By default,
       a pseudo-logger is used that just prints ``[INFO|WARNING|ERROR] {msg}``
       to stdout.


    .. warning::
           Currently, the cube **must** contain a DATA and STAT extension and
           be of type float32. **Otherwise, the program will crash.** This
           limitation is prorably going to be lifted soon, or at least a
           more helpful way of termination will be provided.ß
    """
    def __init__(self,  cubename, step=1,
                 delta=20, fw=(0.26, 0.7, 1., 0.7, 0.26),
                 radius=4.0, ima_size=21, nlines_max=25, clean=0.5,
                 nbcube=True, expmapcube=None,
                 skyclean=((5573.5, 5578.8), (6297.0, 6300.5)),
                 del_sex=False,
                 SNcrit=5.0,
                 workdir='pylet_work', outdir='pylet_out',
                 logger=None):
        if logger is None:
            logger = _PseudoLogger
        self.logger = logger

        self.cubename = cubename

        if len(fw) != 5:
            self.logger.warning("muselet - len(fw) != 5")
        try:
            self.fw = np.array(fw, dtype=np.float)
        except:
            raise TypeError('fw is not an array of float')

        try:
            subprocess.call(['sex', '-v'], stdout=subprocess.DEVNULL)
            self.cmd_sex = 'sex'
        except OSError:
            try:
                subprocess.call(['sextractor', '-v'], stdout=subprocess.DEVNULL)
                self.cmd_sex = 'sextractor'
            except OSError:
                self.logger.warning('SExtractor not found')

        self.delta = delta
        self.radius = radius
        self.ima_size = ima_size
        self.nlines_max = nlines_max
        self.clean = clean
        self.nbcube = nbcube
        self.expmapcube = expmapcube
        self.skyclean = skyclean
        self.SNcrit = SNcrit

        self.workdir = workdir
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        self.outdir = os.path.abspath(outdir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        if self.expmapcube:
            self.logger.warning('Exposure maps are currently not supported.')
            self.expdir = os.path.join(self.workdir, 'exp')
            if not os.path.exists(self.expdir):
                os.makedirs(self.expdir)
        else:
            self.expdir = None

        self.nbdir = os.path.join(self.workdir, 'nb')

        if not os.path.exists(self.nbdir):
            os.makedirs(self.nbdir)

        self.catdir = os.path.join(self.workdir, 'cat')
        if not os.path.exists(self.catdir):
            os.makedirs(self.catdir)

        self.tBGR_name = os.path.join(self.workdir, 'tBGR.cat')
        self.white_name = os.path.join(self.workdir, 'white.fits')
        self.R_name = os.path.join(self.workdir, 'whiter.fits')
        self.G_name = os.path.join(self.workdir, 'whiteg.fits')
        self.B_name = os.path.join(self.workdir, 'whiteb.fits')
        self.inv_var_name = os.path.join(self.workdir, 'inv_variance.fits')
        self.outnbcube = None

        self.left = self.fw.shape[0] // 2
        self.right = self.fw.shape[0] - self.left

    def execute(self, conf_path=None, overwrite=True, fmt='working'):
        """
        Execute all steps of Muselet in turn and then write the output.

        Parameters
        ----------
        conf_path:
           path of the SExtractor configuration files. See :func:`step2`
        overwrite:
           overwrite catalogues if they already exist.
           **Note:** this does not ensure that the intermediate files will be
           overwritten.
        fmt
           format of the output catalogues. See
           `mpdaf.sdetect.SourceList.write
           <http://mpdaf.readthedocs.io/en/latest/api/mpdaf.sdetect.SourceList.html#mpdaf.sdetect.SourceList.write>`_
           for the possible values
        """
        t0 = time()
        self.step1()
        t1 = time()
        self.step2(conf_path=conf_path)
        t2 = time()
        self.step3()
        t3 = time()
        self.logger.info('muselet - run complete on '+self.cubename)
        self.write(overwrite, fmt)
        t4 = time()
        self.logger.info('Execution complete:\n'
                         'total time {:.0f}s ('
                         'step 1: {:.0f}s, '
                         'step 2: {:.0f}s, '
                         'step 3: {:.0f}s, '
                         'writing: {:.0f}s)'
                         .format(t4-t0, t1-t0, t2-t1, t3-t2, t4-t3))

    def step1(self):
        """
        Extract narrow-band images from the cube as well as a "white" and
        "red", "green" and "blue" images. This step also computes the inverse
        variance image, that is computed in step 2 of the original MUSELET.
        """
        self.logger.info('muselet - STEP 1: creates narrow-band images')

        try:
            fnames = (self.cubename+"[DATA]", self.cubename+"[STAT]")

            nbdir = '!' + self.nbdir
            if not nbdir[:-1] == os.sep:
                nbdir += os.sep

            names = tuple(['!'+name for name in (self.white_name, self.R_name, self.G_name, self.B_name, self.inv_var_name)])

            _pylet_step1(fnames, nbdir, names, self.left, self.delta, self.fw)
        except BaseException:
            self.logger.error('muselet - STEP 1: Failed.')
            raise ChildProcessError('Step 1 failed.')
        self.logger.info('muselet - STEP 1: Complete.')

    def step2(self, start=None, end=None, conf_path=None):
        """
        Run SExtractor on extracted images. Start and end determine the range
        in the usual Python way. If they are not given, the whole wavelength
        range is processed.

        Parameters
        ----------
        start
           first narrow-band image to process.
        end
           one beyond the last narrow-band image to process.
        conf_path
           path of the SExtractor configuration files. Default ones will be
           used if this is not provided.


        .. note::
           SExtractor is run in parallel using GNU parallel, if it is found (
           ``parallel`` executable), or using intrinsic parallelism via the
           :mod:`multiprocessing` module. The former option is preferable,
           since it is faster and provides better feedback. GNU parallel can be
           installed from `here <https://www.gnu.org/software/parallel/>`_. If
           you end up using it, please acknowledge Ole Tange, according to the
           disclaimer.
        """
        if start is None:
            start = self.left+1
        if end is None:
            cshape = Cube(self.cubename, copy=False, dtype=np.float32).shape
            end = cshape[0] - self.right

        self.logger.info("muselet - STEP 2: "
                         "running SExtractor on whitelight and RGB images")

        if not self._check_files(
                (self.white_name, self.inv_var_name,
                 self.R_name, self.G_name, self.B_name)):
            self.logger.error('muselet - STEP 2: Exiting with missing files.')
            raise RuntimeError('Missing images. See the log for details.')

        cmd = _gen_sex(self.cmd_sex,
                       self.white_name, os.path.join(self.catdir, 'white.cat'),
                       self.inv_var_name, False, conf_path)
        subprocess.call(cmd, shell=True)

        for fits, cat in ((self.R_name, 'R.cat'),
                          (self.G_name, 'G.cat'),
                          (self.B_name, 'B.cat')):
            cat = os.path.join(self.catdir, cat)
            cmd = _gen_sex(self.cmd_sex, (self.white_name, fits), cat, self.inv_var_name, False, conf_path)
            subprocess.call(cmd, shell=True)

        tB = Table.read(os.path.join(self.catdir, 'B.cat'), format='ascii.sextractor')
        tG = Table.read(os.path.join(self.catdir, 'G.cat'), format='ascii.sextractor')
        tR = Table.read(os.path.join(self.catdir, 'R.cat'), format='ascii.sextractor')

        names = ('NUMBER', 'X_IMAGE', 'Y_IMAGE',
                 'MAG_APER_B', 'MAGERR_APER_B',
                 'MAG_APER_G', 'MAGERR_APER_G',
                 'MAG_APER_R', 'MAGERR_APER_R')
        tBGR = Table([tB['NUMBER'], tB['X_IMAGE'], tB['Y_IMAGE'],
                      tB['MAG_APER'], tB['MAGERR_APER'],
                      tG['MAG_APER'], tG['MAGERR_APER'],
                      tR['MAG_APER'], tR['MAGERR_APER']], names=names)
        tBGR.write(self.tBGR_name, format='ascii.fixed_width_two_line')

        os.remove(os.path.join(self.catdir, 'B.cat'))
        os.remove(os.path.join(self.catdir, 'G.cat'))
        os.remove(os.path.join(self.catdir, 'R.cat'))

        self.logger.info("muselet - STEP 2: "
                         "running SExtractor on narrow-band images")

        if not self._check_files(
                [os.path.join(self.nbdir, 'nb{:04d}.fits'.format(i))
                 for i in range(start, end)]):
            self.logger.error('muselet - STEP 2: Exiting with missing files.')
            raise RuntimeError('Missing narrow-band images. See the log for '
                               'details.')

        if self.expmapcube:
            raise NotImplementedError('Using exposure maps is currently not '
                                      'supported.')
            weight_name = os.path.join(self.expdir, 'exp{:04d}.fits'.format(k))
        else:
            weight_name = self.inv_var_name

        if os.cpu_count() in (None, 1):
            self.logger.info('Running on a single processor.')

            for i in range(start, end):
                _sex_nb(self.nbdir, weight_name, self.cmd_sex,
                        self.catdir, conf_path, i)
                self.logger.info('SExtracted narrow {}'.format(i))
        else:
            try:
                subprocess.check_call(('parallel', '--version'))

                conf, conv, nnw, param = _config(conf_path)

                cmd = self.cmd_sex
                cmd += ' {}'  # parallel placeholder
                cmd += ' -c "{}"'.format(conf)
                cmd += ' -PARAMETERS_NAME "{}"'.format(param)
                cmd += ' -FILTER_NAME "{}"'.format(conv)
                cmd += ' -STARNNW_NAME "{}"'.format(nnw)
                cmd += ' -CATALOG_NAME "{}"'.format(os.path.join(self.catdir, '{/.}.cat'))  # parallel - name only
                cmd += ' -CATALOG_TYPE ASCII_HEAD'
                cmd += ' -WEIGHT_IMAGE "{}"'.format(weight_name)
                cmd += ' -VERBOSE_TYPE QUIET'

                fnames = os.linesep.join(glob.glob(os.path.join(self.nbdir, '*')))
                proc = subprocess.Popen('parallel --bar '+cmd, shell=True,
                                        stdin=subprocess.PIPE,
                                        universal_newlines=True)
                proc.communicate(fnames)

            except OSError:
                self.logger.info('GNU Parallel was not found.'
                                 'Using intrinsic parallelism.')

                args = (self.nbdir, weight_name, self.cmd_sex, self.catdir, conf_path)

                with Pool() as p:
                    p.map(partial(_sex_nb, *args), range(start, end))

        self.logger.info('muselet - STEP 2: Complete.')

    def step3(self):
        """
        Merge SExtractor catalogs and measure redshifts. This step has hardly
        been changed.
        """

        c = Cube(self.cubename)

        if 'CUBE_V' in c.primary_header:
            cubevers = '%s' % c.primary_header['CUBE_V']
        else:
            cubevers = ''

        wlmin = c.wave.get_start(unit=u.angstrom)
        dw = c.wave.get_step(unit=u.angstrom)
        nslices = c.shape[0]

        ima_size = self.ima_size * c.wcs.get_step(unit=u.arcsec)[0]

        fullvar = Image(self.inv_var_name)
        cleanlimit = self.clean * np.ma.median(fullvar.data)

        tBGR = Table.read(self.tBGR_name, format='ascii.fixed_width_two_line')

        maxidc = 0
        # Continuum lines
        C_ll = []
        C_idmin = []
        C_fline = []
        C_eline = []
        C_xline = []
        C_yline = []
        C_magB = []
        C_magG = []
        C_magR = []
        C_emagB = []
        C_emagG = []
        C_emagR = []
        C_catID = []
        # Single lines
        S_ll = []
        S_fline = []
        S_eline = []
        S_xline = []
        S_yline = []
        S_catID = []

        self.logger.info("muselet - STEP 3: merge SExtractor catalogs and measure redshifts")
        self.logger.info("muselet - cleaning below inverse variance " + str(cleanlimit))

        # TODO: figure out 3 and 14??
        for i in range(3, nslices - 14):
            ll = wlmin + dw * i
            flagsky = False

            if self.skyclean:
                for (skylmin, skylmax) in self.skyclean:
                    if ll > skylmin and ll < skylmax:
                        flagsky = True
                        break

            if flagsky: continue

            slicename = os.path.join(self.catdir, 'nb{:04d}.cat'.format(i))
            t = Table.read(slicename, format='ascii.sextractor')
            for line in t:
                xline = line['X_IMAGE']
                yline = line['Y_IMAGE']
                if fullvar.data[int(yline - 1), int(xline - 1)] > cleanlimit:
                    fline = 10.0 ** (0.4 * (25. - float(line['MAG_APER'])))
                    eline = float(line['MAGERR_APER']) * fline * (2.3 / 2.5)
                    flag = 0
                    distmin = -1
                    distlist = ((xline - tBGR['X_IMAGE']) ** 2.0 +
                                (yline - tBGR['Y_IMAGE']) ** 2.0)
                    ksel = np.where(distlist < self.radius ** 2.0)

                    for j in ksel[0]:
                        if fline > self.SNcrit * eline:
                            if flag <= 0 or distlist[j] < distmin:
                                idmin = tBGR['NUMBER'][j]
                                distmin = distlist[j]
                                magB = tBGR['MAG_APER_B'][j]
                                magG = tBGR['MAG_APER_G'][j]
                                magR = tBGR['MAG_APER_R'][j]
                                emagB = tBGR['MAGERR_APER_B'][j]
                                emagG = tBGR['MAGERR_APER_G'][j]
                                emagR = tBGR['MAGERR_APER_R'][j]
                                xline = tBGR['X_IMAGE'][j]
                                yline = tBGR['Y_IMAGE'][j]
                                flag = 1
                        else:
                            if fline < -self.SNcrit * eline:
                                idmin = tBGR['NUMBER'][j]
                                distmin = distlist[j]
                                flag = -2
                            else:
                                flag = -1

                    if flag == 1:
                        C_ll.append(ll)
                        C_idmin.append(idmin)
                        C_fline.append(fline)
                        C_eline.append(eline)
                        C_xline.append(xline)
                        C_yline.append(yline)
                        C_magB.append(magB)
                        C_magG.append(magG)
                        C_magR.append(magR)
                        C_emagB.append(emagB)
                        C_emagG.append(emagG)
                        C_emagR.append(emagR)
                        C_catID.append(i)
                        if (idmin > maxidc):
                            maxidc = idmin

                    if flag == 0 and ll < 9300.0:
                        S_ll.append(ll)
                        S_fline.append(fline)
                        S_eline.append(eline)
                        S_xline.append(xline)
                        S_yline.append(yline)
                        S_catID.append(i)

        self.logger.info('muselet - STEP 3: lines collected.')

        nC = len(C_ll)
        nS = len(S_ll)

        flags = np.ones(nC)
        for i in range(nC):
            fl = 0
            for j in range(nC):
                if ((i != j) and (C_idmin[i] == C_idmin[j]) and
                        (np.abs(C_ll[j] - C_ll[i]) < 3.00)):
                    if C_fline[i] < C_fline[j]:
                        flags[i] = 0
                    fl = 1
            if fl == 0:  # identification of single line emissions
                flags[i] = 2

        # Sources list
        continuum_lines = SourceList()
        origin = ('muselet', __version__, self.cubename, cubevers)

        # write all continuum lines here:
        raw_catalog = SourceList()
        idraw = 0
        for i in range(nC):
            if flags[i] == 1:
                idraw = idraw + 1
                dec, ra = c.wcs.pix2sky([C_yline[i] - 1, C_xline[i] - 1],
                                        unit=u.deg)[0]
                s = Source.from_data(ID=idraw, ra=ra, dec=dec, origin=origin)
                s.add_mag('MUSEB', C_magB[i], C_emagB[i])
                s.add_mag('MUSEG', C_magG[i], C_emagG[i])
                s.add_mag('MUSER', C_magR[i], C_emagR[i])
                lbdas = [C_ll[i]]
                fluxes = [C_fline[i]]
                err_fluxes = [C_eline[i]]
                ima = Image(os.path.join(self.nbdir, 'nb{:04d}.fits'.format(C_catID[i])))
                s.add_image(ima, 'NB{}'.format(int(C_ll[i])), ima_size)


                lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                              names=['LBDA_OBS', 'LBDA_OBS_ERR',
                                     'FLUX', 'FLUX_ERR'],
                              dtype=['<f8', '<f8', '<f8', '<f8'])
                lines['LBDA_OBS'].format = '.2f'
                lines['LBDA_OBS'].unit = u.angstrom
                lines['LBDA_OBS_ERR'].format = '.2f'
                lines['LBDA_OBS_ERR'].unit = u.angstrom
                lines['FLUX'].format = '.4f'
                # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                lines['FLUX_ERR'].format = '.4f'
                s.lines = lines
                raw_catalog.append(s)

        for r in range(maxidc + 1):
            lbdas = []
            fluxes = []
            err_fluxes = []
            for i in range(nC):
                if (C_idmin[i] == r) and (flags[i] == 1):
                    if len(lbdas) == 0:
                        dec, ra = c.wcs.pix2sky([C_yline[i] - 1, C_xline[i] - 1],
                                                unit=u.deg)[0]
                        s = Source.from_data(ID=r, ra=ra, dec=dec, origin=origin)
                        s.add_mag('MUSEB', C_magB[i], C_emagB[i])
                        s.add_mag('MUSEG', C_magG[i], C_emagG[i])
                        s.add_mag('MUSER', C_magR[i], C_emagR[i])
                    lbdas.append(C_ll[i])
                    fluxes.append(C_fline[i])
                    err_fluxes.append(C_eline[i])
                    ima = Image(os.path.join(self.nbdir, 'nb{:04d}.fits'.format(C_catID[i])))
                    s.add_image(ima, 'NB{:04d}'.format(int(C_ll[i])), ima_size)
            if len(lbdas) > 0:
                lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                              names=['LBDA_OBS', 'LBDA_OBS_ERR',
                                     'FLUX', 'FLUX_ERR'],
                              dtype=['<f8', '<f8', '<f8', '<f8'])
                lines['LBDA_OBS'].format = '.2f'
                lines['LBDA_OBS'].unit = u.angstrom
                lines['LBDA_OBS_ERR'].format = '.2f'
                lines['LBDA_OBS_ERR'].unit = u.angstrom
                lines['FLUX'].format = '.4f'
                # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                lines['FLUX_ERR'].format = '.4f'
                s.lines = lines
                continuum_lines.append(s)

        if len(continuum_lines) > 0:
            self.logger.info("muselet - {} continuum lines detected".format(len(continuum_lines)))
        else:
            self.logger.info("muselet - no continuum lines detected")

        singflags = np.ones(nS)
        S2_ll = []
        S2_fline = []
        S2_eline = []
        S2_xline = []
        S2_yline = []
        S2_catID = []

        for i in range(nS):
            fl = 0
            xref = S_xline[i]
            yref = S_yline[i]
            ksel = np.where((xref - S_xline) ** 2.0 + (yref - S_yline) ** 2.0 <
                            (self.radius / 2.0) ** 2.0)  # spatial distance
            for j in ksel[0]:
                if (i != j) and (np.abs(S_ll[j] - S_ll[i]) < 3.0):
                    if S_fline[i] < S_fline[j]:
                        singflags[i] = 0
                    fl = 1
            if fl == 0:
                singflags[i] = 2
            if singflags[i] == 1:
                S2_ll.append(S_ll[i])
                S2_fline.append(S_fline[i])
                S2_eline.append(S_eline[i])
                S2_xline.append(S_xline[i])
                S2_yline.append(S_yline[i])
                S2_catID.append(S_catID[i])

        # output single lines catalogs here:S2_ll,S2_fline,S2_eline,S2_xline,
        # S2_yline,S2_catID
        nlines = len(S2_ll)
        for i in range(nlines):
            idraw = idraw + 1
            dec, ra = c.wcs.pix2sky([S2_yline[i] - 1, S2_xline[i] - 1],
                                    unit=u.deg)[0]
            s = Source.from_data(ID=idraw, ra=ra, dec=dec, origin=origin)
            lbdas = [S2_ll[i]]
            fluxes = [S2_fline[i]]
            err_fluxes = [S2_eline[i]]
            ima = Image(os.path.join(self.nbdir, 'nb{:04d}.fits'.format(S2_catID[i])))
            s.add_image(ima, 'NB{:04d}'.format(int(S2_ll[i])), ima_size)
            lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                          names=['LBDA_OBS', 'LBDA_OBS_ERR',
                                 'FLUX', 'FLUX_ERR'],
                          dtype=['<f8', '<f8', '<f8', '<f8'])
            lines['LBDA_OBS'].format = '.2f'
            lines['LBDA_OBS'].unit = u.angstrom
            lines['LBDA_OBS_ERR'].format = '.2f'
            lines['LBDA_OBS_ERR'].unit = u.angstrom
            lines['FLUX'].format = '.4f'
            # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lines['FLUX_ERR'].format = '.4f'
            s.lines = lines
            raw_catalog.append(s)

        # List of single lines
        # Merging single lines of the same object
        single_lines = SourceList()
        flags = np.zeros(nlines)
        ising = 0
        for i in range(nlines):
            if flags[i] == 0:
                ising = ising + 1
                lbdas = []
                fluxes = []
                err_fluxes = []

                dec, ra = c.wcs.pix2sky([S2_yline[i] - 1, S2_xline[i] - 1],
                                        unit=u.deg)[0]
                s = Source.from_data(ID=ising, ra=ra, dec=dec, origin=origin)
                lbdas.append(S2_ll[i])
                fluxes.append(S2_fline[i])
                err_fluxes.append(S2_eline[i])
                ima = Image(os.path.join(self.nbdir, 'nb{:04d}.fits'.format(S2_catID[i])))
                s.add_image(ima, 'NB{:04d}'.format(int(S2_ll[i])), ima_size)
                ksel = np.where(((S2_xline[i] - S2_xline) ** 2.0 +
                                 (S2_yline[i] - S2_yline) ** 2.0 <
                                 self.radius ** 2.0) & (flags == 0))
                for j in ksel[0]:
                    if (j != i):
                        lbdas.append(S2_ll[j])
                        fluxes.append(S2_fline[j])
                        err_fluxes.append(S2_eline[j])
                        ima = Image(os.path.join(self.nbdir, 'nb{:04d}.fits'.format(S2_catID[i])))
                        s.add_image(ima, 'NB{:04d}'.format(int(S2_ll[i])), ima_size)
                        flags[j] = 1
                lines = Table([lbdas, [dw] * len(lbdas), fluxes, err_fluxes],
                              names=['LBDA_OBS', 'LBDA_OBS_ERR',
                                     'FLUX', 'FLUX_ERR'],
                              dtype=['<f8', '<f8', '<f8', '<f8'])
                lines['LBDA_OBS'].format = '.2f'
                lines['LBDA_OBS'].unit = u.angstrom
                lines['LBDA_OBS_ERR'].format = '.2f'
                lines['LBDA_OBS_ERR'].unit = u.angstrom
                lines['FLUX'].format = '.4f'
                # lines['FLUX'].unit = !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                lines['FLUX_ERR'].format = '.4f'
                s.lines = lines
                single_lines.append(s)
                flags[i] = 1

        if len(single_lines) > 0:
            self.logger.info("muselet - {} single lines detected".format(len(single_lines)))
        else:
            self.logger.info("muselet - no single lines detected")

        # redshift of continuum objects
        eml, eml2 = _get_line_lists()
        self.logger.info("muselet - estimating the best redshift")

        i = 0
        for source in continuum_lines:
            if (len(source.lines) > 3):
                source.crack_z(eml)
            else:
                source.crack_z(eml2)
            source.sort_lines(self.nlines_max)
            i += 1
            self.logger.info('muselet - STEP 3: continuum object {}/{}'.format(i, len(continuum_lines)))
        i = 0
        for source in single_lines:
            if (len(source.lines) > 3):
                source.crack_z(eml, 20)
            else:
                source.crack_z(eml2, 20)
            source.sort_lines(self.nlines_max)
            i+=1
            self.logger.info('muselet - STEP 3: single line object {}/{}'.format(i, len(single_lines)))

        lsource = lambda s: s.lines[0][0]
        single_lines.sort(key=lsource)

        self.cont = continuum_lines
        self.sing = single_lines
        self.raw = raw_catalog

        self.logger.info('muselet - STEP 3: Complete.')

    def write(self, overwrite=True, fmt='working'):
        """
        Write the catalogues produced in step 3.

        :param overwrite:
           Overwrite existing catalogues. Default is True.
        :param fmt:
           format of the output catalogues. See `mpdaf.sdetect.SourceList.write
           <http://mpdaf.readthedocs.io/en/latest/api/mpdaf.sdetect.SourceList.html#mpdaf.sdetect.SourceList.write>`_
           for the possible values.
        """
        self.logger.info('Writing SourceLists to .fits files')
        if len(self.raw) > 0:
            self.raw.write('raw', path=self.outdir, overwrite=overwrite, fmt=fmt)
        if len(self.cont) > 0:
            self.cont.write('cont', path=self.outdir, overwrite=overwrite, fmt=fmt)
        if len(self.sing) > 0:
            self.sing.write('sing', path=self.outdir, overwrite=overwrite, fmt=fmt)
        self.logger.info('Writing of output finished.')
