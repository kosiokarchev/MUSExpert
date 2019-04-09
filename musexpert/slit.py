import numpy as np
from scipy.interpolate import RegularGridInterpolator
from mpdaf.obj import Image


def slit(cube, begin, end, width):
    lvec = end - begin
    length = np.linalg.norm(lvec)

    wvec = np.array((-lvec[0], lvec[1])) * (width/length)

    c = np.swapaxes(
        np.meshgrid(np.linspace(-1, 1, int(2*width) + 1),
                    np.linspace(0, 1, int(length) + 1)),
        0, 2)

    # Invert order of coordinates to fit FITS standard
    coords = begin[::-1] \
             + c[:, :, 0][:, :, np.newaxis] * wvec[np.newaxis, np.newaxis, ::-1] \
             + c[:, :, 1][:, :, np.newaxis] * lvec[np.newaxis, np.newaxis, ::-1]
    c = np.repeat(coords[np.newaxis, :, ::], cube.shape[0], 0)
    coords = np.ndarray((c.shape[0], c.shape[1], c.shape[2], 3))
    coords[:, :, :, 1:] = c
    coords[:, :, :, 0] = np.arange(cube.shape[0])[:, np.newaxis, np.newaxis]
    del c

    endpoints = np.array((begin + wvec, begin - wvec,
                          end + wvec, end - wvec))

    i = ((int(np.min(endpoints[:, 0])), int(np.max(endpoints[:, 0])) + 1),
         (int(np.min(endpoints[:, 1])), int(np.max(endpoints[:, 1])) + 1))

    points = (np.arange(cube.shape[0]),
              np.arange(i[1][0], i[1][1] + 1),
              np.arange(i[0][0], i[0][1] + 1))
    subcube = cube[:, i[1][0]:i[1][1] + 1, i[0][0]:i[0][1] + 1]

    ipol_data = RegularGridInterpolator(points, subcube.data)
    ipol_var = RegularGridInterpolator(points, subcube.var)

    iimg = Image(data=np.swapaxes(ipol_data(coords).mean(axis=1), 0, 1),
                 var=np.swapaxes(ipol_var(coords).mean(axis=1), 0, 1),
                 unit=cube.unit, copy=False)

    return iimg
