# -*- coding: utf-8 -*-

import numpy as np

from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

context = {}

lat, lng, alt = np.meshgrid(np.arange(-89.5, 89.5, 1), np.arange(-180, 181, 1), np.arange(0, 16000, 500.0))

one = np.ones(lng.shape)
zero = np.zeros(lng.shape)

theta = lng * np.pi / 180
phi = lat * np.pi / 180

a = 6371000

g = 9.80665

Omega = 2 * np.pi / (24 * 3600 * 0.99726966323716)

gamma = - 6.49 / 1000
gammad = - 9.80 / 1000
cv = 718.0
R = 8.31447
miu = 1.72e-1
M = 0.0289644 # molar mass of dry air, 0.0289644 kg/mol


SunConst = 1366
StefanBoltzmann = 0.0000000567
WaterHeatCapacity = 4200
RockHeatCapacity = 840


kernel = np.ones([3, 3]) / 9.0


def filter_extreams(array):
    mask = np.isnan(array)
    array[mask] = np.average(array[~mask])

    mx = np.max(array)
    mn = np.min(array)

    xthresh = 0.99 * mx + 0.01 * mn
    xthresh_less = 0.98 * mx + 0.02 * mn
    nthresh = 0.01 * mx + 0.99 * mn
    nthresh_more = 0.02 * mx + 0.98 * mn

    pmask = np.where(array >= xthresh)
    nmask = np.where((array < xthresh) * (array > xthresh_less))
    if len(nmask[1]) != 0:
        array[pmask] = np.average(array[nmask])

    pmask = np.where(array <= nthresh)
    nmask = np.where((array > nthresh) * (array < nthresh_more))
    if len(nmask[1]) != 0:
        array[pmask] = np.average(array[nmask])

    np.copyto(array, ndimage.convolve(array, kernel))


def merge(array):
    filter_extreams(array)
    avg = (array[0, :] + array[-1, :]) / 2
    array[0, :] = avg[:]
    array[-1, :] = avg[:]

    return np.copy(array)


class Grid(object):
    def __init__(self, name, lng_size, lat_size, alt_size, initval=0.0, initfn=None):
        self.lng_size = lng_size
        self.lat_size = lat_size
        self.alt_size = alt_size
        context[name] = self

        self.nxtval = np.zeros([lng_size, lat_size, alt_size])
        if initfn:
            self.curval = initfn()
        else:
            self.curval = np.ones([lng_size, lat_size, alt_size]) * initval

    def evolve(self, dt):
        kwargs = {k: v.curval for k, v in context.iteritems()}
        dval = self.step(**kwargs) * dt
        val = self.curval + dval
        for i in range(32):
            np.copyto(self.nxtval[:, :, i], merge(val[:, :, i]))

    def step(self, ** kwargs):
        return self.nxtval

    def swap(self):
        np.copyto(self.curval, self.nxtval)


class Relation(object):
    def __init__(self, name, lng_size, lat_size, alt_size, initval=0.0):
        self.lng_size = lng_size
        self.lat_size = lat_size
        self.alt_size = alt_size
        context[name] = self

        self.nxtval = np.zeros([lng_size, lat_size, alt_size])
        self.curval = np.ones([lng_size, lat_size, alt_size]) * initval

    def evolve(self):
        kwargs = {k: v.curval for k, v in context.iteritems()}
        val = self.step(**kwargs)
        for i in range(32):
            np.copyto(self.nxtval[:, :, i], merge(val[:, :, i]))

    def step(self, ** kwargs):
        return self.nxtval

    def swap(self):
        np.copyto(self.curval, self.nxtval)




