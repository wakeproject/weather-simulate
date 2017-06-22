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

g = 9.8

Omeaga = 2 * np.pi / (24 * 3600 * 0.99726966323716)

gamma = - 6.49 / 1000
gammad = - 9.80 / 1000
cp = 1003.5
R = 293.0
miu = 18.1 * 0.0001


SunConst = 1366
StefanBoltzmann = 0.0000000567
WaterHeatCapacity = 4200
RockHeatCapacity = 840


def merge(array):
    mask = np.isnan(array)
    array[mask] = np.average(array[~mask])

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




