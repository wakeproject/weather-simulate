# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import random

from solarsys import dlng, dlat, dalt, bottom, top, north_pole, south_pole, theta

context = {}


g = 9.80665

Omega = 2 * np.pi / (24 * 3600 * 0.99726966323716)

gamma = 6.49 / 1000
gammad = 9.80 / 1000
cv = 718.0
cp = 1005.0
R = 287
miu = 1.72e-1
M = 0.0289644 # molar mass of dry air, 0.0289644 kg/mol

niu = 0.1 # friction between air and land surface
niu_matrix = niu * bottom


SunConst = 1366

StefanBoltzmann = 0.0000000567
WaterHeatCapacity = 4185.5
RockHeatCapacity = 840
WaterDensity = 1000
RockDensity = 2650


def inject_random_nearby(i, j, thresh, speed, src, tgt):
    tries = 0
    replacement = thresh
    while tries < 3:
        dx, dy = random(), random()
        while replacement > thresh:
            i, j = i + dx, j + dy
            if j < 0:
                j = 0
                i = (180 / dlng + i) % (360 / dlng)
                dy = - dy
            if j > 179 / dlat:
                j = 179 / dlat
                i = (180 / dlng + i) % (360 / dlng)
                dy = - dy
            if i < 0 or i > 360 / dlng:
                i = (i + 360 / dlng) % (360 / dlng)
            replacement = speed[i, j]
        else:
            tries = 3
            tgt[i, j] = src[i, j]


def filter_extream_scalar(name, array):
    mask = np.isnan(array)
    array[mask] = np.average(array[~mask])

    mx = np.max(array)
    mn = np.min(array)

    xthresh = (1 - 0.001) * mx + 0.001 * mn
    xthresh_less = (1 - 0.002) * mx + 0.002 * mn
    nthresh = 0.001 * mx + (1 - 0.001) * mn
    nthresh_more = 0.002 * mx + (1 - 0.002) * mn

    pmask = np.where(array >= xthresh)
    nmask = np.where((array < xthresh) * (array > xthresh_less))
    if len(nmask[1]) != 0:
        array[pmask] = np.average(array[nmask])

    pmask = np.where(array <= nthresh)
    nmask = np.where((array > nthresh) * (array < nthresh_more))
    if len(nmask[1]) != 0:
        array[pmask] = np.average(array[nmask])

    #np.copyto(array, ndimage.gaussian_filter(array, 0.2 * (np.max(context['T'].curval) - np.min(context['T'].curval)) / 150.0))


def filter_extream_vector(name, array, u, v, w):
    mask = np.isnan(array)
    array[mask] = np.average(array[~mask])

    speed = u * u + v * v + w * w
    mx = np.max(speed)
    xthresh = (1 - 0.01) * mx
    shape = speed.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            s = speed[i, j]
            if s > xthresh:
                if name == 'u':
                    inject_random_nearby(i, j, xthresh, speed, u, array)
                if name == 'v':
                    inject_random_nearby(i, j, xthresh, speed, v, array)
                if name == 'w':
                    inject_random_nearby(i, j, xthresh, speed, w, array)

    np.copyto(array, 0.9 * array)


def combine_scalar(array):
    nval = np.mean(array[:, 0])
    sval = np.mean(array[:, -1])
    array[:, 0] = nval
    array[:, -1] = sval


def combine_vector(name, array, u, v):
    th = theta[:, 0, 0]
    uval = np.mean(np.cos(th) * u + np.sin(th) * v)
    vval = np.mean(- np.sin(th) * u + np.cos(th) * v)
    if name == 'u':
        array[:] = np.cos(th) * uval - np.sin(th) * vval
    if name == 'v':
        array[:] = np.sin(th) * uval + np.cos(th) * vval


def merge(name, array, compu=None, compv=None, compw=None):
    if name not in {'u', 'v', 'w'}:
        filter_extream_scalar(name, array)
    else:
        filter_extream_vector(name, array, compu, compv, compw)

    avg = (array[0, :] + array[-1, :]) / 2
    array[0, :] = avg[:]
    array[-1, :] = avg[:]

    if name in {'u', 'v'}:
        combine_vector(name, array[:, 0], compu[:, 0], compv[:, 0])
        combine_vector(name, array[:, -1], compu[:, -1], compv[:, -1])
    else:
        combine_scalar(array)

    return np.copy(array)


class Grid(object):
    def __init__(self, name, lng_size, lat_size, alt_size, initval=0.0, initfn=None):
        self.lng_size = lng_size
        self.lat_size = lat_size
        self.alt_size = alt_size
        self.name = name
        context[name] = self

        self.drvval = np.zeros([lng_size, lat_size, alt_size])
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
            if self.name in {'u', 'v', 'w'}:
                np.copyto(self.nxtval[:, :, i], merge(self.name, val[:, :, i],
                                                      compu=context['u'].curval[:, :, i],
                                                      compv=context['v'].curval[:, :, i],
                                                      compw=context['w'].curval[:, :, i],))
            else:
                np.copyto(self.nxtval[:, :, i], merge(self.name, val[:, :, i]))
        self.drvval[:, :, :] = (self.nxtval - self.curval) / dt

    def step(self, ** kwargs):
        return self.nxtval

    def swap(self):
        for i in range(32):
            if self.name in {'u', 'v', 'w'}:
                np.copyto(self.curval[:, :, i], merge(self.name, self.nxtval[:, :, i],
                                                      compu=context['u'].nxtval[:, :, i],
                                                      compv=context['v'].nxtval[:, :, i],
                                                      compw=context['w'].nxtval[:, :, i],))
            else:
                np.copyto(self.curval[:, :, i], merge(self.name, self.nxtval[:, :, i]))


class Relation(object):
    def __init__(self, name, lng_size, lat_size, alt_size, initval=0.0, initfn=None):
        self.lng_size = lng_size
        self.lat_size = lat_size
        self.alt_size = alt_size
        self.name = name
        context[name] = self

        self.drvval = np.zeros([lng_size, lat_size, alt_size])
        self.nxtval = np.zeros([lng_size, lat_size, alt_size])
        if initfn:
            self.curval = initfn()
        else:
            self.curval = np.ones([lng_size, lat_size, alt_size]) * initval

    def evolve(self, dt):
        kwargs = {k: v.curval for k, v in context.iteritems()}
        val = self.step(**kwargs)
        for i in range(32):
            np.copyto(self.nxtval[:, :, i], merge(self.name, val[:, :, i], dt))
        self.drvval[:, :, :] = (self.nxtval - self.curval) / dt

    def step(self, ** kwargs):
        return self.nxtval

    def swap(self):
        np.copyto(self.curval, self.nxtval)




