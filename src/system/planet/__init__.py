# -*- coding: utf-8 -*-

import numpy as np

from scipy import ndimage
from numpy.random import random

context = {}

dlng = 3.0
dlat = 3.0
dalt = 500.0

a = 6371000

lat, lng, alt = np.meshgrid(np.arange(-89.5, 89.5, dlat), np.arange(-180, 181, dlng), np.arange(0, 16000, dalt))

shape = lng.shape

one = np.ones(lng.shape)
zero = np.zeros(lng.shape)

bottom = np.zeros(lng.shape)
bottom[:, :, 0] = 1
north_pole = np.zeros(lng.shape)
north_pole[:, 0, :] = 1
south_pole = np.zeros(lng.shape)
south_pole[:, -1, :] = 1

dth = np.pi / 180 * dlng * one
dph = np.pi / 180 * dlat * one
dr = 1 * dalt * one

r = a + alt
theta = lng * np.pi / 180
phi = lat * np.pi / 180

rx = np.sin(phi) * np.cos(theta)
ry = np.sin(phi) * np.sin(theta)
rz = np.cos(phi)
R = np.concatenate([np.expand_dims(rx, 3), np.expand_dims(ry, 3), np.expand_dims(rz, 3)], axis=3)

thx = np.cos(phi) * np.cos(theta)
thy = np.cos(phi) * np.sin(theta)
thz = - np.sin(phi)
Th = np.concatenate([np.expand_dims(thx, 3), np.expand_dims(thy, 3), np.expand_dims(thz, 3)], axis=3)

phx = - np.sin(theta)
phy = np.cos(theta)
phz = zero
Ph = np.concatenate([np.expand_dims(phx, 3), np.expand_dims(phy, 3), np.expand_dims(phz, 3)], axis=3)

dLr = 1 * dr
dLph = r * dph
dLth = r * np.cos(phi) * dth

dSr = r * r * np.cos(phi) * dth * dph
dSph = r * np.cos(phi) * dth * dr
dSth = r * dph * dr

dV = r * r * dr * dth * dph


def dpath(df_th, df_ph, df_r):
    return df_r[:, :, :, np.newaxis] * R + r[:, :, :, np.newaxis] * df_th[:, :, :, np.newaxis] * Th + r[:, :, :, np.newaxis] * np.sin(theta)[:, :, :, np.newaxis] * df_ph[:, :, :, np.newaxis] * Ph


def grad(f):
    f_th, f_ph, f_r = np.gradient(f)
    return f_r[:, :, :, np.newaxis] * R + f_th[:, :, :, np.newaxis] / r[:, :, :, np.newaxis] * Th + f_ph[:, :, :, np.newaxis] / (r * np.sin(phi))[:, :, :, np.newaxis] * Ph


def div(F):
    Fth = F[:, :, :, 0]
    Fph = F[:, :, :, 1] * np.cos(phi)
    Fr = F[:, :, :, 2] * r * r
    val_th, _, _ = np.gradient(Fth)
    _, val_ph, _ = np.gradient(Fph)
    _, _, val_r = np.gradient(Fr)

    return (val_th + val_ph) / r / np.sin(theta) + val_r / r / r


def curl(F):
    Fth = F[:, :, :, 0]
    Fph = F[:, :, :, 1]
    Fr = F[:, :, :, 2]

    val_r = (np.gradient(Fph * np.sin(theta))[0] - np.gradient(Fth)[1]) / r / np.sin(theta)
    val_th = (np.gradient(Fr)[1] / np.sin(theta) - np.gradient(r * Fph)[2]) / r
    val_ph = (np.gradient(r * Fth)[2] - np.gradient(Fr)[0]) / r

    return val_r[:, :, :, np.newaxis] * R + val_th[:, :, :, np.newaxis] * Th + val_ph[:, :, :, np.newaxis] * Ph


def laplacian(f):
    return div(grad(f))


g = 9.80665

Omega = 2 * np.pi / (24 * 3600 * 0.99726966323716)

gamma = - 6.49 / 1000
gammad = - 9.80 / 1000
cv = 718.0
cp = 1005.0
R = 287
miu = 1.72e-1
M = 0.00289644 # molar mass of dry air, 0.00289644 kg/mol

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


def filter_extream_scalar(array):
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

    np.copyto(array, ndimage.gaussian_filter(array, 0.2))


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
        filter_extream_scalar(array)
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
                np.copyto(self.nxtval[:, :, i], merge(self.name, val[:, :, i], dt))

    def step(self, ** kwargs):
        return self.nxtval

    def swap(self):
        np.copyto(self.curval, self.nxtval)


class Relation(object):
    def __init__(self, name, lng_size, lat_size, alt_size, initval=0.0, initfn=None):
        self.lng_size = lng_size
        self.lat_size = lat_size
        self.alt_size = alt_size
        self.name = name
        context[name] = self

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

    def step(self, ** kwargs):
        return self.nxtval

    def swap(self):
        np.copyto(self.curval, self.nxtval)




