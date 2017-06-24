# -*- coding: utf-8 -*-

import numpy as np

from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

context = {}

dlng = 1.0
dlat = 1.0
dalt = 500.0

a = 6371000

lat, lng, alt = np.meshgrid(np.arange(-89.5, 89.5, 1), np.arange(-180, 181, 1), np.arange(0, 16000, 500.0))

one = np.ones(lng.shape)
zero = np.zeros(lng.shape)
bottom = np.zeros(lng.shape)
bottom[:, :, 0] = 1

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
R = 8.31447
miu = 1.72e-1
M = 0.0289644 # molar mass of dry air, 0.0289644 kg/mol

niu = 0.1 # friction between air and land surface
niu_matrix = niu * bottom

SunConst = 1366
StefanBoltzmann = 0.0000000567
WaterHeatCapacity = 4200
RockHeatCapacity = 840
WaterDensity = 1000
RockDensity = 2650


def filter_extreams(array, dt):
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


def merge(array, dt):
    filter_extreams(array, dt)
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
            np.copyto(self.nxtval[:, :, i], merge(val[:, :, i], dt))

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

    def evolve(self, dt):
        kwargs = {k: v.curval for k, v in context.iteritems()}
        val = self.step(**kwargs)
        for i in range(32):
            np.copyto(self.nxtval[:, :, i], merge(val[:, :, i], dt))

    def step(self, ** kwargs):
        return self.nxtval

    def swap(self):
        np.copyto(self.curval, self.nxtval)




