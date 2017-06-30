# -*- coding: utf-8 -*-

import numpy as np
import cv2

from os import path

import solarsys

from solarsys.earth import Relation, Grid
from solarsys import shape, zero, bottom, theta, phi, dSr, alt, lng, lat
from solarsys.earth import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst, WaterDensity, RockDensity


if not path.exists('data/continent.npy'):
    im = cv2.imread('data/earth-continent.png', 0)
    np.save('data/continent', im > 200)

cntndata = np.array(np.load('data/continent.npy'), dtype=np.float64).T
cntndata = (cv2.resize(cntndata, (shape[1], shape[0])))[:, :, np.newaxis]


def continent():
    return (cntndata > 0.9)


def relu(x):
    return x * (x > 0)


def zinit(**kwargs):
    return np.copy(zero)


def tinit():
    return 278.15 * bottom


class TLGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(TLGrd, self).__init__('lt', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        contnt = continent()
        capacity = WaterHeatCapacity * (1 - contnt) + RockHeatCapacity * contnt
        density = WaterDensity * (1 - contnt) + RockDensity * contnt

        return (si + StefanBoltzmann * T * T * T * T / 2 - StefanBoltzmann * lt * lt * lt * lt) / (capacity * density) * bottom


class SIGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(SIGrd, self).__init__('si', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        albedo = 0.7 * (lt > 273.15) + 0.1 * (lt < 273.15) # considering ice and soil

        doy = np.mod(solarsys.t / 3600 / 24, 365.24)
        hod = np.mod(solarsys.t / 3600 - lng / 15.0, 24)
        ha = 2 * np.pi * hod / 24
        decline = - 23.44 / 180 * np.pi * np.cos(2 * np.pi * (doy + 10) / 365)
        sza_coeff = np.sin(phi) * np.sin(decline) + np.cos(phi) * np.cos(decline) * np.cos(ha)

        return albedo * relu(sza_coeff) * SunConst * tc * bottom


class TotalCloudage(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(TotalCloudage, self).__init__('tc', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        dT = solarsys.earth.context['T'].drvval
        cloudage = np.sqrt(q) * (dT < 0) * (q > 0.0001)

        ratio = 1 - cloudage
        ratio_total = np.copy(bottom)
        for ix in range(32):
            ratio_total[:, :, 0] = ratio_total[:, :, 0] * ratio[:, :, ix]

        return 1 - ratio_total

