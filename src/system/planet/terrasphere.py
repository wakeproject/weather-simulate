# -*- coding: utf-8 -*-

import numpy as np
import cv2

from os import path

import system

from system.planet import Relation, Grid, zero, shape, alt, lng, theta, phi, bottom, dSr
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst, WaterDensity, RockDensity
from system.planet import shtAbsorbLand, shtAbsorbAir


if not path.exists('data/continent.npy'):
    im = cv2.imread('data/earth-continent.png', 0)
    np.save('data/continent', im > 250)

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

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        contnt = continent()
        capacity = WaterHeatCapacity * (1 - contnt) + RockHeatCapacity * contnt
        density = WaterDensity * (1 - contnt) + RockDensity * contnt

        return (si + StefanBoltzmann * T * T * T * T / 2 - StefanBoltzmann * lt * lt * lt * lt) / (capacity * density) * bottom


class SIGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(SIGrd, self).__init__('si', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        albedo = 0.7 * (lt > 273.15) + 0.1 * (lt < 273.15) # considering ice and soil

        dT = system.planet.context['T'].drvval
        cloudage = np.sqrt(q) * (dT < 0) * (q > 0.0001)

        ratio_in = 1 - cloudage
        ratio_last = np.copy(bottom)
        for ix in range(32):
            ratio_last[:, :, 0] = ratio_last[:, :, 0] * ratio_in[:, :, ix]
        print 'cloudy', 1 - np.min(ratio_last[:, :, 0]), 1 - np.max(ratio_last[:, :, 0]), 1 - np.mean(ratio_last[:, :, 0])

        doy = np.mod(system.t / 3600 / 24, 365.24)
        hod = np.mod(system.t / 3600 - lng / 15.0, 24)
        ha = 2 * np.pi * hod / 24
        decline = - 23.44 / 180 * np.pi * np.cos(2 * np.pi * (doy + 10) / 365)
        sza_coeff = np.sin(phi) * np.sin(decline) + np.cos(phi) * np.cos(decline) * np.cos(ha)

        return albedo * relu(sza_coeff) * SunConst * ratio_last * bottom

