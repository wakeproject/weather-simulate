# -*- coding: utf-8 -*-

import numpy as np

import system

from system.planet import Relation, Grid, zero, alt, lng, theta, phi, bottom, dSr
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst, WaterDensity, RockDensity


def continent():
    return (np.absolute(2 + theta) < 1.2) * (phi > - np.pi / 9) * (phi < 2 * np.pi / 5) + (np.absolute(2 - theta) < 0.5) * (phi > - 2 * np.pi / 5) * (phi < np.pi / 3)


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

        doy = np.mod(system.t / 3600 / 24, 365.24)
        hod = np.mod(system.t / 3600 - lng / 15.0, 24)
        ha = 2 * np.pi * hod / 24
        decline = - 23.44 / 180 * np.pi * np.cos(2 * np.pi * (doy + 10) / 365)
        sza_coeff = np.sin(phi) * np.sin(decline) + np.cos(phi) * np.cos(decline) * np.cos(ha)

        return albedo * relu(sza_coeff) * SunConst * bottom

