# -*- coding: utf-8 -*-

import numpy as np

import system

from system.planet import Relation, Grid, zero, alt, theta, phi
from system.planet import a, g, Omega
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst


def continent():
    return (np.absolute(2 + theta) < 1.2) * (phi > - np.pi / 9) * (phi < 2 * np.pi / 5) + (np.absolute(2 - theta) < 0.5) * (phi > - 2 * np.pi / 5) * (phi < np.pi / 3)


def relu(x):
    return x * (x > 0)


def zinit(**kwargs):
    return zero


def tinit():
    return 273.15 - 60 * (1 - np.cos(phi)) + 7 * np.cos(theta)


class TLGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(TLGrd, self).__init__('lt', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        contnt = continent()
        capacity = WaterHeatCapacity * 100 * (1 - contnt) + RockHeatCapacity * 100 * contnt
        return (si - StefanBoltzmann * lt * lt * lt * lt) / capacity


class SIGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(SIGrd, self).__init__('si', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return 3 * SunConst * np.cos(phi) * relu(np.sin(theta - Omega * system.t))

