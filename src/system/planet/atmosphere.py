# -*- coding: utf-8 -*-

import numpy as np

import system

from system.planet import Relation, Grid, merge
from system.planet import zero, one, alt, bottom, theta, phi, r, dSr, dV
from system.planet import a, g, Omega, gamma, gammad, cv, cp, R, miu, M, niu_matrix
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst


def zinit(**kwargs):
    return np.copy(zero)


def uinit(**kwargs):
    return np.copy(zero)


def vinit(**kwargs):
    return np.copy(zero)


def winit(**kwargs):
    return np.copy(zero)


def tinit(**kwargs):
    return 288.15 + gamma * alt


def pinit(**kwargs):
    t = tinit()
    return 101325 * (t / 288.15) ** (g * M / R / gamma)


def rinit(**kwargs):
    t = tinit()
    p = pinit()
    return p * M / R / t


class UGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(UGrd, self).__init__('u', lng_size, lat_size, alt_size, initfn=uinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        p_th, _, _ = np.gradient(p)

        return u * v / r * np.tan(phi) - u * w / r - p_th / (rao * r * np.cos(phi)) - 2 * Omega * (w * np.cos(phi) - v * np.sin(phi))


class VGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(VGrd, self).__init__('v', lng_size, lat_size, alt_size, initfn=vinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        _, p_ph, _ = np.gradient(p)

        return - u * u / r * np.tan(phi) - v * w / r - p_ph / (rao * r) - 2 * Omega * u * np.sin(phi)


class WGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(WGrd, self).__init__('w', lng_size, lat_size, alt_size, initfn=winit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        _, _, p_r = np.gradient(p)

        return (u * u + v * v) / r - p_r / rao + 2 * Omega * u * np.cos(phi) - g


class RGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(RGrd, self).__init__('rao', lng_size, lat_size, alt_size, initfn=rinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_th, _, _ = np.gradient(u)
        _, v_ph, _ = np.gradient(v)
        _, _, w_r = np.gradient(w)

        return rao * (v / r * np.tan(phi) - 2 * w / r - u_th / (r * np.cos(phi)) - v_ph / r - w_r)


class TGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(TGrd, self).__init__('T', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return (dH + R * T * system.planet.context['rao'].step(u, v, w, rao, p, T, q, dQ, dH, lt, si)) / (cp - rao * R)


class QGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(QGrd, self).__init__('q', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return dQ


class PRel(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(PRel, self).__init__('p', lng_size, lat_size, alt_size, initfn=pinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return rao * R * T


class dQRel(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(dQRel, self).__init__('dQ', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return zero


coeff = np.copy(one)
for ix in range(32):
    coeff[:, :, ix] *= (0.9 * 0.1 ** (ix + 1))


class dHRel(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(dHRel, self).__init__('dH', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        lt = lt[:, :, 0::32]
        income_l = StefanBoltzmann * lt * lt * lt * lt * dSr[:, :, 0::32]
        outcome = StefanBoltzmann * T * T * T * T * dSr

        income_r = np.copy(zero)
        for ix in range(32):
            if ix == 0:
                income_r[:, :, ix] += outcome[:, :, ix + 1] / 2
            elif ix == 31:
                income_r[:, :, ix] += outcome[:, :, ix - 1] / 2
            else:
                income_r[:, :, ix] += (outcome[:, :, ix - 1] / 2 + outcome[:, :, ix + 1] / 2)

        return (income_l * coeff + income_r - outcome) / dV

