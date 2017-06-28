# -*- coding: utf-8 -*-

import numpy as np

import system

from system.planet import Relation, Grid, div
from system.planet import zero, one, alt, bottom, top, theta, phi, r, dSr, dSth, dSph, dV, dlng, dlat, dalt, Th, Ph, R, dpath
from system.planet import a, g, Omega, gamma, gammad, cv, cp, R, miu, M, niu_matrix
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, WaterDensity, SunConst

from system.planet.terrasphere import continent


def zinit(**kwargs):
    return np.copy(zero)


def uinit(**kwargs):
    return np.copy(zero)


def vinit(**kwargs):
    return np.copy(zero)


def winit(**kwargs):
    return np.copy(zero)


def tinit(**kwargs):
    return 288.15 - gamma * alt + 2 * np.random.random(system.planet.shape) - 1


def pinit(**kwargs):
    return 101325 * np.exp(- g * M * alt / 288.15 / 8.31447) + 100 * np.random.random(system.planet.shape) - 50


def rinit(**kwargs):
    t = tinit()
    p = pinit()
    return p / R / t


class UGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(UGrd, self).__init__('u', lng_size, lat_size, alt_size, initfn=uinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        a_th, _, _ = np.gradient(p * dSth) / rao / dV

        f_th = np.gradient(rao * u * dSth * u)[0]
        f_ph = np.gradient(rao * u * dSth * v)[1]
        f_r = np.gradient(rao * u * dSth * w)[2]

        f = 0.001 * (f_th + f_ph + f_r) / rao / dV

        print '----------------------------------------'
        print 'u', np.max(f), np.min(f), np.mean(f)
        print 'u', np.max(a_th), np.min(a_th), np.mean(a_th)

        return u * v / r * np.tan(phi) - u * w / r - 2 * Omega * (w * np.cos(phi) - v * np.sin(phi)) + a_th - f


class VGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(VGrd, self).__init__('v', lng_size, lat_size, alt_size, initfn=vinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        _, a_ph, _ = np.gradient(p * dSph) / rao / dV

        f_th = np.gradient(rao * v * dSph * u)[0]
        f_ph = np.gradient(rao * v * dSph * v)[1]
        f_r = np.gradient(rao * v * dSph * w)[2]

        f = 0.001 * (f_th + f_ph + f_r) / rao / dV * r

        print '----------------------------------------'
        print 'v', np.max(f), np.min(f), np.mean(f)
        print 'v', np.max(a_ph), np.min(a_ph), np.mean(a_ph)

        return - u * u / r * np.tan(phi) - v * w / r - 2 * Omega * u * np.sin(phi) + a_ph - f


class WGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(WGrd, self).__init__('w', lng_size, lat_size, alt_size, initfn=winit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        _, _, a_r = np.gradient(p * dSr) / rao / dV

        f_th = np.gradient(rao * w * dSr * u)[0]
        f_ph = np.gradient(rao * w * dSr * v)[1]
        f_r = np.gradient(rao * w * dSr * w)[2]

        f = 0.001 * (f_th + f_ph + f_r) / rao / dV * dalt

        print '----------------------------------------'
        print 'w', np.max(f), np.min(f), np.mean(f)
        print 'w', np.max(a_r), np.min(a_r), np.mean(a_r)

        dw = (u * u + v * v) / r + 2 * Omega * u * np.cos(phi) - g + a_r - f
        return dw * (1 - bottom) * (1 - top) + (w > 0) * dw * bottom + (w < 0) * dw * top


class RGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(RGrd, self).__init__('rao', lng_size, lat_size, alt_size, initfn=rinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_th, _, _ = np.gradient(u)
        _, v_ph, _ = np.gradient(v)
        _, _, w_r = np.gradient(w)

        return rao * (u_th * dSth + v_ph * dSph + w_r * dSr * (1 - bottom)) / dV


class TGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(TGrd, self).__init__('T', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        dp = system.planet.context['p'].drvval
        return dH / dV / rao / cp + dp / rao / cp


class QGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(QGrd, self).__init__('q', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_th, _, _ = np.gradient(u)
        _, v_ph, _ = np.gradient(v)
        _, _, w_r = np.gradient(w)

        return q * (u_th * dSth + v_ph * dSph + w_r * dSr * (1 - bottom)) / dV + dQ


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
        dT = system.planet.context['T'].drvval
        cntnt = continent()

        return + 0.00001 * (dT > 0) * cntnt + 0.0001 * (dT > 0) * (1 - cntnt) - 0.0001 * (dT < 0) * (q > 0.0001)


coeff = np.copy(one)
for ix in range(32):
    coeff[:, :, ix] *= (0.1 * 0.9 ** (ix + 1))


class dHRel(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(dHRel, self).__init__('dH', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        lt = lt[:, :, 0::32]
        income_l = StefanBoltzmann * lt * lt * lt * lt * dSr
        outcome = StefanBoltzmann * T * T * T * T * dSr

        fusion = + (lt >= 273.15) * (lt < 275) * dSr * bottom * 0.01 * WaterDensity * 333550 \
                 - (lt > 271) * (lt <= 273.155) * dSr * bottom * 0.01 * WaterDensity * 333550

        income_r = np.copy(zero)
        for ix in range(32):
            if ix == 0:
                income_r[:, :, ix] += outcome[:, :, ix + 1] / 2
            elif ix == 31:
                income_r[:, :, ix] += outcome[:, :, ix - 1] / 2
            else:
                income_r[:, :, ix] += (outcome[:, :, ix - 1] / 2 + outcome[:, :, ix + 1] / 2)

        return (income_l * coeff + income_r - outcome) - 2266000 * dQ * dV + fusion

