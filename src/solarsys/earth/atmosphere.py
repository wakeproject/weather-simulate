# -*- coding: utf-8 -*-

import numpy as np

import solarsys

from solarsys.earth import Relation, Grid
from solarsys import alt, lng, R
from solarsys import dlng, dlat, dalt, a, one, zero, bottom, top, r, theta, phi, Th, Ph, dSr, dSph, dSth, dV, dpath, div
from solarsys.earth import g, Omega, gamma, gammad, cv, cp, R, miu, M, niu_matrix
from solarsys.earth import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, WaterDensity, SunConst

from solarsys.earth.terrasphere import continent


def relu(x):
    return x * (x > 0)


def zinit(**kwargs):
    return np.copy(zero)


def uinit(**kwargs):
    return 20 * np.random.random(solarsys.shape) - 10


def vinit(**kwargs):
    return 20 * np.random.random(solarsys.shape) - 10


def winit(**kwargs):
    return np.copy(zero)


def tinit(**kwargs):
    return 288.15 - gamma * alt + 2 * np.random.random(solarsys.shape) - 1


def pinit(**kwargs):
    return 101325 * np.exp(- g * M * alt / 288.15 / 8.31447) + 100 * np.random.random(solarsys.shape) - 50


def rinit(**kwargs):
    t = tinit()
    p = pinit()
    return p / R / t


class UGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(UGrd, self).__init__('u', lng_size, lat_size, alt_size, initfn=uinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        a_th, _, _ = np.gradient(p * dSth) / (r * np.cos(phi)) / rao / dV

        f_th = np.gradient(rao * u * dSth * u)[0] / (r * np.cos(phi))
        f_ph = np.gradient(rao * u * dSth * v)[1] / r
        f_r = np.gradient(rao * u * dSth * w)[2]

        f = 0.0004 * (f_th + f_ph + f_r) / rao / dV

        return u * v / r * np.tan(phi) - u * w / r - 2 * Omega * (w * np.cos(phi) - v * np.sin(phi)) + a_th - f


class VGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(VGrd, self).__init__('v', lng_size, lat_size, alt_size, initfn=vinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        _, a_ph, _ = np.gradient(p * dSph) / r / rao / dV

        f_th = np.gradient(rao * v * dSph * u)[0] / (r * np.cos(phi))
        f_ph = np.gradient(rao * v * dSph * v)[1] / r
        f_r = np.gradient(rao * v * dSph * w)[2]

        f = 0.0004 * (f_th + f_ph + f_r) / rao / dV

        return - u * u / r * np.tan(phi) - v * w / r - 2 * Omega * u * np.sin(phi) + a_ph - f


class WGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(WGrd, self).__init__('w', lng_size, lat_size, alt_size, initfn=winit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        _, _, a_r = np.gradient(p * dSr) / rao / dV

        f_th = np.gradient(rao * w * dSr * u)[0] / (r * np.cos(phi))
        f_ph = np.gradient(rao * w * dSr * v)[1] / r
        f_r = np.gradient(rao * w * dSr * w)[2]

        f = 0.0004 * (f_th + f_ph + f_r) / rao / dV

        dw = (u * u + v * v) / r + 2 * Omega * u * np.cos(phi) - g + a_r - f
        return dw * (1 - bottom) * (1 - top) + (w > 0) * dw * bottom + (w < 0) * dw * top


class RGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(RGrd, self).__init__('rao', lng_size, lat_size, alt_size, initfn=rinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        u_th, _, _ = np.gradient(u)
        _, v_ph, _ = np.gradient(v)
        _, _, w_r = np.gradient(w)

        return rao * (u_th * dSth + v_ph * dSph + w_r * dSr * (1 - bottom)) / dV


class TGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(TGrd, self).__init__('T', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        dp = solarsys.earth.context['p'].drvval
        return dH / dV / rao / cp + dp / rao / cp


class QGrd(Grid):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(QGrd, self).__init__('q', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        u_th, _, _ = np.gradient(u)
        _, v_ph, _ = np.gradient(v)
        _, _, w_r = np.gradient(w)

        return q * (u_th * dSth + v_ph * dSph + w_r * dSr * (1 - bottom)) / dV + dQ


class PRel(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(PRel, self).__init__('p', lng_size, lat_size, alt_size, initfn=pinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        return rao * R * T


class dQRel(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(dQRel, self).__init__('dQ', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        dT = solarsys.earth.context['T'].drvval
        cntnt = continent()

        return + 0.00001 * T * T * T / 273.15 / (373.15 - T) / (373.15 - T) * (dT > 0) * (1 - cntnt) + 0.000001 * (dT > 0) * cntnt - 0.000001 * (dT < 0) * (q > 0.0001)


class dHRel(Relation):
    def __init__(self, shape):
        lng_size, lat_size, alt_size = shape
        super(dHRel, self).__init__('dH', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None, tc=None):
        doy = np.mod(solarsys.t / 3600 / 24, 365.24)
        hod = np.mod(solarsys.t / 3600 - lng / 15.0, 24)
        ha = 2 * np.pi * hod / 24
        decline = - 23.44 / 180 * np.pi * np.cos(2 * np.pi * (doy + 10) / 365)
        sza_coeff = np.sin(phi) * np.sin(decline) + np.cos(phi) * np.cos(decline) * np.cos(ha)

        dT = solarsys.earth.context['T'].drvval
        absorbS = np.sqrt(q) * (dT < 0) * (q > 0.0001)
        absorbL = np.sqrt(np.sqrt(q)) * (dT < 0) * (q > 0.0001)

        reachnessS = np.ones((solarsys.shape[0], solarsys.shape[1], solarsys.shape[2], solarsys.shape[2]))
        for ix in range(32):
            for jx in range(ix, 32):
                for kx in range(ix, jx):
                    reachnessS[:, :, ix, jx] = reachnessS[:, :, ix, jx] * (1 - absorbS[:, :, kx])

        reachnessL = np.ones((solarsys.shape[0], solarsys.shape[1], solarsys.shape[2], solarsys.shape[2]))
        for ix in range(32):
            for jx in range(ix, 32):
                for kx in range(ix, jx):
                    reachnessL[:, :, ix, jx] = reachnessL[:, :, ix, jx] * (1 - absorbL[:, :, kx])

        income_s = relu(sza_coeff) * SunConst * top

        lt = lt[:, :, 0::32]
        income_l = StefanBoltzmann * lt * lt * lt * lt * dSr * (lt > 0)
        outcome = StefanBoltzmann * T * T * T * T * dSr * (T > 0)

        fusion = + (lt >= 273.15) * (lt < 275) * dSr * bottom * 0.01 * WaterDensity * 333550 \
                 - (lt > 271) * (lt <= 273.155) * dSr * bottom * 0.01 * WaterDensity * 333550

        income = np.copy(zero)
        for ix in range(32):
            for jx in range(32):
                income[:, :, ix] += outcome[:, :, jx] * reachnessL[:, :, ix, jx]
            income[:, :, ix] += income_l[:, :, ix] * reachnessL[:, :, ix, 0]
            income[:, :, ix] += income_s[:, :, -1] * reachnessS[:, :, ix, 31]

        return (income - outcome) - 2266000 * dQ * dV + fusion

