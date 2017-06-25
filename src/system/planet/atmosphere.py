# -*- coding: utf-8 -*-

import numpy as np

from system.planet import Relation, Grid, merge
from system.planet import zero, one, alt, bottom, theta, phi, r, dSr, dV
from system.planet import a, g, Omega, gamma, gammad, cv, R, miu, M, niu_matrix
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst


def zinit(**kwargs):
    return zero


def uinit(**kwargs):
    return 100 * (1.2 - np.cos(6 * phi + alt / 8000 * np.pi / 2)) + 40 * np.random.random([361, 179, 32]) - 20


def vinit(**kwargs):
    return 100 * np.sin(6 * phi + alt / 8000 * np.pi / 2) + 100 * np.random.random([361, 179, 32]) - 50


def winit(**kwargs):
    return 0.02 * np.random.random([361, 179, 32]) - 0.01


def tinit(**kwargs):
    return 288.15 - gamma * alt


def pinit(**kwargs):
    t = tinit()
    return 101325 * (t / 288.15) ** (g * M / R / gamma)


def rinit(**kwargs):
    t = tinit()
    p = pinit()
    return p * M / R / t


class UGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(UGrd, self).__init__('u', lng_size, lat_size, alt_size, initfn=uinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        p_th, _, _ = np.gradient(p)

        #return u * v * np.tan(phi) / a - u * w / a - p_th / (a * rao * np.cos(phi)) - 2 * Omega * (w * np.cos(phi) - v * np.sin(phi)) + miu * (tao_xx + tao_xy + tao_xz) / rao - niu_matrix * u / rao
        return u * v * np.tan(phi) / a - u * w / a - p_th / (a * rao * np.cos(phi)) - 2 * Omega * (w * np.cos(phi) - v * np.sin(phi)) - niu_matrix * u / rao


class VGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(VGrd, self).__init__('v', lng_size, lat_size, alt_size, initfn=vinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        _, p_ph, _ = np.gradient(p)

        #return - u * u * np.tan(phi) / a - v * w / a - p_ph / (a * rao) - 2 * Omega * u * np.sin(phi) + miu * (tao_yx + tao_yy + tao_yz) / rao - niu_matrix * v / rao
        return - u * u * np.tan(phi) / a - v * w / a - p_ph / (a * rao) - 2 * Omega * u * np.sin(phi) - niu_matrix * v / rao


class WGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(WGrd, self).__init__('w', lng_size, lat_size, alt_size, initfn=winit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        _, _, p_r = np.gradient(p)

        #return (u * u + v * v) / a - p_a / rao + 2 * Omega * u * np.cos(phi) - g + miu * (tao_zx + tao_zy + tao_zz) / rao - niu_matrix * w / rao
        #return - p_z / rao - g + miu * (tao_zx + tao_zy + tao_zz) / rao
        return (u * u + v * v) / a - p_r / rao + 2 * Omega * u * np.cos(phi) - g


class RGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(RGrd, self).__init__('rao', lng_size, lat_size, alt_size, initfn=rinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_th, _, _ = np.gradient(u)
        _, v_ph, _ = np.gradient(v)
        _, _, w_r = np.gradient(w)

        return - rao * (u_th / (r * np.cos(phi)) + v_ph / r + w_r) - rao * (v * np.tan(phi) / r + 2 * w / r)


class TGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(TGrd, self).__init__('T', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_t, _, _ = np.gradient(u)
        _, v_p, _ = np.gradient(v)
        _, _, w_a = np.gradient(w)
        return (dH + R * T * (v * np.tan(phi) / r + 2 * w / r - (u_t / (r * np.cos(phi)) + v_p / r + w_a))) / cv


class QGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(QGrd, self).__init__('q', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return dQ


class PRel(Relation):
    def __init__(self, lng_size, lat_size, alt_size):
        super(PRel, self).__init__('p', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return rao * T * R


class dQRel(Relation):
    def __init__(self, lng_size, lat_size, alt_size):
        super(dQRel, self).__init__('dQ', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        return zero


coeff = np.copy(one)
for ix in range(32):
    coeff[:, :, ix] *= (0.9 * 0.1 ** (ix + 1))


class dHRel(Relation):
    def __init__(self, lng_size, lat_size, alt_size):
        super(dHRel, self).__init__('dH', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        lt = lt[:, :, 0::32]
        income_l = StefanBoltzmann * lt * lt * lt * lt * dSr[:, :, 0::32]
        outcome = StefanBoltzmann * T * T * T * T * dSr

        income_a = np.copy(zero)
        for ix in range(32):
            if ix == 0:
                income_a[:, :, ix] += outcome[:, :, ix + 1] / 2
            elif ix == 31:
                income_a[:, :, ix] += outcome[:, :, ix - 1] / 2
            else:
                income_a[:, :, ix] += (outcome[:, :, ix - 1] / 2 + outcome[:, :, ix + 1] / 2)

        return (income_l * coeff + income_a - outcome) / dV

