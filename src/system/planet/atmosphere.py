# -*- coding: utf-8 -*-

import numpy as np

from system.planet import Relation, Grid, zero, one, alt, theta, phi, a, g, Omega, gamma, gammad, cv, R, miu, M
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst


def zinit(**kwargs):
    return zero


def uinit(**kwargs):
    return 2.5 * (1 - np.cos(6 * phi + alt / 8000 * np.pi / 2)) + 0.02 * np.random.random([361, 179, 32]) - 0.01


def vinit(**kwargs):
    return 0.02 * np.random.random([361, 179, 32]) - 0.01


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
        u_x, u_y, u_z = np.gradient(u)
        p_x, p_y, p_z = np.gradient(p)

        tao_xx, _, _ = np.gradient(u_x)
        _, tao_xy, _ = np.gradient(u_y)
        _, _, tao_xz = np.gradient(u_z)

        return u * v * np.tan(phi) / a - u * w / a - p_x / (a * rao * np.cos(phi)) - 2 * Omega * (w * np.cos(phi) - v * np.sin(phi)) + miu * (tao_xx + tao_xy + tao_xz) / rao


class VGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(VGrd, self).__init__('v', lng_size, lat_size, alt_size, initfn=vinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        v_x, v_y, v_z = np.gradient(v)
        p_x, p_y, p_z = np.gradient(p)

        tao_yx, _, _ = np.gradient(v_x)
        _, tao_yy, _ = np.gradient(v_y)
        _, _, tao_yz = np.gradient(v_z)

        return - u * u * np.tan(phi) / a - v * w / a - p_y / (a * rao) - 2 * Omega * u * np.sin(phi) + miu * (tao_yx + tao_yy + tao_yz) / rao


class WGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(WGrd, self).__init__('w', lng_size, lat_size, alt_size, initfn=winit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        w_x, w_y, w_z = np.gradient(u)
        p_x, p_y, p_z = np.gradient(p)

        tao_zx, _, _ = np.gradient(w_x)
        _, tao_zy, _ = np.gradient(w_y)
        _, _, tao_zz = np.gradient(w_z)

        return (u * u + v * v) / a - p_z / rao + 2 * Omega * u * np.cos(phi) - g + miu * (tao_zx + tao_zy + tao_zz) / rao
        #return - p_z / rao - g + miu * (tao_zx + tao_zy + tao_zz) / rao


class RGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(RGrd, self).__init__('rao', lng_size, lat_size, alt_size, initfn=rinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_x, _, _ = np.gradient(u)
        _, v_y, _ = np.gradient(v)
        _, _, w_z = np.gradient(w)
        return - rao * (u_x / (a * np.cos(phi)) + v_y / a + w_z) - rao * (v * np.tan(phi) / a + 2 * w / a)


class TGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(TGrd, self).__init__('T', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_x, _, _ = np.gradient(u)
        _, v_y, _ = np.gradient(v)
        _, _, w_z = np.gradient(w)
        return (dH + R * T * (v * np.tan(phi) / a + 2 * w / a - (u_x / (a * np.cos(phi)) + v_y / a + w_z))) / cv


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
        lt = (lt[:, :, 0]).reshape([361, 179, 1])
        income_l = StefanBoltzmann * lt * lt * lt * lt
        outcome = StefanBoltzmann * T * T * T * T

        income_a = np.copy(zero)
        for ix in range(32):
            if ix == 0:
                income_a[:, :, ix] += outcome[:, :, ix + 1] / 2
            elif ix == 31:
                income_a[:, :, ix] += outcome[:, :, ix - 1] / 2
            else:
                income_a[:, :, ix] += (outcome[:, :, ix - 1] / 2 + outcome[:, :, ix + 1] / 2)

        return income_l * coeff + income_a - outcome

