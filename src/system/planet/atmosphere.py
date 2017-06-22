# -*- coding: utf-8 -*-

import numpy as np

from system.planet import Relation, Grid, zero, one, alt, theta, phi, a, g, Omeaga, gamma, gammad, cp, R, miu
from system.planet import StefanBoltzmann, WaterHeatCapacity, RockHeatCapacity, SunConst


def zinit(**kwargs):
    return zero


def uinit(**kwargs):
    return 0.1 + 0.2 * np.random.random([361, 179, 32])


def vinit(**kwargs):
    return 0.2 * np.random.random([361, 179, 32]) - 0.1


def winit(**kwargs):
    return 0.02 * np.random.random([361, 179, 32]) - 0.01


def tinit(**kwargs):
    return 278.15 - 0.006 * alt - 80 * (1 - np.cos(phi)) + 10 * np.cos(theta) + 0.2 * np.random.random([361, 179, 32])


def rinit(**kwargs):
    return np.exp(- alt / 10000) * (1 - 0.05 * np.cos(6 * theta + alt / 10000 * np.pi / 2)) + 0.2 * np.random.random([361, 179, 32])


class UGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(UGrd, self).__init__('u', lng_size, lat_size, alt_size, initfn=uinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        u_x, u_y, u_z = np.gradient(u)
        p_x, p_y, p_z = np.gradient(p)

        tao_xx, _, _ = np.gradient(u_x)
        _, tao_xy, _ = np.gradient(u_y)
        _, _, tao_xz = np.gradient(u_z)

        return - u * u_x - v * u_y - w * u_z + u * v * np.tan(phi) / a - u * w / a - p_x / rao - 2 * Omeaga * (w * np.sin(phi) - v * np.sin(phi)) + miu * (tao_xx + tao_xy + tao_xz) / rao


class VGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(VGrd, self).__init__('v', lng_size, lat_size, alt_size, initfn=vinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        v_x, v_y, v_z = np.gradient(v)
        p_x, p_y, p_z = np.gradient(p)

        tao_yx, _, _ = np.gradient(v_x)
        _, tao_yy, _ = np.gradient(v_y)
        _, _, tao_yz = np.gradient(v_z)

        return - u * v_x - v * v_y - w * v_z - u * u * np.tan(phi) / a - u * w / a - p_y / rao - 2 * Omeaga * u * np.sin(phi) + miu * (tao_yx + tao_yy + tao_yz) / rao


class WGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(WGrd, self).__init__('w', lng_size, lat_size, alt_size, initfn=winit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        w_x, w_y, w_z = np.gradient(u)
        p_x, p_y, p_z = np.gradient(p)

        tao_zx, _, _ = np.gradient(w_x)
        _, tao_zy, _ = np.gradient(w_y)
        _, _, tao_zz = np.gradient(w_z)

        #return - u * w_x - v * w_y - w * w_z - (u * u + v * v) - p_z / rao + 2 * Omeaga * u * np.cos(phi) - g + miu * (tao_zx + tao_zy + tao_zz) / rao
        return  - p_z / rao - g + miu * (tao_zx + tao_zy + tao_zz) / rao


class TGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(TGrd, self).__init__('T', lng_size, lat_size, alt_size, initfn=tinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        T_x, T_y, T_z = np.gradient(T)
        return - u * T_x - v * T_y + w * (gamma - gammad) + dH / cp


class RGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(RGrd, self).__init__('rao', lng_size, lat_size, alt_size, initfn=rinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        r_x, r_y, r_z = np.gradient(rao)
        u_x, _, _ = np.gradient(u)
        _, v_y, _ = np.gradient(v)
        _, _, w_z = np.gradient(w)
        return - u * r_x - v * r_y - w * r_z - rao * (u_x + v_y + w_z)


class QGrd(Grid):
    def __init__(self, lng_size, lat_size, alt_size):
        super(QGrd, self).__init__('q', lng_size, lat_size, alt_size, initfn=zinit)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        q_x, q_y, q_z = np.gradient(q)
        return - u * q_x - v * q_y - w * q_z + dQ


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
    coeff[:, :, ix] *= (0.5 ** ix)


class dHRel(Relation):
    def __init__(self, lng_size, lat_size, alt_size):
        super(dHRel, self).__init__('dH', lng_size, lat_size, alt_size)

    def step(self, u=None, v=None, w=None, rao=None, p=None, T=None, q=None, dQ=None, dH=None, lt=None, si=None):
        lt = (lt[:, :, 0]).reshape([361, 179, 1])
        income = StefanBoltzmann * lt * lt * lt * lt
        outcome = StefanBoltzmann * T * T * T * T

        return income * coeff - outcome

