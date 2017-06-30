# -*- coding: utf-8 -*-

import numpy as np


t = 0

context = {
    't': t,
}

dlng = 2
dlat = 2
dalt = 500.0

a = 6371000

lat, lng, alt = np.meshgrid(np.arange(-89.5, 89.5, dlat), np.arange(-180, 181, dlng), np.arange(0, 16000, dalt))

shape = lng.shape
one = np.ones(lng.shape)
zero = np.zeros(lng.shape)
bottom = np.zeros(lng.shape)
top = np.zeros(lng.shape)
north_pole = np.zeros(lng.shape)
south_pole = np.zeros(lng.shape)

bottom[:, :, 0] = 1
top[:, :, -1] = 1

north_pole[:, 0, :] = 1
south_pole[:, -1, :] = 1

dth = np.pi / 180 * dlng * one
dph = np.pi / 180 * dlat * one
dr = 1 * dalt * one

r = a + alt
theta = lng * np.pi / 180
phi = lat * np.pi / 180

rx = np.cos(phi) * np.cos(theta)
ry = np.cos(phi) * np.sin(theta)
rz = np.sin(phi)

R = np.concatenate([np.expand_dims(rx, 3), np.expand_dims(ry, 3), np.expand_dims(rz, 3)], axis=3)

thx = np.sin(phi) * np.cos(theta)
thy = np.sin(phi) * np.sin(theta)
thz = - np.cos(phi)

Th = np.concatenate([np.expand_dims(thx, 3), np.expand_dims(thy, 3), np.expand_dims(thz, 3)], axis=3)

phx = - np.sin(theta)
phy = np.cos(theta)
phz = np.copy(zero)

Ph = np.concatenate([np.expand_dims(phx, 3), np.expand_dims(phy, 3), np.expand_dims(phz, 3)], axis=3)

dLr = 1 * dr
dLph = r * dph
dLth = r * np.cos(phi) * dth
dSr = dLph * dLth
dSph = dLr * dLth
dSth = dLr * dLph
dV = dLr * dLph * dLth


def dpath(df_th, df_ph, df_r):
    return df_r[:, :, :, np.newaxis] * R + r[:, :, :, np.newaxis] * df_th[:, :, :, np.newaxis] * Th + r[:, :, :, np.newaxis] * np.cos(phi)[:, :, :, np.newaxis] * df_ph[:, :, :, np.newaxis] * Ph


def grad(f):
    f_th, f_ph, f_r = np.gradient(f)
    return f_r[:, :, :, np.newaxis] * R + f_th[:, :, :, np.newaxis] / r[:, :, :, np.newaxis] * Th + f_ph[:, :, :, np.newaxis] / (r * np.cos(phi))[:, :, :, np.newaxis] * Ph


def div(F):
    Fth = F[:, :, :, 0]
    Fph = F[:, :, :, 1] * np.cos(phi)
    Fr = F[:, :, :, 2] * r * r
    val_th, _, _ = np.gradient(Fth)
    _, val_ph, _ = np.gradient(Fph)
    _, _, val_r = np.gradient(Fr)

    return (val_th + val_ph) / r / np.cos(theta) + val_r / r / r


def curl(F):
    Fth = F[:, :, :, 0]
    Fph = F[:, :, :, 1]
    Fr = F[:, :, :, 2]

    val_r = (np.gradient(Fph * np.cos(phi))[0] - np.gradient(Fth)[1]) / r / np.cos(phi)
    val_th = (np.gradient(Fr)[1] / np.cos(phi) - np.gradient(r * Fph)[2]) / r
    val_ph = (np.gradient(r * Fth)[2] - np.gradient(Fr)[0]) / r

    return val_r[:, :, :, np.newaxis] * R + val_th[:, :, :, np.newaxis] * Th + val_ph[:, :, :, np.newaxis] * Ph


def laplacian(f):
    return div(grad(f))