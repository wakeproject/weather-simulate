# -*- coding: utf-8 -*-

import numpy as np
import pygame
import time

import system

from system.planet.atmosphere import UGrd, VGrd, WGrd, TGrd, RGrd, QGrd, PRel, dHRel, dQRel
from system.planet.terrasphere import TLGrd, SIGrd, continent


u = UGrd(system.planet.shape)
v = VGrd(system.planet.shape)
w = WGrd(system.planet.shape)
T = TGrd(system.planet.shape)
rao = RGrd(system.planet.shape)
q = QGrd(system.planet.shape)

p = PRel(system.planet.shape)
dH = dHRel(system.planet.shape)
dQ = dQRel(system.planet.shape)

tl = TLGrd(system.planet.shape)
si = SIGrd(system.planet.shape)

cntn = continent()


def evolve():
    s = np.sqrt(u.curval * u.curval + v.curval * v.curval + w.curval * w.curval + 0.00001)
    dt = 100 / np.max(s)
    if dt > 1:
        dt = 1
    system.t = system.t + dt
    print '----------------------------------------------------'
    print system.t, dt
    print 'wind: ', np.max(s), np.min(s), np.mean(s)
    print 'temp', np.max(T.curval - 273.15), np.min(T.curval - 273.15), np.mean(T.curval - 273.15)
    print 'pres', np.max(p.curval / 101325), np.min(p.curval / 101325), np.mean(p.curval / 101325)
    print 'rao', np.max(rao.curval), np.min(rao.curval), np.mean(rao.curval)
    print 'humd', np.max(q.curval), np.min(q.curval), np.mean(q.curval)

    u.evolve(dt)
    v.evolve(dt)
    w.evolve(dt)
    T.evolve(dt)
    rao.evolve(dt)
    q.evolve(dt)

    p.evolve(dt)
    dH.evolve(dt)
    dQ.evolve(dt)

    tl.evolve(dt)
    si.evolve(dt)


def flip():
    u.swap()
    v.swap()
    w.swap()
    T.swap()
    rao.swap()
    q.swap()

    p.swap()
    dH.swap()
    dQ.swap()

    tl.swap()
    si.swap()


def normalize(array):
    maxv = np.max(array)
    minv = np.min(array)
    return (array - minv) / (maxv - minv + 0.001) * 255


if __name__ == '__main__':
    map_width = system.planet.shape[0]
    map_height = system.planet.shape[1]

    tile_size = 9
    arrrow_size = tile_size

    pygame.init()
    screen = pygame.display.set_mode((map_width * tile_size, map_height * tile_size))
    background = pygame.Surface(screen.get_size())

    clock = pygame.time.Clock()

    first_gen = True
    timer = 12

    running = True
    lasttile = 0
    while running == True:
        clock.tick(5)
        time.sleep(5)
        pygame.display.set_caption('FPS: ' + str(clock.get_fps()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        evolve()

        tmap = T.curval[:, :, 0]
        umap = 0.5 * u.curval[:, :, 0] + 0.5 * u.curval[:, :, 1]
        vmap = 0.5 * v.curval[:, :, 0] + 0.5 * v.curval[:, :, 1]
        wmap = 0.5 * w.curval[:, :, 0] + 0.5 * w.curval[:, :, 1]
        smap = np.sqrt(umap * umap + vmap * vmap + wmap * wmap + 0.001)
        mxs = np.max(smap)
        mns = np.min(smap)
        mms = np.mean(smap)
        print 'wind(O): ', mxs, mns, mms

        tcmap = normalize(T.curval[:, :, 0])
        scmap = normalize(smap)
        ucmap = normalize(umap)
        vcmap = normalize(vmap)
        wcmap = normalize(wmap)
        for ixlng in range(system.planet.shape[0]):
            for ixlat in range(system.planet.shape[1]):
                tval = tmap[ixlng, ixlat]
                uval = umap[ixlng, ixlat]
                vval = vmap[ixlng, ixlat]
                sval = smap[ixlng, ixlat]

                scolor = scmap[ixlng, ixlat]
                tcolor = tcmap[ixlng, ixlat]
                ucolor = ucmap[ixlng, ixlat]
                vcolor = vcmap[ixlng, ixlat]
                wcolor = wcmap[ixlng, ixlat]
                tile = pygame.Surface((tile_size, tile_size))
                r = (int(tcolor * 2 / 3) + int(72 * cntn[ixlng, ixlat, 0])) * (tval > 273.15) + (128 + int(tcolor / 2) + int(72 * cntn[ixlng, ixlat, 0])) * (tval <= 273.15)
                g = (255 - int(tcolor * 2 / 3) - int(72 * cntn[ixlng, ixlat, 0])) + (255 - int(tcolor / 4) - int(72 * cntn[ixlng, ixlat, 0])) * (tval <= 273.15)
                b = (255 - int(tcolor * 2 / 3) - int(72 * cntn[ixlng, ixlat, 0])) + (255 - int(tcolor / 4) - int(72 * cntn[ixlng, ixlat, 0])) * (tval <= 273.15)
                r = (r > 255) * 255 + (r > 0) * (r < 256) * r
                g = (g > 255) * 255 + (g > 0) * (g < 256) * g
                b = (b > 255) * 255 + (b > 0) * (b < 256) * b
                tile.fill((r, g, b))
                tile.set_alpha(64)

                if ixlng % 2 == 0 and ixlat % 2 == 0:
                    length = int(arrrow_size * sval / mxs)
                    if np.absolute(uval) >= np.absolute(vval):
                        pygame.draw.aaline(tile, (int(wcolor), int(ucolor), int(vcolor)), [arrrow_size / 2.0 - length, arrrow_size / 2.0 - length * vval / uval],
                                                                                          [arrrow_size / 2.0 + length, arrrow_size / 2.0 + length * vval / uval], True)
                    else:
                        pygame.draw.aaline(tile, (int(wcolor), int(ucolor), int(vcolor)), [arrrow_size / 2.0 - length * vval / uval, arrrow_size / 2.0 - length],
                                                                                          [arrrow_size / 2.0 + length * vval / uval, arrrow_size / 2.0 + length], True)

                screen.blit(tile, (ixlng * tile_size, ixlat * tile_size))

        flip()
        pygame.display.flip()

        if first_gen:
            timer -= 1
            if timer < 0:
                first_gen = False

    pygame.quit()
