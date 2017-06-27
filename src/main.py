# -*- coding: utf-8 -*-

import numpy as np
import pygame
import time

import system

from system.planet.atmosphere import UGrd, VGrd, WGrd, TGrd, RGrd, QGrd, PRel, dHRel, dQRel
from system.planet.terrasphere import TLGrd, SIGrd


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


def evolve():
    s = np.sqrt(u.curval * u.curval + v.curval * v.curval + w.curval * w.curval + 0.00001)
    dt = 100000 / np.max(s)
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

    tile_size = 12
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
        time.sleep(2)
        pygame.display.set_caption('FPS: ' + str(clock.get_fps()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        evolve()

        umap = u.curval[:, :, 0]
        vmap = v.curval[:, :, 0]
        wmap = w.curval[:, :, 0]
        smap = np.sqrt(umap * umap + vmap * vmap + wmap * wmap + 0.001)
        mxs = np.max(smap)

        tcmap = normalize(T.curval[:, :, 0])
        scmap = normalize(smap)
        ucmap = normalize(umap)
        vcmap = normalize(vmap)
        wcmap = normalize(wmap)
        for ixlng in range(system.planet.shape[0]):
            for ixlat in range(system.planet.shape[1]):
                uval = umap[ixlng, ixlat]
                vval = vmap[ixlng, ixlat]
                sval = smap[ixlng, ixlat]

                scolor = scmap[ixlng, ixlat]
                tcolor = tcmap[ixlng, ixlat]
                ucolor = ucmap[ixlng, ixlat]
                vcolor = vcmap[ixlng, ixlat]
                wcolor = wcmap[ixlng, ixlat]
                tile = pygame.Surface((tile_size, tile_size))
                tile.fill((int(tcolor * 2 / 3), 255 - int(tcolor * 2 / 3), 255 - int(tcolor * 2 / 3)))
                tile.set_alpha(64)

                if ixlng % 3 == 0 and ixlat % 3 == 0:
                    length = int(arrrow_size * sval / mxs)
                    if np.absolute(uval) >= np.absolute(vval):
                        pygame.draw.aaline(tile, (int(wcolor), int(ucolor), int(vcolor)), [arrrow_size / 2 - length * vval / uval, 0], [arrrow_size / 2 + length * vval / uval, arrrow_size], True)
                    else:
                        pygame.draw.aaline(tile, (int(wcolor), int(ucolor), int(vcolor)), [0, arrrow_size / 2 - length * uval / vval], [arrrow_size, arrrow_size / 2 + length * uval / vval], True)

                screen.blit(tile, (ixlng * tile_size, ixlat * tile_size))

        flip()
        pygame.display.flip()

        if first_gen:
            timer -= 1
            if timer < 0:
                first_gen = False

    pygame.quit()
