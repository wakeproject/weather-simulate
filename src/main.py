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
Q = QGrd(system.planet.shape)

p = PRel(system.planet.shape)
dH = dHRel(system.planet.shape)
dQ = dQRel(system.planet.shape)

tl = TLGrd(system.planet.shape)
si = SIGrd(system.planet.shape)


def evolve():
    s = np.sqrt(u.curval * u.curval + v.curval * v.curval + w.curval * w.curval + 0.00001)
    dt = 100000 / np.max(s)
    if dt > 60:
        dt = 60
    system.t = system.t + dt
    print '----------------------------------------------------'
    print system.t, dt
    print 'wind: ', np.max(s), np.min(s), np.mean(s)
    print 'temp', np.max(T.curval - 273.15), np.min(T.curval - 273.15), np.mean(T.curval - 273.15)
    print 'pres', np.max(p.curval / 101325), np.min(p.curval / 101325), np.mean(p.curval / 101325)

    u.evolve(dt)
    v.evolve(dt)
    w.evolve(dt)
    T.evolve(dt)
    rao.evolve(dt)
    Q.evolve(dt)

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
    Q.swap()

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

    pygame.init()
    screen = pygame.display.set_mode((map_width * tile_size, map_height * tile_size))
    background = pygame.Surface(screen.get_size())

    clock = pygame.time.Clock()

    first_gen = True
    timer = 12

    running = True
    lasttile = 0
    while running == True:
        clock.tick(500)
        time.sleep(3)
        pygame.display.set_caption('FPS: ' + str(clock.get_fps()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        evolve()

        umap = u.curval[:, :, 0]
        vmap = v.curval[:, :, 0]
        tcmap = normalize(tl.curval[:, :, 0])
        scmap = normalize(np.sqrt(umap * umap + vmap * vmap))
        ucmap = normalize(umap)
        vcmap = normalize(vmap)
        for ixlng in range(system.planet.shape[0]):
            for ixlat in range(system.planet.shape[1]):
                uval = umap[ixlng, ixlat]
                vval = vmap[ixlng, ixlat]
                scolor = scmap[ixlng, ixlat]
                tcolor = tcmap[ixlng, ixlat]
                ucolor = ucmap[ixlng, ixlat]
                vcolor = vcmap[ixlng, ixlat]
                tile = pygame.Surface((tile_size, tile_size))
                tile.fill((int(tcolor * 2 / 3), 255 - int(tcolor * 2 / 3), 255 - int(tcolor * 2 / 3)))
                if np.absolute(uval) >= np.absolute(vval):
                    pygame.draw.aaline(tile, (int(scolor), 255 - int(ucolor), 255 - int(vcolor)), [6 - 6 * vval / uval, 0], [6 + 6 * vval / uval, 112], True)
                else:
                    pygame.draw.aaline(tile, (int(scolor), 255 - int(ucolor), 255 - int(vcolor)), [0, 6 - 6 * uval / vval], [12, 6 + 6 * uval / vval], True)

                screen.blit(tile, (ixlng * tile_size, ixlat * tile_size))

        flip()
        pygame.display.flip()

        if first_gen:
            timer -= 1
            if timer < 0:
                first_gen = False

    pygame.quit()
