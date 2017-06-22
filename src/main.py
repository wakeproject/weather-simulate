# -*- coding: utf-8 -*-

import numpy as np
import pygame

import system

from system.planet.atmosphere import UGrd, VGrd, WGrd, TGrd, RGrd, QGrd, PRel, dHRel, dQRel
from system.planet.terrasphere import TLGrd, SIGrd
from system.planet import a

u = UGrd(361, 179, 32)
v = VGrd(361, 179, 32)
w = WGrd(361, 179, 32)
T = TGrd(361, 179, 32)
rao = RGrd(361, 179, 32)
Q = QGrd(361, 179, 32)

p = PRel(361, 179, 32)
dH = dHRel(361, 179, 32)
dQ = dQRel(361, 179, 32)

tl = TLGrd(361, 179, 32)
si = SIGrd(361, 179, 32)


def evolve():
    s = np.sqrt(u.curval * u.curval + v.curval * v.curval + w.curval * w.curval + 0.00001)
    dt = 1 / np.max(s)
    print dt
    system.t = system.t + dt

    u.evolve(dt)
    v.evolve(dt)
    w.evolve(dt)
    T.evolve(dt)
    rao.evolve(dt)
    Q.evolve(dt)

    p.evolve()
    dH.evolve()
    dQ.evolve()

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
    mask = np.isnan(array)
    array[mask] = np.average(array[~mask])
    maxv = np.max(array)
    minv = np.min(array)
    return (array - minv) / (maxv - minv + 0.001) * 255


if __name__ == '__main__':
    map_width = 361
    map_height = 179
    tile_size = 4

    pygame.init()
    screen = pygame.display.set_mode((map_width * tile_size,map_height * tile_size))
    background = pygame.Surface(screen.get_size())

    clock = pygame.time.Clock()

    first_gen = True
    timer = 12

    running = True
    lasttile = 0
    while running == True:
        clock.tick(1)
        pygame.display.set_caption('FPS: ' + str(clock.get_fps()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        evolve()

        pmap = normalize(p.curval[:, :, 0])
        umap = normalize(u.curval[:, :, 0])
        vmap = normalize(v.curval[:, :, 0])
        for ixlng in range(361):
            for ixlat in range(179):
                tval = int(pmap[ixlng, ixlat])
                uval = int(umap[ixlng, ixlat])
                vval = int(vmap[ixlng, ixlat])
                tile = pygame.Surface((4, 4))
                tile.fill((tval, uval, vval))
                screen.blit(tile, (ixlng * tile_size, ixlat * tile_size))

        flip()
        pygame.display.flip()

        if first_gen:
            timer -= 1
            if timer < 0:
                first_gen = False

    pygame.quit()
