# -*- coding: utf-8 -*-

import numpy as np
import pygame
import time

import solarsys

from solarsys.earth.atmosphere import UGrd, VGrd, WGrd, TGrd, RGrd, QGrd, PRel, dHRel, dQRel
from solarsys.earth.terrasphere import TLGrd, SIGrd, continent, TotalCloudage


u = UGrd(solarsys.shape)
v = VGrd(solarsys.shape)
w = WGrd(solarsys.shape)
T = TGrd(solarsys.shape)
rao = RGrd(solarsys.shape)
q = QGrd(solarsys.shape)

p = PRel(solarsys.shape)
dH = dHRel(solarsys.shape)
dQ = dQRel(solarsys.shape)

tl = TLGrd(solarsys.shape)
si = SIGrd(solarsys.shape)
tc = TotalCloudage(solarsys.shape)

cntn = continent()


def evolve():
    s = np.sqrt(u.curval * u.curval + v.curval * v.curval + w.curval * w.curval + 0.00001)
    dt = 100 / np.max(s)
    if dt > 1:
        dt = 1
    solarsys.t = solarsys.t + dt
    print '----------------------------------------------------'
    print solarsys.t, dt
    print 'wind: ', np.max(s), np.min(s), np.mean(s)
    print 'temp', np.max(T.curval - 273.15), np.min(T.curval - 273.15), np.mean(T.curval - 273.15)
    print 'pres', np.max(p.curval / 101325), np.min(p.curval / 101325), np.mean(p.curval / 101325)
    print 'rao', np.max(rao.curval), np.min(rao.curval), np.mean(rao.curval)
    print 'humd', np.max(q.curval), np.min(q.curval), np.mean(q.curval)
    print 'cldg', np.max(tc.curval[:, :, 0]), np.min(tc.curval[:, :, 0]), np.mean(tc.curval[:, :, 0])

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
    tc.evolve(dt)


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
    tc.swap()


def normalize(array):
    maxv = np.max(array)
    minv = np.min(array)
    return (array - minv) / (maxv - minv + 0.001) * 255


if __name__ == '__main__':
    map_width = solarsys.shape[0]
    map_height = solarsys.shape[1]

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
        #time.sleep(1)
        pygame.display.set_caption('FPS: ' + str(clock.get_fps()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        evolve()

        bmap = si.curval[:, :, 0]
        tmap = T.curval[:, :, 0]
        cmap = tc.curval[:, :, 0]
        umap = 0.5 * u.curval[:, :, 0] + 0.5 * u.curval[:, :, 1]
        vmap = 0.5 * v.curval[:, :, 0] + 0.5 * v.curval[:, :, 1]
        wmap = 0.5 * w.curval[:, :, 0] + 0.5 * w.curval[:, :, 1]
        smap = np.sqrt(umap * umap + vmap * vmap + wmap * wmap + 0.001)
        mxs = np.max(smap)
        mns = np.min(smap)
        mms = np.mean(smap)

        bcmap = normalize(bmap)
        tcmap = normalize(tmap)
        ccmap = normalize(cmap)
        scmap = normalize(smap)
        ucmap = normalize(umap)
        vcmap = normalize(vmap)
        wcmap = normalize(wmap)
        for ixlng in range(solarsys.shape[0]):
            for ixlat in range(solarsys.shape[1]):
                tval = tmap[ixlng, ixlat]
                cval = cmap[ixlng, ixlat]
                uval = umap[ixlng, ixlat]
                vval = vmap[ixlng, ixlat]
                sval = smap[ixlng, ixlat]

                bcolor = bcmap[ixlng, ixlat]
                scolor = scmap[ixlng, ixlat]
                tcolor = tcmap[ixlng, ixlat]
                ccolor = ccmap[ixlng, ixlat]
                ucolor = ucmap[ixlng, ixlat]
                vcolor = vcmap[ixlng, ixlat]
                wcolor = wcmap[ixlng, ixlat]
                tile = pygame.Surface((tile_size, tile_size))
                r = (int(tcolor * 2 / 3) + int(72 * cntn[ixlng, ixlat, 0])) * (tval > 273.15) + (128 + int(tcolor / 2) + int(72 * cntn[ixlng, ixlat, 0])) * (tval <= 273.15)
                g = (128 + ccolor - int(72 * cntn[ixlng, ixlat, 0])) + (256 + ccolor - int(72 * cntn[ixlng, ixlat, 0])) * (tval <= 273.15)
                b = (128 + ccolor - int(72 * cntn[ixlng, ixlat, 0])) + (256 + ccolor - int(72 * cntn[ixlng, ixlat, 0])) * (tval <= 273.15)
                m = np.sqrt(r * r + g * g + b * b + 1)
                r = int(r * (255 - ccolor + 0) / m)
                g = int(g * (255 - ccolor + 0) / m)
                b = int(b * (255 - ccolor + 0) / m)
                m = np.sqrt(r * r + g * g + b * b + 1)
                r = int(r * (bcolor + 160) / m)
                g = int(g * (bcolor + 160) / m)
                b = int(b * (bcolor + 160) / m)
                r = (r > 255) * 255 + (r >= 0) * (r < 256) * r
                g = (g > 255) * 255 + (g >= 0) * (g < 256) * g
                b = (b > 255) * 255 + (b >= 0) * (b < 256) * b
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
