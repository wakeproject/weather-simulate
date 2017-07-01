# -*- coding: utf-8 -*-

import numpy as np
import pygame
import time

import solarsys

from solarsys.earth.atmosphere import UGrd, VGrd, WGrd, TGrd, RGrd, QGrd, PRel, dHRel, dQRel
from solarsys.earth.terrasphere import TLGrd, SIGrd, continent, TotalCloudage, SunConst


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
    tmp = 0.5 * s[:, :, 0] + 0.5 * s[:, :, 1]
    print 'wind: ', np.max(tmp), np.min(tmp), np.mean(tmp)
    tmp = T.curval[:, :, 0] - 273.15
    print 'temp', np.max(tmp), np.min(tmp), np.mean(tmp)
    tmp = p.curval[:, :, 0] / 101325
    print 'pres', np.max(tmp), np.min(tmp), np.mean(tmp)
    tmp = rao.curval[:, :, 0]
    print 'rao', np.max(tmp), np.min(tmp), np.mean(tmp)
    tmp = q.curval[:, :, 0]
    print 'humd', np.max(tmp), np.min(tmp), np.mean(tmp)
    tmp = tc.curval[:, :, 0]
    print 'cldg', np.max(tmp), np.min(tmp), np.mean(tmp)

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


def normalize(array, minv, maxv):
    val = (array - minv) / (maxv - minv + 0.001) * 255
    return val * (val > 0) * (val < 256)


if __name__ == '__main__':
    map_width = solarsys.shape[0]
    map_height = solarsys.shape[1]

    tile_size = 6
    gap = int(12 / solarsys.dlng)
    wind_size = tile_size * gap

    pygame.init()
    screen = pygame.display.set_mode((map_width * tile_size, map_height * tile_size))
    background = pygame.Surface(screen.get_size())
    tilep = pygame.Surface((tile_size, tile_size))
    tilew = pygame.Surface((wind_size, wind_size))
    tilew.set_alpha(128)

    clock = pygame.time.Clock()

    first_gen = True
    timer = 12

    running = True
    lasttile = 0
    while running == True:
        clock.tick(5)
        time.sleep(1)
        pygame.display.set_caption('FPS: ' + str(clock.get_fps()))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        evolve()

        mapc = cntn[:, :, 0]
        bmap = si.curval[:, :, 0]
        tmap = T.curval[:, :, 0]
        cmap = tc.curval[:, :, 0]
        umap = 0.5 * u.curval[:, :, 0] + 0.5 * u.curval[:, :, 1]
        vmap = 0.5 * v.curval[:, :, 0] + 0.5 * v.curval[:, :, 1]
        wmap = 0.5 * w.curval[:, :, 0] + 0.5 * w.curval[:, :, 1]
        smap = np.sqrt(umap * umap + vmap * vmap + 0.001)

        bcmap = normalize(bmap, 0, np.max(bmap))
        tcmap = normalize(tmap, 200, 374)
        ccmap = normalize(cmap, 0, 1)
        scmap = normalize(smap, 0, np.max(smap))
        ucmap = normalize(umap, 0, np.max(umap))
        vcmap = normalize(vmap, 0, np.max(vmap))
        wcmap = normalize(wmap, 0, np.max(wmap))

        r = (tcmap * 2 / 3 + 72 * mapc) * (tmap > 273.15) + (128 + tcmap / 2 + 72 * mapc) * (tmap <= 273.15)
        g = (128 + ccmap - 72 * mapc) * (tmap > 273.15) + (256 + ccmap - 72 * mapc) * (tmap <= 273.15)
        b = (128 + ccmap - 72 * mapc) * (tmap > 273.15) + (256 + ccmap - 72 * mapc) * (tmap <= 273.15)
        bcmap = bcmap + 200
        r = r * bcmap / (255 + 200)
        g = g * bcmap / (255 + 200)
        b = b * bcmap / (255 + 200)

        for ixlng in range(solarsys.shape[0]):
            for ixlat in range(solarsys.shape[1]):
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

                rval = r[ixlng, ixlat]
                gval = g[ixlng, ixlat]
                bval = b[ixlng, ixlat]
                rval = rval * (rval > 0) * (rval < 256) + 255 * (rval > 255)
                gval = gval * (rval > 0) * (gval < 256) + 255 * (gval > 255)
                bval = bval * (rval > 0) * (bval < 256) + 255 * (bval > 255)

                try:
                    tilep.fill((rval, gval, bval))
                except:
                    print rval, gval, bval
                tilep.set_alpha((255 - ccolor) / 2)
                screen.blit(tilep, (ixlng * tile_size, ixlat * tile_size))

                if ixlng % gap == 0 and ixlat % gap == 0:
                    length = wind_size / 2 * scolor / 256.0
                    tilew.fill((255, 255, 255))
                    size = length
                    if np.absolute(uval) >= np.absolute(vval):
                        alpha = np.arctan2(vval, uval)
                        pygame.draw.aaline(tilew, (wcolor, ucolor, vcolor), [wind_size / 2.0 - size * np.cos(alpha), wind_size / 2.0 - size * np.sin(alpha)],
                                                                            [wind_size / 2.0 + size * np.cos(alpha), wind_size / 2.0 + size * np.sin(alpha)], True)
                    else:
                        alpha = np.arctan2(uval, vval)
                        pygame.draw.aaline(tilew, (wcolor, ucolor, vcolor), [wind_size / 2.0 - size * np.sin(alpha), wind_size / 2.0 - size * np.cos(alpha)],
                                                                            [wind_size / 2.0 + size * np.sin(alpha), wind_size / 2.0 + size * np.cos(alpha)], True)

                    screen.blit(tilew, (ixlng * tile_size, ixlat * tile_size))

        flip()
        pygame.display.flip()

        if first_gen:
            timer -= 1
            if timer < 0:
                first_gen = False

    pygame.quit()
