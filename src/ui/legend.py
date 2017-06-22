# -*- coding: utf-8 -*-

import pygame

colors16 = {i: pygame.Surface((2, 2)) for i in range(16)}

colors16[0].fill((255, 255, 255))
colors16[1].fill((0, 0, 246))
colors16[2].fill((1, 160, 246))
colors16[3].fill((0, 236, 236))
colors16[4].fill((1, 255, 0))
colors16[5].fill((0, 200, 0))
colors16[6].fill((1, 144, 0))
colors16[7].fill((255, 255, 0))
colors16[8].fill((231, 192, 0))
colors16[9].fill((255, 144, 0))
colors16[10].fill((255, 0, 0))
colors16[11].fill((214, 0, 0))
colors16[12].fill((192, 0, 0))
colors16[13].fill((255, 0, 240))
colors16[14].fill((120, 0, 132))
colors16[15].fill((172, 144, 240))

colors = {i: pygame.Surface((2, 2)) for i in range(256)}
for k, v in colors.iteritems():
    v.fill((k, k, 255 - k))

