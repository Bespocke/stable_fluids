from __future__ import annotations

import pygame
import numpy as np
from matplotlib.colors import hsv_to_rgb

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fluid_numba_buff import Fluid

class Renderer:
    def __init__(self, N: int, scale: int=5):
        self.N = N
        self.scale = scale

        pygame.init()
        self.screen = pygame.display.set_mode((N*scale, N*scale))
        self.clock = pygame.time.Clock()
        self.quit_event = pygame.QUIT

    def render_density(self, fluid: Fluid):

        density = fluid.density
        image = np.clip(density*255, 0, 255).astype(np.uint8)
        rgb = np.stack((image, image, image), axis=-1)
        rgb[fluid.solid_mask] = (250,50,50)

        # il serait intéressant de faire une transpoisition de la matrice rendu afin d'obtenir un résultat cohérent avec ce qui est fait en numpy/phyisique

        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))

        surface = pygame.transform.scale(
            surface,
            (self.N*self.scale,
             self.N*self.scale)
        )

        self.screen.blit(surface, (0, 0))

        pygame.display.flip()
    
    def render_temperature(self, fluid: Fluid):

        T = fluid.temperature

        # normalisation
        Tmin = T.min()
        Tmax = T.max()

        T = (T-Tmin)/(Tmax-Tmin+1e-8)

        # HSV
        hsv = np.zeros(
            (self.N,self.N,3)
        )

        hsv[...,0] = T        # hue
        hsv[...,1] = 1        # saturation
        hsv[...,2] = T        # luminosité

        rgb = hsv_to_rgb(hsv)

        rgb = (rgb*255).astype(np.uint8)


        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))

        surface = pygame.transform.scale(
            surface,
            (
                self.N*self.scale,
                self.N*self.scale
            )
        )

        self.screen.blit(surface,(0,0))
        pygame.display.flip()
    
    def render_pressure(self, fluid: Fluid):

        P = fluid.pressure

        P = (P-P.min())/(P.max()-P.min()+1e-8)

        image = (P*255).astype(np.uint8)

        rgb = np.stack(
            (image,image,image),
            axis=-1
        )

        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))

        surface = pygame.transform.scale(
            surface,
            (self.N*self.scale,
            self.N*self.scale)
        )

        self.screen.blit(surface,(0,0))
        pygame.display.flip()

    def render_curl(self, fluid: Fluid):

        max_curl = np.max(np.abs(fluid.curl))

        C = (fluid.curl/max_curl + 1)/2


        rgb = np.zeros(
            (self.N,self.N,3),
            dtype=np.uint8
        )

        rgb[...,0] = C*255
        rgb[...,1] = 0
        rgb[...,2] = (1-C)*255

        rgb[fluid.solid_mask] = (0,0,0)


        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))

        surface = pygame.transform.scale(
            surface,
            (self.N*self.scale,
            self.N*self.scale)
        )

        self.screen.blit(surface,(0,0))
        pygame.display.flip()

    def events(self):
        return pygame.event.get()

    def close(self):
        pygame.quit()

    def tick(self, fps: int = 60):
        self.clock.tick(fps)

