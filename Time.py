# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:08:23 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import numpy as np

class Time:
    def __init__(self, tend: float=1.0, dt: float=0.001):
        self.dt = dt
        self.t = np.arange(0.0,tend,dt);
        self.nt = np.size(self.t)