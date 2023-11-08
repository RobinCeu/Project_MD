# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:12:32 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import numpy as np

class Grid:
    def __init__(self,Contact,Nodes: int=256):
        self.x = np.linspace(Contact.Domain[0],Contact.Domain[1],Nodes)
        self.dx = self.x[1]-self.x[0]
        self.Nx=Nodes