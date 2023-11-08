# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:39:07 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""


class State:
    def __init__(self,Grid):
        
        #Single-values
        self.h0=1e-6
        self.Hersey=0.0
        self.Lambda=1.0
        self.HydrodynamicLoad=0.0
        self.AsperityLoad=0.0
        self.ViscousFriction=0.0
        self.AsperityFriction=0.0
        self.AsperityContactArea=0.0
        self.AsperityContactPressure=0.0
        self.HertzianContactPressure=0.0
        self.COF=0.0
        self.WearDepthRing=0.0
        
        #Scalar Fields
        self.h=0.0*Grid.x
        self.Pressure=0.0*Grid.x+101325.0
        self.Temperature=0.0*Grid.x + 300.0
        self.WallShearStress=0.0*Grid.x
        self.WearLocationsCylinder=[]
        self.WearDepthCylinder=[]
        
        
    