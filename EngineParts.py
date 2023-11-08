# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:45:22 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

from SolidsLibrary import Solids
import numpy as np


class Engine:
    
    def __init__(self, EngineType): # initiatlization
        self.EngineType=EngineType
        
        if EngineType=='VW 2.0 R4 16v TDI CR 103kW':
           self.Cylinder=Cylinder(EngineType)
           self.Piston=Piston(EngineType)
           self.CompressionRing=CompressionRing(EngineType)
           self.ValveTiming=ValveTiming(EngineType)
           self.CompressionRatio=18.0  # m
           self.MinimumVolume=np.pi *self.Cylinder.Radius**2 * self.Piston.Stroke/(self.CompressionRatio-1) # m^3
           self.Deckh0=self.MinimumVolume/(np.pi * self.Cylinder.Radius**2)
           self.MaximumVolume=self.MinimumVolume*self.CompressionRatio # m^3

        else:
           print("Undefined Engine")
           self.Cylinder=[]
           self.Piston=[]
           self.CompressionRing=[]
           self.ValveTiming=[]
           self.CompressionRatio=[]
           self.MinimumVolume=[]
           self.Deckh0=[]
           self.MaximumVolume=[]

class Cylinder:

  
    def __init__(self, EngineType): # initiatlization
            if EngineType=='VW 2.0 R4 16v TDI CR 103kW':
                self.Radius=(81.0/2.0)/1000.0  # m
                self.Roughness=0.2e-6
                self.Material=Solids("Grey Cast Iron")
            else:
                self.Radius= []
                self.Roughness= []
                self.Material= []
    
   
    
class Piston:
    
  
    def __init__(self, EngineType): # initiatlization
            if EngineType=='VW 2.0 R4 16v TDI CR 103kW':
                self.Radius=(81.0/2.0)/1000.0  # m
                self.Stroke=95.5/1000.0
                self.Cranck=self.Stroke/2.0
                self.Height=0.12
                self.ConRodLength=self.Cranck*2.0
                self.Material=Solids("Grey Cast Iron")
            else:
                self.Radius=[]
                self.Stroke=[]
                self.Cranck=[]
                self.Height=[]
                self.ConRodLength=[]

                


   
class CompressionRing:

  
    def __init__(self, EngineType): # initiatlization
            if EngineType=='VW 2.0 R4 16v TDI CR 103kW':
                self.Thickness=0.0015#m
                self.Width=0.0035 #m
                self.GapSize=0.0005 #m
                self.FreeGapSize=0.012 #m
                self.CrownHeight=10e-6  #Defines barrel shape
                self.Roughness=0.1e-6 #m 
                self.Curvature=((self.Thickness/2.0)**2.0)/(2*self.CrownHeight);
                self.Material=Solids("Nitrided Stainless Steel")
            else:
                self.Thickness=[]
                self.Width=[]
                self.GapSize=[]
                self.FreeGapSize=[]
                self.CrownHeight=[]
                self.Roughness=[]
                self.Curvature=[]
                self.Material=[]
                
                
                

class ValveTiming:

  
    def __init__(self, EngineType): # initiatlization
            if EngineType=='VW 2.0 R4 16v TDI CR 103kW':
                self.IVC=np.pi  # intake valve closing (IVC)
                self.SOC=2.0*np.pi  # start of combustion (SOC)
                self.EOC=2.0*np.pi+np.pi/4.0  # end of combustion (EOC)
                self.EVO=3.0*np.pi-np.pi/8.0  # exhaust valve opening (EVO)
            else:
                self.IVC=[]
                self.SOC=[]
                self.EOC=[]
                self.EVO=[]