# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:26:03 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import numpy as np


class CavitationModel:
    
    
    def __init__(self, Oil,Vapour): # initiatlization
        
       self.Oil=Oil
       self.Vapour=Vapour
       
       self.SaturationFraction=0.98
       self.EvaporationFraction=0.02
       self.Beta  = np.log((-1.0 + self.EvaporationFraction)*self.SaturationFraction/(self.EvaporationFraction*(-1.0 + self.SaturationFraction)))/(-2.0*Oil.VapourPressure + 2.0*Oil.SaturationPressure)
       self.P0 = (-Oil.SaturationPressure*np.log(-self.EvaporationFraction/(-1.0 + self.EvaporationFraction)) + np.log(-self.SaturationFraction/(-1.0 + self.SaturationFraction))*Oil.VapourPressure)/ \
                         np.log((-1.0 + self.EvaporationFraction)*self.SaturationFraction/(self.EvaporationFraction*(-1.0 + self.SaturationFraction)))

    def VapourVolumeFraction(self,State):
        VapourVolumeFraction= 0.5*(1.0-np.tanh(self.Beta*(State.Pressure-self.P0)))
        return VapourVolumeFraction

    def Density(self,State):
        alpha=self.VapourVolumeFraction(State)
        Density= (1.0-alpha)*self.Oil.EOS_Density(State)                   + alpha*self.Vapour.EOS_Density(State)
        return Density

    def DynamicViscosity(self,State):
        alpha=self.VapourVolumeFraction(State)
        DynamicViscosity=(1.0-alpha)*self.Oil.EOS_DynamicViscosity(State) + alpha*self.Vapour.EOS_DynamicViscosity(State)
        return DynamicViscosity
  
    def SpecificHeatCapacity(self,State):
        alpha=self.VapourVolumeFraction(State)
        SpecificHeatCapacity=(1.0-alpha)*self.Oil.SpecificHeatCapacity + alpha*self.Vapour.SpecificHeatCapacity
        return SpecificHeatCapacity

    def ThermalConductivity(self,State):
        alpha=self.VapourVolumeFraction(State)
        ThermalConductivity=(1.0-alpha)*self.Oil.ThermalConductivity   + alpha*self.Vapour.ThermalConductivity
        return ThermalConductivity

