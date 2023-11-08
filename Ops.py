# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:49:18 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import numpy as np
import scipy.integrate as integration


class Ops:
    
    def __init__(self,Time,Engine,EngineRPM,EngineAcceleration,OilTemperature):
        
        self.AtmosphericPressure=101325.0
        self.AdiabaticExpCoeff=1.35
        self.EngineSpeed=EngineRPM*(2.0*np.pi/60.0)
        self.OilTemperature=OilTemperature+273.15


        """Kinematics"""
        self.CranckAngle=self.EngineSpeed*Time.t
        theta = np.arcsin((Engine.Piston.Cranck/Engine.Piston.ConRodLength)*np.sin(self.CranckAngle))
        dthetadphi = (Engine.Piston.Cranck/Engine.Piston.ConRodLength)*np.cos(self.CranckAngle)/np.cos(theta)
        dtheta2dphi2 = (Engine.Piston.Cranck/Engine.Piston.ConRodLength)*((Engine.Piston.Cranck/Engine.Piston.ConRodLength)**2.0-1.0)*np.sin(self.CranckAngle)/np.cos(theta)**3.0
        self.PistonPosition=Engine.Piston.Cranck*np.cos(self.CranckAngle)+Engine.Piston.ConRodLength*np.cos(theta)
        self.PistonVelocity=-self.EngineSpeed*(Engine.Piston.Cranck*np.sin(self.CranckAngle) + Engine.Piston.ConRodLength*np.sin(theta)*dthetadphi)
        self.PistonAcceleration=-Engine.Piston.ConRodLength*(dtheta2dphi2*self.EngineSpeed**2.0 + dthetadphi* EngineAcceleration)*np.sin(theta) \
                               -Engine.Piston.ConRodLength * dthetadphi**2.0 * self.EngineSpeed**2.0 * np.cos(theta) \
                               -Engine.Piston.Cranck*(self.EngineSpeed**2.0 * np.cos(self.CranckAngle)+np.sin(self.CranckAngle)* EngineAcceleration)
        self.ChamberVolume=Engine.MinimumVolume + (np.pi*Engine.Cylinder.Radius**2.0)*(Engine.Piston.Cranck + Engine.Piston.ConRodLength - self.PistonPosition)
        self.SlidingDistance=integration.cumtrapz(np.sqrt(1.0+(self.PistonVelocity)**2.0),Time.t)
        self.SlidingVelocity=-self.PistonVelocity

        
        """Different Stages of 1 Cycle"""
        self.Intake=np.flatnonzero(self.CranckAngle < Engine.ValveTiming.IVC)
        self.AdiabaticCompression=np.flatnonzero(np.logical_and(self.CranckAngle >= Engine.ValveTiming.IVC, self.CranckAngle < Engine.ValveTiming.SOC))
        self.Combustion=np.flatnonzero(np.logical_and(self.CranckAngle >= Engine.ValveTiming.SOC , self.CranckAngle < Engine.ValveTiming.EOC))
        self.AdiabaticExpansion=np.flatnonzero(np.logical_and(self.CranckAngle >= Engine.ValveTiming.EOC , self.CranckAngle < Engine.ValveTiming.EVO))
        self.IsochoricExpansion=np.flatnonzero(np.logical_and(self.CranckAngle >= Engine.ValveTiming.EVO , self.CranckAngle < 3*np.pi))
        self.BlowOut=np.flatnonzero(np.logical_and(self.CranckAngle >= 3*np.pi , self.CranckAngle <= 4*np.pi))
        
        """Combustion & Cylinder Pressure Evolution"""
        self.CombustionPressure=self.AtmosphericPressure*(Engine.MaximumVolume/Engine.MinimumVolume)**self.AdiabaticExpCoeff
        self.CylinderPressure=0.0*Time.t #initialize
        self.CylinderPressure[self.Intake]=self.AtmosphericPressure
        self.CylinderPressure[self.AdiabaticCompression]=self.AtmosphericPressure*(Engine.MaximumVolume/self.ChamberVolume[self.AdiabaticCompression])**self.AdiabaticExpCoeff
        self.CylinderPressure[self.Combustion]=self.CombustionPressure
        self.CylinderPressure[self.AdiabaticExpansion]=self.CombustionPressure*(self.ChamberVolume[self.AdiabaticExpansion[0]]/self.ChamberVolume[self.AdiabaticExpansion])**self.AdiabaticExpCoeff
        self.IsochoricExpCoeff=np.log(self.CylinderPressure[self.AdiabaticExpansion[-1]]/self.AtmosphericPressure)/np.log(Engine.MaximumVolume/self.ChamberVolume[self.AdiabaticExpansion[-1]])
        self.CylinderPressure[self.IsochoricExpansion]=self.AtmosphericPressure*(Engine.MaximumVolume/self.ChamberVolume[self.IsochoricExpansion])**self.IsochoricExpCoeff
        self.CylinderPressure[self.BlowOut]=self.AtmosphericPressure
        self.CompressionRingLoad=((self.CylinderPressure-self.AtmosphericPressure))*Engine.CompressionRing.Thickness + (Engine.CompressionRing.FreeGapSize*Engine.CompressionRing.Material.YoungsModulus*Engine.CompressionRing.Thickness*Engine.CompressionRing.Width**3.0)/(36.0*np.pi*Engine.CompressionRing.Thickness*Engine.Cylinder.Radius**4.0)*Engine.CompressionRing.Thickness #%[N/m]
        
