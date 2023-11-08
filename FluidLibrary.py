# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:14:41 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import numpy as np

class Liquid:

    def __init__(self, Fluid): # initiatlization
        
       if  Fluid=='SAE5W40':
           self.SAE5W40()
       else:
           print("Undefined Liquid")

    def SAE5W40(self):

        self.SpecificHeatCapacity=1670.0 #[J/kgK]
        self.ThermalConductivity=0.15 #
        self.SpeedOfSound=1461.0 #[m/s]
        self.SaturationPressure=100000.0 
        self.VapourPressure=1000.0
        self.EOSDensityParam=[804.5, 0.0007335, 373.15]
        self.EOSViscosityParam=[4.298e-5, 1285.0, 141.5]

    def EOS_Density(self,State):
        #Thermal Expansion law
        Density = (self.EOSDensityParam[0])/(1.0 + (self.EOSDensityParam[1])*(State.Temperature-self.EOSDensityParam[2])) #[kg/m^3]
        return Density
                
    def EOS_DynamicViscosity(self,State):
        # Exponential Correlation
        DynamicViscosity = (self.EOSViscosityParam[0])*np.exp(self.EOSViscosityParam[1]/(State.Temperature-self.EOSViscosityParam[2])) #
        return DynamicViscosity        







class Gas:

    def __init__(self, Fluid): # initiatlization
        
       if  Fluid=='SAE5W40':
           self.SAE5W40()
       else:
           print("Undefined Gas")

    def SAE5W40(self):

        self.SpecificHeatCapacity=1008.0 #[J/kgK]
        self.ThermalConductivity=0.03 #
        self.SpecificGasConstant=[287.05] # Specific Gas Constant
        self.EOSViscosityParam=[0.00001716, 2.0371021481231e+7, 3.8367e+2, 110.56]

    def EOS_Density(self,State):
        #Ideal Gas law
        Density = State.Pressure/(self.SpecificGasConstant[0]*State.Temperature) #[kg/m^3]
        return Density
    
    def EOS_SpeedOfSound(self,State):
        #Ideal Gas law
        SpeedOfSound = np.sqrt(1.4*self.SpecificGasConstant[0]*State.Temperature) #[kg/m^3]
        return SpeedOfSound
                
    def EOS_DynamicViscosity(self,State):
        # SutherLand Law
        DynamicViscosity = (self.EOSViscosityParam[0])*np.sqrt(State.Temperature*State.Temperature*State.Temperature/self.EOSViscosityParam[1])*((self.EOSViscosityParam[2])/(State.Temperature+self.EOSViscosityParam[3]))
        return DynamicViscosity      