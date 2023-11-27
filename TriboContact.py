# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:18:44 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import numpy as np
import scipy.special as special # Sparse Matrix Definitions and operations
import scipy.integrate as integral # Sparse Matrix Definitions and operations
from EngineParts import Engine
from SolidsLibrary import Solids

from scipy.interpolate import interp1d
from scipy.integrate import quad

class TriboContact:
    
    def __init__(self,Engine):

        self.l_c = 2.2239
        self.Engine=Engine
    
        """ Equivalent Young's modulus of Hertzian contact"""
        self.YoungsModulus=1.0/((1.0-Engine.Cylinder.Material.PoissonModulus**2.0)/Engine.Cylinder.Material.YoungsModulus + (1.0-Engine.CompressionRing.Material.PoissonModulus**2)/Engine.CompressionRing.Material.YoungsModulus);
        self.Domain=np.array([-Engine.CompressionRing.Thickness/2,Engine.CompressionRing.Thickness/2])
        
        """ Roughness parameters """
        self.Roughness=np.sqrt(Engine.Cylinder.Roughness**2.0 + Engine.CompressionRing.Roughness**2.0)
        self.Zeta=97.0e9
        self.Kappa=1.56e-6
        self.Tau0=2.0e6
        self.f_b=0.3
        self.RoughnessParameter=self.Zeta*self.Kappa*self.Roughness
        self.RoughnessSlope=self.Roughness/self.Kappa
        
        """Wear Coefficients"""
        self.WearCoefficient_Cylinder=2.5e-10
        self.WearCoefficient_CompressionRing=1.25e-10

    def integrate_discrete_data(self, x_values, y_values):
        interp_func = interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
        def integrand(x):
            return interp_func(x)
        result, error = quad(integrand, x_values[0], x_values[-1])
        return result, error
       

    def I2(self,l): 
        I2 = (0.5*(l**2+1)*special.erfc(l/np.sqrt(2.0)) - (l/np.sqrt(2.0*np.pi))*np.exp(-l**2.0/2.0))/np.sqrt(l)
        return I2
    
    def I52(self,l):
        I52 = ((1.0/(8.0*np.sqrt(np.pi)))*np.exp(-l**2.0/4.0)*(l**(3.0/2.0))*((2.0*l**2.0+3.0)*special.kv(3.0/4.0,l**2.0/4.0)-(2.0*l**2.0+5.0)*special.kv(1.0/4.0,l**2.0/4.0)))/np.sqrt(l)
        return I52
    
    def area_integrand(self, x):
        return self.I2(x)*x**(-0.5)
    
    def load_integrand(self, x):
        return self.I52(x)*x**(-0.5)
    

#################
##### TO DO #####
#################
    def AsperityContact(self,StateVector,time):
        Lambda=StateVector[time].Lambda
        l_0 = min(StateVector[time].h0/self.Roughness,self.l_c)
        StateVector[time].AsperityArea= np.pi**2*(self.RoughnessParameter)**2*np.pi*2*self.Engine.Cylinder.Radius*np.sqrt((self.Engine.CompressionRing.Thickness**2*self.Roughness)/(4*self.Engine.CompressionRing.CrownHeight))*quad(self.area_integrand, l_0, self.l_c)[0]
        StateVector[time].AsperityLoad= 16/15*np.sqrt(2)*np.pi*(self.RoughnessParameter)**2*np.sqrt(self.Roughness/self.Kappa)*self.YoungsModulus*np.sqrt(self.Roughness*self.Engine.CompressionRing.Thickness**2/(4*self.Engine.CompressionRing.CrownHeight))*quad(self.load_integrand,l_0,self.l_c)[0]
        StateVector[time].AsperityFriction=self.Tau0*StateVector[time].AsperityArea/(np.pi*self.Engine.Cylinder.Radius*2)+self.f_b*StateVector[time].AsperityLoad
        StateVector[time].AsperityContactPressure= StateVector[time].AsperityLoad/StateVector[time].AsperityArea
        StateVector[time].HertzianContactPressure=np.pi/4*np.sqrt(StateVector[time].AsperityLoad*self.YoungsModulus/(np.pi*self.Engine.CompressionRing.CrownHeight))

        
        
#################
##### TO DO #####
#################    
   
    def Wear(self,Ops,Time,StateVector,time):
        self.Ops = Ops 

        # Calculate Wear Depth on the Piston Ring  
        # accumulated wear depth on the ring:
        k = 1
        hertzian = [0]
        while k <= time:
            hertzian.append(StateVector[k].HertzianContactPressure)
            k += 1
        
        # sliding velocity doesnt start from zero
        hertzian = self.WearCoefficient_CompressionRing*np.array(hertzian)/Solids('Nitrided Stainless Steel').Hardness
        s = Ops.SlidingDistance[:time]
        s = np.concatenate(([0],s))

        interp_func = interp1d(s, hertzian, kind='linear', fill_value='extrapolate')
        def integrand(x):
            return interp_func(x)
        result, error = quad(integrand, s[0], s[time-1])
        
        StateVector[time].WearDepthRing = result
        
      
        # Calculate The Wear Depth on the Cylinder wall
        # array of unique Positions where the pistion passes by:
        StateVector[time].WearLocationsCylinder= np.unique(np.round(Ops.PistonPosition,8)) 

        StateVector[time].WearDepthCylinder= 35 #incremental wear depth on the positions in the array above

        t = 0
        b = 10e-6 # crownheight
        integrand_l = []

        def integrand_generator(tijd):
            return self.StateVector[tijd].HertzianContactPressure*self.Ops.PistonVelocity/Solids('Grey Cast Iron').Hardness

        for i in range(len(StateVector[time].WearLocationsCylinder)):
            integrand_l.append([])

        for i in range(len(StateVector[time].WearLocationsCylinder)):
            # If ring is currently over the point of intrest: add integrand and timestamp
            if self.Ops.PistonPosition[time] < StateVector[time].WearLocationsCylinder[i] +b/2 or  self.Ops.PistonPosition[time] >StateVector[time].WearLocationsCylinder[i] - b/2:
                integrand_l[i].append([integrand_generator(time), time])
                # If in the previous timestamp the ring wasn't yet over the POI: add integrand and timestamp of when it first came in contact [add interpolation?]
                if self.Ops.PistonPosition[time-1] > StateVector[time].WearLocationsCylinder[i] +b/2 or  self.Ops.PistonPosition[time-1] < StateVector[time].WearLocationsCylinder[i] - b/2:
                    integrand_l[i].append([integrand_generator(time-1), time - abs(self.Ops.PistonPosition[time]+b/2-StateVector[time].WearLocationsCylinder[i])/self.Ops.Pistonvelocity])
            # if previously the ring was over the POI but not now: add integrand and timestamp of when contact ends
            if self.Ops.PistonPosition[time-1] < StateVector[time].WearLocationsCylinder[i] +b/2 or  self.Ops.PistonPosition[time-1] >StateVector[time].WearLocationsCylinder[i] - b/2:
                if self.Ops.PistonPosition[time] > StateVector[time].WearLocationsCylinder[i] +b/2 or  self.Ops.PistonPosition[time] < StateVector[time].WearLocationsCylinder[i] - b/2:
                    integrand_l[i].append([integrand_generator(time), time + abs(self.Ops.PistonPosition[time]+b/2-StateVector[time].WearLocationsCylinder[i])/self.Ops.Pistonvelocity])
        

        StateVector[time].WearDepthCylinder= 35 #incremental wear depth on the positions in the array above

        


 