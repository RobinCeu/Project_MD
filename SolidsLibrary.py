# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:08:13 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

class Solids:
    
    def __init__(self, Material): # initiatlization
        
       if  Material=='Grey Cast Iron':
           self.CastIron()
       elif Material=='Nitrided Stainless Steel':
           self.Steel()
       else:
           print("Undefined Material")
           self.Undefined()
 
   
    def Undefined(self):
        self.Name ='Undefined'
        self.YoungsModulus=0.0  # Pa
        self.PoissonModulus=0.0 
        self.Hardness=0.0   # Pa          
           
    def CastIron(self):
        self.Name ='Grey Cast Iron'
        self.YoungsModulus=120.0e9  # Pa
        self.PoissonModulus=0.28 
        self.Hardness=250.0e6   # Pa

        
    def Steel(self):
        self.Name ='Nitrided Stainless Steel'
        self.YoungsModulus=210.0e9  # Pa
        self.PoissonModulus=0.28 
        self.Hardness=1200.0e6   # Pa
        

