# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:59:11 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""


import numpy as np # Matrix Definitions and operations.
import scipy.sparse as sparse # Sparse Matrix Definitions and operations
import scipy.sparse.linalg as linalg # Sparse Matrix Linear Algebra
import matplotlib.pyplot as plt
import VisualLib as vis

class ReynoldsSolver:
    def __init__(self,Grid,Time,Ops,FluidModel,Discretization):
        self.MaxIter=[]
        self.TolP=[]
        self.UnderRelaxP=[]
        self.SetSolver()
        self.VisualFeedbackLevel=0
        
        self.Grid=Grid
        self.Time=Time
        self.Ops=Ops
        self.FluidModel=FluidModel
        self.Discretization=Discretization
        self.VisualFeedbackLevel=0
    
    def SetSolver(self,MaxIter: int=10000,TolP: float=1.0e-5 ,UnderRelaxP: float=0.001,TolT: float=1.0e-5 ,UnderRelaxT: float=0.001, VisualFeedbackLevel: int=0):
        self.MaxIter=MaxIter
        self.TolP=TolP
        self.UnderRelaxP=UnderRelaxP
        self.TolT=TolT
        self.UnderRelaxT=UnderRelaxT
        self.VisualFeedbackLevel=VisualFeedbackLevel

 
#################
##### TO DO #####
#################       
    def SolveReynolds(self,StateVector,time): # StateVector is both in and output
        
        #1. reset convergence
        epsP=np.zeros(self.MaxIter+1)
        epsP[0]=1.0
        epsT=np.zeros(self.MaxIter+1)
        epsT[0]=1.0
        
        
        #2. Predefine variables outside loop for Computational Efficiency
        DensityFunc    =self.FluidModel.Density
        ViscosityFunc  =self.FluidModel.DynamicViscosity
        SpecHeatFunc   =self.FluidModel.SpecificHeatCapacity
        ConducFunc     =self.FluidModel.ThermalConductivity      
        PreviousDensity    =self.FluidModel.Density(StateVector[time-1])
        
        DDX=self.Discretization.DDXCentral
        DDXBackward=self.Discretization.DDXBackward
        DDXForward=self.Discretization.DDXForward
        D2DX2=self.Discretization.D2DX2
        SetDirichletLeft=self.Discretization.SetDirichletLeft
        SetDirichletRight=self.Discretization.SetDirichletRight
        SetNeumannLeft=self.Discretization.SetNeumannLeft
        SetNeumannRight=self.Discretization.SetNeumannRight
        
        #define your own when desired
        P = np.zeros(self.MaxIter+1)
             '''StateVector[time] is the State at time 
             The pressure distribution where the iteration starts is 
             the pressure distribution at the State corresponding to 
             the previous time step '[time-1]'
             Which is expressed as StateVector[time-1]'''
        P[0] = StateVector[time-1].Pressure 
        #3. Iterate

        k=0
        while ((((epsP[k]>self.TolP)) or (epsT[k]>self.TolT)) and (k<self.MaxIter)):
        
     
            #0. Calc Properties

            

        
            #1. LHS Pressure
               
        
            #2. RHS Pressure
    
   
            #3. Set Boundary Conditions Pressure
           
            
            #4. Solve System for Pressure + Update

                '''result of the solver is P_star'''
                '''In Delta_P equation (68) is the calculated P_star compared to zero-values
                The dimensions of P_star and the zero array need to be the same 
                That's why the zero array is defined in function of the Grid.x array.
                In 'SolutionState.py', the pressure state is also defined with Grid.x
                This way, every array (here: P_star, State.pressure, and 0.0*self.Grid.x) 
                has the same dimensions.'''
            Delta_P = np.maximum(P_star, 0.0*self.Grid.x) - P[k]
            P[k+1] = P[k]

                '''Update Statevector variable for current time step 'time' '''
            StateVector[time].Pressure 

                '''Later on: Update Statevector variable temperature as well'''



            #5. LHS Temperature

            
            #6. RHS Temperature

            
            #7. Solve System for Temperature + Update

            
            #8. Calculate other quantities
 
            
            #9. Residuals & Report
         
           
            #10. Provide a plot of the solution
            # 9. Provide a plot of the solution
            if (k % 500 == 0):
                CFL=np.max(Uaveraged)*self.Time.dt/self.Grid.dx
                print("ReynoldsSolver:: CFL", np.round(CFL,2) ,"Residual [P,T] @Time:",round(self.Time.t[time]*1000,5),"ms & Iteration:",k,"-> [",np.round(epsP[k],6),",",np.round(epsT[k],6),"]")
                if self.VisualFeedbackLevel>2:
                    fig=vis.Report_PT(self.Grid,StateVector[time]) 
                    plt.close(fig)

                
            if (epsP[k]<=self.TolP) and (epsT[k]<=self.TolT):
                print("ReynoldsSolver:: Convergence [P,T] to the predefined tolerance @Time:",round(self.Time.t[time]*1000,5),"ms & Iteration:",k,"-> [",np.round(epsP[k],6),",",np.round(epsT[k],6),"]")
                
            if k>=self.MaxIter:
                print("ReynoldsSolver:: Residual [P,T] @Time:",round(self.Time.t[time]*1000,5),"ms & Iteration:",k,"-> [",np.round(epsP[k],6),",",np.round(epsT[k],6),"]")
                print("ReynoldsSolver:: Warning: Maximum Iterations without converging to the predefined tolerance]")


            
        #11. Calculate other quantities (e.g. Wall Shear Stress, Hydrodynamic Load, ViscousFriction)
