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
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve

import copy as copy
from scipy.interpolate import interp1d
from scipy.integrate import quad

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
        DensityFunc    =self.FluidModel.Density(StateVector[time])
        ViscosityFunc  =self.FluidModel.DynamicViscosity(StateVector[time])
        SpecHeatFunc   =self.FluidModel.SpecificHeatCapacity(StateVector[time])
        ConducFunc     =self.FluidModel.ThermalConductivity(StateVector[time])  
        PreviousDensity    =self.FluidModel.Density(StateVector[time-1]) #" state on time - 1"
        
        " calls FiniteDifferences class "
        DDX=self.Discretization.DDXCentral
        DDXBackward=self.Discretization.DDXBackward
        DDXForward=self.Discretization.DDXForward
        D2DX2=self.Discretization.D2DX2Central
        SetDirichletLeft=self.Discretization.SetDirichletLeft
        SetDirichletRight=self.Discretization.SetDirichletRight
        SetNeumannLeft=self.Discretization.SetNeumannLeft
        SetNeumannRight=self.Discretization.SetNeumannRight
        
        #define your own when desired
        Identity=sparse.identity(self.Grid.Nx, dtype='float', format="csr")
        Phi=sparse.identity(self.Grid.Nx, dtype='float', format="csr")
        DPhiDX=sparse.identity(self.Grid.Nx, dtype='float', format="csr")
        I = self.Discretization.Identity
        D1=sparse.identity(self.Grid.Nx, dtype='float', format="csr")
        D2=sparse.identity(self.Grid.Nx, dtype='float', format="csr")
        E=sparse.identity(self.Grid.Nx, dtype='float', format="csr")
        #3. Iterate

        k=0
        while ((((epsP[k]>self.TolP)) or (epsT[k]>self.TolT)) and (k<self.MaxIter)):
        
            " these will be the same values as the predefined values for the first iteration "
            " when not in the first iteration, the values will be calculated based on the newly found state (press, temp) and calculated using the " 
            " cavitationmodel class in TwoPhaseModel.py "
            #0. Calc Properties
            DensityFunc    =self.FluidModel.Density(StateVector[time])
            ViscosityFunc  =self.FluidModel.DynamicViscosity(StateVector[time])
            SpecHeatFunc   =self.FluidModel.SpecificHeatCapacity(StateVector[time])
            ConducFunc     =self.FluidModel.ThermalConductivity(StateVector[time])

            p_carter = self.Ops.AtmosphericPressure
        
            " set M = A+B using the discretizing schemes (see appendix assignment) "
            #1. LHS Pressure
            " find A "
            " phi based on formula (16) "
            CurState = StateVector[time]
            h3 = np.power(CurState.h,3) 
            " state corresponding to t (and iteration k) --> height (array) "
            Densh3 = DensityFunc*h3
            phi = Densh3/ViscosityFunc/12

            " phi as sparse matrix "
            #Phi = sparse.diags(phi)
            #A = Phi*D2DX2
            # print("phi", phi)
            Phi.data=phi
            # print("Phi", Phi)
            A = Phi @ D2DX2
            # print("A",A)
             
            " find B "
            " use central diff to find derivatives of phi in each node "
            #d_phi = DDX*phi
            #D_Phi = sparse.diags(d_phi)
            dphidx=DDX @ phi
            DPhiDX.data=dphidx
            #B = D_Phi*DDX
            B=DPhiDX @ DDX
            #print(B)

            " find M "
            M = A+B
            
            #2. RHS Pressure
            U = self.Ops.SlidingVelocity[time]
            h_Density = CurState.h*DensityFunc

            " squeeze term "
            PrevState = copy.deepcopy(StateVector[time-1])
            dt = self.Time.dt
            h_Density_Diff = (h_Density  - PrevState.h*PreviousDensity)/dt

            RHS = (U/2)*DDX @  h_Density + h_Density_Diff

            #3. Set Boundary Conditions Pressure
            " Dirichlet on M as coded in FiniteDifferences "
            " M is new M with left (right) Dirichlet? "
            #print(M)
            #M = sparse.csr_matrix(M)
            #print(M)
            SetDirichletLeft(M)
            SetDirichletRight(M)
            #print(M)
            " Boundary conditions on RHS "
            RHS[0] = self.Ops.AtmosphericPressure
            RHS[-1] = self.Ops.CylinderPressure[time]

            #4. Solve System for Pressure + Update
            P_sol = spsolve(M,RHS)
            P_old = CurState.Pressure
            delta_P = np.maximum(P_sol,0)-P_old
            StateVector[time].Pressure = P_old + self.UnderRelaxP*delta_P
            

            "Create LHS = I+E+D"
            #5. LHS Temperature
            " set I "
            " set D "
            " use central diff to find derivatives of pressure in each node "
            d_pressure = DDX @ StateVector[time].Pressure

            " substitute this in the average velocity equation (22) "
            u = -(CurState.h**2)*d_pressure/ViscosityFunc/12+U/2 # correct????
            u_plus = np.maximum(u,0)
            u_min = np.minimum(u,0)
            D1.data = (u_plus*dt)
            D2.data = (u_min*dt)
            D = D1 @ DDXBackward + D2 @ DDXForward

            " set E "
            E.data = (ConducFunc*dt/DensityFunc/SpecHeatFunc)

            M2 = I+D - E @ D2DX2

            #6. RHS Temperature
            T_old = PrevState.Temperature
         
            " substitute the shear heating equation (23) "
            Q = (CurState.h**2)*d_pressure**2/ViscosityFunc/12 + (ViscosityFunc*U**2)/(CurState.h**2)

            RHS_T = T_old + Q*dt/DensityFunc/SpecHeatFunc

            #7. Set Boundary Conditions Temperature
            if U <= 0:
                " left Neumann, right Dirichlet "
                SetDirichletRight(M2)
                SetNeumannLeft(M2)
                " boundary conditions on RHS "
                RHS_T[0] = 0.0
                RHS_T[-1] = self.Ops.OilTemperature
            else:
                " right Neumann, left Dirichlet "
                SetDirichletLeft(M2)
                SetNeumannRight(M2)
                " boundary conditions on RHS "
                RHS_T[0] = self.Ops.OilTemperature
                RHS_T[-1] = 0.0

            #8. Solve System for Temperature + Update
            T_sol = spsolve(M2,RHS_T)
            T_sol = np.maximum(np.minimum(T_sol,2*self.Ops.OilTemperature),self.Ops.OilTemperature)
            delta_T = T_sol-T_old
            StateVector[time].Temperature += self.UnderRelaxT*delta_T
            
            
            k += 1


            " update statevector? "
            #9. Calculate other quantities
            
            
            
            #10. Residuals & Report
            epsP[k] = np.linalg.norm(delta_P/StateVector[time].Pressure)/self.Grid.Nx
            epsT[k] = np.linalg.norm(delta_T/StateVector[time].Temperature)/self.Grid.Nx
           
            #11. Provide a plot of the solution
            # 10. Provide a plot of the solution

            Uaveraged = np.mean(u)
             
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


            
        #12. Calculate other quantities (e.g. Wall Shear Stress, Hydrodynamic Load, ViscousFriction)
        
        # Hydronamic load
        x_values = self.Grid.x
        y_values = StateVector[time].Pressure
        StateVector[time].HydrodynamicLoad = np.trapz(y_values,x_values)

        # Wall shear stress (Course 9.24)
        Poisseuille = -(StateVector[time].h/2)*(DDX @ StateVector[time].Pressure)
        Couette = self.FluidModel.DynamicViscosity(StateVector[time])*self.Ops.SlidingVelocity[time]/StateVector[time].h
       
        StateVector[time].WallShearStress = Poisseuille + Couette

        # ViscousFriction
        x_values = self.Grid.x
        y_values = StateVector[time].WallShearStress
        StateVector[time].ViscousFriction = np.trapz(y_values,x_values)
    