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
        D2DX2=self.Discretization.D2DX2Central
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

        ''' The sliding velocity of the cylinder liner, U,
        needs to be defined for assembling the RHS-matrix 
        see equation (67).
        Above equation (19), the sliding velocity of the 
        cyslinder liner U is defined as 
        U = -y_dot
        Where y_dot = Ops.PistonVelocity'''
        y_dot =self.Ops.PistonVelocity

        '''The cylinder pressure pcc(related to cranckangle) is defined as Ops.CylinderPressure in the Ops file.'''
        pcc = self.Ops.CylinderPressure
        '''The atmospheric pressure is also defined in the Ops file '''
        pa = self.Ops.AtmosphericPressure
        '''The absolute carter pressure is equal to the atmospheric pressure.'''
        pcarter = pa

        '''dt is the squeeze term is defined in the Time class as self.dt'''
        dt = self.Time.dt 

        T = np.zeros(self.MaxIter+1)
        '''StateVector[time] is the State at time 
        The temperature distribution where the iteration starts is 
        the temperature distribution at the State corresponding to 
        the previous time step '[time-1]'
        Which is expressed as StateVector[time-1]'''
        T[0] = StateVector[time-1].Temperature

        '''The oil temperature is defined in the Ops file'''
        T_oil = self.Ops.OilTemperature

        #3. Iterate

        k=0
        while ((((epsP[k]>self.TolP)) or (epsT[k]>self.TolT)) and (k<self.MaxIter)):
        
     
            #0. Calc Properties
            
            ''' Each element in the vector DensityFunc(StateVector[time])
            is the density at time=time at the x-position of the element.
            example: The density at x1 at time=time: DensityFunc(StateVector[time])[0]
            example 2: The density at x5 at time=time: DensityFunc(StateVector[time])[4]'''


            ''' Setup the phi-array (phi as defined on page 4, equation 16).
            This is done by iterating over the length of the Grid.x-array,
            and taking the density, film thickness and viscosity for that 
            point in space (x1, x2, ...) '''
            phi = np.zeros(len(self.Grid.x))
            for i in range(len(self.Grid.x)):  
                phi[i] = (DensityFunc(StateVector[time])[i] * StateVector[time].h[i])/(12 * ViscosityFunc(StateVector[time])[i])
            
            
            #1. LHS Pressure

            A_mat1 = sparse.diags(phi, [0], shape=(self.Grid.Nx, self.Grid.Nx), dtype='float', format="csr")
            B_mat1 = np.matmul(DDX, phi)

            A = np.matmul(A_mat1, D2DX2)
            B = np.matmul(B_mat1, DDX)
            M = A + B 
        
            #2. RHS Pressure

            '''sliding velocity of the cylinder liner is U'''
            U = -y_dot

            ''' The array mat_rho_h is the second matrix in the RHS equation'''
            mat_rho_h = np.zeros(len(self.Grid.x))
            for i in range(len(self.Grid.x)):  
                mat_rho_h[i] = (DensityFunc(StateVector[time])[i] * StateVector[time].h[i])

            '''RHS_term2 = The squeeze term. Setup squeeze term array.'''
            squeeze_term = np.zeros(len(self.Grid.x))
            for i in range(len(self.Grid.x)): 
                if i == 0:
                    squeeze_term[i] = (DensityFunc(StateVector[time])[1]*StateVector[time].h[i] - DensityFunc(StateVector[time-1])[1]*StateVector[time-1].h[i])/dt
                else:
                    squeeze_term[i] = (DensityFunc(StateVector[time])[i]*StateVector[time].h[i] - DensityFunc(StateVector[time-1])[i]*StateVector[time-1].h[i])/dt
                    
            RHS = U/2 * np.matmul(DDX, mat_rho_h) + squeeze_term

            #3. Set Boundary Conditions Pressure
           
            '''Boundary conditions should be imposed on matrix M and RHS (page 17)'''
            M = SetDirichletLeft(M)
            M = SetDirichletRight(M)
            RHS[0] = pcarter - pa
            RHS[self.Grid.Nx-1] = pcc - pa

            #4. Solve System for Pressure + Update

            '''result of the solver is P_star.
            scipy.sparse.linalg.spsolve() solves the linear system  at a single iteration.'''
            '''[M][p] = [RHS]'''
            P_star = sparse.linalg.spsolve(M, RHS)


            '''In Delta_P equation (68) is the calculated P_star compared to zero-values
            The dimensions of P_star and the zero array need to be the same 
            That's why the zero array is defined in function of the Grid.x array.
            In 'SolutionState.py', the pressure state is also defined with Grid.x
            This way, every array (here: P_star, State.pressure, and 0.0*self.Grid.x) 
            has the same dimensions.'''
            Delta_P = np.maximum(P_star, 0.0*self.Grid.x) - P[k]

            '''self.UnderRelaxP is theta_P in the assignment 
            (see page 12 where it is stated that theta_P 
            is the 'under-relaxation-factor for pressure')'''
            P[k+1] = P[k] + self.UnderRelaxP * Delta_P 

            '''Update Statevector variable for pressure for current time step 'time' '''
            StateVector[time].Pressure = P[k+1]


            #5. LHS Temperature

            '''the film-thickness averaged velocity u_avg (equation (22)) needs to be calculated for every x: 
            u_avg(x) need to be known to setup matrix D (page 19).'''

            '''In this equation (22), dp/dx_i is used, the just calculated p = P[k+1] is used.'''
            dp_dx = np.matmul(DDX, P[k+1])

            u_avg_x = np.zeros(len(self.Grid.x))
            for i in range(len(self.Grid.x)):  
                u_avg_x[i] = -(((StateVector[time].h[i])**2)/(12 * ViscosityFunc(StateVector[time])[i]))*dp_dx[i] + U/2

            u_avg_x_plus = np.maximum(u_avg_x, 0.0*self.Grid.x)
            u_avg_x_min = np.minimum(u_avg_x, 0.0*self.Grid.x)

            D_mat1_array = u_avg_x_plus*dt
            D_mat3_array = u_avg_x_min*dt
            D_mat1 = sparse.diags(D_mat1_array, [0], shape=(self.Grid.Nx, self.Grid.Nx), dtype='float', format="csr")
            D_mat3 = sparse.diags(D_mat3_array, [0], shape=(self.Grid.Nx, self.Grid.Nx), dtype='float', format="csr")

            D_term1 = np.matmul(D_mat1, DDXBackward)
            D_term2 = np.matmul(D_mat3, DDXForward)

            D = D_term1 + D_term2

            I = sparse.diags(np.ones(len(self.Grid.x)), [0], shape=(self.Grid.Nx, self.Grid.Nx), dtype='float', format="csr")
            
            E_mat1_array = np.zeros(len(self.Grid.x)) 
            for i in range(len(self.Grid.x)):  
                E_mat1_array[i] = dt*(ConducFunc(StateVector[time])[i])/(DensityFunc(StateVector[time])[i]*SpecHeatFunc(StateVector[time])[i])
            
            E_mat1 = sparse.diags(E_mat1_array, [0], shape=(self.Grid.Nx, self.Grid.Nx), dtype='float', format="csr")
            E = -1 * np.matmul(E_mat1, D2DX2)

            M = I + D + E

            '''The boundary conditions, as stated in eq (71) and (72), need to be applied on matrix M. 
            First, an if/else statement checks if U is < or > than 0. 
            The Dirichlet BC applies to the non-derived function
            The Neumann BC applies to the first derivative.
            'Left' applies to [0,0]
            'Right' applies to [Nx-1, Nx-1].'''

            if U <= 0:
                SetNeumannLeft(M)
                SetDirichletRight(M)

            else:
                SetDirichletLeft(M)
                SetNeumannRight(M)

            #6. RHS Temperature

            RHS_term1 = StateVector[time-1].Temperature
            
            Q_avg = np.zeros(len(self.Grid.x))
            for i in range(len(self.Grid.x)):  
                Q_avg[i] = (((StateVector[time].h[i])**2)/(12 * ViscosityFunc(StateVector[time])[i]))*(dp_dx[i])**2 + ViscosityFunc(StateVector[time])[i] * (U**2)/((StateVector[time].h[i])**2)

            RHS_term2 = np.zeros(len(self.Grid.x))
            for i in range(len(self.Grid.x)):  
                RHS_term2[i] = (dt*Q_avg[i])/((DensityFunc(StateVector[time])[i])*(SpecHeatFunc(StateVector[time])[i]))

            RHS_T = RHS_term1 + RHS_term2

            '''The boundary conditions are manually inputted in the RHS_T marix.'''

            if U <= 0:
                RHS_T[0] = 0
                RHS_T[self.Grid.Nx-1] = T_oil

            else:
                RHS_T[0] = T_oil
                RHS_T[self.Grid.Nx-1] = 0

            #7. Solve System for Temperature + Update

            T_star = sparse.linalg.spsolve(M, RHS_T)
            Delta_T = T_star - T[k]
            T[k+1] = T[k] + self.UnderRelaxT * Delta_T 
            
            '''Update Statevector variable for temperature for current time step 'time' '''
            StateVector[time].Temperature = T[k+1]
            
            #8. Calculate other quantities
 
            '''Uaveraged for plot.'''
            '''Uaveraged='''

            '''Update value for k'''
            k += 1

            #9. Residuals & Report

            '''Calculate the residuals with the updated values for pressure and temperature.'''
            '''Keep in mind that th value for k is already updated (thus P[k] is the herabove calculated P[k+1])'''

            epsP[k] = np.linalg.norm(np.divide(Delta_P, P[k]))/len(self.Grid.x)
            epsT[k] = np.linalg.norm(np.divide(Delta_T, T[k]))/len(self.Grid.x)
                                                             
            #10. Provide a plot of the solution
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

        '''Update state quantities'''

        '''Hydrodynamic force per unit length is calculated as in equation (37).'''
        '''StateVector[time].HydrodynamicLoad = '''

        '''From slideshow 3.2, slide 25: Shear stress @ wall'''
        '''u1 = cylinder wall velocity (=U), u2 = 0 (relative to the cylinder liner).
        The shear stress @ wall is the shear stress at cylinder wall.
        Thus the shear stress formula used is the one at z=0 (because we take u1 as the cylinder liner velocity) (formula 52 @slide25)'''
        
        Wall_shear = np.zeros(len(self.Grid.x))
        for i in range(len(self.Grid.x)):  
            Wall_shear[i] = ViscosityFunc(StateVector[time])[i] * (0-U)/(StateVector[time].h[i]) - (dp_dx[i])*(StateVector[time].h[i])/2
        StateVector[time].WallShearStress = Wall_shear

        '''StateVector[time].ViscousFriction = '''

