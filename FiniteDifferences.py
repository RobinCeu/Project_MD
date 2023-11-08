# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:03:02 2020
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""

import numpy as np # Matrix Definitions and operations.
import scipy.sparse as sparse # Sparse Matrix Definitions and operations


class FiniteDifferences:
    
    
#################
##### TO DO #####
#################
    def __init__(self,Grid):
        
        #Schemes for first-order (D) and second order ( DD) derivatives
        self.CentralStencilD=np.array([-1.0,0.0,1.0])/(2.0*Grid.dx)
        self.BackwardStencilD=
        self.ForwardStencilD=
        
        self.CentralStencilDD=

        
        # Sparse Matrix Operators
        self.Identity=sparse.identity(Grid.Nx, dtype='float', format="csr")
        
        self.DDXCentral=sparse.diags(self.CentralStencilD, [-1, 0, 1], shape=(Grid.Nx, Grid.Nx), dtype='float', format="csr")
        self.DDXCentral #Define right boundary stencil
        self.DDXCentral #Define left boundary stencil

        self.DDXBackward=
        self.DDXBackward #Define boundary stencil

        self.DDXForward=
        self.DDXForward  #Define boundary stencil
        
        self.D2DX2=
        self.D2DX2 #Define right boundary stencil
        self.D2DX2 #Define left boundary stencil
        

    # Efficient Implementation for 1D csr type FD matrix
    #do not Change implementation below!
    def SetDirichletLeft(self,M):
        M.data[[0,1,2]]=[1.0, 0.0, 0.0]

    def SetDirichletRight(self,M):
        M.data[[-3,-2,-1]]=[0.0, 0.0, 1.0]
    
    def SetNeumannLeft(self,M):
        M.data[[0,1,2]]=self.BackwardStencilD
    
    def SetNeumannRight(self,M):
        M.data[[-3,-2,-1]]=self.ForwardStencilD
