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
        
        # Schemes for first-order (D) and second order (DD) derivatives
        """ define arrays of values that have to be assigned to a certain fixed diagonal (the diagonal is specified in the next step), the first
        value in the array will be the value of the first diagonal specified in sparse.diag """
        
        self.CentralStencilD=np.array([-1.0,1.0,0.0])/(2.0*Grid.dx)
        self.BackwardStencilD=np.array([-1.0,1.0,0.0])/(1.0*Grid.dx)
        self.ForwardStencilD=np.array([-1.0,1.0,0])/(1.0*Grid.dx)
        
        self.CentralStencilDD=np.array([1.0,-2.0,1.0])/(Grid.dx**2)
        self.BackwardStencilDD=np.array([1.0,-2.0,1.0])/(Grid.dx**2)
        self.ForwardStencilDD=np.array([1.0,-2.0,1.0])/(Grid.dx**2)

        
        # Sparse Matrix Operators
        """ specify the exact diagonal that gets the value assigned in the array + adjust matrices for the boundary grid positions"""
        self.Identity=sparse.identity(Grid.Nx, dtype='float', format="csr")
        
        self.DDXCentral=sparse.diags(self.CentralStencilD, [-1, 1, 0], shape=(Grid.Nx, Grid.Nx), dtype='float', format="csr")
        self.DDXCentral[0,0] = -1.0/Grid.dx #Define right boundary stencil    # SIGN DIFFERENCE WITH RESPECT TO EXAMPLE IN ASSIGNMENT
        self.DDXCentral[0,1] = 1.0/Grid.dx #Define right boundary stencil     # SIGN DIFFERENCE WITH RESPECT TO EXAMPLE IN ASSIGNMENT
        self.DDXCentral[Grid.Nx-1,Grid.Nx-2] = -1.0/Grid.dx #Define left boundary stencil
        self.DDXCentral[Grid.Nx-1,Grid.Nx-1] = 1.0/Grid.dx #Define left boundary stencil

        self.DDXBackward=sparse.diags(self.BackwardStencilD, [-1, 0, 1], shape=(Grid.Nx, Grid.Nx), dtype='float', format="csr")
        self.DDXBackward[0,0] = -1.0/Grid.dx #Define boundary stencil
        self.DDXBackward[0,1] = 1.0/Grid.dx #Define boundary stencil

        self.DDXForward=sparse.diags(self.ForwardStencilD, [0,1,-1], shape=(Grid.Nx, Grid.Nx), dtype='float', format="csr")
        self.DDXForward[Grid.Nx-1,Grid.Nx-2] = -1.0/Grid.dx #Define boundary stencil
        self.DDXForward[Grid.Nx-1,Grid.Nx-1] = 1.0/Grid.dx #Define boundary stencil


        self.D2DX2Central = sparse.diags(self.CentralStencilDD, [-1,0,1] , shape=(Grid.Nx, Grid.Nx), dtype='float', format="csr")
        self.D2DX2Central[0,0] = 1.0/Grid.dx**2
        self.D2DX2Central[0,1] = -2.0/Grid.dx**2
        self.D2DX2Central[0,2] = 1.0/Grid.dx**2
        self.D2DX2Central[Grid.Nx-1,Grid.Nx-3] = 1.0/Grid.dx**2
        self.D2DX2Central[Grid.Nx-1,Grid.Nx-2] = -2.0/Grid.dx**2
        self.D2DX2Central[Grid.Nx-1,Grid.Nx-1] = 1.0/Grid.dx**2


        """ don't know where these are needed? In the linear systems found for Reynolds and Temp., only the second order central is used
        for completeness, apply boundaries in the same way as done for the first order matrices (dont assume this correct as this is a pure guess) """

        self.D2DX2Backward = sparse.diags(self.ForwardStencilDD,[0,-1,-2] , shape=(Grid.Nx, Grid.Nx), dtype='float', format="csr")
        self.D2DX2Backward[0,1] = -2.0/Grid.dx**2
        self.D2DX2Backward[0,2] = 1.0/Grid.dx**2
        self.D2DX2Backward[1,0] = 0.0
        self.D2DX2Backward[1,1] = 1.0/Grid.dx**2
        self.D2DX2Backward[1,2] = -2.0/Grid.dx**2
        self.D2DX2Backward[1,3] = 1.0/Grid.dx**2

        self.D2DX2Forward = sparse.diags(self.ForwardStencilDD,[0,1,2] , shape=(Grid.Nx, Grid.Nx), dtype='float', format="csr")
        self.D2DX2Forward[Grid.Nx-2,Grid.Nx-4] = 1.0/Grid.dx**2
        self.D2DX2Forward[Grid.Nx-2,Grid.Nx-3] = -2.0/Grid.dx**2
        self.D2DX2Forward[Grid.Nx-2,Grid.Nx-2] = 1.0/Grid.dx**2
        self.D2DX2Forward[Grid.Nx-2,Grid.Nx-1] = 0.0
        self.D2DX2Forward[Grid.Nx-1,Grid.Nx-3] = 1.0/Grid.dx**2
        self.D2DX2Forward[Grid.Nx-1,Grid.Nx-2] = -2.0/Grid.dx**2
        self.D2DX2Forward[Grid.Nx-1,Grid.Nx-1] = 1.0/Grid.dx**2

    

    # Efficient Implementation for 1D csr type FD matrix
    #   do not Change implementation below!
    def SetDirichletLeft(self,M):
        M.data[[0,1,2]]=[1.0, 0.0, 0.0]

    def SetDirichletRight(self,M):
        M.data[[-3,-2,-1]]=[0.0, 0.0, 1.0]
    
    def SetNeumannLeft(self,M):
        M.data[[0,1,2]]=self.BackwardStencilD
    
    def SetNeumannRight(self,M):
        M.data[[-3,-2,-1]]=self.ForwardStencilD

    """


    def SetDirichletLeft(self,M):
        M.sort_indices()
        M.data[M.indptr[0]:M.indptr[1]]=[1.0, 0.0, 0.0]

    def SetDirichletRight(self,M):
        M.sort_indices()
        M.data[M.indptr[-2]:M.indptr[-1]]=[0.0, 0.0, 1.0]
    
    def SetNeumannLeft(self,M):
        M.sort_indices()
        M.data[M.indptr[0]:M.indptr[1]]=self.BackwardStencilD
    
    def SetNeumannRight(self,M):
        M.sort_indices()
        M.data[M.indptr[-2]:M.indptr[-1]]=self.ForwardStencilD

    """