
"""
Import libraries
"""
import socket
import sys

hostname=socket.gethostname()
Tier2List=['swallot', 'skitty', 'victini', 'slaking', 'kirlia', 'doduo']
if any(list(i in hostname for i in Tier2List)):
    import matplotlib
    matplotlib.use('Agg')
    print('HPC detected: Matplotlib used as backend')

import matplotlib.pyplot as plt
import numpy as np 
import copy as copy

from scipy.interpolate import interp1d
from scipy.integrate import quad

from EngineParts import Engine  #import all classes from file
from SolidsLibrary import Solids
from TriboContact import TriboContact #import all classes from file
from Grid import Grid #import all classes from file
from Time import Time #import all classes from file
from Ops import Ops #import all classes from file
from FluidLibrary import Liquid,Gas #import all classes from file
from TwoPhaseModel import CavitationModel 
from SolutionState import State #import all classes from file
from FiniteDifferences import FiniteDifferences
from ReynoldsOliver import ReynoldsSolver
from IOHDF5 import IOHDF5
import time as TimeKeeper
import VisualLib as vis


"""General Settings for Input and Output """
VisualFeedbackLevel=1 # [0,1,2,3] = [none, per time step, per load iteration, per # reynolds iterations]
SaveFig2File=False # Save figures to file? True/False
LoadInitialState=False # Load The IntialSate? True/False
InitTime=0.0 #Initial Time to Load?
SaveStates=False # Save States to File? True/False

"""I/O Operator"""
IO=IOHDF5()

""" Input Parameters"""
EngineType='VW 2.0 R4 16v TDI CR 103kW'
OilTemperature=95.0 #C
EngineRPM=2400.0 #rpm
EngineAcceleration=0.0


""" Define Engine Geometry"""
Engine=Engine(EngineType)


"""Define Dry Contact parameters"""
Contact=TriboContact(Engine)


"""1D Computational Grid"""
Nodes=64
Grid=Grid(Contact,Nodes)


""" Spatial Discretization by Finite Differences """
Discretization=FiniteDifferences(Grid)

u=np.sin(8000*Grid.x)
dudx=8000*np.cos(8000*Grid.x)         # GOOD CONVERGENCE FOR 256 NODES
d2udx2=-8000**2*np.sin(8000*Grid.x)

# u = Grid.x**3
# dudx = 3*Grid.x**2                    # GOOD CONVERGENCE FOR 256 NODES
# d2udx2 = 6*Grid.x

#u = (Grid.x)**2
#dudx = 2*Grid.x                 #  GOOD CONVERGENCE ONLY FOR SMALL NUMBER OF NODES (10) --> INCREASE NUMERCIAL INSTABILITY
#d2udx2 = 0*Grid.x + 2



# u = np.exp(8000*Grid.x)
# dudx = 8000*np.exp(8000*Grid.x)       # GOOD CONVERGENCE FOR 256 NODES
# d2udx2 = 8000**2*np.exp(8000*Grid.x)

DUDX=Discretization.DDXCentral @ u
D2UDX2=Discretization.D2DX2Central @ u

DUDX_Forward = Discretization.DDXForward @ u
D2UDX2_Forward = Discretization.D2DX2Forward @ u

DUDX_Backward = Discretization.DDXBackward @ u
D2UDX2_Backward = Discretization.D2DX2Backward @ u

plt.figure()
plt.plot(Grid.x,dudx,'+',Grid.x,DUDX,'x')
plt.title("Central")
plt.show()


plt.figure()
plt.plot(Grid.x,d2udx2,'+',Grid.x,D2UDX2,'x')
plt.title("Central second order")
plt.show()


# plt.figure()
# plt.plot(Grid.x,dudx,Grid.x,DUDX_Forward)
# plt.title("Forward")
# plt.show()


# plt.figure()
# plt.plot(Grid.x,d2udx2,Grid.x,D2UDX2_Forward)
# plt.title("Forward second order")
# plt.show()


# plt.figure()
# plt.plot(Grid.x,dudx,Grid.x,DUDX_Backward)
# plt.title("Backward")
# plt.show()


# plt.figure()
# plt.plot(Grid.x,d2udx2,Grid.x,D2UDX2_Backward)
# plt.title("Backward second order")
# plt.show()
