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
Nodes=256
Grid=Grid(Contact,Nodes)

"""Temporal Discretization"""
TimeStep=0.01#1e-5 # Choose Temperal Resolution 
EndTime=4.0*np.pi/(EngineRPM*(2.0*np.pi/60.0))
Time=Time(EndTime,TimeStep)

"""Define Operational Conditions""" 
Ops=Ops(Time,Engine,EngineRPM,EngineAcceleration,OilTemperature)


"""Define Two-Phase Lubricant-Vapour flow"""
Oil=Liquid('SAE5W40')
Vapour=Gas('SAE5W40')
Mixture=CavitationModel(Oil,Vapour)


"""Define the State Vector = List of All States over time"""
StateVector=[]
for t in range(Time.nt):
    StateVector.append(State(Grid))


""" Spatial Discretization by Finite Differences """
Discretization=FiniteDifferences(Grid)


""" Initialize Reynolds Solver"""
MaxIterReynolds=5000
TolP=1e-4
UnderRelaxP=0.001
TolT=1e-4
UnderRelaxT=0.01
Reynolds=ReynoldsSolver(Grid,Time,Ops,Mixture,Discretization)
Reynolds.SetSolver(MaxIterReynolds,TolP,UnderRelaxP,TolT,UnderRelaxT,VisualFeedbackLevel)

""" Set Load Balance loop"""
MaxIterLoad=15
Tolh0=1e-3
UnderRelaxh0=0.25

# For this Test-file you have to re-activate the squeeze term in the ReynoldsOliver.py file
# other than this, nothing changes (same code as the Test_Main_LoadBalance file)

"""Start from Initial guess or Load Initial State"""

time=(np.abs(Time.t - InitTime)).argmin()

if LoadInitialState:
    
    """Start from previous solution: Load Data at t=0"""
    FileName='Data/Time_'+str(round(Time.t[time]*1000,5))+'ms.h5'
    Data=IO.ReadData(FileName)
    StateVector[time].h0=float(Data['State']['h0'])
    StateVector[time].Hersey=float(Data['State']['Hersey'])
    StateVector[time].Lambda=float(Data['State']['Lambda'])
    StateVector[time].HydrodynamicLoad=float(Data['State']['HydrodynamicLoad'])
    StateVector[time].ViscousFriction=float(Data['State']['ViscousFriction'])
    StateVector[time].AsperityLoad=float(Data['State']['AsperityLoad'])
    StateVector[time].AsperityFriction=float(Data['State']['AsperityFriction'])
    StateVector[time].AsperityContactArea=float(Data['State']['AsperityContactArea'])
    StateVector[time].AsperityContactPressure=float(Data['State']['AsperityContactPressure'])
    StateVector[time].HertzianContactPressure=float(Data['State']['HertzianContactPressure'])
    StateVector[time].COF=float(Data['State']['COF'])
    StateVector[time].WearDepthRing=float(Data['State']['WearDepthRing'])
    
    StateVector[time].h= Data['State']['h']
    StateVector[time].Pressure=Data['State']['Pressure']
    StateVector[time].Temperature=Data['State']['Temperature']
    StateVector[time].WallShearStress=Data['State']['WallShearStress']
    StateVector[time].WearLocationsCylinder=Data['State']['WearLocationsCylinder']
    StateVector[time].WearDepthCylinder=Data['State']['WearDepthCylinder']


else:    
    
    """Start from Scratch: Set State Initial Conditions"""
    StateVector[time].Lambda=1.758; 
    StateVector[time].h0=StateVector[time].Lambda*Contact.Roughness
    StateVector[time].h= StateVector[time].h0 + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
    StateVector[time].Hersey=0.001*np.abs(Ops.PistonVelocity[time])/np.abs(Ops.CompressionRingLoad[time])
    StateVector[time].Pressure=Ops.AtmosphericPressure+0.0*Grid.x
    StateVector[time].Temperature=Ops.OilTemperature+0.0*Grid.x
    StateVector[time].ViscousFriction=0.0
    
    Contact.AsperityContact(StateVector,time)
    StateVector[time].COF=0.0
    StateVector[time].WearDepthRing=0.0
    StateVector[time].WearLocationsCylinder=np.unique(np.round(Ops.PistonPosition,8)) 
    StateVector[time].WearDepthCylinder=0.0*StateVector[time].WearLocationsCylinder
    
    if SaveStates:
        FileName='Data/Time_'+str(round(Time.t[time]*1000,5))+'ms.h5'
        #Data2File={'Grid': Grid,'Time': Time,'State': StateVector[time]}
        Data2File={'State': StateVector[time]}
        IO.SaveData(FileName,Data2File)

"""Start Time Loop"""
start_time = TimeKeeper.time()
while time<Time.nt:
    time += 1

    if time == Time.nt:
        break

    " use previous state for the initial guesses "
    StateVector[time] = StateVector[time-1]
    
    " initialize residual h0 and iterations "
    eps_h_0 = np.ones(MaxIterLoad+1)
    i = 1

    " initialize/guess h0 by using the last h0 from the previous timestep and 0.99*h0  "
    h0 = np.zeros(MaxIterLoad+1)
    h0[0] = StateVector[time-1].h0
    h0[1] = 0.99*h0[0]

    " elastic tension and compression on ring "
    F_el = 16*Engine.CompressionRing.FreeGapSize*Solids('Nitrided Stainless Steel').YoungsModulus*(Engine.CompressionRing.Thickness*Engine.CompressionRing.Width**3/12)/(3*np.pi*(Engine.Cylinder.Radius*2)**4)
    F_comp = Engine.CompressionRing.Thickness*(Ops.CylinderPressure[time]-Ops.AtmosphericPressure)

    " initialize load imbalance "
    DW = np.zeros(MaxIterLoad+1)

    " calculate pressure and h for initial guess (last timestep) "
    StateVector[time].h= h0[0] + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
    Reynolds.SolveReynolds(StateVector,time)
    " calculate asperity model for initial guess (last timestep) "
    # StateVector[time].Lambda =  h0[0]/Contact.Roughness
    # Contact.AsperityContact(StateVector,time)
    " calculate load imbalance for initial guess (last timestep) "
    DW[0] = StateVector[time].HydrodynamicLoad  - F_el - F_comp #+ StateVector[time].AsperityLoad

    " calculate pressure and h for second guess (0.99*last timestep) "
    StateVector[time].h= h0[1] + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
    Reynolds.SolveReynolds(StateVector,time)
    " calculate asperity model for second guess (0.99*last timestep) "
    # StateVector[time].Lambda =  h0[1]/Contact.Roughness
    # Contact.AsperityContact(StateVector,time)
    " calculate load imbalance for initial guess (last timestep) "
    DW[1] = StateVector[time].HydrodynamicLoad  - F_el - F_comp #+ StateVector[time].AsperityLoad


    # Load balance iterations : Quasi-Newton method


    while (eps_h_0[i] > Tolh0 and i < MaxIterLoad):

        """a. Calculate Film Thickness Profile"""
        StateVector[time].h= StateVector[time].h0 + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
        StateVector[time].h= h0[i] + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0

        """b. Calculate Asperity Load"""
        # StateVector[time].Lambda =  h0[i]/Contact.Roughness
        # Contact.AsperityContact(StateVector,time)

        """c. Solve Reynolds""" 
        Reynolds.SolveReynolds(StateVector,time)

        """d. update h0 with Quasi Newton method"""
        DW[i+1] = StateVector[time].HydrodynamicLoad  - F_el - F_comp #+ StateVector[time].AsperityLoad
        h0[i+1] = max(h0[i]- UnderRelaxh0*(DW[i]/(DW[i]-DW[i-1]))*(h0[i]-h0[i-1]) , 0.1*np.sqrt(Engine.Cylinder.Roughness**2+Engine.CompressionRing.Roughness**2))
        StateVector[time].h0 = h0[i+1]

        """update iteration parameters"""
        i += 1 
        eps_h_0[i] = np.abs(h0[i]/h0[i-1]-1)


    " update statevector "
    # StateVector[time].Lambda =  h0[i]/Contact.Roughness
    # Contact.AsperityContact(StateVector,time)
    StateVector[time].h= StateVector[time].h0 + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
    Reynolds.SolveReynolds(StateVector,time)

   
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('position')
    ax1.set_ylabel('height h', color=color)
    ax1.plot(Grid.x, StateVector[time].h, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('pressure', color=color)
    ax2.plot(Grid.x, StateVector[time].Pressure, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    variable_value = np.round(Ops.SlidingVelocity[time],2)
    text_to_display = f'Sliding Velocity: {variable_value} m/s at {time}'
    plt.title(text_to_display, fontsize=10, color='red')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    #print(StateVector[time].HydrodynamicLoad)
