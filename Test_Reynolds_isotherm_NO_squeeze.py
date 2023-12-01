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
from matplotlib.animation import FuncAnimation


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
TimeStep=1e-5 # Choose Temperal Resolution 
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

# de-activate squeeze term and temperature code in ReynoldsOliver
h0 = 10e-6
time = 10
StateVector[time].h= h0 + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
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





# ITERATE OVER PLOTS

# for time in range(1,Time.nt,250):
#     StateVector[time].h= h0 + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
#     Reynolds.SolveReynolds(StateVector,time)
#     fig, ax1 = plt.subplots()
#     color = 'tab:red'
#     ax1.set_xlabel('position')
#     ax1.set_ylabel('height h', color=color)
#     ax1.plot(Grid.x, StateVector[time].h, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()
#     color = 'tab:blue'
#     ax2.set_ylabel('pressure', color=color)
#     ax2.plot(Grid.x, StateVector[time].Pressure, color=color)
#     ax2.tick_params(axis='y', labelcolor=color)

#     variable_value = np.round(Ops.SlidingVelocity[time],2)
#     text_to_display = f'Sliding Velocity: {variable_value} m/s at {time}'
#     plt.title(text_to_display, fontsize=10, color='red')

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.show()






# ANIMATION: TAKES A LOTTT OF TIME 

# time = 0
# while time < Time.nt:
#     h0 = 10e-6
#     StateVector[time].h= h0 + (4.0*Engine.CompressionRing.CrownHeight/Engine.CompressionRing.Thickness**2.0)*Grid.x**2.0
#     Reynolds.SolveReynolds(StateVector,time)
#     time +=1


# # Create the initial plot
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('position')
# ax1.set_ylabel('height h', color=color)
# line1, = ax1.plot(Grid.x, StateVector[0].h, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('pressure', color=color)
# line2, = ax2.plot(Grid.x, StateVector[0].Pressure, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# variable_value = np.round(Ops.SlidingVelocity[0], 2)
# text_to_display = f'Sliding Velocity: {variable_value} m/s'
# title = ax1.set_title(text_to_display, fontsize=10, color='red')

# # Animation update function
# def update(frame):
#     line1.set_ydata(StateVector[frame].h)
#     line2.set_ydata(StateVector[frame].Pressure)
#     variable_value = np.round(Ops.SlidingVelocity[frame], 2)
#     text_to_display = f'Sliding Velocity: {variable_value} m/s'
#     title.set_text(text_to_display)
#     return line1, line2, title

# # Create the animation
# animation = FuncAnimation(fig, update, frames=Time.nt, interval=1000, blit=True)

# # Display the animation
# plt.show()


