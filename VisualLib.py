# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:21:10 2021
@ProjectTitle : Thermal Mixed-Hydrodynamic Lubrication Solver for simulation of a compression ring - cylinder liner contact in an IC Engine.
@author: Dieter Fauconnier
@version : 1.0
"""


import numpy as np 
import matplotlib.pyplot as plt


    
def Report_PT(Grid,State): # initiatlization

    f1, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('$x [m]$')
    ax1.set_ylabel('$P [MPa]$',color=color)
    ax1.plot(Grid.x,State.Pressure/1e6,'x-', linewidth=1,color=color)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('$T [^\circ C]$',color=color)  # we already handled the x-label with ax1
    ax2.plot(Grid.x,State.Temperature-273.15,'x-', linewidth=1,color=color)
    ax2.tick_params(axis='y')
    f1.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return f1
    

    


def Report_Ops(Time,Ops,time):     
    
    f2, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('$t [s]$')
    ax1.set_ylabel('$U [m/s]$',color=color)
    ax1.plot(Time.t,Ops.SlidingVelocity,'-',Time.t[time],Ops.SlidingVelocity[time],'ko', linewidth=1,color=color)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('$F [N/m]$',color=color)  # we already handled the x-label with ax1
    ax2.plot(Time.t,Ops.CompressionRingLoad,'-',Time.t[time],Ops.CompressionRingLoad[time],'ko',linewidth=1,color=color)
    ax2.tick_params(axis='y')
    f2.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return f2
