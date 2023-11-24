import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.special as special 

def I2(l): 
    I2 = (0.5*(l**2+1)*special.erfc(l/np.sqrt(2.0)) - (l/np.sqrt(2.0*np.pi))*np.exp(-l**2.0/2.0))/np.sqrt(l)
    return I2

def I52(l):
    I52 = ((1.0/(8.0*np.sqrt(np.pi)))*np.exp(-l**2.0/4.0)*(l**(3.0/2.0))*((2.0*l**2.0+3.0)*special.kv(3.0/4.0,l**2.0/4.0)-(2.0*l**2.0+5.0)*special.kv(1.0/4.0,l**2.0/4.0)))/np.sqrt(l)
    return I52

def functie(x):
    return I2(x)*x**(-0.5)
a = quad(functie, 0.002, 5.0003)
print(a)