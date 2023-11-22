import numpy as np

"hallo"

"hallo"

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Suppose you have x values and corresponding y values in arrays
y_values = press
x_values = Grid.x

# Create an interpolating function
interp_func = interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')

# Define the function to integrate
def integrand(x):
    return interp_func(x)

# Integrate the function using quad
W_hyd, W_hyd_error = quad(integrand, x_values[0], x_values[-1])
