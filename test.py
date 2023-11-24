import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

def integrate_discrete_data(x_values, y_values):
    # Create an interpolation function
    interp_func = interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')

    # Define the integration function
    def integrand(x):
        return interp_func(x)

    # Integrate the function using quad
    result, error = quad(integrand, x_values[0], x_values[-1])

    return result, error

# Example usage
x_values = np.array([0, 1, 2, 3, 4])
y_values = np.array([1, 4, 6, 8, 10])

result, error = integrate_discrete_data(x_values, y_values)

print("Result of integration:", result)
print("Estimated error:", error)