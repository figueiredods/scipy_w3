from scipy import constants
from scipy.optimize import root
from numpy import cos

print(constants.liter)

# Constants
print("\nConstants\n")

print(constants.pi)

# constants units
print(dir(constants))

# Optimazers
print("\nOptimazers\n")

def eqn(x):
    return x + cos(x)

myroot = root(eqn, 0)
print(myroot.x)
print(f"\n{myroot}\n")