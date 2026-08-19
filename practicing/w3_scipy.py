from scipy import constants
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.csgraph import bellman_ford
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cosine
from scipy.spatial.distance import hamming
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import Rbf
from scipy.stats import ttest_ind
from scipy.stats import kstest
from scipy.stats import describe
from scipy.stats import skew, kurtosis, normaltest
from scipy import io
from numpy import cos
import numpy as np
import matplotlib.pyplot as plt

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

# minimazing a function

# finding minima
def eqn(x):
    return x**2 + x + 2

mymin = minimize(eqn, 0, method = "BFGS")
print(mymin)

# Sparse Data
print("\nSparce Data\n")

arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])
print(csr_matrix(arr))

# sparce matrix methods

arr = np.array([[0, 0, 0], (0, 0, 1), (1, 0, 2)])
print(csr_matrix(arr).data) # viewing stored data

print(csr_matrix(arr).count_nonzero()) # counting nonzeros

mat = csr_matrix(arr)
mat.eliminate_zeros() # removing zero-entries
print(mat)

mat = csr_matrix(arr)
mat.sum_duplicates() # eliminating duplicate entries
print(mat)

newarr = csr_matrix(arr).tocsc()

print(newarr)

# Graphs
print("\nGraphs\n")

arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)
print(connected_components(newarr)) # find all the connected components

# dijkstra

print(dijkstra(newarr, return_predecessors = True, indices = 0)) # find the shortest path in a graph from one elemento to another

# floyd warshall
print(floyd_warshall(newarr, return_predecessors = True)) # find shortest path between all pairs od elements

# bellman ford
print(bellman_ford(newarr, return_predecessors = True, indices = 0)) # shortest path between all pairs of elements, but this method can handle negative weights as well

# depth first order

arr = np.array([
    [0, 1, 0, 1],
    [1, 1, 1, 1],
    [2, 1, 1, 0],
    [0, 1, 0, 1]
])

newarr = csr_matrix(arr)
print(depth_first_order(newarr, 1)) # return a depth first traversal from a node

# breadth first order
print(breadth_first_order(newarr, 1)) # returns a breadth first traversal from a node

# Spacial Data
print("\nSpacial Data\n")

# triangulation
points = np.array([
    [2, 4],
    [3, 4],
    [3, 0],
    [2, 2],
    [4, 1]
])

simplices = Delaunay(points).simplices # simplices property creates a generalization

plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color = "r")

plt.show()

# convex hull
points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1],
  [1, 2],
  [5, 0],
  [3, 1],
  [1, 2],
  [0, 2]
])

hull = ConvexHull(points) # smallest polygon that covers all of the given points
hull_points = hull.simplices

plt.scatter(points[:, 0], points[:, 1])
for simplex in hull_points:
    plt.plot(points[simplex, 0], points[simplex, 1], "k-")

plt.show()

# kdtrees
points = [(1, -1), (2, 3), (-2, 3), (2, -3)]

kdtree = KDTree(points)
res = kdtree.query((1, 1)) # find the nearest neighbor to point (1, 1)

print(res)

# euclidean distance
p1 = (1, 0)
p2 = (10, 2)

res = euclidean(p1, p2) # euclidean distance between given points

print(res)

# cityblock distance (Manhattan distance)

res = cityblock(p1, p2) # cityblock distance between given points

print(res)

# cosine distance
res = cosine(p1, p2) # cosine distance ...
print(res)

# hamming distance
p1 = (True, False, True)
p2 = (False, True, True)

res = hamming(p1, p2)
print(res)

# Matlab Arrays
print("\nMatlab Arrays\n")

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# export:
io.savemat("arr.mat", {"vec": arr})

# import:
mydata = io.loadmat("arr.mat")

print(mydata)
print("")
print(mydata["vec"]) # display only the array
mydata = io.loadmat("arr.mat", squeeze_me = True)
print(f"\n{mydata["vec"]}\n")

# Interpolation
print("\nInterpolation\n")

# 1d interpolation
xs = np.arange(10)
ys = 2*xs + 1

interp_func = interp1d(xs, ys)

newarr = interp_func(np.arange(2.1, 3, 0.1))

print(newarr)

# spline interpolation
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1

interp_func = UnivariateSpline(xs, ys)

newarr = interp_func(np.arange(2.1, 3, 0.1))

print(newarr)

# interpolation with radial basis function
interp_func = Rbf(xs, ys)

newarr = interp_func(np.arange(2.1, 3, 0.1))

print(newarr)

# Statistical Significance Tests
print("\nStatistical Significance Tests\n")

v1 = np.random.normal(size = 100)
v2 = np.random.normal(size = 100)

res = ttest_ind(v1, v2)

print(res)

res = ttest_ind(v1, v2).pvalue

print(f"\npvalue: {res}\n")


# ks-test
v = np.random.normal(size = 100)

res = kstest(v, "norm")

print(f"{res}")

# statistical description of data
v = np.random.normal(size = 100)
res = describe(v)

print(f"\n{res}\n")

# normality tests (skewness and kurtosis)

print(f"\n{skew(v)}\n")
print(f"\n{kurtosis(v)}\n")

print(f"\n{normaltest(v)}\n")