import numpy as np
from matplotlib import pyplot as plt
from firedrake import *

# In the simplest case you can use assemble to evaluate integrals with no
# unknowns (0-forms):

# set up a really dull mesh and function space
m = UnitSquareMesh(5, 5)
V = FunctionSpace(m, 'CG', 1)
# Define a Function in the FunctionSpace - this is a UFL `Coefficient`
f = Function(V)
# Get symbolic representation of coordinates in mesh
x, y = SpatialCoordinate(m)
# Set f to be some really boring function (here a plane)
f.interpolate(2.*x)
# Define an integral of the boring function over the interior of the 
# mesh as a UFL form. This is a 0-form since there are no unknown
# functions (UFL `Argument`s) only the known function f (a UFL 
# `Coefficient`).
zeroform = f*dx 
# Call assemble to evaluate this 0-form as a rank zero tensor (a real 
# number).
# This is done in two steps:
#   - 1. assemble._assemble is called to yield callable functions 
#        (typically PyOP2 loops)
#       - a. tsfc_interface.compile_form is used to generate kernels to
#            execute on the mesh with PyOP2
#       - b. PyOP2 is configured to execute these loops, with the 
#            functions to execute to make this happen being yielded by 
#            assemble._assemble
#   - 2. the loops are called to yield the final result - here a real number.
assembled = assemble(zeroform) 
assert np.isclose(assembled, 1.)

# Now consider a linear variational problem within a finite element 
# function space V: find u in V such that a(u,v) = F(v) where v is in V.
# Given that `u = \sum_i u_i \phi_i` and `v = sum_i v_i\phi_i` where phi_i
# are the n basis functions in V, we can turn this into an equivalent matrix-
# vector equation 
#
# `\sum_j a(\phi_i, \phi_j)u_j = F(\phi_i) i = 1, ..., n`
#
# where \phi_i are known basis functions and u_j are the unknown 
# coefficients of u.

# Lets look at the linear form F(v) first.
# Note that `TestFunction` constructs a special case of a UFL `Argument`
# defined on our function space.
v = TestFunction(V)
# Since v is unknown, F(v) is a 1-form: a linear map from the test 
# function v to a set of n numbers `F(\phi_i)` where i = 1, ..., n 
# (a rank 1 tensor). 
# Below is the simplest possible example:
oneform = v*dx
# Assemble works as for the 0-form but with the final result being the 
# rank 1 tensor `F(\phi_i)` where i = 1, ..., n.
assembled = assemble(oneform)
# Note that in Firedrake, we reuse the `Function` type to store each 
# `F(\phi_i)` where we usually store the basis coefficient u_i 
# corresponding to basis function `\phi_i`.
print(type(assembled))

# Next lets look at the bilinear form a(u,v).
# `TrialFunction` also constructs a special case of a UFL `Argument`
# defined on our function space.
u = TrialFunction(V)
# Since u and v are unknown a(u,v) is a 2-form: a bilinear map from the 
# trial function u and test function v to a set of numbers `a(\phi_i, \phi_j)` 
# where i = 1, ..., n and j = 1, ..., n (a rank 2 tensor).
# Below is the simplest possible example:
twoform = v*u*dx
# Assemble again works as for the 0-form but with the final result being
# the rank 2 tensor `a(\phi_i, \phi_j)` where i = 1, ..., n and j = 1, ..., n
assembled = assemble(twoform)
# Assembled matrices are saved in their own `Matrix` type.
print(type(assembled))

