import numpy as np
from firedrake import *

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
# `Coefficient`)
zeroform = f*dx 
# Call assemble to evaluate this 0-form as a rank zero tensor (a real 
# number).
# This is done in two steps:
#   - 1. assemble._assemble is called to yield callable functions 
#        (typically PYOP2 loops)
#       - a. tsfc_interface.compile_form is used to generate kernels to
#            execute on the mesh with PYOP2
#       - b. PYOP2 is configured to execute these loops, with the 
#            functions to execute to make this happen being yielded by 
#            assemble._assemble
#   - 2. the loops are called to yield the final result
assembled = assemble(zeroform) 
assert np.isclose(assembled, 1.)

# Now let's add an unknown TestFunction (a special case of a UFL 
# `Argument`) to our form to create a 1-form
v = TestFunction(V)
oneform = v*dx
# Call assemble to evaluate this 1-form as a rank one tensor (a vector)
assembled = assemble(oneform)
print(assembled)