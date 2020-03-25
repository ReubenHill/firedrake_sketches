import numpy as np
from firedrake.petsc import PETSc, OptionsManager

from firedrake import *

m = UnitSquareMesh(5,5)
V = FunctionSpace(m, 'DG', 0)
points = [(.1, .1), (.2, .3), (.7, .8)]
# requires firedrake branch 0d-mesh
swarm = mesh._pic_swarm_in_plex(m.topology._plex, points)
