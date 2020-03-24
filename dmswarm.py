import numpy as np
from firedrake.petsc import PETSc, OptionsManager

from firedrake import *

def PointCloud(mesh, points, comm=COMM_WORLD):
    """
    Create a point cloud mesh (vertex only mesh) from a mesh and set of points.
    """
    points = np.asarray(points, dtype=np.double)
    tdim = mesh.topological_dimension()
    gdim = mesh.geometric_dimension()
    pdim = np.shape(points)[1]
    if pdim != tdim:
        raise ValueError(f"Mesh topological dimension {tdim} must match point list dimension {pdim}")

    # Create a DMSWARM 
    swarm = PETSc.DMSwarm().create(comm=comm)

    # Set swarm DM dimension to match mesh dimension
    swarm.setDimension(mesh._plex.getDimension())

    # Set coordinates dimension
    swarm.setCoordinateDim(pdim)

    # Link to mesh information for when swarm.migrate() is used
    swarm.setCellDM(mesh._plex)

    # Set to Particle In Cell (PIC) type
    swarm.setType(PETSc.DMSwarm.Type.PIC)

    # START TEMPORARY

    # Setup particle information
    # blocksize = 0
    # swarm.registerField("tempfield", blocksize)
    swarm.finalizeFieldRegister() # required to set point coordinates

    # END TEMPORARY

    # Add point coordinates - note we set redundant mode to true which forces
    # all ranks to search for the points within their cell
    swarm.setPointCoordinates(points, redundant=False, mode=PETSc.InsertMode.INSERT_VALUES)


mesh = UnitSquareMesh(5,5)
V = FunctionSpace(mesh, 'CG', 1)
points = [(.1, .1), (.2, .3), (.7, .8)]
point_cloud = PointCloud(mesh, points)

