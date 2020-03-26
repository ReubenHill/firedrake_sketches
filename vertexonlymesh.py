from firedrake import *

m = UnitSquareMesh(5,5)
vertexcoords = [(.1, .1), (.2, .3), (.7, .8)]
m2 = VertexOnlyMesh(m, vertexcoords)
m2.init()