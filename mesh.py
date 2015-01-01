

import vtk

import numpy
#TODO: cleanup these imports
from numpy import  array, asarray, arange, polyadd, polymul, polysub, polyval,dot, inf, roots, zeros, meshgrid, sqrt,where, abs, isreal

def mesh_from_arcs(arcs):
    # Define points, triangles and colors
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    
    # Build the meshgrid manually
    count = 0
    for i in range(1, len(arcs)):
        parc = arcs[i-1]
        arc = arcs[i]
        
        for j in range(0, len(arc)-1):

            # Triangle 1
            points.InsertNextPoint(arc[j][0], arc[j][1], arc[j][2])
            points.InsertNextPoint(arc[j+1][0], arc[j+1][1], arc[j+1][2])
            points.InsertNextPoint(parc[j+1][0], parc[j+1][1], parc[j+1][2])

            triangle = vtk.vtkTriangle()
            pointIds = triangle.GetPointIds()
            pointIds.SetId(0, count)
            pointIds.SetId(1, count + 1)
            pointIds.SetId(2, count + 2)

            count += 3

            triangles.InsertNextCell(triangle)

            # Triangle 2
            points.InsertNextPoint(arc[j][0], arc[j][1], arc[j][2])
            points.InsertNextPoint(parc[j+1][0], parc[j+1][1], parc[j+1][2])
            points.InsertNextPoint(parc[j][0], parc[j][1], parc[j][2])

            triangle = vtk.vtkTriangle()
            pointIds = triangle.GetPointIds()
            pointIds.SetId(0, count)
            pointIds.SetId(1, count + 1)
            pointIds.SetId(2, count + 2)

            count += 3

            triangles.InsertNextCell(triangle)

    # Create a polydata object
    trianglePolyData = vtk.vtkPolyData()

    # Add the geometry and topology to the polydata
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)

    # Clean the polydata so that the edges are shared !
    cleanPolyData = vtk.vtkCleanPolyData()
    cleanPolyData.SetInput(trianglePolyData)
    cleanPolyData.Update()
    return cleanPolyData.GetOutput()

class Mesh(object):
    """
    Basically just a collection of oriented triangles.
    Should not contain any degenerate triangles.
    Not manifold (because it's probably not closed)
    Most important things that this can do is raycast and render.
    """
    
    def __init__(self, mesh):
        self._mesh = mesh
        
        #smooth_loop = vtk.vtkLoopSubdivisionFilter()
        #smooth_loop.SetNumberOfSubdivisions(3)
        #smooth_loop.SetInput(cleanPolyData.GetOutput())
        #self.mesh = smooth_loop.GetOutput()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInput(self._mesh)
        normals.ComputeCellNormalsOn()
        output = normals.GetOutput()
        output.Update();
        cellData = output.GetCellData();
        self._normals = cellData.GetNormals();

        self._caster = vtk.vtkOBBTree()
        #set the 'mesh' as the caster's dataset
        self._caster.SetDataSet(self._mesh)
        #build a caster locator
        self._caster.BuildLocator()
    
    def export(self, filename):
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetFileName(filename)
        stlWriter.SetInput(self._mesh)
        stlWriter.Write()
        
    def intersection_plus_normal(self, start, end):
        """
        Note: if you're firing a ray with start or end very close to the surface, be prepared for bad surprises
        Basically, it's hard to say whether something is exactly on a surface. So either move your point farther past the surface
        or away from it if you know you're going to be close.
        """
        #create a 'vtkPoints' object to store the intersection points
        pointsVTKintersection = vtk.vtkPoints()
        #and a list for the cells
        cellIds = vtk.vtkIdList()

        #perform ray-casting (intersect a line with the mesh)
        code = self._caster.IntersectWithLine(start, end, pointsVTKintersection, cellIds)

        # Interpret the 'code'. If "0" is returned then no intersections points
        # were found so return an empty list
        if code == 0:
            return [None, None]
        point = numpy.array(pointsVTKintersection.GetData().GetTuple3(0))
        normal = numpy.array(self._normals.GetTuple(cellIds.GetId(0)))
        return (point, normal)
    
    def _repr_(self):
        '''
        Return an string with the representation of the mesh surface
        '''
        return "Mesh(numPolys="+str(self._normals.GetNumberOfTuples())+")"

