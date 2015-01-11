

import vtk
import numpy
import pyglet.gl
import pyglet.graphics

from optics.base import *

def mesh_from_arcs(arcs):
    #create all of the points so they have sensible, shared indices
    points = vtk.vtkPoints()
    num_points_in_arc = len(arcs[0])
    for i in range(0, len(arcs)):
        arc = arcs[i]
        assert len(arc) == num_points_in_arc, "All arcs must be the same length"
        for j in range(0, len(arc)):
            point = arc[j]
            points.InsertNextPoint(point[0], point[1], point[2])
    
    # Build the meshgrid manually
    triangles = vtk.vtkCellArray()
    for i in range(1, len(arcs)):
        for j in range(0, len(arc)-1):
            
            triangle = vtk.vtkTriangle()
            pointIds = triangle.GetPointIds()
            pointIds.SetId(0, i * num_points_in_arc + j)
            pointIds.SetId(1, i * num_points_in_arc + j + 1)
            pointIds.SetId(2, (i-1) * num_points_in_arc + j + 1)
            triangles.InsertNextCell(triangle)

            triangle = vtk.vtkTriangle()
            pointIds = triangle.GetPointIds()
            pointIds.SetId(0, i * num_points_in_arc + j)
            pointIds.SetId(1, (i-1) * num_points_in_arc + j + 1)
            pointIds.SetId(2, (i-1) * num_points_in_arc + j)
            triangles.InsertNextCell(triangle)

    # Create a polydata object
    trianglePolyData = vtk.vtkPolyData()

    # Add the geometry and topology to the polydata
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)
    trianglePolyData.Update()

    return trianglePolyData

#NOTE: this would be much cleaner if I had not inlined the logic about how far away the points are from the light cone
def trim_mesh_with_cone(mesh, cone_point, cone_normal, cone_radius):
    """
    returns a mesh that contains only the triangles that where inside of the cone
    """
    cone_end = cone_point + cone_normal*100.0
    sq_radius = cone_radius*cone_radius
    w = cone_point
    v = cone_end
    n = w - v
    v_w_sq_len = dist2(v, w)
   
    points = mesh.GetPoints().GetData()
    cell_array = mesh.GetPolys()
    polygons = cell_array.GetData()
    triangles = vtk.vtkCellArray()
    for i in xrange(0,  cell_array.GetNumberOfCells()):
        triangle = [polygons.GetValue(j) for j in xrange(i*4+1, i*4+4)]
        
        p = points.GetTuple(triangle[0])
        delta = p - (v + (((p - v).dot(n)) / v_w_sq_len) * n)
        if delta.dot(delta) > sq_radius:
            continue
        
        p = points.GetTuple(triangle[1])
        delta = p - (v + (((p - v).dot(n)) / v_w_sq_len) * n)
        if delta.dot(delta) > sq_radius:
            continue
        
        p = points.GetTuple(triangle[2])
        delta = p - (v + (((p - v).dot(n)) / v_w_sq_len) * n)
        if delta.dot(delta) > sq_radius:
            continue
        
        cell = vtk.vtkTriangle()
        pointIds = cell.GetPointIds()
        pointIds.SetId(0, triangle[0])
        pointIds.SetId(1, triangle[1])
        pointIds.SetId(2, triangle[2])
        triangles.InsertNextCell(cell)
            
    # Create a polydata object
    trianglePolyData = vtk.vtkPolyData()

    # Add the geometry and topology to the polydata
    trianglePolyData.SetPoints(mesh.GetPoints())
    trianglePolyData.SetPolys(triangles)
    trianglePolyData.Update()
    
    #run the clean function here to remove the points that are not used
    cleanPolyData = vtk.vtkCleanPolyData()
    cleanPolyData.SetInput(trianglePolyData)
    cleanPolyData.Update()
    trimmed_mesh = cleanPolyData.GetOutput()

    return trimmed_mesh

def merge_meshes(meshes):
    point_offset = 0
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for m in meshes:
        mesh = m._mesh
        
        #add every point
        point_data = mesh.GetPoints().GetData()
        for i in range(0, point_data.GetSize()):
            point = point_data.GetTuple(i)
            points.InsertNextPoint(point[0], point[1], point[2])
    
        #add every triangle with the correct point offset
        cell_array = mesh.GetPolys()
        polygons = cell_array.GetData()
        for i in xrange(0,  cell_array.GetNumberOfCells()):
            triangle = [polygons.GetValue(j) for j in xrange(i*4+1, i*4+4)]
            
            new_triangle = vtk.vtkTriangle()
            pointIds = new_triangle.GetPointIds()
            pointIds.SetId(0, triangle[0] + point_offset)
            pointIds.SetId(1, triangle[1] + point_offset)
            pointIds.SetId(2, triangle[2] + point_offset)
            triangles.InsertNextCell(new_triangle)
        
        point_offset += point_data.GetSize()
        
    # Create a polydata object
    trianglePolyData = vtk.vtkPolyData()

    # Add the geometry and topology to the polydata
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)
    trianglePolyData.Update()
    return trianglePolyData

def load_stl(filename):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    return reader.GetOutput()

class Mesh(object):
    """
    Basically just a collection of oriented triangles.
    Should not contain any degenerate triangles.
    Not manifold (because it's probably not closed)
    Most important things that this can do is raycast and render.
    """
    
    def __init__(self, mesh):
        assert mesh != None
        
        self._mesh = mesh
        self._batch = None
        self._color = (0.0, 1.0, 0.0, 1.0)
        
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
        
    def points(self):
        point_data = self._mesh.GetPoints().GetData()
        return [numpy.array(point_data.GetTuple(i)) for i in range(0, point_data.GetSize())]
        
    def render(self):
        if self._batch == None:
            self._batch = pyglet.graphics.Batch()
            points = self._mesh.GetPoints().GetData()
            cell_array = self._mesh.GetPolys()
            polygons = cell_array.GetData()
            for i in xrange(0,  cell_array.GetNumberOfCells()):
                triangle = [polygons.GetValue(j) for j in xrange(i*4+1, i*4+4)]
                a = points.GetTuple(triangle[0])
                b = points.GetTuple(triangle[1])
                c = points.GetTuple(triangle[2])
                self._batch.add(3, pyglet.gl.GL_TRIANGLES, None, ('v3f', (
                    a[0], a[1], a[2],
                    b[0], b[1], b[2],
                    c[0], c[1], c[2]
                )))
        pyglet.gl.glColor4f(self._color[0], self._color[1], self._color[2], self._color[3])
        pyglet.gl.glPolygonMode(pyglet.gl.GL_FRONT_AND_BACK, pyglet.gl.GL_LINE)
        self._batch.draw()
        pyglet.gl.glPolygonMode(pyglet.gl.GL_FRONT_AND_BACK, pyglet.gl.GL_FILL)
        
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

