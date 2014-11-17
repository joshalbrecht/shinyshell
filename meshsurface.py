
import vtk

import numpy
from numpy import  array, asarray, arange, polyadd, polymul, polysub, polyval,\
     dot, inf, roots, zeros, meshgrid, sqrt,where, abs,  isreal

from pyoptools.raytrace.surface.surface import Surface
from pyoptools.raytrace.ray.ray import Ray

class MeshSurface(Surface):

    def __init__(self, arcs, *args, **kwargs):
        Surface.__init__(self, *args, **kwargs)
        self.arcs = arcs
        self._build_mesh(arcs)

    def _build_mesh(self, arcs):
        # Define points, triangles and colors
        points = vtk.vtkPoints()
        triangles = vtk.vtkCellArray()

        # Build the meshgrid manually
        count = 0
        for i in range(0, len(arcs)):
            arc = arcs[i]
            parc = arcs[i-1]

            # first triangle
            points.InsertNextPoint(arc[0][0], arc[0][1], arc[0][2])
            points.InsertNextPoint(arc[1][0], arc[1][1], arc[1][2])
            points.InsertNextPoint(parc[1][0], parc[1][1], parc[1][2])

            triangle = vtk.vtkTriangle()
            pointIds = triangle.GetPointIds()
            pointIds.SetId(0, count)
            pointIds.SetId(1, count + 1)
            pointIds.SetId(2, count + 2)

            count += 3

            triangles.InsertNextCell(triangle)

            for j in range(1, len(arc)-1):

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
                points.InsertNextPoint(arc[j+1][0], arc[j+1][1], arc[j+1][2])
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
        self.mesh = cleanPolyData.GetOutput()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInput(self.mesh)
        normals.ComputeCellNormalsOn()
        output = normals.GetOutput()
        output.Update();
        cellData = output.GetCellData();
        self.normals = cellData.GetNormals();

        self.caster = vtk.vtkOBBTree()
        #set the 'mesh' as the caster's dataset
        self.caster.SetDataSet(self.mesh)
        #build a caster locator
        self.caster.BuildLocator()

    def topo(self, x, y):
        """**Returns the Z value for a given X and Y**

        This method returns the topography of the polynomical surface to be
        used to plot the surface.
        """
        return self.eval_poly(x, y)

    def _intersection(self, A):
        '''**Point of intersection between a ray and the polynomical surface**

        This method returns the point of intersection  between the surface
        and the ray. This intersection point is calculated in the coordinate
        system of the surface.

           iray -- incident ray

        iray must be in the coordinate system of the surface
        '''

        pointRaySource = A.pos
        pointRayTarget = (1000000.0 * numpy.array(A.dir)) + numpy.array(A.pos)

        #create a 'vtkPoints' object to store the intersection points
        pointsVTKintersection = vtk.vtkPoints()

        cellIds = vtk.vtkIdList()

        #perform ray-casting (intersect a line with the mesh)
        code = self.caster.IntersectWithLine(pointRaySource,
                                             pointRayTarget,
                                             pointsVTKintersection, cellIds)

        # Interpret the 'code'. If "0" is returned then no intersections points
        # were found so return an empty list
        if code == 0:
            #log.info(
            #    "No intersection points found for 'pointRaySource': " + str(
            #        pointRaySource) + " and 'pointRayTarget': " + str(
            #        pointRayTarget))
            return []
        # If code == -1 then 'pointRaySource' lies outside the surface
        #elif code == -1:
        #    log.info("The point 'pointRaySource': " + str(
        #        pointRaySource) + "lies inside the surface")

        return pointsVTKintersection.GetData().GetTuple3(0)

        ##get the actual data of the intersection points (the point tuples)
        #pointsVTKIntersectionData = pointsVTKintersection.GetData()
        ##get the number of tuples
        #noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

        ##create an empty list that will contain all list objects
        #pointsIntersection = []

        ## Convert the intersection points to a list of list objects.
        #for idx in range(noPointsVTKIntersection):
        #    _tup = pointsVTKIntersectionData.GetTuple3(idx)
        #    pointsIntersection.append(_tup)

        ##return the list of list objects
        #return pointsIntersection


    def normal(self, int_p):
        """**Return the vector normal to the surface**

        This method returns the vector normal to the polynomical surface at a
        point ``int_p=(x,y,z)``.

        Note: It uses ``x`` and ``y`` to calculate the ``z`` value and the normal.
        """

        return n


    def _repr_(self):
        '''
        Return an string with the representation of the mesh surface
        '''
        return "MeshSurface(numPolys="+str(self.triangles)+")"



