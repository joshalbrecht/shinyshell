
import optics.arcplane

class Patch(object):
    def __init__(self, shell_point, screen_point, pre_rho_arc, prev_mu_arc, rho_start_plane, rho_end_plane, mu_start_plane, mu_end_plane):
        self.shell_point = shell_point
        self.screen_point = screen_point
        self.pre_rho_arc = pre_rho_arc
        self.prev_mu_arc = prev_mu_arc
        self.rho_start_plane = rho_start_plane
        self.rho_end_plane = rho_end_plane
        self.mu_start_plane = mu_start_plane
        self.mu_end_plane = mu_end_plane
        
        #calculated or loaded
        self.grid = None
        
    def make_grid(self, arc_along_mu, num_slices=10):
        
        #create a bunch of arcs along the correct direction (in patch space)
        projected_eye_point = blah
        projected_shell_point = blah
        projected_screen_point = blah
        projected_plane = blah
        #will need to convert from our space to world space, then to their space, get the derivative, and then come back to our space
        previous_normal_function = blah
        falloff = blah
        if arc_along_mu:
            arc_slice_angles = numpy.linspace(self.rho_start_plane.angle, self.rho_end_plane.angle, num_slices)[1:]
            cross_arc_slice_angles = numpy.linspace(self.mu_start_plane.angle, self.mu_end_plane.angle, num_slices)[1:]
        else:
            arc_slice_angles = numpy.linspace(self.mu_start_plane.angle, self.mu_end_plane.angle, num_slices)[1:]
            cross_arc_slice_angles = numpy.linspace(self.rho_start_plane.angle, self.rho_end_plane.angle, num_slices)[1:]
        #TODO: will have to convert these to local space
        cross_planes = [optics.arcplane.ArcPlane() for blah in blah]
        #does not contain the start points (eg, starting arcs)
        grid = numpy.zeros((num_slices, num_slices))
        for i in range(0, len(arc_slice_angles)):
            angle = arc_slice_angles[i]
            if arc_along_mu:
                arc_plane = optics.arcplane.ArcPlane(rho=angle)
            else:
                arc_plane = optics.arcplane.ArcPlane(mu=angle)
            arc = grow_arc(projected_eye_point, projected_shell_point, projected_screen_point, projected_plane, previous_normal_function, falloff)
            #intersect each of the cross planes with the arc to get actual points
            for j in range(0, len(cross_planes)):
                cross_plane = cross_planes[j]
                intersection = arc.intersect_poly(cross_plane)
                if arc_along_mu:
                    mu = i
                    rho = j
                else:
                    mu = j
                    rho = i
                grid[i][j] = intersection
        
        return grid
    
    def get_corner(self, mu, rho):
        """
        :returns: the point from this patch at the intersection of the planes closest to those mu and rho values
        """
        return blah
    
    def reflect_rays_no_bounds(self, rays):
        """
        :returns: the rays that have been reflected off of our taylor poly surface.
        Note: these rays will are not restricted to bounce only off of the in-domain section of the poly
        This is to prevent weirdness at the edges
        """
        return reflected_rays
    
    
    