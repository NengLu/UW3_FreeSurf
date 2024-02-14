# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Rayleigh-Taylor instability
# This notebook models the Rayleigh-Taylor instability outlined in Kaus et al. (2010), for further free surface and FSSA implementation tests.
#
# ### References
# Kaus, B. J., Mühlhaus, H., & May, D. A. (2010). A stabilization algorithm for geodynamic numerical simulations with a free surface. Physics of the Earth and Planetary Interiors, 181(1-2), 12-20.

# ### Ex in uw3
#
# https://github.com/underworldcode/underworld3/blob/development/Jupyterbook/Notebooks/Examples-StokesFlow/Ex_Stokes_Sinker.py
# https://github.com/underworld-community/UW3-benchmarks
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size

# +
u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

# # scaling 1: drho
# dRho =   100. * u.kilogram / u.meter**3 # matprop.ref_density
# g    =   9.81 * u.meter / u.second**2   # modprop.gravity
# H    = 500. * u.kilometer #  modprop.boxHeight
# bodyforce    = dRho*g

# ref_stress = dRho * g * H
# ref_viscosity = 1e21 * u.pascal * u.seconds

# ref_time        = ref_viscosity/ref_stress
# ref_length      = H
# ref_mass        = (ref_viscosity*ref_length*ref_time).to_base_units()

# KL = ref_length       
# KM = ref_mass         
# Kt = ref_time

# scaling_coefficients = uw.scaling.get_coefficients()
# scaling_coefficients["[length]"] = KL
# scaling_coefficients["[time]"] = Kt
# scaling_coefficients["[mass]"] = KM


# # scaling 2: litho rho
# ref_density = 3300. * u.kilogram / u.meter**3 # matprop.ref_density
# dRho =   3300. * u.kilogram / u.meter**3 # matprop.ref_density
# g    =   9.81 * u.meter / u.second**2   # modprop.gravity
# H    = 100. * u.kilometer #  modprop.boxHeight
# bodyforce    = dRho*g

# ref_stress = dRho * g * H
# ref_viscosity = 1e21 * u.pascal * u.seconds

# ref_time   = ref_viscosity/ref_stress
# ref_length = H
# ref_mass   = (ref_viscosity*ref_length*ref_time).to_base_units()

# KL = ref_length       
# KM = ref_mass         
# Kt = ref_time

# scaling_coefficients = uw.scaling.get_coefficients()
# scaling_coefficients["[length]"] = KL
# scaling_coefficients["[time]"] = Kt
# scaling_coefficients["[mass]"] = KM

# scaling 3: vel
H = 100.  * u.kilometer
velocity     = 1e-9 * u.meter / u.second
g    =   10.0 * u.meter / u.second**2  
bodyforce    = 3300  * u.kilogram / u.metre**3 * g 
mu           = 1e21  * u.pascal * u.second

KL = H
Kt = KL / velocity
KM = mu * KL * Kt

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM

# +
#case_bc = "freeslip"  
#case_bc = "noslip"
case_bc = "freesurf"

render = True
max_time  = ndim(6.0*u.megayear)
dt_set    = ndim(2.5e3*u.year)
save_every = 1

yres = 50 
xmin, xmax = ndim(-250 * u.kilometer), ndim(250 * u.kilometer)
ymin, ymax = ndim(-500 * u.kilometer), ndim(0 * u.kilometer)
boxl = xmax-xmin
boxh = ymax-ymin
xres = int(boxl/boxh*yres)

amplitude = ndim(5*u.kilometer)
offset = ndim(-100.*u.kilometer)   # L
L = ndim(100.*u.kilometer) 
wavelength = ndim(500.*u.kilometer)
k = 2.0 * np.pi / wavelength
npoints = xres*2+1

densityI = ndim(3200 * u.kilogram / u.metre**3)   # for an
densityD = ndim(3300 * u.kilogram / u.metre**3)   # for litho
ND_gravity = ndim(9.81 * u.meter / u.second**2)

viscI = ndim(1e20 * u.pascal * u.second)
viscD = ndim(1e21 * u.pascal * u.second)

outputPath = "op_uw3_q1dq0_" +case_bc+ "_yres{:n}_wl{:n}_boxl{:n}/".format(yres,wavelength,boxl)
if uw.mpi.size == 1:
    # delete previous model run
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+ i)
            
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

print(ndim(max_time),ndim(dt_set),ndim(bodyforce))
print(outputPath)

# +
import underworld3 as uw
from scipy.interpolate import interp1d
import numpy as np

def getVertexSet(mesh,label_name):  
    """
    Returns the set of indices on the special boundary for a Cartesian mesh. 
    Parameters
    ----------
    mesh : Cartesian mesh
    label_name : 'Top','Bottom','Left','Right'

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(4,4), minCoords=(0.,0.), maxCoords=(1.0,1.0))
    >>> index_points = getVertexSet(mesh,'Top') 
        array([7, 8, 9, 2, 3])
    """  
    label = mesh.dm.getLabel(label_name)
    if not label:
        if mesh.verbose == True:
            print(f"Discarding bc {boundary} which has no corresponding mesh / dm label")
    
    iset = label.getNonEmptyStratumValuesIS()
    if iset:
        label_values = iset.getIndices()
        if len(label_values > 0):
            value = label_values[0]   
            ind = value
        else:
            value = -1
            ind = -1
    pStart,pEnd = mesh.dm.getDepthStratum(0) 
    ind_points = mesh.dm.getStratumIS(label_name,ind).array
    ind_points = ind_points[ind_points<pEnd]
    ind_points = np.array(ind_points)-pStart
    ind_box_vertices = []
    if label_name == "Top":
        ind_box_vertices = np.array([2,3]).astype(int)
    if label_name == "Bottom":
        ind_box_vertices = np.array([0,1]).astype(int)
    if label_name == "Left":
        ind_box_vertices = np.array([0,2]).astype(int)
    if label_name == "Right":
        ind_box_vertices = np.array([1,3]).astype(int)
    ind_points = np.append(ind_points,ind_box_vertices)
    return ind_points


# +
mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))
init_mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))
#meshbox = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax),cellSize=1.0/res,regular=False,qdegree=2,)
#mesh.view()

botwall = getVertexSet(mesh,'Bottom')
topwall = getVertexSet(mesh,'Top')


# +
def plot_mesh(title,mesh,showFig=True):
    import numpy as np
    import pyvista as pv
    import vtk
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_title(title,font_size=11)
    if showFig:
        pl.show(cpos="xy")
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data() 
    pvmesh.clear_point_data()


def plot_meshswarm(title,mesh,swarm,material,showSwarm=False,showFig=True):
    import numpy as np
    import pyvista as pv
    import vtk
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    pl.add_mesh(pvmesh,'Black', 'wireframe')

    if showSwarm:
        with swarm.access():
            points = np.zeros((swarm.data.shape[0],3))
            points[:,0] = swarm.data[:,0]
            points[:,1] = swarm.data[:,1]
        point_cloud = pv.PolyData(points)
        with swarm.access():
            point_cloud.point_data["M"] = material.data.copy()
        #pl.add_points(point_cloud, color="red",render_points_as_spheres=False, point_size=3, opacity=0.5)
        pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars='M',
                        use_transparency=False, opacity=0.95, point_size= 3)
    pl.add_title(title,font_size=11)
    if showFig:
        pl.show(cpos="xy")
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data()
    pvmesh.clear_point_data()


# -

from underworld3.utilities._api_tools import uw_object
from scipy.spatial import distance
class PopulationControl_DeformMesh(uw_object):
    def __init__(self, swarm):
        self._swarm = swarm
        with swarm.access():
            self._swarm_coords     = np.copy(np.ascontiguousarray( self._swarm.data))
            self._swarm_cellid     = np.copy(np.ascontiguousarray( self._swarm.particle_cellid.data[:,0]))
            self._particleNumber   = self._swarm.data.shape[0]
            self._mesh_cID, self._cIDcount  = np.unique(self._swarm_cellid, return_counts=True)
            self._swarm_ppc = self._cIDcount[0]

    def repopulate(self,mesh,updateField=None):
    # similar method as PopulationControl.redisturibute, 
    # but update the owning cell frist, using new swarm build on the deformed mesh to add particles, 
    # and delete the over_populated particles 
        self._updateField = updateField
        self._mesh = mesh
        
        # update the owning cell
        cellid = self._swarm.dm.getField("DMSwarm_cellid")
        coords = self._swarm.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm.dim))
        cellid[:] = self._swarm.mesh.get_closest_cells(coords).reshape(-1)
        self._swarm.dm.restoreField("DMSwarmPIC_coor")
        self._swarm.dm.restoreField("DMSwarm_cellid")
        self._swarm.dm.migrate(remove_sent_points=True)
        self._swarm._index = None
        self._swarm._nnmapdict = {}

        self._updateField = updateField

        ### get the particle positions in the cell from the deformed mesh
        _newswarm = uw.swarm.Swarm(self._mesh)
        _newswarm.populate_petsc(fill_param=fill_parameter,layout=uw.swarm.SwarmPICLayout.GAUSS)
        with _newswarm.access():
            original_swarm_coords = np.copy(np.ascontiguousarray(_newswarm.data))

        ### get the current swarm positions in the cell
        with self._swarm.access():
            current_swarm_coords = self._swarm.data
            current_swarm_cells = self._swarm.particle_cellid.data[:,0]

        current_particle_cellID, current_ppc  = np.unique(current_swarm_cells, return_counts=True)

        ### find cells that are empty
        empty_cells = self._mesh_cID[np.isin(self._mesh_cID, current_particle_cellID, invert=True)]
        empty_cell_coords = original_swarm_coords[np.isin(self._swarm_cellid,  empty_cells)]

        ### find under-populated cells
        underpopulated_cells = current_particle_cellID[current_ppc < self._swarm_ppc]
        ### number of particles missing from each cell
        underpopulated_cell_coords = []

        for cell in underpopulated_cells:
            ### get the current number of particles in the cell
            current_particles = current_ppc[current_particle_cellID == cell][0]
            ### get the number of particles to add
            no_particles_to_add = self._swarm_ppc - current_particles

            original_cell_coords = original_swarm_coords[self._swarm_cellid == cell]
            current_cell_coords  = current_swarm_coords[current_swarm_cells == cell]

            ### caculate the distances between the original and current particle positions
            point_distance = distance.cdist(original_cell_coords, current_cell_coords, 'euclidean')
            ### add the ones with the largest distance
            underpopulated_cell_coords.append(original_cell_coords[np.min(point_distance, axis=1).argsort()[::-1]][:no_particles_to_add])
 

        if underpopulated_cell_coords:  # Check if the list is not empty
            underpopulated_cell_coords = np.concatenate(underpopulated_cell_coords)
        else:
            underpopulated_cell_coords = np.empty(shape=(0,2))  # Create an empty numpy array

        ### Combination of the empty cells and the underpopulated cells
        particles_to_add = np.vstack([underpopulated_cell_coords, empty_cell_coords])

        uw.mpi.barrier()
        new_cells = self._swarm.mesh.get_closest_local_cells(particles_to_add)
        
        if isinstance(new_cells, int):
            new_cells = np.array([new_cells])

        # Ensure that 'valid' has the same length as 'particles_to_add'
        if len(particles_to_add) != len(new_cells):
            new_cells = np.resize(new_cells, len(particles_to_add))

        valid = new_cells != -1

        valid_coords = particles_to_add[valid]
        valid_cells = new_cells[valid]

        all_local_coords = np.vstack([current_swarm_coords, valid_coords])

        all_local_cells = np.hstack([current_swarm_cells, valid_cells])

        ### do interp before adding particles, otherwise interp happens from the newly added particles
        if (self._updateField != None):
            ### defualt to the nnn value
            new_particle_mat = np.rint( self._updateField.rbf_interpolate(all_local_coords) )#, nnn=self._swarm_ppc))
        
        swarm_new_size = all_local_coords.data.shape[0]

        self._swarm.dm.addNPoints(swarm_new_size - current_swarm_coords.shape[0])

        cellid = self._swarm.dm.getField("DMSwarm_cellid")
        coords = self._swarm.dm.getField("DMSwarmPIC_coor").reshape((-1, self._swarm.dim))

        coords[...] = all_local_coords[...]
        cellid[:]   = all_local_cells[:]

        self._swarm.dm.restoreField("DMSwarmPIC_coor")
        self._swarm.dm.restoreField("DMSwarm_cellid")

        if (self._updateField != None):
            with self._swarm.access(self._updateField):
                self._updateField.data[:,0] = new_particle_mat[:,0]

        uw.mpi.barrier()
        
        ### get the current swarm positions in the cell
        with self._swarm.access():
            current_swarm_coords = self._swarm.data
            current_swarm_cells = self._swarm.particle_cellid.data[:,0]

        current_particle_cellID, current_ppc  = np.unique(current_swarm_cells, return_counts=True)

        ### find over-populated cells
        overpopulated_cells = current_particle_cellID[current_ppc > self._swarm_ppc]

        overpopulated_cell_coords = []

        for cell in overpopulated_cells:
            ### get the current particles in the cell
            current_particles = current_ppc[current_particle_cellID == cell][0]
            ### number of particles to delete
            no_particles_to_remove = current_particles - self._swarm_ppc
            ### randomly select particles to remove from the current cell
            #rng = np.random.default_rng()
    
            #### Selection of coords doesn't delete all particles (?)
            current_cell_coords  = current_swarm_coords[current_swarm_cells == cell]
            current_cell_coords_local = self._swarm.mesh._centroids[cell].reshape(1,-1)
            point_distance = distance.cdist(current_cell_coords,current_cell_coords_local, 'euclidean')
    
            np.min(point_distance, axis=1).argsort()[::-1][:int(no_particles_to_remove)]
            overpopulated_cell_coords.append(current_cell_coords[np.min(point_distance, axis=1).argsort()[::-1][:int(no_particles_to_remove)]])

        if overpopulated_cell_coords:  # Check if the list is not empty
            particles_to_remove = np.concatenate(overpopulated_cell_coords)
        else:
            particles_to_remove = np.empty(shape=(0,2))  # Create an empty numpy array

        # Check if all_current_coords and particles_to_remove are not empty
        with self._swarm.access(self._swarm.particle_coordinates):
            ### get current coords
            all_current_coords = self._swarm.data

            if particles_to_remove.size > 0:
                ### get number of particles to remove
                no_particles_to_remove = particles_to_remove.shape[0]

                    
                # Calculate point_distance
                point_distance = distance.cdist(all_current_coords, particles_to_remove, 'euclidean')

                # Get the index of particles to remove
                index_condition = np.min(point_distance, axis=1).argsort()[:no_particles_to_remove]

                # Set coords to 1.0e100 so they are deleted by the DM, following the same logic as the remeshing example
                self._swarm.data[index_condition] = 1.0e100

        return

# +
# # dq2dq1
# v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
# p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# q1dq0
v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1,continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0,continuous=False)

material_mesh = uw.discretisation.MeshVariable("M_mesh", mesh, 1, degree=1, continuous=True)
timeField     = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)

swarm  = uw.swarm.Swarm(mesh)
material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=1)  
fill_parameter= 3 # swarm fill parameter
swarm.populate_petsc(fill_param=fill_parameter,layout=uw.swarm.SwarmPICLayout.GAUSS)
#pop_control = uw.swarm.PopulationControl(swarm)
#pop_control_dm = PopulationControl_DeformMesh(swarm,swarm)

interfaceSwarm = uw.swarm.Swarm(mesh)
x = np.linspace(mesh.data[:,0].min(), mesh.data[:,0].max(), npoints)
y = offset + amplitude * np.cos(k * x)
interface_coords = np.ascontiguousarray(np.array([x,y]).T)
interfaceSwarm.add_particles_with_coordinates(interface_coords)

lightIndex = 0
denseIndex = 1
with swarm.access(material):
    perturbation = offset + amplitude * np.cos(k * swarm.particle_coordinates.data[:, 0])
    material.data[:, 0] = np.where(swarm.particle_coordinates.data[:, 1] < perturbation, lightIndex, denseIndex)

density_fn = material.createMask([densityI, densityD])
visc_fn = material.createMask([viscI,viscD])

def build_stokes_solver(mesh,v,p):
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])
    stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
    stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.shear_viscosity_0
    stokes.add_dirichlet_bc((0.0,0.0), "Left", (0,))
    stokes.add_dirichlet_bc((0.0,0.0), "Right", (0,))
    stokes.add_dirichlet_bc((0.0,0.0), "Bottom", (0,1))

    # if case_bc == "noslip":
    #     stokes.add_dirichlet_bc((0.0,0.0), "Left", (0,))
    #     stokes.add_dirichlet_bc((0.0,0.0), "Right", (0,))
    #     stokes.add_dirichlet_bc((0.0,0.0), "Top", (0,1))
    #     stokes.add_dirichlet_bc((0.0,0.0), "Bottom", (0,1))
    
    # if case_bc == "freeslip":
    #     stokes.add_dirichlet_bc((0.0,0.0), "Left", (0,))
    #     stokes.add_dirichlet_bc((0.0,0.0), "Right", (0,))
    #     stokes.add_dirichlet_bc((0.0,0.0), "Top", (1,))
    #     stokes.add_dirichlet_bc((0.0,0.0), "Bottom", (0,1))
        
    if uw.mpi.size == 1:
        stokes.petsc_options['pc_type'] = 'lu'
    
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["ksp_rtol"] = 1.0e-6
    stokes.petsc_options["ksp_atol"] = 1.0e-6
    stokes.petsc_options["snes_converged_reason"] = None
    stokes.petsc_options["snes_monitor_short"] = None
    return stokes
# -





# +
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
ax1.plot(interface_coords[:,0],interface_coords[:,1])

with interfaceSwarm.access():
    xcoord,ycoord = interfaceSwarm.data[:,0],interfaceSwarm.data[:,1]

fname = outputPath + "surfaceSwarm_test"
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
ax1.plot(xcoord,ycoord)
plt.savefig(fname,dpi=150,bbox_inches='tight')
plt.close()
# -

pop_control = PopulationControl_DeformMesh(swarm)

plot_meshswarm("swarm_01",mesh,swarm,material,showSwarm=True,showFig=True)


def _adjust_time_units(val):
    """ Adjust the units used depending on the value """
    if isinstance(val, u.Quantity):
        mag = val.to(u.years).magnitude
    else:
        val = dim(val, u.years)
        mag = val.magnitude
    exponent = int("{0:.3E}".format(mag).split("E")[-1])

    if exponent >= 9:
        units = u.gigayear
    elif exponent >= 6:
        units = u.megayear
    elif exponent >= 3:
        units = u.kiloyears
    elif exponent >= 0:
        units = u.years
    elif exponent > -3:
        units = u.days
    elif exponent > -5:
        units = u.hours
    elif exponent > -7:
        units = u.minutes
    else:
        units = u.seconds
    return val.to(units)


# +
from scipy.interpolate import interp1d
from scipy.interpolate import CloughTocher2DInterpolator


class FreeSurfaceProcessor(object): 
    def __init__(self,v,dt):
        """
        Parameters
        ----------
        _init_mesh : the original mesh
        mesh : the updating model mesh
        vel : the velocity field of the model
        dt : dt for advecting the surface
        """
        if mesh.dim == 2:
            self.init_mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))
        else:
            self.init_mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres),int(zres)), minCoords=(xmin, ymin, zmin), maxCoords=(xmax, ymax, zmax))
        self.Tmesh = uw.discretisation.MeshVariable("Tmesh", self.init_mesh, 1, degree=1)
        self.Bmesh = uw.discretisation.MeshVariable("Bmesh", self.init_mesh, 1, degree=1)
        self.mesh_solver = uw.systems.Poisson(self.init_mesh , u_Field=self.Tmesh, solver_name="FreeSurf_solver")
        self.mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
        self.mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
        self.mesh_solver.f = 0.0
        self.mesh_solver.add_dirichlet_bc(self.Bmesh.sym[0], "Top",0)
        self.mesh_solver.add_dirichlet_bc(self.Bmesh.sym[0], "Bottom",0)

        self.v = v
        self._dt   = dt
        self.top = topwall
        self.bottom = botwall

    def _solve_sle(self):
        self.mesh_solver.solve()
     
    def _advect_surface(self):
        if self.top.size > 0:
            if mesh.dim == 2:
                coords = mesh.data[self.top]
                x = coords[:,0]
                y = coords[:,-1]
                vx = uw.function.evalf(self.v.sym[0], coords)
                vy = uw.function.evalf(self.v.sym[1], coords)
                
                # Advect top surface
                x2 = x + vx * self._dt
                y2 = y + vy * self._dt
        
                # Spline top surface
                f = interp1d(x2, y2, kind='cubic', fill_value='extrapolate')
    
                with self.init_mesh.access(self.Bmesh):
                    self.Bmesh.data[self.bottom, 0] = mesh.data[self.bottom, -1]
                    self.Bmesh.data[self.top, 0] = f(x)        
             # TO DO : with mpi
             # need reordering y according x then interp?
            else:
                coords = mesh.data[self.top]
                x = coords[:,0]
                y = coords[:,1]
                z = coords[:,-1]
                vx = uw.function.evalf(self.v.sym[0], coords)
                vy = uw.function.evalf(self.v.sym[1], coords)
                vz = uw.function.evalf(self.v.sym[2], coords)
                
                # Advect top surface
                x2 = x + vx * self._dt
                y2 = y + vy * self._dt
                z2 = z + vz * self._dt
        
                # Spline top surface
                f = CloughTocher2DInterpolator((x2, y2), z2)
    
                with self.init_mesh.access(self.Bmesh):
                    self.Bmesh.data[self.bottom, 0] = mesh.data[self.bottom, -1]
                    self.Bmesh.data[self.top, 0] = f((x,y)) 
                
         # uw.mpi.barrier()
         # self.Bmesh.syncronise()
         
    def _update_mesh(self):
        with self.init_mesh.access():
            new_mesh_coords = self.init_mesh.data
            new_mesh_coords[:,-1] = self.Tmesh.data[:,0]   
        #mesh.deform_mesh(new_mesh_coords)
        return new_mesh_coords
    
    def solve(self):
        self._advect_surface()
        self._solve_sle()
        new_mesh_coords = self._update_mesh()
        return new_mesh_coords


# +
step      = 0
max_steps = 1
time      = 0
dt        = 0

if rank == 0:
    fw = open(outputPath + "ParticlePosition.txt","w")
    fw.write("Time \t W \t dWdT \n")
    fw.close()
uw.mpi.barrier()
w = []
dwdt = []
times = []

while time < max_time:
#while step < max_steps:
    
    if uw.mpi.rank == 0:
        string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
        step, _adjust_time_units(time),
        _adjust_time_units(dt),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.stdout.write(string)
        sys.stdout.flush()
    
    stokes = build_stokes_solver(mesh,v,p)
    stokes.solve(zero_init_guess=False)

    with interfaceSwarm.access():
        particle_coordinates = interfaceSwarm.particle_coordinates.data
        w = -(particle_coordinates[0][1]+L)
        dwdt = -uw.function.evaluate(v.fn,particle_coordinates)[0,1]

        fw = open(outputPath + "ParticlePosition.txt","a")
        fw.write("%.4e \t %.4e \t %.4e \n" %(time,w,dwdt))
        fw.close()


    if step%save_every ==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data:')
        with mesh.access(timeField):
            timeField.data[:,0] = dim(time, u.megayear).m

        # with surfaceSwarm.access():
        #     xcoord,ycoord = surfaceSwarm.data[:,0],surfaceSwarm.data[:,1]
        # fname = outputPath + "surfaceSwarm_step"+str(step)
        # #import matplotlib.pyplot as plt
        # fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
        # ax1.plot(xcoord,ycoord)
        # plt.savefig(fname,dpi=150,bbox_inches='tight')
        # plt.close()

        mesh.petsc_save_checkpoint(index=step, outputPath=outputPath+"mesh.")
        mesh.petsc_save_checkpoint(meshVars=[v, p, timeField], index=step, outputPath=outputPath)
        swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputPath) 
        swarm.write_timestep(filename='material',swarmname='swarm',index=step,swarmVars=[material],outputPath=outputPath)
        interfaceSwarm.petsc_save_checkpoint(swarmName='interfaceSwarm', index=step, outputPath=outputPath) 

        #filename = "mat_step"+str(step)+"_time_"+str(np.round(dim(time,u.megayear).m,3))+"Ma"
        #plot_meshswarm(filename,mesh,swarm,material,showSwarm=True,showFig=False)
        #plot_mat(filename,figshow=False)

    times.append(time)
    dt_solver = stokes.estimate_dt()
    dt = min(dt_solver,dt_set)
    
    swarm.advection(V_fn=stokes.u.sym, delta_t=dt)
    interfaceSwarm.advection(V_fn=stokes.u.sym, delta_t=dt)

    freesuface = FreeSurfaceProcessor(v,dt)
    new_mesh_coords=freesuface.solve()
    mesh.deform_mesh(new_mesh_coords)
    pop_control.repopulate(mesh,material)
    #repopulate(swarm,mesh,updateField=material)
    #pop_control.redistribute(material)
    #pop_control.repopulate(material)

    step += 1
    time += dt
# -



#plot_meshswarm("swarm_test",mesh,swarm,material,showSwarm=True,showFig=True)







# +
# index = step
# filename =  outputPath+ f"mesh.{index:05}.h5"
# mesh.write(filename=filename, index=step)
# -



# +
data = swarm.mesh.data

x1 = data[topwall,0]
y1 = data[topwall,1]
zipxy = zip(x1,y1)
zipxy = sorted(zipxy)
x1,y1 = zip(*zipxy) 

title = "mesh topwall_uw3" 
fig, ax = plt.subplots(nrows=1, figsize=(5,3))
ax.set_title(title)
ax.plot(y1,'--r',label = 'new_mesh')
#ax.axhline(0.05,color='black',linestyle='-')
ax.legend(loc='best',prop = {'size':8})
# -

dt,dt_solver,dt_set

# +
# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
# ax1.plot(interface_coords[:,0],interface_coords[:,1])

with interfaceSwarm.access():
    xcoord,ycoord = interfaceSwarm.data[:,0],interfaceSwarm.data[:,1]

title = "interface_uw3" 
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(nrows=1, figsize=(5,3))
ax1.set_title(title)
ax1.plot(xcoord,ycoord)
ax1.set_ylim([-1.06,-0.94])
# plt.savefig(fname,dpi=150,bbox_inches='tight')
# plt.close()
# -

plot_meshswarm("swarm_test",mesh,swarm,material,showSwarm=True,showFig=True)

# +
x1 = mesh.data[topwall,0]
y1 = mesh.data[topwall,1]
zipxy = zip(x1,y1)
zipxy = sorted(zipxy)
x1,y1 = zip(*zipxy) 

x0 = init_mesh.data[topwall,0]
y0 = init_mesh.data[topwall,1]
zipxy = zip(x0,y0)
zipxy = sorted(zipxy)
x0,y0 = zip(*zipxy) 


x2 = swarm.mesh.data[topwall,0]
y2 = swarm.mesh.data[topwall,1]
zipxy = zip(x2,y2)
zipxy = sorted(zipxy)
x2,y2 = zip(*zipxy) 

#fname = "Topography of the box mid"
title = "mesh topwall_uw3" 
fig, ax = plt.subplots(nrows=1, figsize=(5,3))
ax.set_title(title)
ax.plot(y1,'--r',label = 'new_mesh')
ax.plot(y0,'-k',label='old_mesh')
ax.plot(y2,'-.b',label='swarm_mesh')
#ax.axhline(0.05,color='black',linestyle='-')
ax.legend(loc='best',prop = {'size':8})
# -

plot_mesh0()

if rank == 0:
    import matplotlib.pyplot as plt
    data = np.loadtxt(outputPath + "ParticlePosition.txt",skiprows=1)
    arrTime = data[:,0]
    arrW = data[:,1]
    arrdWdt = data[:,2]

    #fig, ax1 = plt.subplots(nrows=1, figsize=(6.7,5))
    fname = "w dwdt versus time"
    plt.clf()
    plt.plot(arrTime,arrW,label='w tracer')
    #plt.plot(arrTime,amplitudes,label='w fourier')
    plt.plot(arrTime,arrdWdt,label='dw/dt')
    plt.xlabel('Time')
    plt.ylabel('Perturbation Displacement / Velocity')
    plt.legend(loc=2)
    plt.savefig(outputPath+fname,dpi=150,bbox_inches='tight')

    fname = "dwdt versus w"
    plt.clf()
    plt.scatter(arrW,arrdWdt,label='dw/dt')
    plt.xlabel('w')
    plt.ylabel('dWdt')
    plt.legend(loc=2)
    plt.savefig(outputPath+fname,dpi=150,bbox_inches='tight')


    arrTau = arrdWdt / arrW
    # k = 2 * pi * n / (domain width)
    k = 2. * np.pi * perturbation_n[0]/(xmax-xmin) * L
    print("Tau at each time: ")
    print(arrTau)
    avTau = np.average(arrTau)
    print("Average Tau is %.2e" %avTau)


def analytic_growthrate(k):
    q = (np.cosh(k) * np.sinh(k) - k)/(k**2. + np.cosh(k)**2.) / (2.*k)
    return q
import scipy
fitfn = lambda t,a,b,c: a+b*np.exp(c*t)
import scipy.optimize
# a decent first guess is important, as np.exp can explode easily.
guess = (0., perturbation_a[0], 2.*analytic_growthrate(k))  
#fit = scipy.optimize.curve_fit(fitfn,  times,  amplitudes, p0=guess)

# +
# import psutil
# process = psutil.Process()
# virtual_memory = psutil.virtual_memory()
# print(f"Total Available Memory: {virtual_memory.total / (1024 ** 3):.2f} GB")
# print(f"Used Memory: {virtual_memory.used / (1024 ** 3):.2f} GB")
# print(f"Free Memory: {virtual_memory.available / (1024 ** 3):.2f} GB")
# print(f"Percent Used: {virtual_memory.percent:.2f}%")

# print(f"Jupyter Notebook Process ID: {process.pid}")
# print(f"Jupyter Notebook Memory Usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")
# -

if rank == 0:
    plt.clf()
    
    # The solution is undefined at 0, so start at 0.01
    arrK = np.linspace(0.01,10,100)
    arrATau = np.zeros(len(arrK))
    plt.xlim(0,10)
    plt.ylim(0,0.2)
    for i in range(len(arrK)):
        arrATau[i] = analytic_growthrate(arrK[i])

    fname = "Growth Rate versus k"
    plt.plot(arrK,arrATau,label="Analytic Solution")
    plt.scatter(k,avTau,label="Numerical Calculation (tracer)")
    #plt.scatter(k,fit[0][2],label="Numerical Calculation (Fourier)")
    plt.legend()
    plt.xlabel('Wavenumber (proportional to frequency) ' + r'$k = 2 \pi L / \lambda$')
    plt.ylabel('Growth Rate ' + r'$\tau$')
    plt.savefig(outputPath+fname,dpi=150,bbox_inches='tight')

    fname = "Growth Rate versus time"
    plt.clf()
    plt.scatter(arrTime,arrTau,label="Numerical Calculation (tracer)")
    plt.xlabel('Time')
    plt.ylabel('Growth ate ' + r'$\tau$')
    plt.savefig(outputPath+fname,dpi=150,bbox_inches='tight')










