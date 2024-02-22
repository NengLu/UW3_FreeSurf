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

# Loading and unloading of a viscous half-space
#
# see the similar case in uw2: [ViscoElasticHalfSpace](https://github.com/underworldcode/underworld2/blob/master/docs/UWGeodynamics/examples/1_08_ViscoElasticHalfSpace.ipynb)
#
# ======
#
# Loading of Earth's surface can be described with an initial periodic surface displacement of a viscous fluid within an infinite half space, the solution of which is outlined in Turcotte and Schubert (1982), 6.10 Postglacial Rebound.  The surface decreases exponentially with time and is dependent on the magnitude, $w_m$, and wavelength $\lambda$ of the perturbation, and the viscosity, $\eta$ and density, $\rho$ of the fluid,
#
# $$ w = w_m exp\Big(\frac{-\lambda \rho g t}{4\pi\eta}\Big) $$
#
# where $w$ is displacement, $w_m$ the initial load magnitude, $g$ gravity, $t$ time. This solution can be charaterised by the relaxation time, $t_{relax} = 4\pi\eta / \rho g \lambda $, the time taken for the initial load to decrease by $e^{-1}$. The solution for an elastic material with the equivalent load produces the same magnitude of displacement instantaneously.
#

#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
#from underworld3.systems import Stokes
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

from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local

# +
u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

# scaling 3: vel
half_rate = 1.0 * u.centimeter / u.year
model_length = 100. * u.kilometer
bodyforce = 3300 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM

# +
yres = 8
xres = yres*4
npoints = 50

xmin, xmax = ndim(-200 * u.kilometer), ndim(200 * u.kilometer)
ymin, ymax = ndim(-100 * u.kilometer), ndim(0 * u.kilometer)

eta = ndim(1e21  * u.pascal * u.second)
density = ndim(3300 * u.kilogram / u.metre**3)
gravity = ndim(9.81 * u.meter / u.second**2)
w_m    =   ndim(5.0 * u.kilometer)
Lambda = ndim(100.0 * u.kilometer) 

densityM = density
viscM = eta
ND_gravity = gravity

def perturbation(x):
    return w_m * np.cos(2.*np.pi*(x)/Lambda)

# analytic solution
xMax = xmax - xmin
x = np.linspace(0, xMax, 200+1)
w_0 = perturbation(x)
t_relax = 4 * np.pi * eta / (Lambda * density * gravity)
tMax = t_relax * 5 
t = np.linspace(0, tMax, 100 * 10 + 1)
w_t = w_m * np.exp(-1.*t/t_relax)

max_time =  tMax
#max_time =  t_relax*1e-2*2
dt_set = t_relax*1e-2
save_every = 5

outputPath = "op_Ex_TopoRelaxation_FreeSurface_uw3_swamrepo_mpi_yres_test"+str(yres)+"/"
if uw.mpi.rank == 0:
    #delete previous model run
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+ i)
            
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

# +
# #!pip install scipy

# +
# petsc_dm_find_labeled_points_local

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


# +
mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))      
init_mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))

# # dq2dq1
# v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
# p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# q1dq0
v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1,continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0,continuous=False)
timeField     = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)

# botwall = getVertexSet(mesh,'Bottom')
# topwall = getVertexSet(mesh,'Top')

botwall = petsc_dm_find_labeled_points_local(mesh.dm,"Bottom")
topwall = petsc_dm_find_labeled_points_local(mesh.dm,"Top")
def find_ind(value):
    topwall_x = mesh.data[topwall,0]
    idx = np.abs(topwall_x-value).argmin()
    return idx
idx = find_ind(0)
topwallmid_Ind = topwall[idx] 

# +
# if uw.mpi.rank == 0:
#     plot_mesh("mesh0",mesh)

# +
# Tmesh = uw.discretisation.MeshVariable("Tmesh", init_mesh, 1, degree=1)
# Bmesh = uw.discretisation.MeshVariable("Bmesh", init_mesh, 1, degree=1)

# mesh_solver = uw.systems.Poisson(init_mesh, u_Field=Tmesh, solver_name="FreeSurf_solver")
# mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
# mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
# mesh_solver.f = 0.0
# mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Top",0)
# mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Bottom",0)

# x = init_mesh.data[topwall,0]
# with init_mesh.access(Bmesh):
#     Bmesh.data[topwall, 0] = perturbation(x)
#     Bmesh.data[botwall, 0] = init_mesh.data[botwall,-1]
# mesh_solver.solve()

# def update_mesh():
#     with init_mesh.access():
#         new_mesh_coords = init_mesh.data
#         new_mesh_coords[:,-1] = Tmesh.data[:,0]
#     return new_mesh_coords
# new_mesh_coords = update_mesh()
# mesh.deform_mesh(new_mesh_coords)
# #update_mesh(mesh)
#plot_mesh("mesh01",mesh)

# +
swarm  = uw.swarm.Swarm(mesh)
material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=1, proxy_degree=1)  
fill_parameter= 2 # swarm fill parameter
swarm.populate_petsc(fill_param=fill_parameter,layout=uw.swarm.SwarmPICLayout.GAUSS)
# pop_control = uw.swarm.PopulationControl(swarm)

MIndex = 0
with swarm.access(material):
    material.data[:] = MIndex

density_fn = material.createMask([densityM])
visc_fn = material.createMask([viscM])

def build_stokes_solver(mesh,v,p):
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])
    stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
    stokes.add_dirichlet_bc((0.0,0.0), "Left", (0,))
    stokes.add_dirichlet_bc((0.0,0.0), "Right", (0,))
    stokes.add_dirichlet_bc((0.0,0.0), "Bottom", (0,1))
    
    # if uw.mpi.size == 1:
    #     stokes.petsc_options['pc_type'] = 'lu'
    
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["ksp_rtol"] = 1.0e-6
    stokes.petsc_options["ksp_atol"] = 1.0e-6
    stokes.petsc_options["snes_converged_reason"] = None
    stokes.petsc_options["snes_monitor_short"] = None
    return stokes
#stokes = build_stokes_solver(mesh,v,p)

# stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
# stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])
# stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
# stokes.add_dirichlet_bc((0.0,0.0), "Left", (0,))
# stokes.add_dirichlet_bc((0.0,0.0), "Right", (0,))
# stokes.add_dirichlet_bc((0.0,0.0), "Bottom", (0,1))

# if uw.mpi.size == 1:
#     stokes.petsc_options['pc_type'] = 'lu'

# stokes.tolerance = 1.0e-6
# stokes.petsc_options["ksp_rtol"] = 1.0e-6
# stokes.petsc_options["ksp_atol"] = 1.0e-6
# stokes.petsc_options["snes_converged_reason"] = None
# stokes.petsc_options["snes_monitor_short"] = None


# +
# plot_mesh('swarm0',mesh,showSwarm=True)

# +
Tmesh = uw.discretisation.MeshVariable("Tmesh", init_mesh, 1, degree=1)
Bmesh = uw.discretisation.MeshVariable("Bmesh", init_mesh, 1, degree=1)

mesh_solver = uw.systems.Poisson(init_mesh, u_Field=Tmesh, solver_name="FreeSurf_solver")
mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
mesh_solver.f = 0.0
mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Top",0)
mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Bottom",0)

x = init_mesh.data[topwall,0]
with init_mesh.access(Bmesh):
    Bmesh.data[topwall, 0] = perturbation(x)
    Bmesh.data[botwall, 0] = init_mesh.data[botwall,-1]
mesh_solver.solve()

def update_mesh():
    with init_mesh.access():
        new_mesh_coords = init_mesh.data
        new_mesh_coords[:,-1] = Tmesh.data[:,0]
    return new_mesh_coords
new_mesh_coords = update_mesh()
mesh.deform_mesh(new_mesh_coords)
#update_mesh(mesh)
#if uw.mpi.rank == 0:
#    plot_mesh("mesh01",mesh)
#stokes = build_stokes_solver(mesh,v,p)
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
# if uw.mpi.rank == 0:
#     plot_meshswarm("swarm_norepo",mesh,swarm,material,showSwarm=True,showFig=True)
# -

pop_control = PopulationControl_DeformMesh(swarm)

# +
# with pop_control._swarm.access():
#     current_swarm_coords = pop_control._swarm.data
#     current_swarm_cells = pop_control._swarm.particle_cellid.data[:,0]
# current_particle_cellID, current_ppc  = np.unique(current_swarm_cells, return_counts=True)
# current_particle_cellID, current_ppc

# +
# pop_control.repopulate(mesh,material)
# with pop_control._swarm.access():
#     current_swarm_coords = pop_control._swarm.data
#     current_swarm_cells = pop_control._swarm.particle_cellid.data[:,0]
# current_particle_cellID, current_ppc  = np.unique(current_swarm_cells, return_counts=True)
# current_particle_cellID, current_ppc
# -

pop_control.repopulate(mesh,material)
# if uw.mpi.rank == 0:
#     plot_meshswarm("swarm_repo",mesh,swarm,material,showSwarm=True,showFig=True)

# +
## some test

# with swarm.access():
#     current_swarm_coords = swarm.data
#     current_swarm_cells = swarm.particle_cellid.data[:,0]

# current_particle_cellID, current_ppc  = np.unique(current_swarm_cells, return_counts=True)

# ### find over-populated cells
# overpopulated_cells = current_particle_cellID[current_ppc > pop_control._swarm_ppc]


# current_particle_cellID, current_ppc 
# -

# ### repopulate 
# ### similar to Ben's PopControl.redistribute
#
# - **_update the the swarm's owning cell_**
# - add particles to the underpopulated_cells from the new swarm build on the deformed mesh
# - **_delete particles to the overpopulated cells which are closed to each other_** (random in redistribute)





# +
# cellid = swarm.dm.getField("DMSwarm_cellid")
# coords = swarm.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm.dim))
# cellid[:] = swarm.mesh.get_closest_cells(coords).reshape(-1)
# swarm.dm.restoreField("DMSwarmPIC_coor")
# swarm.dm.restoreField("DMSwarm_cellid")
# # now migrate.
# swarm.dm.migrate(remove_sent_points=True)
# # void these things too
# swarm._index = None
# swarm._nnmapdict = {}

# plot_meshswarm("swarm_newnore",mesh,swarm,material,showSwarm=True,showFig=True)
# -

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
from scipy.interpolate import pchip_interpolate

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
                #fx = pchip_interpolate(x2,y2,x)
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
        uw.mpi.barrier()
        #self.Bmesh.syncronise()
         
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
max_steps = 5
time      = 0
dt        = 0

# if rank == 0:
#     fw = open(outputPath + "ParticlePosition.txt","w")
#     fw.write("Time \t W \t dWdT \n")
#     fw.close()
# uw.mpi.barrier()
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

    #particle_coordinates = mesh.data[topwallmid_Ind]
    #w = particle_coordinates[1]
    #dwdt = uw.function.evaluate(v.fn,mesh.data[topwallmid_Ind:topwallmid_Ind+1,:])[0,1]
    #print(dwdt)
  
    #fw = open(outputPath + "ParticlePosition.txt","a")
    #fw.write("%.4e \t %.4e \t %.4e \n" %(time,w,dwdt))
    #fw.close()
    #amplitudes.append(get_amplitudes()[0])
    times.append(time)

    if step%save_every ==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data:')
        with mesh.access(timeField):
            timeField.data[:,0] = dim(time, u.megayear).m
        
        mesh.petsc_save_checkpoint(meshVars=[v, p, timeField], index=step, outputPath=outputPath)
        #swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputPath) 
        #interfaceSwarm.petsc_save_checkpoint(swarmName='interfaceSwarm', index=step, outputPath=outputPath) 

        filename = "mat_step"+str(step)+"_time_"+str(np.round(dim(time,u.megayear).m,3))+"Ma"
        #plot_mesh(filename,mesh,showSwarm=True,showFig=False)
        #plot_mat(filename,figshow=False)
    
    dt_solver = stokes.estimate_dt()
    dt = min(dt_solver,dt_set)

    #swarm.advection(V_fn=stokes.u.sym, delta_t=dt,order=2)
    #interfaceSwarm.advection(V_fn=stokes.u.sym, delta_t=dt,order=2)
    
    # coords = mesh.data[topwall]
    # vx = uw.function.evalf(v.sym[0], coords)
    # vy = uw.function.evalf(v.sym[1], coords)
    freesuface = FreeSurfaceProcessor(v,dt)
    new_mesh_coords=freesuface.solve()
    mesh.deform_mesh(new_mesh_coords)
    #repopulate(swarm,mesh,updateField=material)
    #pop_control.repopulate(mesh,material)
    #pop_control.repopulate(material)

    # #in uw2
    # advector.integrate(dt, update_owners=False)
    # freesuface =  FreeSurfaceProcessor_MV()
    # freesuface.solve(dt)
    # swarm.update_particle_owners()
    # pop_control.repopulate()

             
    step += 1
    time += dt
# -








