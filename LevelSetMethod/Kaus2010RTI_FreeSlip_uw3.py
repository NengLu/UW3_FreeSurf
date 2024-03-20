#!/usr/bin/env python
# coding: utf-8

# Rayleigh-Taylor instability
# This notebook models the Rayleigh-Taylor instability outlined in Kaus et al. (2010), for further free surface and FSSA implementation tests.
# 
# ### References
# Kaus, B. J., Mühlhaus, H., & May, D. A. (2010). A stabilization algorithm for geodynamic numerical simulations with a free surface. Physics of the Earth and Planetary Interiors, 181(1-2), 12-20.

# In[1]:


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


# In[2]:


outputPath = 'output_Kaus2010RTI_FreeSlip_uw3_GAUSS/'
if uw.mpi.size == 1:
    ## delete previous model run
    # if os.path.exists(outputPath):
    #     for i in os.listdir(outputPath):
    #         os.remove(outputPath+ i)
            
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)


# In[3]:


u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

# scaling 3: vel
H = 100.  * u.kilometer
velocity     = 1e-9 * u.meter / u.second
g    =   9.81 * u.meter / u.second**2 
bodyforce    = 3300  * u.kilogram / u.metre**3 * g 
mu           = 1e21  * u.pascal * u.second

KL = H
Kt = KL / velocity
KM = mu * KL * Kt

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM


ND_gravity = ndim(9.81 * u.meter / u.second**2)


# In[4]:


render = True  # plot images

xres,yres = 50,50 

xmin, xmax = ndim(-250 * u.kilometer), ndim(250 * u.kilometer)
ymin, ymax = ndim(-500 * u.kilometer), ndim(0 * u.kilometer)
mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))
#meshbox = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax),cellSize=1.0/res,regular=False,qdegree=2,)


# In[5]:


def plot_mesh0():
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
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)
    pl.show(cpos="xy")
plot_mesh0()


# In[6]:


mesh.view()


# In[7]:


v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
material_mesh = uw.discretisation.MeshVariable("M_mesh", mesh, 1, degree=1, continuous=True)
timeField     = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

swarm  = uw.swarm.Swarm(mesh)
#materialVariable  = swarm.add_variable(name="material", size=1, dtype=PETSc.IntType, proxy_degree=1)
material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=1)  
# ax1.plot(times1/1000,depth1_l,'-k',label = "FreeSurface_uw2 dt=2500 yr")
#swarm.populate(fill_param=swarmGPC)  # with k-d tree
swarmGPC = 3 # swarm fill parameter
swarm.populate_petsc(fill_param=swarmGPC,layout=uw.swarm.SwarmPICLayout.GAUSS)

lightIndex = 0
denseIndex = 1
amplitude = ndim(5*u.kilometer)
offset = ndim(-100.*u.kilometer)
wavelength = ndim(500.*u.kilometer)
k = 2.0 * np.pi / wavelength

surfaceSwarm = uw.swarm.Swarm(mesh)
npoints = 101
x = np.linspace(mesh.data[:,0].min(), mesh.data[:,0].max(), npoints)
y = offset + amplitude * np.cos(k * x)
surface_coords = np.ascontiguousarray(np.array([x,y]).T)
surfaceSwarm.add_particles_with_coordinates(surface_coords)

with swarm.access(material):
    perturbation = offset + amplitude * np.cos(k * swarm.particle_coordinates.data[:, 0])
    material.data[:, 0] = np.where(swarm.particle_coordinates.data[:, 1] < perturbation, lightIndex, denseIndex)

# densityI = ndim(3200 * u.kilogram / u.metre**3)
# densityD = ndim(3300 * u.kilogram / u.metre**3)
# density_fn = material.createMask([densityI, densityD])

# drho - backgroud_density
densityI = ndim(3200 * u.kilogram / u.metre**3)
densityD = ndim(3300 * u.kilogram / u.metre**3)
density_fn = material.createMask([densityI, densityD])

viscI = ndim(1e20 * u.pascal * u.second)
viscD = ndim(1e21 * u.pascal * u.second)
visc_fn = material.createMask([viscI,viscD])


ND_gravity = ndim(9.81 * u.meter / u.second**2)
stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])
#stokes.bodyforce =  sympy.Matrix([0, -1])
#stokes.bodyforce =  sympy.Matrix([0, -1 *density_fn])

stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.shear_viscosity_0


stokes.add_dirichlet_bc((0.0,0.0), "Left", (0,))
stokes.add_dirichlet_bc((0.0,0.0), "Right", (0,))
stokes.add_dirichlet_bc((0.0,0.0), "Top", (1,))
stokes.add_dirichlet_bc((0.0,0.0), "Bottom", (0,1))

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

#snes_rtol = 1.0e-5
#stokes.tolerance = snes_rtol
stokes.petsc_options["ksp_rtol"] = 1.0e-6
stokes.petsc_options["ksp_atol"] = 1.0e-6
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None

# stokes.petsc_options["ksp_type"] = "gmres"
# stokes.petsc_options["ksp_rtol"] = 1.0e-9
# stokes.petsc_options["ksp_atol"] = 1.0e-12
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 1.0e-8
# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-8
# stokes.petsc_options["snes_atol"] = 0.1 * snes_rtol # by inspection
# stokes.petsc_options["ksp_monitor"] = None

#stokes.solve(zero_init_guess=False)
#stokes.solve(zero_init_guess=True) #stokes.solve(zero_init_guess=False)ax1.plot(times0/1000,depth0_l,'--k',label = "FreeSlip_uw2")
# ax1.plot(times1/1000,depth1_l,'-k',label = "FreeSurface_uw2 dt=2500 yr")


# In[8]:


# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
# ax1.plot(surface_coords[:,0],surface_coords[:,1])


# with surfaceSwarm.access():
#     xcoord,ycoord = surfaceSwarm.data[:,0],surfaceSwarm.data[:,1]

# fname = outputPath + "surfaceSwarm_test"
# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
# ax1.plot(xcoord,ycoord)
# plt.savefig(fname,dpi=150,bbox_inches='tight')
# plt.close()


# In[9]:


def plot_mat(filename,figshow = True):
    import numpy as np
    import pyvista as pv
    import vtk


    figsize = [750, 750]
    pv.global_theme.background = 'white'
    pv.global_theme.window_size = figsize 
    pv.global_theme.anti_aliasing = None #"ssaa", "msaa", "fxaa", or None
    #"static", "client", "server", "trame", "none"
    pv.global_theme.jupyter_backend = 'static'
    pv.global_theme.smooth_shading = True 
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]
    # pl.camera_position = 'xy'
    # pl.camera_zoom = 1.0  # Adjust the zoom level if needed

    mesh.vtk("tempMsh.vtk")
    pvmesh = pv.read("tempMsh.vtk") 

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
    point_cloud = pv.PolyData(points)
    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()
    with surfaceSwarm.access():
        surfaceCloud = pv.PolyData(np.vstack((surfaceSwarm.data[:,0],surfaceSwarm.data[:,1], np.zeros(len(surfaceSwarm.data)))).T)


    v_dpoint = 2
    v_vectors = np.zeros((mesh.data[::v_dpoint].shape[0], 3))
    v_vectors[:, 0] = uw.function.evalf(v.sym[0], mesh.data[::v_dpoint], mesh.N)
    v_vectors[:, 1] = uw.function.evalf(v.sym[1], mesh.data[::v_dpoint], mesh.N)
    v_max = v_vectors.max()
    v_vectors = v_vectors/v_max
    
    v_points = np.zeros((mesh.data[::v_dpoint].shape[0], 3))
    v_points[:,0] = mesh.data[::v_dpoint][:,0]
    v_points[:,1] = mesh.data[::v_dpoint][:,1]
    
    # pvstream = pvmesh.streamlines_from_source(
    #     point_cloud,
    #     vectors="V",
    #     integration_direction="both",
    #     max_steps=10,
    #     surface_streamlines=True,
    #     max_step_length=0.05,
    # )
    pl = pv.Plotter(window_size=figsize)
    #pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_points(point_cloud, cmap="coolwarm",render_points_as_spheres=False, point_size=20, opacity=0.5)

    #pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",use_transparency=False, opacity=0.95)
    pl.remove_scalar_bar("M")
    pl.add_points(surfaceCloud, color='black',render_points_as_spheres=False, point_size=10, opacity=0.5)
    
    #pl.add_mesh(v_points)
    arrows = pl.add_arrows(v_points,v_vectors, color="black",mag=0.05,opacity=1.0, show_scalar_bar=False)
    
    #streamlines = pl.add_mesh(pvstream, opacity=0.25)
    # streamlines.SetVisibility(False)
    pl.screenshot(filename="{}.png".format(filename),window_size=figsize,return_img=False) 
    if figshow:
        pl.show(cpos="xy")
    pvmesh.clear_data()
    pvmesh.clear_point_data()
if render == True:
    plot_mat("test",figshow=True)


# In[10]:


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


# In[11]:


step      = 0
max_steps = 2
time      = 0
dt        = 0
dt_set    = ndim(5e4*u.year)
#timing setup
#viewer.getTimestep()
#viewer.setTimestep(1)

#while step<max_steps:
w = []
dwdt = []
times = []
while time < ndim(6.0*u.megayear):
    
    if uw.mpi.rank == 0:
        string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
        step, _adjust_time_units(time),
        _adjust_time_units(dt),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.stdout.write(string)
        sys.stdout.flush()
    
    save_every=1
    if step%save_every ==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data:')
        ### updates projection of fields to the mesh
        #updateFields()
        ## saves the mesh and swarm
        with mesh.access(timeField):
            timeField.data[:,0] = dim(time, u.megayear).m

        with surfaceSwarm.access():
            xcoord,ycoord = surfaceSwarm.data[:,0],surfaceSwarm.data[:,1]
        
        fname = outputPath + "surfaceSwarm_step"+str(step)
        #import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
        ax1.plot(xcoord,ycoord)
        plt.savefig(fname,dpi=150,bbox_inches='tight')
        plt.close()
        
        mesh.petsc_save_checkpoint(meshVars=[v, p, timeField], index=step, outputPath=outputPath)
        swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputPath) 
        surfaceSwarm.petsc_save_checkpoint(swarmName='surfaceSwarm', index=step, outputPath=outputPath) 
    
    # filename = outputPath+"mat_step"+str(step)+"_time_"+str(np.round(dim(time,u.megayear).m,2))+"Ma"
    # plot_mat(filename,figshow=False)
    with surfaceSwarm.access():
          particle_coordinates = surfaceSwarm.particle_coordinates.data
          w_ = - (particle_coordinates[0][1]-offset)
    dwdt_ = -uw.function.evaluate(v.fn,particle_coordinates)[0,1]
    w.append(w_)
    dwdt.append(dwdt_)
    times.append(time)
        
    stokes.solve(zero_init_guess=False)
    dt_solver = stokes.estimate_dt()
    dt = min(dt_solver,dt_set)
    swarm.advection(V_fn=stokes.u.sym, delta_t=dt)
    surfaceSwarm.advection(V_fn=stokes.u.sym, delta_t=dt)
    step += 1
    time += dt


# In[12]:


plot_mat("test_final",figshow=True)


# In[13]:


cab = "times;w;dwdt"
fname_save = outputPath + "ParticlePosition.txt"
data_save = np.column_stack((times,w,dwdt))
np.savetxt(fname_save,data_save,fmt='%3.8f %3.8f %3.8f', header=cab) 


# In[14]:


plot_mat(outputPath+"test_final")


# In[15]:


plot_mesh0()


# In[ ]:





# In[ ]:




