#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld as uw
from underworld import function as fn
import underworld.visualisation as vis
import math
import numpy as np
import os


# In[2]:


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


# In[3]:


run = [0,3]

cases_bc = ["freesurf","freesurf_fixVx"]  
cases_swarm = ["gv","gg","sv","sg"] # guassfill/spacefill, voronori integration/guass integration

case_bc = cases_bc[run[0]]
case_swarm = cases_swarm[run[1]] 

render = True
max_time  = ndim(6.0*u.megayear)
dt_set    = ndim(2.5e3*u.year)
save_every = 20

yres = 32  
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

outputPath = "op_uw2_" +case_bc+"_"+ case_swarm +"_yres{:n}_wl{:n}_boxl{:n}/".format(yres,wavelength,boxl)
if uw.mpi.size == 1:
    ## delete previous model run
    # if os.path.exists(outputPath):
    #     for i in os.listdir(outputPath):
    #         os.remove(outputPath+ i)
            
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

print(ndim(max_time),ndim(dt_set),ndim(bodyforce))
print(outputPath)


# In[4]:


mesh = uw.mesh.FeMesh_Cartesian(elementType=("Q1/dQ0"), elementRes=(xres,yres), minCoord=(xmin, ymin),maxCoord=(xmax, ymax))


# In[5]:


velocityField = mesh.add_variable(nodeDofCount=2)
pressureField = mesh.subMesh.add_variable(nodeDofCount=1)
timeField= mesh.add_variable(nodeDofCount=1)
velocityField.data[:] = [0.0, 0.0]
pressureField.data[:] = 0.0
timeField.data[...] = 0.0

swarm = uw.swarm.Swarm(mesh=mesh)
materialIndex = swarm.add_variable(dataType="int", count=1)
# swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm=swarm, particlesPerCell=swarmGPC)
# swarm.populate_using_layout(layout=swarmLayout)

particlesPerCell = 16
gaussPointCount = 4
if case_swarm =="gv" or case_swarm =="gg":
    swarmLayout = uw.swarm.layouts.PerCellGaussLayout( swarm=swarm, gaussPointCount=gaussPointCount)
elif case_swarm =="sv" or case_swarm =="sg":
    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=particlesPerCell)
else:
    print("no layout type")
swarm.populate_using_layout(layout=swarmLayout)
pop_control = uw.swarm.PopulationControl(swarm,aggressive=True,splitThreshold=0.15, maxDeletions=2,maxSplits=10,particlesPerCell=particlesPerCell)

lightIndex = 0
denseIndex = 1

coord = fn.coord()
perturbationFn = offset + amplitude * fn.math.cos(k * coord[0])
conditions = [(perturbationFn > coord[1], lightIndex), (True, denseIndex)]
materialIndex.data[:] = fn.branching.conditional(conditions).evaluate(swarm)

densityMap = {lightIndex: densityI, denseIndex: densityD}
densityFn = fn.branching.map(fn_key=materialIndex, mapping=densityMap)

viscosityMap = {lightIndex: viscI , denseIndex:viscD}
fn_viscosity = fn.branching.map(fn_key=materialIndex, mapping=viscosityMap)

interfaceSwarm = uw.swarm.Swarm(mesh)
x = np.linspace(mesh.data[:,0].min(), mesh.data[:,0].max(), npoints)
y = offset + amplitude * np.cos(k * x)
interface_coords = np.ascontiguousarray(np.array([x,y]).T)
interfaceSwarm.add_particles_with_coordinates(interface_coords)

markerSwarm = uw.swarm.Swarm( mesh=mesh)
x_marker = interface_coords[0][0]
y_marker = interface_coords[0][1]
markerSwarm.add_particles_with_coordinates(np.array([(x_marker,y_marker)]))

# def InterfaceY(x,arrN,arrA):
#     pertX = 0.
#     for i in range(len(arrN)):
#         pertX += arrA[i] * -uw.function.math.cos(2.*np.pi * float(arrN[i]) / (xmax-xmin) * x )
#     interface = ymax - L + pertX
#     return interface  
# fn_height_integrand = uw.function.branching.map(materialIndex, {asth_id:1.,lith_id:0.})
# x = fn.input()[0]
# fn_mode = -uw.function.math.cos(x*2.*np.pi * perturbation_n / (xmax-xmin) )
# def get_amplitudes():
#     voronoiSwarm = uw.swarm.VoronoiIntegrationSwarm(swarm)
#     voronoiSwarm.repopulate()
#     transform = np.array(uw.utils.Integral(fn_mode*fn_height_integrand,
#                                               mesh,
#                                               integrationSwarm=voronoiSwarm,
#                                               integrationType=None,).evaluate())
#     return 2*transform/(xmax-xmin)


# In[6]:


#update_interface_swarm()
Fig = vis.Figure(resolution=(500,500),rulers=False,margin = 30) 
Fig.Points(markerSwarm, 1, fn_size=10.0,colourBar=False,colours= ["black"])
Fig.Points(interfaceSwarm, 1, fn_size=10.0,colourBar=False,colours= ["grey"])
Fig.Points(swarm, materialIndex, fn_size=2.0,colourBar=False,colours= ["blue","red"])
#Fig.Mesh(mesh)
#Fig.Surface(mesh,fn.math.dot(velocityField, velocityField),colourBar=False)
#Fig.Surface(mesh,velocityField.data[:,1],colourBar=False)
Fig.show()
Fig.save_image(outputPath+"test_initial")


# In[7]:


# xcoord,ycoord = interfaceSwarm.data[:,0],interfaceSwarm.data[:,1]

# fname = outputPath + "interfaceSwarm_step"+str(step)
# #import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots(nrows=1, figsize=(10,6))
# ax1.plot(xcoord,ycoord)
# #plt.savefig(fname,dpi=150,bbox_inches='tight')
# #plt.close()


# In[8]:


z_hat = (0.0, 1.0)
buoyancyFn = -densityFn * z_hat*ND_gravity

iWalls = mesh.specialSets["Left_VertexSet"] + mesh.specialSets["Right_VertexSet"]
jWalls = mesh.specialSets["Bottom_VertexSet"] + mesh.specialSets["Top_VertexSet"]
allWalls = iWalls + jWalls
botwall = mesh.specialSets["Bottom_VertexSet"]
topwall = mesh.specialSets["Top_VertexSet"]

if case_bc == "freesurf":
    stokesBC = uw.conditions.DirichletCondition(variable=velocityField, indexSetsPerDof=(iWalls+botwall, botwall))
elif case_bc == "freesurf_fixVx":
    stokesBC = uw.conditions.DirichletCondition(variable=velocityField, indexSetsPerDof=(iWalls+jWalls, botwall))
else:
    print("no bcs type")


if case_swarm =="gv" or case_swarm =="sv":
    stokes = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    conditions=stokesBC,
    fn_viscosity=fn_viscosity,
    fn_bodyforce=buoyancyFn,
    voronoi_swarm=swarm,)
elif case_swarm =="gg" or case_swarm =="sg":
    stokes = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    conditions=stokesBC,
    fn_viscosity=fn_viscosity,
    fn_bodyforce=buoyancyFn,
    voronoi_swarm=None,)
else:
    print("no specific integration type")


solver = uw.systems.Solver(stokes)
if uw.mpi.size == 1:
    solver.set_inner_method("lu")
stokes_inner_tol = 1e-6
stokes_outer_tol = 1e-6
solver.set_inner_rtol(stokes_inner_tol)
solver.set_outer_rtol(stokes_outer_tol)

# Create a system to advect the swarm
advector = uw.systems.SwarmAdvector(swarm=swarm, velocityField=velocityField, order=2)
advector_tracer1 = uw.systems.SwarmAdvector(swarm=interfaceSwarm, velocityField=velocityField, order=2)
advector_tracer2  = uw.systems.SwarmAdvector(swarm=markerSwarm, velocityField=velocityField, order=2)

# top = mesh.specialSets["MaxJ_VertexSet"]
# surfaceArea = uw2.utils.Integral(fn=1.0, mesh=mesh, integrationType="surface", surfaceIndexSet=top)
# surfacePressureIntegral = uw2.utils.Integral(fn=pressureField, mesh=mesh, integrationType="surface", surfaceIndexSet=top)

# def pressure_calibrate():
#     (area,) = surfaceArea.evaluate()
#     (p0,) = surfacePressureIntegral.evaluate()
#     offset = p0 / area
#     if rank == 0:
#         print(
#             "Zeroing pressure using mean upper surface pressure {}".format(offset)
#         )
#     pressureField.data[:] -= offset


# In[9]:


from datetime import datetime
import sys
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


# In[10]:


# save mesh
# define checkpointing file
def checkpoint(output_path):
    meshHnd  = mesh.save(output_path+'mesh.'+ str(step).zfill(5) +'.h5',time=time)
    swarmHnd = swarm.save(output_path+'swarm.'     + str(step).zfill(5) +'.h5')
    # save swarm variables
    #materialIndexHnd = materialIndex.save(output_path+'materialIndex.' + str(step).zfill(5) +'.h5')
    
    # save mesh variable
    velocityHnd = velocityField.save(output_path+'velocityField.'+ str(step).zfill(5) +'.h5', meshHnd)
    pressureHnd = pressureField.save(output_path+'pressureField.'+ str(step).zfill(5) +'.h5', meshHnd)
    timeHnd = timeField.save(output_path+'timeField.'+ str(step).zfill(5) +'.h5', meshHnd)
    #projviscosityHnd = _projviscosityField.save(output_path+'projviscosityField.'+ str(step).zfill(5) +'.h5', meshHnd)

    # and the xdmf files
    velocityField.xdmf(output_path+'velocityField.' +str(step).zfill(5)+'.xdmf',velocityHnd,      "velocity",      meshHnd, "mesh", modeltime=time)
    pressureField.xdmf(output_path+'pressureField.' +str(step).zfill(5)+'.xdmf',pressureHnd,      "pressure",      meshHnd, "mesh", modeltime=time)
    #materialIndex.xdmf(output_path+'materialIndex.' +str(step).zfill(5)+'.xdmf',materialIndexHnd, "materialIndex", swarmHnd,"swarm",modeltime=time)
    
    tacerSwarmHnd1 = interfaceSwarm.save(output_path+'interfaceSwarm.' + str(step).zfill(5) +'.h5')
    tacerSwarmHnd2 = markerSwarm.save(output_path+'markerSwarm.' + str(step).zfill(5) +'.h5')


# In[11]:


from scipy.interpolate import interp1d
#import underworld as uw
from mpi4py import MPI as _MPI

comm = _MPI.COMM_WORLD
rank = comm.rank
size = comm.size

class FreeSurfaceProcessor_MV(object):
    """FreeSurfaceProcessor"""

    def __init__(self):
        """Create a Freesurface processor
        """
        # Create the tools
        
        mesh0 = uw.mesh.FeMesh_Cartesian(elementType=("Q1/dQ0"), elementRes=(xres,yres), minCoord=(xmin, ymin),maxCoord=(xmax, ymax))
        
        self.TField = mesh0.add_variable(nodeDofCount=1)
        self.TField.data[:, 0] = mesh0.data[:, 1].copy()
        
        self.TField_x = mesh0.add_variable(nodeDofCount=1)
        self.TField_x.data[:, 0] = mesh0.data[:, 1].copy()

        self.top = mesh0.specialSets["Top_VertexSet"]
        self.bottom = mesh0.specialSets["Bottom_VertexSet"]
        #self.block_top = IndexSet_block1_top + IndexSet_block2_top
        
        # self.interface0 = IndexSet_moho
        # self.interface1 = IndexSet_lab
        
        # Create boundary condition
        self._conditions = uw.conditions.DirichletCondition(
            variable=self.TField,
            indexSetsPerDof=(self.top + self.bottom,))
        # Create Eq System
        self._system = uw.systems.SteadyStateHeat(
            temperatureField=self.TField,
            fn_diffusivity=1.0,
            conditions=self._conditions)

        self._solver = uw.systems.Solver(self._system)

    def _solve_sle(self):
        self._solver.solve()
        #self._solver2.solve()

    def _advect_surface(self, dt):

        #if self.top:
            # Extract top surface
        x = mesh.data[self.top.data, 0]
        y = mesh.data[self.top.data, 1]

        # Extract velocities from top
        vx = velocityField.data[self.top.data, 0]
        vy = velocityField.data[self.top.data, 1]

        # Advect top surface
        x2 = x + vx * dt
        y2 = y + vy * dt

        # Spline top surface
        f = interp1d(x2, y2, kind='cubic', fill_value='extrapolate')
        self.TField.data[self.top.data, 0] = f(x)
        
        comm.Barrier()
        self.TField.syncronise()
        
    def _update_mesh(self):
        with  mesh.deform_mesh():
            mesh.data[:, -1] = self.TField.data[:, 0]  
            #mesh.data[:, 0] = self.TField_x.data[:, 0]
            
    def solve(self, dt):
        """ Advect free surface through dt and update the mesh """
        # First we advect the surface
        self._advect_surface(dt)
        #self._advect_surface_x(vel,time)
        # Then we solve the system of linear equation
        self._solve_sle()
        # Finally we update the mesh
        self._update_mesh()


# In[ ]:


step      = 0
max_steps = 2
time      = 0
dt        = 0
#timing setup
#viewer.getTimestep()
#viewer.setTimestep(1)

#while step<max_steps:
w = []
dwdt = []
times = []

while time < max_time:
    
    if uw.mpi.rank == 0:
        string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
        step, _adjust_time_units(time),
        _adjust_time_units(dt),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.stdout.write(string)
        sys.stdout.flush()

    solver.solve()
    if markerSwarm.particleLocalCount > 0:
        w = -(markerSwarm.particleCoordinates.data[0][1]+L)
        dwdt = -velocityField.evaluate(tuple(markerSwarm.particleCoordinates.data[0]))[0,1]

        fw = open(outputPath + "ParticlePosition.txt","a")
        fw.write("%.4e \t %.4e \t %.4e \n" %(time,w,dwdt))
        fw.close()
    #amplitudes.append(get_amplitudes()[0])
    times.append(time)

    if step%save_every ==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data:')
        timeField.data[...]= dim(time, u.megayear).m    
        checkpoint(outputPath)

        filename = outputPath+"mat_step"+str(step)+"_time_"+str(np.round(dim(time,u.megayear).m,2))+"Ma"
        Fig = vis.Figure(resolution=(500,500),rulers=False,margin = 80) 
        Fig.Points(swarm, materialIndex, fn_size=5.0,colourBar=False,colours= ["blue","red"])
        Fig.Mesh(mesh)
        Fig.VectorArrows(mesh, velocityField)
        #Fig.Surface(mesh,fn.math.dot(velocityField, velocityField),colourBar=False)
        #Fig.Surface(mesh,velocityField.data[:,1],colourBar=False)
        #Fig.show()
        Fig.save_image(filename)

    dt_solver = advector.get_max_dt()
    dt = min(dt_solver,dt_set)
    # advector.integrate(dt, update_owners=True)
    # advector_tracer.integrate(dt,update_owners=True)

    advector.integrate(dt, update_owners=False)
    advector_tracer1.integrate(dt,update_owners=False)
    advector_tracer2.integrate(dt,update_owners=False)

    freesuface =  FreeSurfaceProcessor_MV()
    freesuface.solve(dt)

    swarm.update_particle_owners()
    interfaceSwarm.update_particle_owners()
    markerSwarm.update_particle_owners()
    pop_control.repopulate()

    step += 1
    time += dt


# In[ ]:





# In[ ]:


Fig = vis.Figure(resolution=(500,500),rulers=False,margin = 80) 
Fig.Points(swarm, materialIndex, fn_size=5.0,colourBar=False,colours= ["blue","red"])
Fig.Points(markerSwarm, 1, fn_size=10.0,colourBar=False,colours= ["black"])
Fig.Mesh(mesh)
#Fig.Surface(mesh,fn.math.dot(velocityField, velocityField),colourBar=False)
#Fig.Surface(mesh,velocityField.data[:,1],colourBar=False)
Fig.show()
Fig.save_image(outputPath+"test_final")


# In[ ]:


### !


# In[ ]:





# In[ ]:





# In[ ]:


# import h5py   
# import matplotlib.pyplot as plt
# import numpy as np
# import math

# def load_surf_swarm(fdir,step):
#     fname = fdir+'surfaceSwarm.' + str(step).zfill(5) +'.h5'
#     fh5   = h5py.File(fname ,'r')  
#     fdata = fh5["data"][()]
#     xcoord = fdata[:,0]
#     ycoord = fdata[:,1]
    
#     fname = fdir+'timeField.'+ str(step).zfill(5) +'.h5'
#     fh5   = h5py.File(fname ,'r')  
#     fdata = fh5["data"][()]
#     time = fdata[0]
#     return xcoord,ycoord,time

# def load_depth(fdir,maxstep,dstep):
#     depth_l = []
#     depth_r = []
#     times   = []
#     for step in range(0,maxstep+1,dstep):
#         xcoord,ycoord,time = load_surf_swarm(fdir,step)
#         depth_l.append(ycoord[0])   
#         depth_r.append(ycoord[-1])  
#         times.append(time)  
#     return np.array(depth_l),np.array(depth_r),np.array(times)


# In[ ]:


# dt0,maxsteps0,dstep0,= 50,145,1
# times0 = np.arange(0,dt0*maxsteps0+dt0*dstep0/2,dt0*dstep0)

# fdir = outputPath
# depth0_l,depth0_r,times= load_depth(fdir,maxsteps0,dstep0)


# In[ ]:


# # Fig 3 in Kaus et al., 2010

# fname = "Depth of the interface at x=−250km versus time for the free surface simulations_FreeSlip"
# fig, ax1 = plt.subplots(nrows=1, figsize=(7,5))
# ax1.set(xlabel='Time [Myrs]', ylabel='Interface Depth [km]') 
# ax1.plot(times,depth0_l*500,'-k')
# #ax1.plot(depth0_l*500,'-k')
# ax1.set_ylim([-500,-100])
# ax1.set_xlim([0,6])
# ax1.grid()
# #ax1.legend(loc = 'lower right',prop = {'size':8})
# plt.savefig(outputPath+fname,dpi=150,bbox_inches='tight')


# In[ ]:


# ### make gif

# import imageio.v2 as imageio
# #import imageio 
# images = [] 
# for i,time in enumerate(time_l):
#     step = i
#     filename = outputPath+"mat_step"+str(step)+"_time_"+str(np.round(time,2))+"Ma"+'.png' 
#     images.append(imageio.imread(filename )) 
# imageio.mimsave(outputPath+'mat.gif', images, duration=1.0) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




