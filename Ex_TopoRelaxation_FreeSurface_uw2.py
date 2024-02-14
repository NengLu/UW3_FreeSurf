#!/usr/bin/env python
# coding: utf-8

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


# In[3]:


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
dt_set = t_relax*1e-2
save_every = 5

outputPath = "op_TopoRelaxation_FreeSurf_uw2_yres"+str(yres)+"/"
if uw.mpi.size == 1:
    # delete previous model run
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+ i)
            
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)


# In[4]:


mesh = uw.mesh.FeMesh_Cartesian(elementType=("Q1/dQ0"), elementRes=(xres,yres), minCoord=(xmin, ymin),maxCoord=(xmax, ymax))
init_mesh = uw.mesh.FeMesh_Cartesian(elementType=("Q1/dQ0"), elementRes=(xres,yres), minCoord=(xmin, ymin),maxCoord=(xmax, ymax))

velocityField = mesh.add_variable(nodeDofCount=2)
pressureField = mesh.subMesh.add_variable(nodeDofCount=1)
timeField= mesh.add_variable(nodeDofCount=1)
velocityField.data[:] = [0.0, 0.0]
pressureField.data[:] = 0.0
timeField.data[...] = 0.0

iWalls = mesh.specialSets["Left_VertexSet"] + mesh.specialSets["Right_VertexSet"]
jWalls = mesh.specialSets["Bottom_VertexSet"] + mesh.specialSets["Top_VertexSet"]
botwall = mesh.specialSets["Bottom_VertexSet"]
topwall = mesh.specialSets["Top_VertexSet"]

def find_ind(value):
    topwall_x = mesh.data[topwall,0]
    idx = np.abs(topwall_x-value).argmin()
    return idx
idx = find_ind(0)
topwallmid_Ind = topwall[:][idx] 
print(mesh.data[topwallmid_Ind])


# In[5]:


figsize = (800,400)


Fig = vis.Figure(resolution=figsize,rulers=True,margin = 80)
Fig.Mesh(mesh)
Fig.show()
Fig.save(outputPath+"mesh_00.png")


# In[6]:


# coords = fn.input()
# perturbation = w_m * fn.math.cos(2. * np.pi * coords[0] / Lambda)

TField = init_mesh.add_variable(nodeDofCount=1)
TField.data[:, 0] = init_mesh.data[:, 1].copy()

conditions = uw.conditions.DirichletCondition(variable=TField,indexSetsPerDof=(topwall + botwall,))
system = uw.systems.SteadyStateHeat(
    temperatureField=TField,
    fn_diffusivity=1.0,
    conditions=conditions)
solver = uw.systems.Solver(system)

x = init_mesh.data[topwall,0]
TField.data[topwall, 0] = perturbation(x)

solver.solve()
with mesh.deform_mesh():
     mesh.data[:, -1] = TField.data[:, 0].copy()


# In[7]:


Fig = vis.Figure(resolution=figsize,rulers=True,margin = 80)
Fig.Mesh(mesh)
Fig.show()
Fig.save(outputPath+"mesh_01.png")


# In[8]:


swarm = uw.swarm.Swarm(mesh=mesh,particleEscape=True)
material = swarm.add_variable(dataType="int", count=1)
particlesPerCell = 9
gaussPointCount = 3
swarmLayout = uw.swarm.layouts.PerCellGaussLayout( swarm=swarm, gaussPointCount=gaussPointCount)
swarm.populate_using_layout(layout=swarmLayout)
pop_control = uw.swarm.PopulationControl(swarm,aggressive=True,splitThreshold=0.15, maxDeletions=2,maxSplits=10,particlesPerCell=particlesPerCell)

# interfaceSwarm = uw.swarm.Swarm(mesh,particleEscape=True)
# pop_control2 = uw.swarm.PopulationControl(interfaceSwarm ,aggressive=True,splitThreshold=0.15, maxDeletions=2,maxSplits=10,particlesPerCell=4)
# topwallmid_Ind = int((topwall[:].max()-topwall[:].min())/2+topwall[:].min())
# #x = 0
# #y = w
# #x = np.linspace(xmin,xmax, npoints)
# #y = w_m * np.cos(2.*np.pi*(x)/Lambda)
# x = mesh.data[topwallmid_Ind,0] 
# y = mesh.data[topwallmid_Ind,1] 
# interface_coords = np.array([(x, y)]) #np.ascontiguousarray(np.array([x,y]))
# interfaceSwarm.add_particles_with_coordinates(interface_coords)

MIndex = 0
material.data[:] = MIndex

densityMap = {MIndex:densityM}
densityFn = fn.branching.map(fn_key=material, mapping=densityMap)

viscosityMap = {MIndex:viscM}
fn_viscosity = fn.branching.map(fn_key=material, mapping=viscosityMap)

z_hat = (0.0, 1.0)
buoyancyFn = -densityFn * z_hat*ND_gravity

stokesBC = uw.conditions.DirichletCondition(variable=velocityField, indexSetsPerDof=(iWalls+botwall, botwall))
stokes = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    conditions=stokesBC,
    fn_viscosity=fn_viscosity,
    fn_bodyforce=buoyancyFn,
    voronoi_swarm=None,)
solver = uw.systems.Solver(stokes)

if uw.mpi.size == 1:
    solver.set_inner_method("lu")
stokes_inner_tol = 1e-6
stokes_outer_tol = 1e-6
solver.set_inner_rtol(stokes_inner_tol)
solver.set_outer_rtol(stokes_outer_tol)

# Create a system to advect the swarm
advector = uw.systems.SwarmAdvector(swarm=swarm, velocityField=velocityField, order=2)
#advector_tracer = uw.systems.SwarmAdvector(swarm=interfaceSwarm, velocityField=velocityField, order=2)


# In[9]:


Fig = vis.Figure(resolution=figsize,rulers=True,margin = 80)
Fig.Mesh(mesh)
Fig.Points(swarm, material, fn_size=5.0,colourBar=False,colours= ["red"])
#Fig.Points([0,0.05], 1, fn_size=10.0,colourBar=False,colours= ["black"])
Fig.show()
Fig.save(outputPath+"swarm_01.png")


# In[10]:


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

# save mesh
# define checkpointing file
def checkpoint(output_path):
    meshHnd  = mesh.save(output_path+'mesh.'+ str(step).zfill(5) +'.h5',time=time)
    swarmHnd = swarm.save(output_path+'swarm.'     + str(step).zfill(5) +'.h5')
    # save swarm variables
    materialIndexHnd = material.save(output_path+'materialIndex.' + str(step).zfill(5) +'.h5')
    
    # save mesh variable
    velocityHnd = velocityField.save(output_path+'velocityField.'+ str(step).zfill(5) +'.h5', meshHnd)
    pressureHnd = pressureField.save(output_path+'pressureField.'+ str(step).zfill(5) +'.h5', meshHnd)
    timeHnd = timeField.save(output_path+'timeField.'+ str(step).zfill(5) +'.h5', meshHnd)
    #projviscosityHnd = _projviscosityField.save(output_path+'projviscosityField.'+ str(step).zfill(5) +'.h5', meshHnd)

    # and the xdmf files
    velocityField.xdmf(output_path+'velocityField.' +str(step).zfill(5)+'.xdmf',velocityHnd,      "velocity",      meshHnd, "mesh", modeltime=time)
    pressureField.xdmf(output_path+'pressureField.' +str(step).zfill(5)+'.xdmf',pressureHnd,      "pressure",      meshHnd, "mesh", modeltime=time)
    material.xdmf(output_path+'materialIndex.' +str(step).zfill(5)+'.xdmf',materialIndexHnd, "materialIndex", swarmHnd,"swarm",modeltime=time)

    #tacerSwarmHnd1 = interfaceSwarm.save(output_path+'interfaceSwarm.' + str(step).zfill(5) +'.h5')


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
 
    #if markerSwarm.particleLocalCount > 0:

    particle_coordinates = mesh.data[topwallmid_Ind]
    w = particle_coordinates[1]
    dwdt = velocityField.evaluate(tuple(particle_coordinates))[0,1]

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

        # filename = outputPath+"mat_step"+str(step)+"_time_"+str(np.round(dim(time,u.megayear).m,2))+"Ma"
        # Fig = vis.Figure(resolution=figsize,rulers=False,margin = 80) 
        # Fig.Points(swarm, material, fn_size=5.0,colourBar=False,colours= ["blue","red"])
        # Fig.Mesh(mesh)
        # Fig.VectorArrows(mesh, velocityField)
        # #Fig.Surface(mesh,fn.math.dot(velocityField, velocityField),colourBar=False)
        # #Fig.Surface(mesh,velocityField.data[:,1],colourBar=False)
        # #Fig.show()
        # Fig.save_image(filename)

    dt_solver = advector.get_max_dt()
    dt = min(dt_solver,dt_set)
    # advector.integrate(dt, update_owners=True)
    # advector_tracer.integrate(dt,update_owners=True)

    advector.integrate(dt, update_owners=False)
    #advector_tracer.integrate(dt,update_owners=False)
    # #advector_tracer2.integrate(dt,update_owners=False)
    freesuface =  FreeSurfaceProcessor_MV()
    freesuface.solve(dt)

    swarm.update_particle_owners()
    #interfaceSwarm.update_particle_owners()
    #markerSwarm.update_particle_owners()
    pop_control.repopulate()
    #pop_control2.repopulate()

    step += 1
    time += dt


# In[ ]:


Fig = vis.Figure(resolution=(800,400),rulers=True,margin = 80)
Fig.Mesh(mesh)
Fig.Points(swarm, material, fn_size=5.0,colourBar=False,colours= ["red"])
#Fig.Points(interfaceSwarm, 1, fn_size=10.0,colourBar=False,colours= ["black"])
Fig.show()
Fig.save(outputPath+"swarm_final.png")


# In[ ]:


data = np.loadtxt(outputPath + "ParticlePosition.txt") #,skiprows=1)
t1 = data[:,0]
w1 = data[:,1]
dwdt1 = data[:,2]


# In[ ]:


import matplotlib.pyplot as plt

fname = "Topography of the box mid"
fig, ax = plt.subplots(nrows=1, figsize=(8,5))
ax.plot(dim(t,u.kiloyears), dim(w_t,u.kilometer),
         label='Analytic solution', color="k",
         linestyle="-", linewidth=2)
ax.plot(dim(t1,u.kiloyears), dim(w1,u.kilometer), 
        label='Numeric solution_uw2', color="red",
        linestyle="--", linewidth=2)
#ax.axhline(0.,color='black',linestyle='--')
ax.set_xlabel('Time (yrs)')
ax.set_ylabel('Deflection (km)')
ax.legend(loc='best',prop = {'size':8})
ax.grid()
ax.set_xlim([0,600])
plt.show()
plt.savefig(fname,dpi=150,bbox_inches='tight')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




