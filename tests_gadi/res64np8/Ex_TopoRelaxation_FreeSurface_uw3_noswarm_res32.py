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
yres = 64
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


density_fn = densityM  
visc_fn = viscM 


# +
Tmesh = uw.discretisation.MeshVariable("Tmesh", init_mesh, 1, degree=1)
Bmesh = uw.discretisation.MeshVariable("Bmesh", init_mesh, 1, degree=1)

mesh_solver = uw.systems.Poisson(init_mesh, u_Field=Tmesh, solver_name="FreeSurf_solver")
mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
mesh_solver.f = 0.0
mesh_solver.add_dirichlet_bc((Bmesh.sym[0],), "Top")
mesh_solver.add_dirichlet_bc((Bmesh.sym[0],), "Bottom")

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


stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.shear_viscosity_0
stokes.add_essential_bc((0.0,None), "Left")
stokes.add_essential_bc((0.0,None), "Right")
stokes.add_essential_bc((0.0,0.0), "Bottom")

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = 1.0e-6
stokes.petsc_options["ksp_rtol"] = 1.0e-6
stokes.petsc_options["ksp_atol"] = 1.0e-6
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None

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
        self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Top")
        self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Bottom")

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
        self.init_mesh.update_lvec()
        #a_global = self.init_mesh.dm.getGlobalVec()
        #self.init_mesh.dm.localToGlobal(self.init_mesh.lvec, a_global)
         
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

while time < max_time:   
#while step < max_steps:
    
    if uw.mpi.rank == 0:
        string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
        step, _adjust_time_units(time),
        _adjust_time_units(dt),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.stdout.write(string)
        sys.stdout.flush()

    stokes.solve(zero_init_guess=False,_force_setup=True)
    #stokes.solve(zero_init_guess=False)


    if step%save_every ==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data:')
        with mesh.access(timeField):
            timeField.data[:,0] = dim(time, u.megayear).m
        mesh.petsc_save_checkpoint(meshVars=[v, p, timeField], index=step, outputPath=outputPath)

    
    dt_solver = stokes.estimate_dt()
    dt = min(dt_solver,dt_set)

   
    freesuface = FreeSurfaceProcessor(v,dt)
    new_mesh_coords=freesuface.solve()
    mesh.deform_mesh(new_mesh_coords)
   
             
    step += 1
    time += dt
# -








