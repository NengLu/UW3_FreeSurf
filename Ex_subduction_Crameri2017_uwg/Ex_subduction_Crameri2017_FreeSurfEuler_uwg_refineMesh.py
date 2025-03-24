#!/usr/bin/env python
# coding: utf-8

# 2D Subduction from Crameri et al 2017
# ======
# 
# Use this setup to run the model from [Crameri et al, 2017](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GC006821) (Figure 4).
# 
# **References**
# 
# 1. Crameri, F., Lithgow‐Bertelloni, C. R., & Tackley, P. J. (2017). The dynamical control of subduction parameters on surface topography. Geochemistry, Geophysics, Geosystems, 18(4), 1661-1687.

# In[1]:


import underworld as uw
import math
from underworld import function as fn
#import underworld.visualisation as vis
import numpy as np
import os

from underworld import UWGeodynamics as GEO
u = GEO.UnitRegistry

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size


# In[2]:


# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-3
GEO.rcParams['initial.nonlinear.max.iterations'] = 100
GEO.rcParams["nonlinear.tolerance"] = 1e-3
GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 16
GEO.rcParams["swarm.particles.per.cell.2D"] = 16
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1


# In[3]:


#u = uw.scaling.units
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

refDensity = 3300. * u.kilogram / u.meter**3  
refGravity = 9.81 * u.meter / u.second**2    
refLength  = 2890. * u.kilometer #  
refViscosity = 1e23 * u.pascal * u.seconds
refTempBot  = 2800 * u.degK
refTempSurf = 300 *  u.degK
refTempLAB = 1700 *  u.degree_Kelvin

# KL = ref_length  
# KM = ref_density * KL**3
# Kt = KM/ ( KL * ref_viscosity) 

bodyforce = refDensity *refGravity  #* (1-3e-5*1600)
refStress = bodyforce * refLength * 1e-6
Kt = (refViscosity/refStress).to_base_units()
KL = refLength
KM = (refViscosity*KL*Kt).to_base_units()


# KL = refLength  
# KM = refDensity * KL**3
# Kt = KM/ ( KL * refViscosity)

KT = refTempBot - refTempSurf

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
GEO.scaling_coefficients["[temperature]"] = KT

if uw.mpi.size == 1:
    print('Length, km = ', GEO.dimensionalise(1., u.kilometer))
    print('Time, Myr = ',GEO.dimensionalise(1., u.megayear))
    print('Pressure, MPa = ',GEO.dimensionalise(1577., u.megapascal))
    print('Temperature, K = ',GEO.dimensionalise(1., u.degK))
    print('Velocity, cm/yr = ',GEO.dimensionalise(1., u.centimeter / u.year))
    print('Viscosity, Pa S = ',GEO.dimensionalise(1.,u.pascal * u.second))
    print('Density, kg / m^3 = ',GEO.dimensionalise(1.,u.kilogram / u.meter**3 ))
    print(ndim(bodyforce))


# In[4]:


refDiffusivity  = 1e-6 * u.meter**2/u.second
refExpansivity = 3e-5 / u.kelvin

Ra = ((refExpansivity*refDensity*refGravity*(refTempBot - refTempSurf)*refLength**3).to_base_units() /(refViscosity*refDiffusivity).to_base_units()).magnitude
print(Ra)


# In[10]:


yRes = 256 
xRes = yRes*2
th_st = 0.05
boxLength = 2.0
boxHeight = 1.0

yRes2 = int(yRes+np.round(th_st/(1/yRes)))

boxHalf = boxLength*0.5
xmin,ymin = -boxHalf, -boxHeight,
xmax,ymax2 = boxHalf,th_st 

Model = GEO.Model(elementRes=(xRes, yRes2),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax2),
                  gravity=(0.0, -refGravity))
#Model.outputDir= "output_fslip_yres{}_uwg_sideFslip".format(yRes)
Model.outputDir= "output_fsurfEuler_yres{}_uwg_refine".format(yRes)
Model.minStrainRate = 1e-18 / u.second


# In[6]:


meshRefineFactor = 0.8

with Model.mesh.deform_mesh():
    Model.mesh.data[:,1] =  Model.mesh.data[:,1] 
    normYs = -1.* Model.mesh.data[:,1]/( Model.mesh.maxCoord[1] -  Model.mesh.minCoord[1])
    Model.mesh.data[:,1] =  Model.mesh.data[:,1] * np.exp(meshRefineFactor*normYs**2)/np.exp(meshRefineFactor*1.0**2)


# In[ ]:


from underworld.swarm import Swarm
from collections import OrderedDict
Model.swarm_variables = OrderedDict()
Model.swarm = Swarm(mesh=Model.mesh, particleEscape=True)
Model.swarm.allow_parallel_nn = True
if Model.mesh.dim == 2:
    particlesPerCell = GEO.rcParams["swarm.particles.per.cell.2D"]
else:
    particlesPerCell = GEO.rcParams["swarm.particles.per.cell.3D"]
Model._swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(
    swarm=Model.swarm,
    particlesPerCell=particlesPerCell)

Model.swarm.populate_using_layout(layout=Model._swarmLayout)
Model._initialize()


# In[42]:


WBL0 = 0.03

depthFn = 0.-Model.y
WBLFn = WBL0 * fn.math.sqrt(fn.math.abs(Model.x-boxHalf)) # platethickness 
platethickness = fn.branching.conditional([(Model.x > 0., WBLFn ),
                                          (True,         WBL0)]) 
def thickness_x(x,th0):
    return th0 * np.sqrt(np.abs(x-boxHalf)) # platethickness

def erf_custom(x):
    # Define the number of steps for integration
    n = 100000  # More steps for better accuracy
    step = x / n
    integral = 0.0

    # Trapezoidal integration
    for i in range(n):
        t = i * step
        integral += np.exp(-t**2) * step

    return (2 / np.sqrt(np.pi)) * integral


# In[43]:


AirShape = GEO.shapes.Layer(top=Model.top, bottom=0.)
MantleShape = GEO.shapes.Layer(top=Model.top, bottom=Model.bottom)

npoints = xRes*2+1
slab_bot = np.zeros([npoints,2]) 
slab_bot_x = np.linspace(xmin,xmax,npoints)
slab_bot[:,0] = slab_bot_x
slab_bot[:,1] = -WBL0
slab_bot[slab_bot_x>0,1]  = -thickness_x(slab_bot_x[slab_bot_x>0],WBL0) 

slabshape_polygon1 = np.vstack(([slab_bot,[xmax,0],[xmin,0],[xmin,-WBL0]]))

slab_theta = 30
# slab_sintheta = np.sin(np.radians(slab_theta))
slab_sintheta =  0.5 #np.sin(np.radians(slab_theta))
slab_costheta =  np.cos(np.radians(slab_theta))
slab_length = 0.25 # 722.5km ~700km
slab_depth = -slab_length*slab_sintheta 
slabe_x0 = -slab_length*slab_costheta 

slab_dx = WBL0*slab_sintheta
slab_dy = WBL0*slab_costheta
slabshape_polygon2 = [(slabe_x0,slab_depth),(0,0),(slab_dx,-slab_dy),(slabe_x0+slab_dx,slab_depth-slab_dy)]

SlabShape = GEO.shapes.Polygon(slabshape_polygon1) | GEO.shapes.Polygon(slabshape_polygon2)

thcrust = 0.005

npoints = xRes+1
crust_bot = np.zeros([npoints,2]) 
crust_bot_x = np.linspace(0.,xmax,npoints)
crust_bot[:,0] = crust_bot_x
crust_bot[:,1] = -thcrust 
crust_bot[crust_bot_x>0,1]  = -thickness_x(crust_bot_x[crust_bot_x>0],thcrust) 
shape_polygon1 = np.vstack(([crust_bot,[xmax,0],[0.,0],[0.,-thcrust]]))

crust_dx = thcrust*slab_sintheta
crust_dy = thcrust*slab_costheta
shape_polygon2 = [(slabe_x0,slab_depth),(0,0),(crust_dx,-crust_dy),(slabe_x0+crust_dx,slab_depth-crust_dy)]

CrustShape = GEO.shapes.Polygon(shape_polygon1) | GEO.shapes.Polygon(shape_polygon2)

thair = 0.05 


# In[44]:


Mantle = Model.add_material(name="Mantle", shape=MantleShape)
Slab   = Model.add_material(name="Slab", shape=SlabShape)
Crust  = Model.add_material(name="Crust", shape=CrustShape)
Air    = Model.add_material(name="Air", shape=AirShape)


# In[45]:


import numpy as np

npoints = xRes*2+1

coords = np.zeros([npoints,2])
coords[:,0] = np.linspace(xmin,xmax,npoints)
coords[:,1] = 0.

         
surf_tracers = Model.add_passive_tracers(name="Surface", vertices=coords) 


# In[10]:


# # x1, y1 = slab_x3 ,slab_
# # x2, y2 = slab_x3 ,drip_y1
# # x3, y3 = slab_x3 ,drip_y2 
# # coords = np.ndarray((3, 2))
# # coords[:, 0] = np.array([x1, x2, x3])
# # coords[:, 1] = np.array([y1, y2, y3])
# # Model.add_passive_tracers(name="Tip", vertices=coords)
# x1, y1 = slab_x4 ,slab_y2
# x2, y2 = slab_x5 ,slab_y3 
# coords = np.ndarray((2, 2))
# coords[:, 0] = np.array([x1, x2])
# coords[:, 1] = np.array([y1, y2])
# Model.add_passive_tracers(name="Tip", vertices=coords)


# In[46]:


# if uw.mpi.rank == 0:
#     from underworld import visualisation as vis
#     fig_res = (800,300)

#     Fig = vis.Figure(resolution=fig_res,rulers=False,margin = 20,rulerticks=7,quality=2,clipmap=False)
#     Fig.Points(Model.Surface_tracers, pointSize=2.0)
#     Fig.Points(Model.swarm, Model.materialField,fn_size=2.0,discrete=True,colourBar=False,colours='white orange red blue')
#     Fig.show()
#     Fig.save("Modelsetup_freesurfEuler.png")


# $ \sigma_{y,brittle} = C  + \mu P$
# 
# $ \sigma_y = min [\sigma_{y,brittle}, \sigma_{y,ductile}]$
# 
# ### Note in UWG
# in _rheology line 311 should be :  cohesion = fn.misc.constant(nd(self.cohesion)) 
# 

# In[12]:


class DruckerPrager_Byerlee(object):
    def __init__(self, name=None, cohesion=None, frictionCoefficient=None):
        """
        using a Drucker-Prager yield criterion with the pressure-dependent yield stress sigma_y based on Byerlee’s law

        Drucker Prager yield Rheology.
         The pressure-dependent Drucker-Yield criterion is defined as follow:
         .. math::
            $ \sigma_{y,brittle} = C  + \mu P$
            
        Parameters
        ----------
            cohesion :
                Cohesion for the pristine material(initial cohesion)
            frictionCoefficient :
                friction angle for a pristine material
            ductileYieldStress :    
        Returns
        -------
        An UWGeodynamics DruckerPrager class
        """
        self.name = name
        self._cohesion = cohesion
        self._frictionCoefficient = frictionCoefficient
    
        self.plasticStrain = None
        self.pressureField = Model.pressureField
          
    @property
    def cohesion(self):
        return self._cohesion
    
    @cohesion.setter
    def cohesion(self, value):
        self._cohesion = value
    
    @property
    def frictionCoefficient(self):
        return self._frictionCoefficient
    
    @frictionCoefficient.setter
    def frictionCoefficient(self, value):
        self._frictionCoefficient = value
    
    def _frictionFn(self):
        friction = fn.misc.constant(self.frictionCoefficient)
        return friction
    
    def _cohesionFn(self):
        cohesion = fn.misc.constant(ndim(self.cohesion))
        return cohesion
        
    def _get_yieldStress2D(self):
        f = self._frictionFn()
        C = self._cohesionFn()
        P = self.pressureField
        self.yieldStress = C + P * f 
        return self.yieldStress
        
    def _get_yieldStress3D(self):
        print("no settings")
        return


# In[13]:


Model.maxViscosity = refViscosity*1e5
Model.minViscosity = refViscosity*1e-4

ref_viscosity_A = refViscosity/ np.exp((240 *1e3/(1600*8.314)))  
ductileViscosity = GEO.ViscousCreep(name='ductile',
                                 preExponentialFactor=1./ref_viscosity_A,
                                 stressExponent=1.0,
                                 activationVolume=0.,activationEnergy=240 * u.kilojoules/u.mole,
                                 f=2.0)  # Crameri et al., 2017
Mantle.viscosity = refViscosity  # Isoviscous
Slab.viscosity   = ductileViscosity  
Crust.viscosity  = ductileViscosity 
Air.viscosity = 1e20 * u.pascal * u.seconds

mplasticity  = DruckerPrager_Byerlee(name='mantlepl',cohesion=10. * u.megapascal,frictionCoefficient = 0.25)
cplasticity  = DruckerPrager_Byerlee(name='crustpl',cohesion=10. * u.megapascal,frictionCoefficient = 0.001)
Slab.plasticity  = mplasticity
Crust.plasticity = cplasticity
Slab.stressLimiter = ndim(600. * u.megapascal)
Crust.stressLimiter = ndim(600. * u.megapascal)


# $\alpha =\frac{k}{\rho c_{p}}$
# 
# Where $\alpha$ is the thermal diffusivity $(m^{2}/s)$, $k$ is thermal conductivity $(W/(m·K))$, $c_{p}$ is specific heat capacity $(J/(kg·K))$, $\rho$  is density $(kg/m³)$
# 
# 
# ( Q ) is the internal heating rate (W/m³), which can be derived from your given internal heating rate of (5.44 \times 10^{-12} , \text{W/kg}) by multiplying it by the density (( \rho )):
# 
# 
# [
# Q = H \cdot \rho
# ]
# where ( H ) is the internal heating rate per unit mass (W/kg).

# In[14]:


internalHeatingRate = 5.44*1e-12 * u.watt/u.kilogram
radiogenicHeatProd = internalHeatingRate*refDensity*(4/3.3) 
#radiogenicHeatProd = internalHeatingRate*(refDensity*((1-0.075)))
print(dimen(ndim(radiogenicHeatProd),u.microwatt / u.meter**3 ))


# In[15]:


#from _density import Density
#from _density import LinearDensityT

r_tE = 3e-5 / u.kelvin
r_tem = 300 * u.kelvin
# r_beta = 0. / u.pascal
# r_pre = 0.1e6 * u.pascal

#alldensity = GEO.LinearDensityT(3300. * u.kilogram / u.metre**3,thermalExpansivity = r_tE, reference_temperature = r_tem)   
alldensity  = GEO.LinearDensity(3300. * u.kilogram / u.metre**3,thermalExpansivity = r_tE, reference_temperature = r_tem )     
alldensity.temperatureField = Model.temperature

Mantle.density = alldensity 
Slab.density   = alldensity    
Crust.density  = alldensity  
Air.density = 0.


Model.diffusivity = 1.0e-6 * u.metre**2 / u.second
Model.capacity    = 1200. * u.joule / (u.kelvin * u.kilogram)
Mantle.diffusivity = 1.0e-6 * u.metre**2 / u.second
Slab.diffusivity = 1.0e-6 * u.metre**2 / u.second 
Crust.diffusivity = 1.0e-6 * u.metre**2 / u.second
Air.diffusivity = 20.0e-6 * u.metre**2 / u.second 

Mantle.capacity  = 1200. * u.joule / (u.kelvin * u.kilogram)
Slab.capacity    = 1200. * u.joule / (u.kelvin * u.kilogram)
Crust.capacity   = 1200. * u.joule / (u.kelvin * u.kilogram)
Air.capacity = 100. * u.joule / (u.kelvin * u.kilogram)

Mantle.radiogenicHeatProd  = 0.022 * u.microwatt / u.meter**3 
Slab.radiogenicHeatProd    = 0.022 * u.microwatt / u.meter**3 
Crust.radiogenicHeatProd   = 0.022 * u.microwatt / u.meter**3 


# In[16]:


Temp_surf = 300/2500
Temp_bot = 2800/2500
Temp0 = 0.64
Temp_mantle = Temp0+Temp_surf

halfSpaceTemp = Temp0*fn.math.erf((depthFn)/platethickness)+Temp_surf 
geotherm_fn = fn.branching.conditional([(depthFn <= 0.,Temp_surf),
                                        (depthFn <= -platethickness ,Temp_mantle),
                                       (True,       halfSpaceTemp)])
Model._init_temperature_variables()
#Model.temperature.data[...] = 0.
Model.temperature.data[...] = geotherm_fn.evaluate(Model.mesh)

slab_index = GEO.shapes.Polygon(slabshape_polygon2).evaluate(Model.mesh.data)[:,0]
slab_coords = Model.mesh.data[slab_index]

temth = np.zeros(slab_coords.shape[0])

for index in range(slab_coords.shape[0]):
    coord = slab_coords[index][:]
    coordx = coord[0]
    coordy = coord[1]
    if coordx<0:
        temth[index] = (coordx/slab_costheta*slab_sintheta-coordy)*slab_costheta
    else:
        temth[index] = coordx/slab_sintheta+ (-coordy-coordx/slab_sintheta*slab_costheta)*slab_costheta  

Model.temperature.data[slab_index,0]= Temp0*erf_custom(temth/WBL0)+Temp_surf 


# In[17]:


Model.set_velocityBCs(left=[0.,None],right=[0,None],bottom=[None,0.], top=[None, None])
Model.set_temperatureBCs(top=refTempSurf, materials=[(Air, Temp_surf)]) 
#Model.set_heatFlowBCs(bottom=(0. * u.milliwatt / u.metre**2,Mantle))


# In[18]:


# def testnan(A):
#     return A[np.isnan(A)]

# gg = Model.temperature.data.copy()
# testnan(gg)
# print(Model.mesh.data[np.where(np.isnan(gg)==True),:])
# print(np.where(np.isnan(gg)==True))

# plt.imshow( np.flipud(gg.reshape(yRes2+1,xRes+1)))

# Model.temperature.data[80600]


# In[19]:


#erf_custom((93/2890)/(0.03*np.sqrt(1))),erf_custom(0), erf_custom((1)/(0.03*np.sqrt(1)))
## (0.87073151579686947, 0.0, 1.0001880631945124)
#0.64*0.87*2500+300


# In[20]:


#Model.init_model(pressure='lithostatic',temperature=None)
#Model.initialize_pressure_to_lithostatic()
Model.pressureField.data[...] = 0.
Model.init_model(pressure='lithostatic',temperature=None,defaultStrainRate=1e-15*u.second)
#Model.pressureField.data[...] = 0.


# In[21]:


# import matplotlib.pyplot as plt
# #plt.scatter(slab_coords[:,0],slab_coords[:,1],c=temth)
# plt.scatter(slab_coords[:,0],slab_coords[:,1],c=Model.temperature.data[slab_index,0])


# In[22]:


# rh = GEO.ViscousCreepRegistry() 
# rh.Dry_Olivine_Dislocation_Karato_and_Wu_1993


# In[23]:


# pl = GEO.PlasticityRegistry()
# pl.Huismans_et_al_2011_Crust


# In[24]:


# Fig = vis.Figure(resolution=fig_res,rulers=False,margin = 20,rulerticks=7,quality=2,clipmap=False)
# #Fig.Points(Model.Tip_tracers, pointSize=8.0)
# cmap = vis.lavavu.matplotlib_colourmap("bwr")
# Fig.Points(Model.swarm, Model.materialField,discrete=True,fn_size=2.0,colourBar=True,colours=cmap)

# # visc.colourbar(size=[0.95,15], align="bottom")
# # cbar1.colourmap([(0, 'green'), (0.75, 'yellow'), (1, 'red')], reverse=True)
# Fig.show()
# Fig.save("Modelsetup_ma.png")


# In[25]:


# Fig = vis.Figure(resolution=fig_res,rulers=False,margin = 20,rulerticks=7,quality=2,clipmap=False)
# #Fig.Points(Model.Tip_tracers, pointSize=8.0)
# cmap = vis.lavavu.matplotlib_colourmap("bwr")
# Fig.Points(Model.swarm, Model.temperature*2500.,discrete=False,fn_size=2.0,colourBar=True,colours=cmap)

# # visc.colourbar(size=[0.95,15], align="bottom")
# # cbar1.colourmap([(0, 'green'), (0.75, 'yellow'), (1, 'red')], reverse=True)
# Fig.show()
# Fig.save("Modelsetup_temp.png")


# In[26]:


# Fig = vis.Figure(resolution=fig_res,rulers=False,margin = 20,rulerticks=7,quality=2,clipmap=False)
# #Fig.Points(Model.Tip_tracers, pointSize=8.0)
# cmap = vis.lavavu.matplotlib_colourmap("bwr")
# Fig.Points(Model.swarm, Model.densityField/ndim(refDensity)*3300.,discrete=False,fn_size=2.0,colourBar=True,colours=cmap)

# # visc.colourbar(size=[0.95,15], align="bottom")
# # cbar1.colourmap([(0, 'green'), (0.75, 'yellow'), (1, 'red')], reverse=True)
# Fig.show()
# Fig.save("Modelsetup_density.png")


# In[27]:


# Fig = vis.Figure(resolution=fig_res,rulers=False,margin = 20,rulerticks=7,quality=2,clipmap=False)
# #Fig.Points(Model.Tip_tracers, pointSize=8.0)
# cmap = vis.lavavu.matplotlib_colourmap("RdYlBu_r")
# Fig.Points(Model.swarm, fn.math.log10(Model.viscosityField*refViscosity.m),discrete=False,fn_size=2.0,colourBar=True,colours=cmap)
# # visc.colourbar(size=[0.95,15], align="bottom")
# # cbar1.colourmap([(0, 'green'), (0.75, 'yellow'), (1, 'red')], reverse=True)
# Fig.show()
# Fig.save("Modelsetup_visc.png")


# In[28]:


# coords_wall = np.zeros((npoints,2))
# coords_wall[:,0] = 0.
# coords_wall[:,1] = np.linspace(ymin,ymax2,npoints)
# visc_wall = Model.viscosityField.evaluate(coords_wall)
# temp_wall = Model.temperature.evaluate(coords_wall) 
# rho_wall =  Model.densityField.evaluate(coords_wall) 

# plt.plot(np.log10(dimen(visc_wall,u.pascal*u.second).m),coords_wall[:,1])


# In[29]:


# plt.plot(dimen(rho_wall,u.kilogram / u.meter**3).m,coords_wall[:,1])


# In[30]:


# plt.plot(dimen(temp_wall,u.degK).m,coords_wall[:,1])


# In[31]:


#Model.checkpoint(0)


# In[32]:


Model.solver.set_inner_method("mumps")
Model.solver.set_penalty(1e6)


# In[33]:


maxTime = 2.01*u.megayear
dt_set = 2.5*u.kiloyear 
checkpoint_interval = 10*u.kiloyear


# In[ ]:


Model.run_for(maxTime, checkpoint_interval=checkpoint_interval,dt=dt_set)


# In[ ]:


1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




