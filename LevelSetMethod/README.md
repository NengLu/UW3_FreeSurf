### Level-set method
The level set method \citep{Osher1988} is using the level set function (the sighed distance function) to tracing the material interface. It is generally build within the Eulerian framework and has been extended to the hybrid Eulerian‐Lagrangian framework as the Particle Level Set method \citep{Enright2002}. See details in comprehensive review \citep{Gibou2018}. There are serval applied cases of this method in the geodynamics community, such as modeling incompressible two-phase flow \citep{Sussman1994}, mantle plumes \citep{Gross2007}, gravitational instabilities \citep{Zlotnik2009}, and benchmarks with level set method \citep{Hillebrand2014}; modelling geophsyics flow with particle level set \citep{Samuel2010,Wu2023}; application in geodynamics codes \citep{Braun2008}.


### Level-set representation and equations

$\Gamma$ denotes the interface that is to be associated and tracked with the level set function $\phi$, and $\Omega$ is a bounded region. $\phi$ is defined as the signed distance function to $\Gamma$:

\begin{equation}
\left\{\begin{matrix}\begin{aligned}
		\phi(r,t) = d && (r \in \Omega) \\ 
		\phi(r,t) = -d && (r \notin \Omega) \\ 
		\phi(r,t) = 0 && (r \in \partial \Omega = \Gamma (t))\\  
\end{aligned}\end{matrix}\right.
\end{equation}
where $d$ is the Euclidian distance to $\Gamma$.

For an arbitrary material parameter C can be written based on $\phi$ as:
\begin{equation}
C = \left\{\begin{matrix}\begin{aligned}
		&C_1     && (\phi \leq 0) \\ 
		&C_2  && (\phi > 0)\\ 
\end{aligned}\end{matrix}\right.
\end{equation}

We can smooth the material parameters across the sharp boundary by the diffuse boundary method \citep{Hillebrand2014}:
\begin{equation}
		C = \left\{\begin{matrix}\begin{aligned}
		&C_1     && (\phi \leq -\alpha h) \\ 
		&C_2  && (\phi \geq \alpha h)\\ 
		&\frac{(C_2-C_1)\phi}{2\alpha h} +\frac{C_1+C_2}{2} && (\left | \phi \right | < \alpha h)\\  
\end{aligned}\end{matrix}\right.
\end{equation}
where  $\alpha = 1$ and $h$ represents the element size.

$\phi$ is advected by the velocity field $v$ (in geodynamics models, it is provided by the stokes solver):
\begin{equation}
\frac{\partial \phi}{\partial t}+\mathbf{v}\cdot \nabla \phi = 0
\end{equation}

Another way to update the level set function is recomputed it from the updated surface geometry at each time step \citep{Braun2008}.

To correct the deviation of $phi$ from its signed distance property during the advection, we need reinitialize it:
\begin{equation}
\frac{\partial \phi}{\partial \tau}+sgn(\phi_0) (| \nabla\phi|-1))= 0
\end{equation}
where $\tau$ represents a fictitious time, $sgn(\phi)$ is a smoothed signed distance function:
\begin{equation}
sgn(\phi)=\frac{\phi}{\sqrt{\phi^{2}+\varepsilon^2}}
\end{equation}
where $\varepsilon$ is usually taken as the grid spacing.


### Reference
- Hillebrand, B., Thieulot, C., Geenen, T., Van Den Berg, A. P., & Spakman, W. (2014). Using the level set method in geodynamical modeling of multi-material flows and Earth's free surface. Solid Earth, 5(2), 1087-1098.

- Samuel, H., & Evonuk, M. (2010). Modeling advection in geophysical flows with particle level sets. Geochemistry, Geophysics, Geosystems, 11(8).