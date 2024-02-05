Open quantum dynamics
=====================

Dynamics of open quantum systems requries to solve the quantum master equaiton 

.. math:: 
	
	i \dot{\rho} = \del{\mathcal{L}_0 + \mathcal{L}_1} \rho 

Spin-boson model
----------------

.. math::
	H = \frac{\Delta}{2} \sigma_x  + \sum_k \sigma_z \del{g_k a_k^\dag + g_k^*a_k} + \sum_k \omega_k a^\dag_k a_k 

The environmental influence to the system dynamics is encoded in the so-called spectral density, 
.. math::
	J(\omega) = \sum_k \abs{g_k}^2 \delta(\omega - \omega_k). 

We here implement the Lorentz-Drude form 
.. math::
	J(\omega) = \frac{2\lambda \omega \gamma}{\omega^2 + \gamma^2}
with a single exponential for the time-correlation function. 
.. math::
	D(t) = \pi^{-1} \int_0^\infty \dif \omega J(\omega)    


