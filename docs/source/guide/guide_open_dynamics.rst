Open quantum dynamics
=====================

Dynamics of open quantum systems requries to solve the quantum master equaiton 

.. math:: 
	
	i \dot{\rho} = (\mathcal{L}_0 + \mathcal{L}_1) \rho 


Lindblad quantum master equaiton
--------------------------------

Redfield equaiton
-----------------

Time-convolutionless master equation
------------------------------------


HEOM
----

Models 
------

* Spin-boson model


.. math::
	H = \frac{\Delta}{2} \sigma_x  + \sum_k \sigma_z {g_k a_k^\dagger + g_k^*a_k} + \sum_k \omega_k a^\dagger_k a_k 

The environmental influence to the system dynamics is encoded in the so-called spectral density, 

.. math::
	J(\omega) = \sum_k |{g_k}|^2 \delta(\omega - \omega_k). 

We here implement the Lorentz-Drude form 

.. math::

	J(\omega) = \frac{2\lambda \omega \gamma}{\omega^2 + \gamma^2}

with a single exponential for the time-correlation function. 

.. math::

	D(t) = \pi^{-1} \int_0^\infty d \omega J(\omega)(\coth(\beta\omega/2) - i \sin(\omega t)) \approx \lambda (2T - i \gamma) e^{-\gamma t}


