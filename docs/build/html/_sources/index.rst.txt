.. lime documentation master file, created by
   sphinx-quickstart on Fri May  6 14:23:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to lime's documentation!
================================

The goal is to provide a simple-to-use package to study how light interacts with matter.  

Check docs/manual.pdf for documentations.

Main modules


Nonlinear molecular spectroscopy 
--------------------------------

#. Sum-of-states (SOS) for multilevel system

	Pros: computationally cheap and intuitive

	.. note:: In the SOS method, decay is introduced phenomelogically. 

#. correlation function approach --  Direct computing the many-point correlation functions with the quantum regression theorem 


	Valid for open quantum systems where environment effects can be rigirously described with quantum master equations. 

	Visulize with double-sided Feynman diagrams instead of the time-loop diagrams. 


#. non-perturbative approah -- valid for ALL systems provided a quantum dynamics solver is provided.  

	Simulating the laser-driven dynamics including explicitly all laser pulses    

Molecular quantum dynamics 
--------------------------

- Adiabatic wavepacket dynamics 
	* Split-operator method 
	* Discrete variable representation 

- Nonadiabatic wavepacket dynamics 
	* Split-operator method - For the exact nonadiabatic dynamics of vibronic models in the diabatic representation. 
	* RK4 -  For the exact nonadiabatic wavepacket dynamics in the adiabatic representation.


# Semiclassical quantum trajectory method 

# Quantum chemistry 

Open quantum systems 
--------------------
* Lindblad quantum master equation
* Redfield theory  
* second-order time-convolutionless master equation 
* hierarchical equation of motion 

# Quantum transport 
- Landauer transport 

Soid state materials 
--------------------
- Band structure from tight-binding Hamiltonians 

Periodically driven matter
--------------------------
* Floquet spectrum 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   lime
   modules
   heom
   Floquet


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
