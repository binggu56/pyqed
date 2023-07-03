.. lime documentation master file, created by
   sphinx-quickstart on Fri May  6 14:23:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyQED's documentation!
================================

The goal is to provide a simple-to-use package to study ``how light interacts with matter``.  

Check docs/manual.pdf for theoretical details.

Main modules


* Nonlinear molecular spectroscopy 
 

* Molecular quantum dynamics 
--------------------------

- Adiabatic wavepacket dynamics 
	* Split-operator method 
	* Discrete variable representation 

- Nonadiabatic wavepacket dynamics 
	* Split-operator method - For the exact nonadiabatic dynamics of vibronic models in the diabatic representation. 
	* RK4 -  For the exact nonadiabatic wavepacket dynamics in the adiabatic representation.


# Semiclassical quantum trajectory method 

Quantum chemistry
-----------------
* TDDFT core-level excitation 
** reduced excitation space
** restricted energy window with full/reduced excitation space

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

   guide/guide.rst
   pyqed
   developers.rst
   heom
   Floquet



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
