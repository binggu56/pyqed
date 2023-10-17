Nonlinear molecular spectroscopy 
================================


Sum-of-states (SOS) for multilevel system
-------------------

Computationally cheap and intuitive.

.. note:: In the SOS method, dissipation and decoherence are introduced phenomelogically. 

Correlation function approach 
-----------------------------
Direct computing the many-point correlation functions with the quantum regression theorem 

* Closed quantum systems


* Open quantum systems 

	Valid for open quantum systems where environment effects can be rigirously described with quantum master equations. 

	Visulize with double-sided Feynman diagrams instead of the time-loop diagrams. 

* Many-body systems
	Solve Dyson equaiton for one-particle Green's function

	.. math:: G = G_0 + G_0 \Sigma G 

Non-perturbative approah 
-------------------------

valid for ALL systems if a quantum dynamics solver is provided.  

	Simulating the laser-driven dynamics including explicitly all laser pulses. This usually requires phase clcyling to remove unwanted pathways.     