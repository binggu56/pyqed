---------------------
DESCRIPTION:

The program simulates a quantum solid (helium4) within the dissipative quantum trajecotry framework. 

The details of method can be found in the paper (Bing Gu, R.J Hinde, Vitaly Rassolov, Sophya Garashchuk JCTC, 2015). 

Paralleled with MPI, the main advantage comes from the distribution of trajectories into different processors and computation of the 
matrix that is needed for the fitting of quantum potential. 


----------------------
SUBROUTINES 

	lattice-file-180 : the Cartesian configuration file for solid helium 
	
	vinit.f : This subroutine setup the linear interpolation array for the pariwise potential. 

	gasdev.f, ran1.f : normal random sampling subroutins of trajectories 

	dervis.f : classical force fields

	fit.f : solve the fitting of quantum potential (two-step fitting)  


-----------------------
To use this code for systems other than solid helium4, things you have to change: 


	vinit.f : This subroutine setup the linear interpolation array for the pariwise potential. 
		    Simply change the interparticle potential to your potential 

	IN      : Includes parameters that define the mass, initial wavefunction, interparticle distance, 
		    cutoff distance, changes to the values corresponding to 
		    your system 

	Makefile : add source files if you have extra 

----------------------
command to compile : make qm.x 
command to run     : ./qm.x 
command to submit to POPLE : qsub qsub.sh 


	
