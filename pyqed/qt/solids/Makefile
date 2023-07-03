objects = mod.o long_range.o qm.o derivs.o vinit.o gasdev.o ran1.o fit.o
#FC = gfortran
FC = mpif90
switch = -O3 -w
EXEC = qm
#FLAGS = -heap-arrays
#EXEC2 = cont
#LIB = /usr/lib64/atlas/liblapack.so.3.0
#LIB = -llapack  
LIB = -mkl=sequential 

$(EXEC): $(objects)
	$(FC) -o $(EXEC) $(FLAGS) $(LIB) $(switch) $(objects)

cdat.mod: mod.f
	$(FC) -c mod.f

%.o: %.f
	$(FC) -c $<

%.o: %.f90
	$(FC) -c $<

clean:
	rm $(objects) *.mod
veryclean:
	rm *.o *.dat *.mod
