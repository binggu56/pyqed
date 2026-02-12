objects = mod.o qm.o derivs.o gasdev.o ran1.o fit.o
#objects2 = long_range.o reduce.o cont.o derivs.o input.o tstamp.o vinit.o qpot.o gasdev.o ran2.o
FC = gfortran
#FC = ifort
switch = -O3
EXEC = qm.x
#FLAGS = -heap-arrays
#EXEC2 = cont
LIB = /usr/lib64/atlas/liblapack.so.3.0
#LIB = -llapack  
#LIB = -mkl=sequential 

$(EXEC): $(objects)
	$(FC) -o $(EXEC) $(FLAGS) $(LIB) $(switch) $(objects)
#$(EXEC2): $(objects2)
#	$(FC) -o $(EXEC2) $(LIB) $(switch) $(objects2)

derivs.o : derivs.f90
	$(FC) -c derivs.f90 

%.o: %.f 
	$(FC)  -c  $<
%.o: %.f90
	$(FC) -c $<

clean:
	rm $(objects)
veryclean:
	rm *.o *.dat
