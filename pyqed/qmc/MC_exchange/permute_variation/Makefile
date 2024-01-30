objects = mod.o local.o monte.o gasdev.o ran1.o 

FC = gfortran 
#FC = ifort
switch = -g -O3
EXEC = mc.exe
#FLAGS = -heap-arrays
#EXEC2 = cont
LIB = /usr/lib64/atlas/liblapack.so.3.0
#LIB = -llapack  
#LIB = -mkl=sequential 

$(EXEC): $(objects)
	$(FC) -o $(EXEC) $(FLAGS) $(LIB) $(switch) $(objects)
#$(EXEC2): $(objects2)
#	$(FC) -o $(EXEC2) $(LIB) $(switch) $(objects2)

cdat.mod: mod.o mod.f90 
	$(FC) -g -c mod.f90 
	
%.o: %.f
	$(FC)  -g -c  $<
%.o: %.f90
	$(FC) -g -c $<

clean:
	rm $(objects)
veryclean:
	rm *.o *.dat
