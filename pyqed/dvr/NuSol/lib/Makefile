CC=icc
FCC=gfortran
CFLAGS=-O3
INCL=-I/$(FEASTROOT)/include
LIB=-L/$(FEASTROOT)/lib/x64 -lfeast_sparse -lfeast -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -limf -lsvml -lirc -liomp5 -lpthread -lm  

all: NuSol_FEAST

NuSol_FEAST:
		echo $(CC) $(CFLAGS) -static $(INCL) -c NuSol_FEAST.c;\
		$(CC) $(CFLAGS) $(INCL) -c NuSol_FEAST.c ;\
		echo $(CC) -static -o NuSol_FEAST NuSol_FEAST.o $(LIB) ;\
		$(CC)   -o NuSol_FEAST NuSol_FEAST.o  $(LIB) ;\

clean: 
	-@rm  NuSol_FEAST *.o *.pyc
