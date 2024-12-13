CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O3 -march=native -g
OMPFLAGS = -fopenmp

# Targets
all: main main_OMP main_OMP_MPI clean_obj

# Sequential version
main: io.o main.o darboux.o
	$(CC) $(CFLAGS) -o main darboux.o io.o main.o

# OpenMP version
main_OMP: darboux_OMP.o io.o main_omp.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o main_OMP darboux_OMP.o io.o main_omp.o

# MPI + OpenMP version
main_OMP_MPI: darboux_OMP_MPI.o io.o main_mpi.o
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o main_OMP_MPI darboux_OMP_MPI.o io.o main_mpi.o

# Object files
io.o: io.c io.h check.h type.h
	$(CC) $(CFLAGS) -c io.c -o io.o

main.o: main.c io.h check.h type.h
	$(CC) $(CFLAGS) -c main.c -o main.o

darboux.o: darboux.c darboux.h
	$(CC) $(CFLAGS) -c darboux.c -o darboux.o

darboux_OMP.o: darboux.c darboux.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -DOMP -c darboux.c -o darboux_OMP.o

main_omp.o: main.c io.h check.h type.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -DOMP -c main.c -o main_omp.o

darboux_OMP_MPI.o: darboux.c darboux.h
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -DMPI -DOMP -c darboux.c -o darboux_OMP_MPI.o

main_mpi.o: main.c io.h check.h type.h
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -DMPI -DOMP -c main.c -o main_mpi.o

# Clean targets
clean_obj:
	rm -f *.o

clean_exe: 
	rm -f main main_OMP main_OMP_MPI

clean: 
	rm -f *.o main main_OMP main_OMP_MPI text_*