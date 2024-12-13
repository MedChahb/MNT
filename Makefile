CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O3 -march=native -g
OMPFLAGS = -fopenmp

# Targets
all: main main_OMP main_OMP_MPI clean_obj

main: io.o main.o darboux.o
	$(CC) $(CFLAGS) -o main darboux.o io.o main.o

main_OMP: darboux_OMP.o io_omp.o main_omp.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o main_OMP darboux_OMP.o io_omp.o main_omp.o

main_OMP_MPI: darboux_OMP_MPI.o io_mpi.o main_mpi.o
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o main_OMP_MPI darboux_OMP_MPI.o io_mpi.o main_mpi.o

# Object files
io.o: io.c io.h check.h type.h
	$(CC) $(CFLAGS) -c io.c -o io.o

main.o: main.c io.h check.h type.h
	$(CC) $(CFLAGS) -c main.c -o main.o

darboux.o: darboux.c darboux.h
	$(CC) $(CFLAGS) -c darboux.c -o darboux.o

darboux_OMP.o: darboux_OMP.c darboux.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -c darboux_OMP.c -o darboux_OMP.o

main_omp.o: main.c io.h check.h type.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -c main.c -o main_omp.o

io_omp.o: io.c io.h check.h type.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -c io.c -o io_omp.o

darboux_OMP_MPI.o: darboux_OMP_MPI.c darboux.h
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -c darboux_OMP_MPI.c -o darboux_OMP_MPI.o

main_mpi.o: main_MPI.c io.h check.h type.h
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -c main_MPI.c -o main_mpi.o

io_mpi.o: io.c io.h check.h type.h
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -c io.c -o io_mpi.o

# Clean
clean_obj:
	rm -f *.o

clean_exe : 
	rm main main_OMP main_OMP_MPI

clean_all : 
	rm -f *.o main main_OMP main_OMP_MPI text_*