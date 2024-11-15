CFLAGS=-Wall -O3 -march=native -g
OMP_CFLAGS=$(CFLAGS) -fopenmp
LDFLAGS=$(CFLAGS)
OMP_LDFLAGS=$(OMP_CFLAGS)

# Define the source files for each version
SEQ_SRC=$(filter-out darboux_OMP.c, $(wildcard *.c))
OMP_SRC=$(filter-out darboux.c, $(wildcard *.c))

# Define object files for each version
SEQ_OBJ=$(SEQ_SRC:.c=.o)
OMP_OBJ=$(OMP_SRC:.c=.omp.o)

all: main main_OMP

# Sequential version
main: $(SEQ_OBJ)
	$(CC) $(LDFLAGS) -o $@ $(SEQ_OBJ)

# Parallel version with OpenMP
main_OMP: $(OMP_OBJ)
	$(CC) $(OMP_LDFLAGS) -o $@ $(OMP_OBJ)

# Compile .o files for the sequential version
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile .omp.o files for the OpenMP version
%.omp.o: %.c
	$(CC) $(OMP_CFLAGS) -c $< -o $@

clean:
	rm -f $(SEQ_OBJ) $(OMP_OBJ) main main_OMP

test: main
	./main input/mini.mnt

test_OMP: main_OMP
	./main_OMP input/mini.mnt

# Recompile objects if headers or Makefile change
$(SEQ_OBJ) $(OMP_OBJ): $(wildcard *.h) Makefile