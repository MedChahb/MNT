#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "type.h"
#include "darboux.h"
#include "io.h"
#include "check.h"

int main(int argc, char **argv) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input filename> [<output filename>]\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    mnt *m = NULL, *d = NULL;

    // Read the input file on the master process (rank 0)
    if (rank == 0) {
        m = mnt_read(argv[1]);
    }

    // Broadcast the dimensions and terrain data to all processes
    int nrows = 0, ncols = 0;
    if (rank == 0) {
        nrows = m->nrows;
        ncols = m->ncols;
    }

    // Broadcast the number of rows and columns
    MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        m = malloc(sizeof(mnt));
        m->nrows = nrows;
        m->ncols = ncols;
        m->no_data = 0;
        CHECK((m->terrain = malloc(nrows * ncols * sizeof(float))) != NULL);
    }
    
    // Broadcast terrain data to all processes
    MPI_Bcast(m->terrain, nrows * ncols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Perform the Darboux computation (all processes will work on this)
    d = darboux(m);

    // Write output on the master process (rank 0)
    if (rank == 0) {
        FILE *out = (argc == 3) ? fopen(argv[2], "w") : stdout;
        mnt_write(d, out);
        if (argc == 3) fclose(out);

        // Free memory on the master process
        free(m->terrain);
        free(m);
        free(d->terrain);
        free(d);
    } else {
        // Free memory on worker processes
        free(m->terrain);
        free(m);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
