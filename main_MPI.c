// programme principal
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "type.h"
#include "io.h"
#include "darboux.h"

int main(int argc, char **argv)
{
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  mnt *m, *d;

  if(argc < 2)
  {
    fprintf(stderr, "Usage: %s <input filename> [<output filename>]\n", argv[0]);
    exit(1);
  }

  // READ INPUT
  m = mnt_read(argv[1]);

  double t1 = omp_get_wtime();
  

  // COMPUTE
  d = darboux(m);

  double t2 = omp_get_wtime();
 if (rank == 0){
  // WRITE OUTPUT
  FILE *out;
  if(argc == 3)
    out = fopen(argv[2], "w");
  else
    out = stdout;
  mnt_write(d, out);
  if(argc == 3)
    fclose(out);
  else
    mnt_write_lakes(m, d, stdout);
  }


  double t3 = omp_get_wtime();
  // free
  free(m->terrain);
  free(m);
  free(d->terrain);
  free(d);

  if(rank == 0){
    printf("\nCompute: %lfs, Savefile: %lf\n", t2-t1, t3-t2);
  }
   MPI_Finalize();
  return(0);
}
