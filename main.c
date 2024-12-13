// programme principal
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "type.h"
#include "io.h"
#include "darboux.h"

#ifdef MPI
#include "mpi.h"
#endif

int main(int argc, char **argv)
{
  #ifdef MPI
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  #endif

  mnt *m, *d;

  if(argc < 2)
  {
    fprintf(stderr, "Usage: %s <input filename> [<output filename>]\n", argv[0]);
    exit(1);
  }

  // READ INPUT
  m = mnt_read(argv[1]);

  // COMPUTE
  d = darboux(m);

  
  #ifdef MPI
  if (rank == 0){
  #endif

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
  
  #ifdef MPI
  }
  #endif

  // free
  free(m->terrain);
  free(m);
  free(d->terrain);
  free(d);


  #ifdef MPI
  MPI_Finalize();
  #endif
  
  return(0);
}
