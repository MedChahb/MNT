// programme principal
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "type.h"
#include "io.h"
#include "darboux.h"

#ifdef OMP
#include <omp.h>
#endif

#ifdef MPI
#include "mpi.h"
#include <omp.h>
#endif

#define SEQ_EXEC_TIME 188.876  // temps sequentiel de calcul avec large.mnt

int main(int argc, char **argv)
{
  double start_time, end_time;
  
  #ifdef MPI
  int rank, size;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_processor_name(processor_name, &name_len);
  #endif

  mnt *m, *d;

  if(argc < 2)
  {
    fprintf(stderr, "Usage: %s <input filename> [<output filename>]\n", argv[0]);
    exit(1);
  }

  // Start timing
  #ifdef MPI
  start_time = MPI_Wtime();
  #elif defined(OMP)
  start_time = omp_get_wtime();
  #else
  start_time = (double)clock() / CLOCKS_PER_SEC;
  #endif

  // READ INPUT
  m = mnt_read(argv[1]);

  // COMPUTE
  d = darboux(m);

  // End timing
  #ifdef MPI
  end_time = MPI_Wtime();
  #elif defined(OMP)
  end_time = omp_get_wtime();
  #else
  end_time = (double)clock() / CLOCKS_PER_SEC;
  #endif
  
  double execution_time = end_time - start_time;

  #ifdef MPI
  if (rank == 0) {
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

  // Print total execution time
  fprintf(stderr, "\nTotal execution time: %.3f seconds\n", execution_time);
  
  // Print speedup for parallel versions
  #if defined(OMP) || defined(MPI)
  fprintf(stderr, "Speedup: %.2fx\n", SEQ_EXEC_TIME / execution_time);
  #endif

  #ifdef MPI
  }

  // Print MPI process timing information
  fprintf(stderr, "Process %d on %s: Execution time = %.3f seconds\n", 
          rank, processor_name, execution_time);
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