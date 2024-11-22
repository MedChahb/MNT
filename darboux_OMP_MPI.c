#include <string.h>
#include <omp.h>
#include <mpi.h> // Required for MPI functions
#include "check.h"
#include "type.h"
#include "darboux.h"

// si ce define n'est pas commenté, l'exécution affiche sur stderr la hauteur
// courante en train d'être calculée (doit augmenter) et l'itération du calcul
#define DARBOUX_PPRINT

#define PRECISION_FLOTTANT 1.e-5

// pour accéder à un tableau de flotant linéarisé (ncols doit être défini) :
#define WTERRAIN(w,i,j) (w[(i)*ncols+(j)])

// calcule la valeur max de hauteur sur un terrain
float max_terrain(const mnt *restrict m)
{
  float max = m->terrain[0];
  for(int i = 0 ; i < m->ncols * m->nrows ; i++)
    if(m->terrain[i] > max)
      max = m->terrain[i];
  return(max);
}

// initialise le tableau W de départ à partir d'un mnt m
float *init_W(const mnt *restrict m)
{
  const int ncols = m->ncols, nrows = m->nrows;
  float *restrict W;
  CHECK((W = malloc(ncols * nrows * sizeof(float))) != NULL);

  // initialisation W
  const float max = max_terrain(m) + 10.;
  for(int i = 0 ; i < nrows ; i++)
  {
    for(int j = 0 ; j < ncols ; j++)
    {
      if(i==0 || i==nrows-1 || j==0 || j==ncols-1 || TERRAIN(m,i,j) == m->no_data)
        WTERRAIN(W,i,j) = TERRAIN(m,i,j);
      else
        WTERRAIN(W,i,j) = max;
    }
  }

  return(W);
}

// variables globales pour l'affichage de la progression
#ifdef DARBOUX_PPRINT
float min_darboux=9999.; // ça ira bien, c'est juste de l'affichage
int iter_darboux=0;
// fonction d'affichage de la progression
void dpprint()
{
  if(min_darboux != 9999.)
  {
    fprintf(stderr, "%.3f %d\r", min_darboux, iter_darboux++);
    fflush(stderr);
    min_darboux = 9999.;
  }
  else
    fprintf(stderr, "\n");
}
#endif


// pour parcourir les 8 voisins :
const int VOISINS[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};

// cette fonction calcule le nouveau W[i,j] en utilisant Wprec[i,j]
// et ses 8 cases voisines : Wprec[i +/- 1 , j +/- 1],
// ainsi que le MNT initial m en position [i,j]
// inutile de modifier cette fonction (elle est sensible...):
int calcul_Wij(float *restrict W, const float *restrict Wprec, const mnt *m, const int i, const int j)
{
  const int nrows = m->nrows, ncols = m->ncols;
  int modif = 0;

  // on prend la valeur précédente...
  WTERRAIN(W,i,j) = WTERRAIN(Wprec,i,j);
  // ... sauf si :
  if(WTERRAIN(Wprec,i,j) > TERRAIN(m,i,j))
  {
    // parcourir les 8 voisins haut/bas + gauche/droite
    for(int v=0; v<8; v++)
    {
      const int n1 = i + VOISINS[v][0];
      const int n2 = j + VOISINS[v][1];

      // vérifie qu'on ne sort pas de la grille.
      // ceci est théoriquement impossible, si les bords de la matrice Wprec
      // sont bien initialisés avec les valeurs des bords du mnt
      CHECK(n1>=0 && n1<nrows && n2>=0 && n2<ncols);

      // si le voisin est inconnu, on l'ignore et passe au suivant
      if(WTERRAIN(Wprec,n1,n2) == m->no_data)
        continue;

      CHECK(TERRAIN(m,i,j)>m->no_data);
      CHECK(WTERRAIN(Wprec,i,j)>m->no_data);
      CHECK(WTERRAIN(Wprec,n1,n2)>m->no_data);

      // il est important de mettre cette valeur dans un temporaire, sinon le
      // compilo fait des arrondis flotants divergents dans les tests ci-dessous
      const float Wn = WTERRAIN(Wprec,n1,n2) + EPSILON;
      if(TERRAIN(m,i,j) >= Wn)
      {
        WTERRAIN(W,i,j) = TERRAIN(m,i,j);
        modif = 1;
        #ifdef DARBOUX_PPRINT
        if(WTERRAIN(W,i,j)<min_darboux)
          min_darboux = WTERRAIN(W,i,j);
        #endif
      }
      else if(WTERRAIN(Wprec,i,j) > Wn)
      {
        WTERRAIN(W,i,j) = Wn;
        modif = 1;
        #ifdef DARBOUX_PPRINT
        if(WTERRAIN(W,i,j)<min_darboux)
          min_darboux = WTERRAIN(W,i,j);
        #endif
      }
    }
  }
  return(modif);
}

/*****************************************************************************/
/*           Fonction de calcul principale - À PARALLÉLISER                  */
/*****************************************************************************/
// applique l'algorithme de Darboux sur le MNT m, pour calculer un nouveau MNT
mnt *darboux(const mnt *restrict m)
{
  // MPI setup
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int ncols = m->ncols, nrows = m->nrows;

  // Start timing for the entire function
  double start_total_time = omp_get_wtime();

  // Initialize rows per process
  int rows_per_process = nrows / size;
  int extra_rows = nrows % size; // Handle extra rows
  int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
  int num_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

  // Allocate local W and Wprec
  float *local_W = malloc((num_rows + 2) * ncols * sizeof(float));  // +2 for ghost rows
  float *local_Wprec = malloc((num_rows + 2) * ncols * sizeof(float));
  CHECK(local_W != NULL && local_Wprec != NULL);

  // Initialize Wprec locally
  for (int i = 0; i < num_rows; i++)
  {
    for (int j = 0; j < ncols; j++)
    {
      int global_row = start_row + i;
      WTERRAIN(local_Wprec, i + 1, j) = (global_row == 0 || global_row == nrows - 1 || j == 0 || j == ncols - 1 || TERRAIN(m, global_row, j) == m->no_data)
                                            ? TERRAIN(m, global_row, j)
                                            : max_terrain(m) + 10.0;
    }
  }

  int modif = 1, global_modif;

  // Main calculation loop
  while (modif)
  {
    modif = 0;

    // Share ghost rows with neighbors
    if (rank > 0) // Send top row to above neighbor
      MPI_Send(local_Wprec + ncols, ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
    if (rank < size - 1) // Send bottom row to below neighbor
      MPI_Send(local_Wprec + num_rows * ncols, ncols, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
    if (rank > 0) // Receive top ghost row
      MPI_Recv(local_Wprec, ncols, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank < size - 1) // Receive bottom ghost row
      MPI_Recv(local_Wprec + (num_rows + 1) * ncols, ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Compute new W
    #pragma omp parallel for reduction(|:modif) schedule(dynamic) // Keep existing OpenMP parallelization
    for (int i = 1; i <= num_rows; i++) // Exclude ghost rows
    {
      for (int j = 0; j < ncols; j++)
      {
        modif |= calcul_Wij(local_W, local_Wprec, m, start_row + i - 1, j);
      }
    }

    // Swap W and Wprec
    float *tmp = local_W;
    local_W = local_Wprec;
    local_Wprec = tmp;

    // Reduce across all processes to check if we need another iteration
    MPI_Allreduce(&modif, &global_modif, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    modif = global_modif;
  }

  // Gather results back to root
  float *global_W = NULL;
  if (rank == 0)
    global_W = malloc(nrows * ncols * sizeof(float));

  MPI_Gather(local_Wprec + ncols, num_rows * ncols, MPI_FLOAT, global_W, num_rows * ncols, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Create result structure
  mnt *res = NULL;
  if (rank == 0)
  {
    res = malloc(sizeof(*res));
    memcpy(res, m, sizeof(*res));
    res->terrain = global_W;
  }

  // End timing for the entire function
  double end_total_time = omp_get_wtime();

  // Print thread times and total execution time
  #ifdef DARBOUX_PPRINT
  for (int i = 0; i < omp_get_max_threads(); i++)
  {
    printf("Thread %d execution time: %.6f seconds\n", i, 0.0); // Example, can accumulate individual thread times if needed
  }
  #endif

  printf("Total execution time: %.6f seconds\n", end_total_time - start_total_time);

  // Cleanup
  free(local_W);
  free(local_Wprec);

  return res;
}
