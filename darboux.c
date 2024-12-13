// fonction de calcul principale : algorithme de Darboux
// (remplissage des cuvettes d'un MNT)
#include <string.h>
#include "check.h"
#include "type.h"
#include "darboux.h"

#if defined(OMP) || defined(MPI)
#include <omp.h>
struct ThreadTiming {
    int thread_id;
    double time;
};
#endif

#ifdef MPI
#include "mpi.h"
#endif

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
  for (int i = 0; i < m->ncols * m->nrows; i++) {
    if (m->terrain[i] > max) {
      max = m->terrain[i];
    }
  }
  return max;
}

// initialise le tableau W de départ à partir d'un mnt m
float *init_W(const mnt *restrict m)
{
  const int ncols = m->ncols, nrows = m->nrows;
  float *restrict W;
  CHECK((W = malloc(ncols * nrows * sizeof(float))) != NULL);

  // initialisation W
  const float max = max_terrain(m) + 10.;
  for (int i = 0; i < nrows; i++) 
  {
    for (int j = 0; j < ncols; j++) 
    {
      if (i == 0 || i == nrows - 1 || j == 0 || j == ncols - 1 || TERRAIN(m, i, j) == m->no_data)
        WTERRAIN(W, i, j) = TERRAIN(m, i, j);
      else
        WTERRAIN(W, i, j) = max;
    }
  }

  return W;
}

// variables globales pour l'affichage de la progression
#ifdef DARBOUX_PPRINT
float min_darboux = 9999.; // ça ira bien, c'est juste de l'affichage
int iter_darboux = 0;
int printed = 0; // Avec MPI, on print plusieurs newLines indésirables

// fonction d'affichage de la progression
void dpprint()
{
    if (min_darboux != 9999.) 
    {
        fprintf(stderr, "%.3f %d\r", min_darboux, iter_darboux++);
        fflush(stderr);
        min_darboux = 9999.;
        printed = 1;
    } 
    else if (printed) 
    {
        fprintf(stderr, "\n"); // Print newLine une seule fois
        printed = 0; // Reset flag
    }
}
#endif

// pour parcourir les 8 voisins :
const int VOISINS[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

// cette fonction calcule le nouveau W[i,j] en utilisant Wprec[i,j]
// et ses 8 cases voisines : Wprec[i +/- 1 , j +/- 1],
// ainsi que le MNT initial m en position [i,j]
// inutile de modifier cette fonction (elle est sensible...):
int calcul_Wij(float *restrict W, const float *restrict Wprec, const mnt *m, const int i, const int j)
{
  const int nrows = m->nrows, ncols = m->ncols;
  int modif = 0;

  // on prend la valeur précédente...
  WTERRAIN(W, i, j) = WTERRAIN(Wprec, i, j);
  // ... sauf si :
  if (WTERRAIN(Wprec, i, j) > TERRAIN(m, i, j)) {
    // parcourir les 8 voisins haut/bas + gauche/droite
    for (int v = 0; v < 8; v++) {
      const int n1 = i + VOISINS[v][0];
      const int n2 = j + VOISINS[v][1];

      // vérifie qu'on ne sort pas de la grille.
      // ceci est théoriquement impossible, si les bords de la matrice Wprec
      // sont bien initialisés avec les valeurs des bords du mnt
      CHECK(n1 >= 0 && n1 < nrows && n2 >= 0 && n2 < ncols);

      // si le voisin est inconnu, on l'ignore et passe au suivant
      if (WTERRAIN(Wprec, n1, n2) == m->no_data)
        continue;

      CHECK(TERRAIN(m, i, j) > m->no_data);
      CHECK(WTERRAIN(Wprec, i, j) > m->no_data);
      CHECK(WTERRAIN(Wprec, n1, n2) > m->no_data);

      // il est important de mettre cette valeur dans un temporaire, sinon le
      // compilo fait des arrondis flotants divergents dans les tests ci-dessous
      const float Wn = WTERRAIN(Wprec, n1, n2) + EPSILON;
      if (TERRAIN(m, i, j) >= Wn) {
        WTERRAIN(W, i, j) = TERRAIN(m, i, j);
        modif = 1;
        #ifdef DARBOUX_PPRINT
        if (WTERRAIN(W, i, j) < min_darboux)
          min_darboux = WTERRAIN(W, i, j);
        #endif
      } else if (WTERRAIN(Wprec, i, j) > Wn) {
        WTERRAIN(W, i, j) = Wn;
        modif = 1;
        #ifdef DARBOUX_PPRINT
        if (WTERRAIN(W, i, j) < min_darboux)
          min_darboux = WTERRAIN(W, i, j);
        #endif
      }
    }
  }
  return modif;
}

#ifdef MPI
void exchange_boundaries(float *W, int start, int end, int ncols, int rank, int size)
{
    MPI_Request requests[4];
    int req_count = 0;

    // Post receives first to avoid deadlock
    if (rank > 0) {
        MPI_Irecv(W + (start - 1) * ncols, ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (rank < size - 1) {
        MPI_Irecv(W + end * ncols, ncols, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Then post sends
    if (rank > 0) {
        MPI_Isend(W + start * ncols, ncols, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (rank < size - 1) {
        MPI_Isend(W + (end - 1) * ncols, ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait for all communications to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }
}

void gather_data(float *W, int blockSize, int ncols, int nrows, int rank, int size)
{
    // Use non-blocking communication for gathering data
    if (rank == 0) {
        MPI_Request *requests = malloc((size - 1) * sizeof(MPI_Request));
        int req_count = 0;

        for (int i = 1; i < size; i++) {
            int recv_size = (i == size - 1) ? (nrows * ncols - i * blockSize * ncols) : blockSize * ncols;
            MPI_Irecv(W + i * blockSize * ncols, recv_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }

        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        free(requests);
    } else {
        MPI_Send(W + rank * blockSize * ncols, blockSize * ncols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
}
#endif

mnt *darboux(const mnt *restrict m)
{
    #if defined(OMP) || defined(MPI)
    int max_threads = omp_get_max_threads();
    struct ThreadTiming *thread_timings = malloc(max_threads * sizeof(struct ThreadTiming));
    for (int i = 0; i < max_threads; i++) {
        thread_timings[i].time = 0.0;
        thread_timings[i].thread_id = i;
    }
    #endif

    #ifdef MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif

    const int ncols = m->ncols, nrows = m->nrows;

    float *restrict W;
    CHECK((W = malloc(ncols * nrows * sizeof(float))) != NULL);
    float *restrict Wprec = init_W(m);

    #ifdef MPI
    MPI_Bcast(Wprec, ncols * nrows, MPI_FLOAT, 0, MPI_COMM_WORLD);
    int blockSize = nrows / size;
    int start = rank * blockSize;
    int end = (rank == size - 1) ? nrows : start + blockSize;
    #else
    int start = 0;
    int end = nrows;
    #endif

    int modif = 1;
    while (modif) {
        modif = 0;

        #if defined(OMP) || defined(MPI)
        int tid;
        double thread_start;

        // Process interior points first
        #pragma omp parallel private(tid, thread_start)
        {
            tid = omp_get_thread_num();
            thread_start = omp_get_wtime();

            // Use guided schedule for better load balancing
            #pragma omp for reduction(|:modif) schedule(guided, 64)
            for (int i = start + 1; i < end - 1; i++) {
                for (int j = 1; j < ncols - 1; j++) {
                    modif |= calcul_Wij(W, Wprec, m, i, j);
                }
            }

            double thread_end = omp_get_wtime();
            thread_timings[tid].time += thread_end - thread_start;
        }
        #else
        for (int i = start; i < end; i++) {
            for (int j = 0; j < ncols; j++) {
                modif |= calcul_Wij(W, Wprec, m, i, j);
            }
        }
        #endif

        #ifdef MPI
        // Handle boundaries
        exchange_boundaries(W, start, end, ncols, rank, size);

        // Process boundary points after exchange
        #pragma omp parallel sections reduction(|:modif)
        {
            #pragma omp section
            if (rank > 0) {
                for (int j = 0; j < ncols; j++) {
                    modif |= calcul_Wij(W, Wprec, m, start, j);
                }
            }

            #pragma omp section
            if (rank < size - 1) {
                for (int j = 0; j < ncols; j++) {
                    modif |= calcul_Wij(W, Wprec, m, end - 1, j);
                }
            }
        }

        // Global reduction
        int global_modif;
        MPI_Allreduce(&modif, &global_modif, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        modif = global_modif;
        #endif

        #ifdef DARBOUX_PPRINT
        #ifdef MPI
        if (rank == 0)
        #endif
            dpprint();
        #endif

        // Swap buffers
        float *tmp = W;
        W = Wprec;
        Wprec = tmp;
    }

    #ifdef MPI
    gather_data(W, blockSize, ncols, nrows, rank, size);
    #endif

    #if defined(OMP) || defined(MPI)
    #ifdef MPI
    fprintf(stderr, "\nProcess %d Thread Timings:\n", rank);
    #else
    fprintf(stderr, "\nThread Timings:\n");
    #endif
    for (int i = 0; i < max_threads; i++) {
        fprintf(stderr, "Thread %d: %.3f seconds\n", thread_timings[i].thread_id, thread_timings[i].time);
    }
    free(thread_timings);
    #endif

    free(Wprec);
    mnt *res;
    CHECK((res = malloc(sizeof(*res))) != NULL);
    memcpy(res, m, sizeof(*res));
    res->terrain = W;
    return res;
}