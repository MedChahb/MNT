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
int printed = 0; // Avec MPI, on print plusieurs newLines indésirables
// fonction d'affichage de la progression
void dpprint()
{
    if(min_darboux != 9999.) 
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
#ifdef MPI
void exchange_boundaries(float *W, int start, int end, int ncols, int rank, int size)
{
    int start_offset = start * ncols;
    int end_offset = end * ncols;
    if (rank % 2 == 0) {
          if (rank + 1 < size) {
              MPI_Ssend(&W[end_offset - ncols], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
              MPI_Recv(&W[end_offset], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
          if (rank >= 1) {
              MPI_Ssend(&W[start_offset], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
              MPI_Recv(&W[start_offset - ncols], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        } 
        else {
            if (rank >= 1) {
                MPI_Recv(&W[start_offset - ncols], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&W[start_offset], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
            }
            if (rank + 1 < size) {
                MPI_Recv(&W[end_offset], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&W[end_offset - ncols], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            }
        }
}

void gather_data(float *W, int blockSize, int ncols, int nrows, int rank, int size)
{
    int start = rank * blockSize;
    int end = (rank == size - 1)? nrows : start + blockSize; 

    if (rank == 0) {
      for (int i = 1; i < size; i++) {
        if (i != size - 1) {
          MPI_Recv(W + i * blockSize * ncols, blockSize*ncols, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (i == size - 1) {
          MPI_Recv(W + i * blockSize * ncols, nrows*ncols-(i * blockSize) * ncols, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }
    else {
       MPI_Send(W + rank * blockSize * ncols,end*ncols-start* ncols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
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

  // initialisation
  float *restrict W, *restrict Wprec;
  CHECK((W = malloc(ncols * nrows * sizeof(float))) != NULL);
  Wprec = init_W(m);
  #ifdef MPI
  MPI_Bcast(Wprec, ncols * nrows, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #endif

  // calcul : boucle principale
  int modif = 1;
  #ifdef MPI
  int blockSize = nrows / size;
  int start = rank * blockSize;
  int end = (rank == size - 1)? nrows : start + blockSize;
  #elif defined(OMP)
  int start = 0;
  int end = nrows;
  #endif

  while (modif) {
      modif = 0;  // sera mis à 1 s'il y a une modification

      #if defined(OMP) || defined(MPI)
      int tid;
      double thread_start;
      #pragma omp parallel private(tid, thread_start)
      {
          tid = omp_get_thread_num();
          thread_start = omp_get_wtime();

          #pragma omp for reduction(|:modif) schedule(dynamic)
          // calcule le nouveau W fonction de l'ancien (Wprec) en chaque point [i,j]
          for (int i = start; i < end; i++) {
              for (int j=0; j<ncols; j++){
                // calcule la nouvelle valeur de W[i,j]
                // en utilisant les 8 voisins de la position [i,j] du tableau Wprec
                  modif |= calcul_Wij(W, Wprec, m, i, j);
              }
          }
          
          double thread_end = omp_get_wtime();
          thread_timings[tid].time += thread_end - thread_start;
      }
      #else
      // Sequential version
      for (int i = 0; i < nrows; i++) {
          for (int j = 0; j < ncols; j++) {
              modif |= calcul_Wij(W, Wprec, m, i, j);
          }
      }
      #endif

    #ifdef MPI
    exchange_boundaries(W, start, end, ncols, rank, size);
    #endif

    #ifdef DARBOUX_PPRINT
    #ifdef MPI
    if (rank == 0)
    #endif
      dpprint();
    #endif

    // échange W et Wprec
    // sans faire de copie mémoire : échange les pointeurs sur les deux tableaux
    float *tmp = W;
    W = Wprec;
    Wprec = tmp;

    #ifdef MPI
    int global_modif = 0;
    MPI_Allreduce(&modif, &global_modif, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    modif |= global_modif;
    #endif
  }
  // fin du while principal

  #ifdef MPI
  gather_data(W, blockSize, ncols, nrows, rank, size);
  #endif

  #if defined(OMP) || defined(MPI)
  // Print thread timing information
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
  // fin du calcul, le résultat se trouve dans W
  free(Wprec);
  // crée la structure résultat et la renvoie
  mnt *res;
  CHECK((res=malloc(sizeof(*res))) != NULL);
  memcpy(res, m, sizeof(*res));
  res->terrain = W;
  return(res);
}
