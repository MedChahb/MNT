#include <string.h>
#include <omp.h>
#include <mpi.h>
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
    const int ncols = m->ncols, nrows = m->nrows;
    float *restrict W, *restrict Wprec;

    int rank, size;

    // Get MPI rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide rows among processes
    int rows_per_proc = nrows / size;
    int extra_rows = nrows % size;
    int local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);

    // Allocate memory for local matrices, +2 for boundary rows
    CHECK((W = malloc((local_rows + 2) * ncols * sizeof(float))) != NULL);
    CHECK((Wprec = malloc((local_rows + 2) * ncols * sizeof(float))) != NULL);

    // Initialize Wprec (including boundary rows)
    for (int i = 0; i < (local_rows + 2) * ncols; i++) {
        Wprec[i] = 0.0f; // Initialize Wprec to zeros
    }

    // Set boundary rows from terrain
    for (int j = 0; j < ncols; j++) {
        if (rank == 0) {
            Wprec[j] = TERRAIN(m, 0, j);  // Top boundary
        }
        if (rank == size - 1) {
            Wprec[(local_rows + 1) * ncols + j] = TERRAIN(m, nrows - 1, j);  // Bottom boundary
        }
    }

    // Iteration flag and counter
    int modif = 1;
    int iter = 0;
    while (modif) {
        modif = 0;
        iter++;

        // Debug: Print iteration progress
        if (rank == 0 && iter % 10 == 0) {
            printf("Iteration %d, modif = %d\n", iter, modif);
        }

        // MPI communication: Exchange boundary data
        MPI_Status status;
        if (rank > 0) {
            // Send the first row and receive the last row from the previous process
            MPI_Sendrecv(&Wprec[ncols], ncols, MPI_FLOAT, rank - 1, 0,
                         &Wprec[0], ncols, MPI_FLOAT, rank - 1, 0,
                         MPI_COMM_WORLD, &status);  // Receive into the first row
        }
        if (rank < size - 1) {
            // Send the last row and receive the first row from the next process
            MPI_Sendrecv(&Wprec[(local_rows) * ncols], ncols, MPI_FLOAT, rank + 1, 0,
                         &Wprec[(local_rows + 1) * ncols], ncols, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, &status);  // Receive into the last row
        }

        // Update W using the neighboring cells
        #pragma omp parallel for reduction(|:modif) schedule(dynamic)
        for (int i = 1; i < local_rows + 1; i++) {  // Use local rows + boundary rows
            for (int j = 0; j < ncols; j++) {
                // Check if the new value is different from the old one
                int new_value = calcul_Wij(W, Wprec, m, i, j);  // Returns 1 if modification occurs
                modif |= new_value;
            }
        }

        // Reduce across all processes to check for global modifications
        int global_modif = 0;
        MPI_Allreduce(&modif, &global_modif, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        modif = global_modif;

        // Swap pointers for next iteration
        float *tmp = W;
        W = Wprec;
        Wprec = tmp;
    }

    // Gather results at the master process
    if (rank == 0) {
        float *final_result = malloc(nrows * ncols * sizeof(float));
        CHECK(final_result != NULL);
        memcpy(final_result, Wprec + ncols, local_rows * ncols * sizeof(float)); // Skip boundary rows

        for (int p = 1; p < size; p++) {
            int recv_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
            int offset = (p * rows_per_proc + (p < extra_rows ? p : extra_rows)) * ncols;
            MPI_Recv(&final_result[offset], recv_rows * ncols, MPI_FLOAT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        free(W);
        W = final_result;
    } else {
        MPI_Send(Wprec + ncols, local_rows * ncols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);  // Skip boundary rows
    }

    // Free local resources
    free(Wprec);
    mnt *res = NULL;
    if (rank == 0) {
        CHECK((res = malloc(sizeof(*res))) != NULL);
        memcpy(res, m, sizeof(*res));
        res->terrain = W;
    }
    return res;
}
