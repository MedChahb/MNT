// fonction de calcul principale : algorithme de Darboux
// (remplissage des cuvettes d'un MNT)
#include <string.h>
#include <omp.h>
#include "mpi.h"
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
mnt *darboux(const mnt *restrict m )
{

  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  const int ncols = m->ncols, nrows = m->nrows;

  float *restrict W, *restrict Wprec;
  CHECK((W = malloc(ncols * nrows * sizeof(float))) != NULL);
  Wprec = init_W(m);
  MPI_Bcast(Wprec, ncols*nrows, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int modif = 1;
  int chunk_size = (int)((nrows / size));
  int start = rank * chunk_size;
  int end; 
  if (rank == size - 1){
    end = nrows;
  }
  else{
    end = start + chunk_size; 
  }

  while(modif)
    {
      
      modif = 0; // sera mis à 1 s'il y a une modification

      #pragma omp parallel for reduction(|:modif) schedule(dynamic)
      for(int i=start; i<end; i++)
      {
        for(int j=0; j<ncols; j++)
        {
          modif |= calcul_Wij(W, Wprec, m, i, j);
        }
      }

      #ifdef DARBOUX_PPRINT
      dpprint();
      #endif
      

     // Gestion des échanges de frontières
      if (rank % 2 == 0) {
          if (rank + 1 < size) {
              MPI_Ssend(&W[(end - 1) * ncols], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
              MPI_Recv(&W[end * ncols], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
          if (rank - 1 >= 0) {
              MPI_Ssend(&W[start * ncols], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
              MPI_Recv(&W[(start -1)* ncols], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        } 
        else {
            if (rank - 1 >= 0) {
                MPI_Recv(&W[(start - 1) * ncols], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&W[start * ncols], ncols, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
            }

            if (rank + 1 < size) {
                MPI_Recv(&W[end * ncols], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&W[(end-1) * ncols], ncols, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            }

        }
       float *tmp = W;

        W = Wprec;
        Wprec = tmp;
      
      int global_modif = 0;
      MPI_Allreduce(&modif, &global_modif, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
      modif |= global_modif;
    }
    if (rank == 0) {
      for (int i = 1; i < size; i++) {
        if (i != size - 1) {
          MPI_Recv(W + (i * chunk_size) * ncols, chunk_size*ncols, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (i == size - 1) {
          MPI_Recv(W + (i * chunk_size) * ncols, nrows*ncols-(i * chunk_size) * ncols, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }
    else {
       MPI_Send(W + (rank * chunk_size) * ncols,end*ncols-start* ncols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

  free(Wprec);
  mnt *res;
  CHECK((res=malloc(sizeof(*res))) != NULL);
  memcpy(res, m, sizeof(*res));
  res->terrain = W;
  return(res);
} 
