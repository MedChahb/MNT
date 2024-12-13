#include <string.h>
#include <omp.h>
#include <mpi.h>
#include "check.h"
#include "type.h"
#include "darboux.h"

static const int VOISINS[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};

static void calculate_band_size(int rank, int size, int total_rows, int *start_row, int *num_rows) {
    int base_rows = total_rows / size;
    int extra = total_rows % size;
    
    if (rank < extra) {
        *num_rows = base_rows + 1;
        *start_row = rank * (*num_rows);
    } else {
        *num_rows = base_rows;
        *start_row = extra * (base_rows + 1) + (rank - extra) * base_rows;
    }
}



static int calcul_Wij(float *restrict W, const float *restrict Wprec, const mnt *m, 
                     const int i, const int j, const float *top_recv, const float *bottom_recv,
                     int start_row, int num_rows, int rank, int size)
{
    const int ncols = m->ncols;
    int modif = 0;
    int local_i = i - start_row;
    
    // Utiliser la même macro que dans type.h
    float current = Wprec[local_i * ncols + j];
    float terrain_val = TERRAIN(m, i, j);
    
    W[local_i * ncols + j] = current;

    if(current > terrain_val) {
        for(int v = 0; v < 8; v++) {
            int n1 = i + VOISINS[v][0];
            int n2 = j + VOISINS[v][1];
            
            if(n2 < 0 || n2 >= ncols || n1 < 0 || n1 >= m->nrows)
                continue;

            float neighbor_val;
            
            if(n1 < start_row && rank > 0) {
                if(top_recv == NULL) continue;
                neighbor_val = top_recv[n2];
            }
            else if(n1 >= start_row + num_rows && rank < size - 1) {
                if(bottom_recv == NULL) continue;
                neighbor_val = bottom_recv[n2];
            }
            else if(n1 >= start_row && n1 < start_row + num_rows) {
                neighbor_val = Wprec[(n1 - start_row) * ncols + n2];
            }
            else {
                continue;
            }

            if(neighbor_val == m->no_data)
                continue;

            const float Wn = neighbor_val + EPSILON;
            if(terrain_val >= Wn) {
                W[local_i * ncols + j] = terrain_val;
                modif = 1;
            }
            else if(current > Wn) {
                W[local_i * ncols + j] = Wn;
                modif = 1;
            }
        }
    }
    return modif;
}

static float max_terrain_band(const mnt *m, int start_row, int num_rows) {
    float max_val = TERRAIN(m, start_row, 0);
    
    #pragma omp parallel reduction(max:max_val)
    {
        #pragma omp for collapse(2)
        for(int i = start_row; i < start_row + num_rows; i++) {
            for(int j = 0; j < m->ncols; j++) {
                float val = TERRAIN(m, i, j);
                if(val > max_val) max_val = val;
            }
        }
    }
    return max_val;
}

static float *init_W_band(const mnt *m, int start_row, int num_rows) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    float *W;
    CHECK((W = malloc(num_rows * m->ncols * sizeof(float))) != NULL);

    float local_max = max_terrain_band(m, start_row, num_rows);
    float global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    global_max += 10.0;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < m->ncols; j++) {
            int global_i = start_row + i;
            if(global_i == 0 || global_i == m->nrows-1 || j == 0 || j == m->ncols-1 || 
               TERRAIN(m, global_i, j) == m->no_data) {
                W[i * m->ncols + j] = TERRAIN(m, global_i, j);
            } else {
                W[i * m->ncols + j] = global_max;
            }
        }
    }

    return W;
}

mnt *darboux(const mnt *restrict m) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int start_row, num_rows;
    calculate_band_size(rank, size, m->nrows, &start_row, &num_rows);

    float *W = malloc(num_rows * m->ncols * sizeof(float));
    float *Wprec = init_W_band(m, start_row, num_rows);
    CHECK(W != NULL);

    float *top_recv = NULL, *bottom_recv = NULL;
    float *top_send = NULL, *bottom_send = NULL;
    
    if (rank > 0) {
        top_send = malloc(m->ncols * sizeof(float));
        top_recv = malloc(m->ncols * sizeof(float));
        CHECK(top_send != NULL && top_recv != NULL);
    }
    if (rank < size - 1) {
        bottom_send = malloc(m->ncols * sizeof(float));
        bottom_recv = malloc(m->ncols * sizeof(float));
        CHECK(bottom_send != NULL && bottom_recv != NULL);
    }

    int global_modif = 1;
    while(global_modif) {
        int local_modif = 0;

        MPI_Request requests[4];
        int req_count = 0;

        if (rank > 0) {
            memcpy(top_send, Wprec, m->ncols * sizeof(float));
            MPI_Isend(top_send, m->ncols, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &requests[req_count]);
            MPI_Irecv(top_recv, m->ncols, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &requests[req_count+1]);
            req_count += 2;
        }
        if (rank < size - 1) {
            memcpy(bottom_send, &Wprec[(num_rows-1)*m->ncols], m->ncols * sizeof(float));
            MPI_Isend(bottom_send, m->ncols, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &requests[req_count]);
            MPI_Irecv(bottom_recv, m->ncols, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &requests[req_count+1]);
            req_count += 2;
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }

        #pragma omp parallel reduction(|:local_modif)
        {
            #pragma omp for schedule(dynamic)
            for(int i = start_row; i < start_row + num_rows; i++) {
                for(int j = 0; j < m->ncols; j++) {
                    if(i == 0 || i == m->nrows-1 || j == 0 || j == m->ncols-1 || 
                       TERRAIN(m, i, j) == m->no_data) {
                        W[(i-start_row)*m->ncols + j] = TERRAIN(m, i, j);
                        continue;
                    }
                    local_modif |= calcul_Wij(W, Wprec, m, i, j, top_recv, bottom_recv,
                                            start_row, num_rows, rank, size);
                }
            }
        }

        MPI_Allreduce(&local_modif, &global_modif, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        float *tmp = W;
        W = Wprec;
        Wprec = tmp;
    }

    mnt *res = NULL;
    if (rank == 0) {
        CHECK((res = malloc(sizeof(*res))) != NULL);
        memcpy(res, m, sizeof(*res));
        CHECK((res->terrain = malloc(m->nrows * m->ncols * sizeof(float))) != NULL);
    }

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int proc_start, proc_num_rows;
            calculate_band_size(i, size, m->nrows, &proc_start, &proc_num_rows);
            recvcounts[i] = proc_num_rows * m->ncols;
            displs[i] = offset;
            offset += proc_num_rows * m->ncols;
        }
    }

    MPI_Gatherv(Wprec, num_rows * m->ncols, MPI_FLOAT,
                res ? res->terrain : NULL, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    free(W);
    free(Wprec);
    if (top_send) free(top_send);
    if (top_recv) free(top_recv);
    if (bottom_send) free(bottom_send);
    if (bottom_recv) free(bottom_recv);

    return res;
}