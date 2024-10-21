#include <mpi.h>

#include <stdlib.h>
#include <stdio.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

#define mod(x, N) (x % N + N) % N

double *h, *u, *v;
double *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;

int *memcounts, *displcounts;
int nx, ny, rank, num_procs, num_cols;

double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    rank = rank_;
    num_procs = num_procs_;

    nx = nx_;
    ny = ny_;

    // The column this node starts at, inclusive
    int start_col = nx * rank / num_procs;
    // The column this node ends at, exclusive
    int end_col = (rank == num_procs - 1) ? nx : nx * (rank + 1) / num_procs;
    // The number of columns this node has
    num_cols = end_col - start_col;

    printf("Rank %d has %d columns from %d to %d\n", rank, num_cols, start_col, end_col);

    if (rank == 0)
    {
        h = h0;
        u = u0;
        v = v0;

        memcounts = (int *)calloc(num_procs, sizeof(int));
        displcounts = (int *)calloc(num_procs, sizeof(int));

        for (int i = 0; i < num_procs; i++)
        {
            // The start column of the i-th node
            int start_col_i = nx * i / num_procs;
            // The end column of the i-th node
            int end_col_i = (i == num_procs - 1) ? nx : nx * (i + 1) / num_procs;
            // The number of columns the i-th node has
            int num_cols_i = end_col_i - start_col_i;

            memcounts[i] = num_cols_i * (ny + 1);
            displcounts[i] = (i == 0) ? 0 : displcounts[i - 1] + memcounts[i - 1];

            printf("Rank %d has %d columns from %d to %d\n", i, num_cols_i, start_col_i, end_col_i);
        }

        MPI_Scatterv(h, memcounts, displcounts, MPI_DOUBLE, MPI_IN_PLACE, num_cols * (ny + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        h = (double *)calloc((num_cols + 1) * (ny + 1), sizeof(double));
        u = (double *)calloc((num_cols + 1) * ny, sizeof(double));
        v = (double *)calloc(num_cols * (ny + 1), sizeof(double));

        MPI_Scatterv(nullptr, memcounts, displcounts, MPI_DOUBLE, h, num_cols * (ny + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    dh = (double *)calloc(num_cols * ny, sizeof(double));
    du = (double *)calloc(num_cols * ny, sizeof(double));
    dv = (double *)calloc(num_cols * ny, sizeof(double));

    dh1 = (double *)calloc(num_cols * ny, sizeof(double));
    du1 = (double *)calloc(num_cols * ny, sizeof(double));
    dv1 = (double *)calloc(num_cols * ny, sizeof(double));

    dh2 = (double *)calloc(num_cols * ny, sizeof(double));
    du2 = (double *)calloc(num_cols * ny, sizeof(double));
    dv2 = (double *)calloc(num_cols * ny, sizeof(double));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;
}

void compute_dh()
{
    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }
}

void compute_du()
{
    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }
}

void compute_dv()
{
    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

void multistep_h(double a1, double a2, double a3)
{
    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
        }
    }
}

void multistep_u(double a1, double a2, double a3)
{
    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
        }
    }
}

void multistep_v(double a1, double a2, double a3)
{
    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
        }
    }
}

void compute_ghost_vertical()
{
    for (int i = 0; i < num_cols; i++)
    {
        h(i, ny) = h(i, 0);
    }
}

void compute_boundaries_vertical()
{
    for (int i = 0; i < num_cols; i++)
    {
        v(i, 0) = v(i, ny);
    }
}

void swap_buffers()
{
    double *tmp;

    tmp = dh2;
    dh2 = dh1;
    dh1 = dh;
    dh = tmp;

    tmp = du2;
    du2 = du1;
    du1 = du;
    du = tmp;

    tmp = dv2;
    dv2 = dv1;
    dv1 = dv;
    dv = tmp;
}

int t = 0;

void step()
{
    MPI_Request send_h_start;
    MPI_Isend(&h(0, 0), ny + 1, MPI_DOUBLE, mod(rank - 1, num_procs), 0, MPI_COMM_WORLD, &send_h_start);

    MPI_Request recv_h_end;
    MPI_Irecv(&h(num_cols, 0), ny + 1, MPI_DOUBLE, mod(rank + 1, num_procs), 0, MPI_COMM_WORLD, &recv_h_end);

    compute_ghost_vertical();

    double a1, a2, a3;

    if (t == 0)
    {
        a1 = 1.0;
    }
    else if (t == 1)
    {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
    }
    else
    {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }

    compute_dh();
    MPI_Wait(&send_h_start, MPI_STATUS_IGNORE);
    multistep_h(a1, a2, a3);

    compute_dv();
    multistep_v(a1, a2, a3);

    MPI_Wait(&recv_h_end, MPI_STATUS_IGNORE);
    compute_du();
    multistep_u(a1, a2, a3);

    MPI_Request send_u_start;
    MPI_Isend(&u(0, 0), ny, MPI_DOUBLE, mod(rank - 1, num_procs), 0, MPI_COMM_WORLD, &send_u_start);

    MPI_Request recv_u_end;
    MPI_Irecv(&u(num_cols, 0), ny, MPI_DOUBLE, mod(rank + 1, num_procs), 0, MPI_COMM_WORLD, &recv_u_end);

    compute_boundaries_vertical();
    MPI_Wait(&send_u_start, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_u_end, MPI_STATUS_IGNORE);

    swap_buffers();

    t++;
}

void transfer(double *h_recv)
{
    if (rank == 0)
    {
        MPI_Gatherv(MPI_IN_PLACE, num_cols * (ny + 1), MPI_DOUBLE, h, memcounts, displcounts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(h, num_cols * (ny + 1), MPI_DOUBLE, nullptr, memcounts, displcounts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void free_memory()
{
    printf("Rank %d freeing memory\n", rank);

    if (rank == 0)
    {
        free(memcounts);
        free(displcounts);
    }
    else
    {
        free(h);
        free(u);
        free(v);
    }

    free(dh);
    free(du);
    free(dv);

    free(dh1);
    free(du1);
    free(dv1);

    free(dh2);
    free(du2);
    free(dv2);
}