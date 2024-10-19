#include <mpi.h>

#include "common.h"

double *h, *u, *v;
double *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;

int nx, ny;

int rank, num_procs, start_row, end_row, num_rows;
double H, g, dx, dy, dt;

int *recvcounts, *displcounts;

void init_mpi(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    h = h0;
    u = u0;
    v = v0;

    rank = rank_;
    num_procs = num_procs_;

    nx = nx_;
    ny = ny_;

    start_row = nx * rank / num_procs;
    end_row = rank == num_procs - 1 ? nx : nx * (rank + 1) / num_procs - 1;
    num_rows = end_row - start_row + 1;

    if (rank == 0)
    {
        recvcounts = (int *)calloc(num_procs, sizeof(int));

        for (int i = 0; i < num_procs; i++)
        {
            int start_row_i = nx_ * i / num_procs;
            int end_row_i = rank == num_procs - 1 ? nx : nx * (rank + 1) / num_procs - 1;
            int num_rows_i = end_row_i - start_row_i + 1;

            recvcounts[i] = num_rows_i * (ny + 1);
            displcounts[i] = i == 0 : 0 ? displcounts[i - 1] + recvcounts[i - 1];
        }
    }

    dh = (double *)calloc(num_rows * ny, sizeof(double));
    du = (double *)calloc((num_rows + 1) * ny, sizeof(double));
    dv = (double *)calloc(num_rows * (ny + 1), sizeof(double));

    dh1 = (double *)calloc(num_rows * ny, sizeof(double));
    du1 = (double *)calloc((num_rows + 1) * ny, sizeof(double));
    dv1 = (double *)calloc(num_rows * (ny + 1), sizeof(double));

    dh2 = (double *)calloc(num_rows * ny, sizeof(double));
    du2 = (double *)calloc((num_rows + 1) * ny, sizeof(double));
    dv2 = (double *)calloc(num_rows * (ny + 1), sizeof(double));

    H = H_;
    g = g_;

    dx = length / nx;
    dy = width / nx;

    dt = dt_;
}

void compute_dh()
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }
}

void compute_du()
{
    for (int i = start_row; i < end_row + 1; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }
}

void compute_dv()
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < ny + 1; j++)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

void compute_boundaries_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        u(start_row, j) = u(start_row - 1 % (nx + 1), j);
        u(end_row + 1, j) = u(start_row + 1, j);

        v(end_row, j) = v(0, j);

        h(end_row, j) = h(0, j);
    }
}

void compute_boundaries_vertical()
{
    for (int i = start_row; i < end_row; i++)
    {
        u(i, ny) = u(i, 0);

        v(i, 0) = v(i, ny);
        v(i, ny + 1) = v(i, 1);

        h(i, ny) = h(i, 0);
    }
}

void step()
{
    MPI_Request h_right_r;
    MPI_Irecv(&h(end_row + 1, 0), ny + 1, MPI_DOUBLE, (rank + 1) & num_procs, 0, MPI_COMM_WORLD, &h_right_r);

    MPI_Request u_right_r;
    MPI_Irecv(&u(end_row + 2, 0), ny + 1, MPI_DOUBLE, (rank + 1) & num_procs, 0, MPI_COMM_WORLD, &u_right_r);

    compute_boundaries_vertical();
    compute_dv();

    MPI_Wait(&h_right_r, MPI_STATUS_IGNORE);
    compute_dh();

    MPI_Wait(&u_right_r, MPI_STATUS_IGNORE);
    compute_du();

    compute_boundaries_horizontal();

    euler();

    swap_buffers();
}

void gather_h(double *h_recv)
{
    MPI_Gatherv(h + start_row * (ny + 1), num_rows * (ny + 1), MPI_DOUBLE, h_recv, recvcounts, displcounts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}