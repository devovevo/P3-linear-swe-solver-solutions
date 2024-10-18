#include <mpi.h>

#define h(i, j) h[(i) * (ny + 1) + (j)]
#define u(i, j) u[(i) * (ny + 1) + (j)]
#define v(i, j) v[(i) * (ny + 2) + (j)]

#define dh(i, j) dh[(i) * ny + (j)]
#define du(i, j) du[(i) * ny + (j)]
#define dv(i, j) dv[(i) * (ny + 1) + (j)]

#define dh1(i, j) dh1[(i) * ny + (j)]
#define du1(i, j) du1[(i) * ny + (j)]
#define dv1(i, j) dv1[(i) * (ny + 1) + (j)]

#define dh2(i, j) dh2[(i) * ny + (j)]
#define du2(i, j) du2[(i) * ny + (j)]
#define dv2(i, j) dv2[(i) * (ny + 1) + (j)]

#define dh_dx(i, j) (h(i + 1, j) - h(i, j)) / dx
#define dh_dy(i, j) (h(i, j + 1) - h(i, j)) / dy

#define du_dx(i, j) (u(i + 1, j) - u(i, j)) / dx
#define dv_dy(i, j) (v(i, j + 1) - v(i, j)) / dy

double *h, *u, *v;
double *h_left, *u_left, *h_right, *u_right, *v_right;

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

    if (num_procs_ > 1)
    {
        h_left = (double *)calloc(ny + 1, sizeof(double));
        u_left = (double *)calloc(ny + 1, sizeof(double));

        h_right = (double *)calloc(ny + 1, sizeof(double));
        u_right = (double *)calloc(ny + 1, sizeof(double));
        v_right = (double *)calloc(ny + 1, sizeof(double));
    }

    if (rank == 0)
    {
        recvcounts = (int *)calloc(num_procs, sizeof(int));

        for (int i = 0; i < num_procs; i++)
        {
            int start_row_i = nx_ * i / num_procs;
            int end_row_i = rank == num_procs - 1 ? ny : nx * (rank + 1) / num_procs - 1;
            int num_rows_i = end_row_i - start_row_i + 1;

            recvcounts[i] = num_rows_i * (ny + 1);
            displcounts[i] = i == 0 : 0 ? displcounts[i - 1] + recvcounts[i - 1];
        }

        recvbuf = (double *)calloc((nx + 1) * (ny + 1), sizeof(double));
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

compute_dh()
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }

    for (int j = 0; j < ny; j++)
    {
        double du_dx = (u_right[j] - u(end_row, j)) / dx;
        dh(end_row, j) = -H * (du_dx + dv_dy(end_row, j));
    }
}

compute_du()
{
    for (int i = start_row; i < end_row + 1; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }

    for (int j = 0; j < ny; j++)
    {
        double dh_dx = (h_right[j] - h(end_row, j)) / dx;
        du(end_row + 1, j) = -g * dh_dx;
    }
}

compute_dv()
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < ny + 1; j++)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

void compute_boundaries_left()
{
    for (int j = 0; j < ny + 1; j++)
    {
        u(start_row, j) = u_up[j];
    }
}

void compute_boundaries_right()
{
    for (int j = 0; j < ny + 1; j++)
    {
        u(end_row + 1, j) = u_down[j];
        v(end_row, j) = v(0, j);
    }
}

void compute_boundaries_vertical()
{
    for (int i = start_row; i < end_row + 1; i++)
    {
        u(i, ny) = u(i, 0);

        v(i, 0) = v(i, ny);
        v(i, ny + 1) = v(i, 1);

        h(i, ny) = h(i, 0);
    }
}

void gather_h(double *h_recv)
{
    MPI_Gatherv(h + start_row * (ny + 1), num_rows * (ny + 1), MPI_DOUBLE, h_recv, recvcounts, displcounts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}