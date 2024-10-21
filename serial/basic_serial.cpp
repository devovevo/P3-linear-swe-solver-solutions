#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

int nx, ny;

double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    h = h0;
    u = u0;
    v = v0;

    nx = nx_;
    ny = ny_;

    dh = (double *)calloc(nx * ny, sizeof(double));
    du = (double *)calloc(nx * ny, sizeof(double));
    dv = (double *)calloc(nx * ny, sizeof(double));

    dh1 = (double *)calloc(nx * ny, sizeof(double));
    du1 = (double *)calloc(nx * ny, sizeof(double));
    dv1 = (double *)calloc(nx * ny, sizeof(double));

    dh2 = (double *)calloc(nx * ny, sizeof(double));
    du2 = (double *)calloc(ny * ny, sizeof(double));
    dv2 = (double *)calloc(nx * ny, sizeof(double));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;
}

void compute_dh()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }
}

void compute_du()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }
}

void compute_dv()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

void multistep(double a1, double a2, double a3)
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
        }
    }
}

void compute_ghost_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        h(nx, j) = h(0, j);
    }
}

void compute_ghost_vertical()
{
    for (int i = 0; i < nx; i++)
    {
        h(i, ny) = h(i, 0);
    }
}

void compute_boundaries_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        u(0, j) = u(nx, j);
    }
}

void compute_boundaries_vertical()
{
    for (int i = 0; i < nx; i++)
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
    compute_ghost_horizontal();
    compute_ghost_vertical();

    compute_dh();
    compute_du();
    compute_dv();

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

    multistep(a1, a2, a3);

    compute_boundaries_horizontal();
    compute_boundaries_vertical();

    swap_buffers();

    t++;
}

void transfer(double *h)
{
    return;
}

void free_memory()
{
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