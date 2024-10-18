#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

int nx, ny;

double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_)
{
    h = h0;
    u = u0;
    v = v0;

    nx = nx_;
    ny = ny_;

    dh = (double *)calloc(nx * ny, sizeof(double));
    du = (double *)calloc((nx + 1) * ny, sizeof(double));
    dv = (double *)calloc(nx * (ny + 1), sizeof(double));

    dh1 = (double *)calloc(nx * ny, sizeof(double));
    du1 = (double *)calloc((nx + 1) * ny, sizeof(double));
    dv1 = (double *)calloc(nx * (ny + 1), sizeof(double));

    dh2 = (double *)calloc(nx * ny, sizeof(double));
    du2 = (double *)calloc((nx + 1) * ny, sizeof(double));
    dv2 = (double *)calloc(nx * (ny + 1), sizeof(double));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;
}

void compute_dh(double *h, double *u, double *v)
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
        }
    }
}

void compute_du(double *h, double *u, double *v)
{
    for (int i = 0; i < nx + 1; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            du(i, j) = -g * dh_dx(i, j);
        }
    }
}

void compute_dv(double *h, double *u, double *v)
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny + 1; j++)
        {
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

void compute_boundaries_horizontal(double *h, double *u, double *v)
{
    for (int j = 0; j < ny + 1; j++)
    {
        u(0, j) = u(nx, j);
        u(nx + 1, j) = u(1, j);

        v(nx, j) = v(0, j);

        h(nx, j) = h(0, j);
    }
}

void compute_boundaries_vertical(double *h, double *u, double *v)
{
    for (int i = 0; i < nx + 1; i++)
    {
        u(i, ny) = u(i, 0);

        v(i, 0) = v(i, ny);
        v(i, ny + 1) = v(i, 1);

        h(i, ny) = h(i, 0);
    }
}

int t = 0;

void euler(double *h, double *u, double *v)
{
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

void swap_buffers()
{
    double *tmp;

    tmp = dh;
    dh = dh1;
    dh1 = dh2;
    dh2 = tmp;

    tmp = du;
    du = du1;
    du1 = du2;
    du2 = tmp;

    tmp = dv;
    dv = dv1;
    dv1 = dv2;
    dv2 = tmp;
}

void step(double *h, double *u, double *v)
{
    compute_boundaries_horizontal(h, u, v);
    compute_boundaries_vertical(h, u, v);

    compute_dh(h, u, v);
    compute_du(h, u, v);
    compute_dv(h, u, v);

    euler(h, u, v);

    swap_buffers();

    t++;
}

void free_memory()
{
    free(dh);
    free(du);
    free(dv);
}