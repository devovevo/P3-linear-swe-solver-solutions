#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

#define hb(i, j) hb[(i) * (ny + 1) + (j)]
#define ub(i, j) ub[(i) * (ny) + (j)]
#define vb(i, j) vb[(i) * (ny + 1) + (j)]

#define dhb_dx(i, j) (hb(i + 1, j) - hb(i, j)) / dx
#define dhb_dy(i, j) (hb(i, j + 1) - hb(i, j)) / dy

#define dub_dx(i, j) (ub(i + 1, j) - ub(i, j)) / dx
#define dvb_dy(i, j) (vb(i, j + 1) - vb(i, j)) / dy

// Here we hold the number of cells we have in the x and y directions
int nx, ny;

// This is where all of our points are. We need to keep track of our active
// height and velocity grids, but also the corresponding derivatives. The reason
// we have 2 copies for each derivative is that our multistep method uses the
// derivative from the last 2 time steps.
float *hb, *ub, *vb, *dh, *du, *dv, *dh1, *du1, *dv1;
float H, g, dx, dy, dt;

void init(float *h, float *u, float *v, float length_, float width_, int nx_, int ny_, float H_, float g_, float dt_)
{
    // We store the number of points we have in the x and y directions
    nx = nx_;
    ny = ny_;

    // We allocate an extra space so that it's easier to compute boundaries
    // without needing branches. Here, the 'b' versions of our fields
    // represent them with boundaries.
    hb = (float *)calloc((nx + 1) * (ny + 1), sizeof(float));
    ub = (float *)calloc((nx + 1) * ny, sizeof(float));
    vb = (float *)calloc(nx * (ny + 1), sizeof(float));

    // We copy these fields into our new expanded array. We offset
    // our 'ub' and 'vb' fields by one since in these cases our boundaries
    // are the first elements
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            hb(i, j) = h(i, j);
            ub(i + 1, j) = u(i, j);
            vb(i, j + 1) = v(i, j);
        }
    }

    // We allocate memory for the derivatives
    dh = (float *)calloc(nx * ny, sizeof(float));
    du = (float *)calloc(nx * ny, sizeof(float));
    dv = (float *)calloc(nx * ny, sizeof(float));

    dh1 = (float *)calloc(nx * ny, sizeof(float));
    du1 = (float *)calloc(nx * ny, sizeof(float));
    dv1 = (float *)calloc(nx * ny, sizeof(float));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / ny;

    dt = dt_;
}

/**
 * This function computes our derivatives.
 */
void compute_derivs()
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (dub_dx(i, j) + dvb_dy(i, j));
            du(i, j) = -g * dhb_dx(i, j);
            dv(i, j) = -g * dhb_dy(i, j);
        }
    }
}

/**
 * This function computes the next time step using a multistep method.
 * The coefficients a1, a2, and a3 are used to determine the weights
 * of the current and previous time steps.
 */
void multistep(int t)
{
    // We set the coefficients for our multistep method
    float a1, a2;
    switch (t)
    {
    case 0:
        a1 = 1.0;
        break;
    default:
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
        break;
    }

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            hb(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j)) * dt;
            ub(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j)) * dt;
            vb(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j)) * dt;
        }
    }
}

/**
 * This function computes the horizontal boundary conditions. We populate our
 * height ghost cells according to a periodic boundary condition, and do the
 * same for our u values
 */
void compute_boundaries_horizontal()
{
    for (int j = 0; j < ny; j++)
    {
        hb(nx, j) = hb(0, j);
        ub(0, j) = ub(nx, j);
    }
}

/**
 * This function computes the vertical boundary conditions.
 */
void compute_ghost_vertical()
{
    for (int i = 0; i < nx; i++)
    {
        hb(i, ny) = hb(i, 0);
        vb(i, 0) = vb(i, ny);
    }
}

/**
 * This function swaps the buffers for the derivatives of our different fields.
 * This is done so that we can use the derivatives from the previous time steps
 * in our multistep method.
 */
void swap_buffers()
{
    std::swap(dh, dh1);
    std::swap(du, du1);
    std::swap(dv, dv1);
}

int t = 0;

void step()
{
    // First we compute our boundary conditions as we need them for our
    // derivative calculations.
    compute_boundaries_horizontal();
    compute_ghost_vertical();

    // Next, we compute the derivatives of our fields
    compute_derivs();

    // Finally, we compute the next time step using our multistep method
    multistep(t);

    // We swap the buffers for our derivatives so that we can use the derivatives
    // from the previous time steps in our multistep method, then increment
    // the time step counter
    swap_buffers();

    t++;
}

// We transfer our created height field back to the original buffer
void transfer(float *h)
{
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            h(i, j) = hb(i, j);
        }
    }
}

// We free all of the memory that we allocated. We didn't create the initial
// height or velocity fields, so we don't need to free them. They are the
// responsibility of the calling code.
void free_memory()
{
    free(hb);
    free(ub);
    free(vb);

    free(dh);
    free(du);
    free(dv);

    free(dh1);
    free(du1);
    free(dv1);
}