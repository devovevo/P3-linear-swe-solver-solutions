#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

int nx, ny;

double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    nx = nx_;
    ny = ny_;

    cudaMalloc((void **)&h, (nx + 1) * (ny + 1) * sizeof(double));
    cudaMalloc((void **)&u, (nx + 1) * ny * sizeof(double));
    cudaMalloc((void **)&v, nx * (ny + 1) * sizeof(double));

    cudaMemcpy(h, h0, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u, u0, (nx + 1) * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v, v0, nx * (ny + 1) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dh, nx * ny * sizeof(double));
    cudaMalloc((void **)&du, nx * ny * sizeof(double));
    cudaMalloc((void **)&dv, nx * ny * sizeof(double));

    cudaMalloc((void **)&dh1, nx * ny * sizeof(double));
    cudaMalloc((void **)&du1, nx * ny * sizeof(double));
    cudaMalloc((void **)&dv1, nx * ny * sizeof(double));

    cudaMalloc((void **)&dh2, nx * ny * sizeof(double));
    cudaMalloc((void **)&du2, nx * ny * sizeof(double));
    cudaMalloc((void **)&dv2, nx * ny * sizeof(double));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;
}

void __global__ compute_derivs(double *h, double *u, double *v, double *du, double *dv, double *dh, int nx, int ny, double dx, double dy, double g, double H)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= nx || j >= ny)
        return;

    if (i == nx - 1)
    {
        h(nx, j) = h(0, j);
    }

    if (j == ny - 1)
    {
        h(i, ny) = h(i, 0);
    }

    dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
    du(i, j) = -g * dh_dx(i, j);
    dv(i, j) = -g * dh_dy(i, j);
}

void __global__ update_fields(double *h, double *u, double *v, double *dh, double *du, double *dv, double *dh1, double *du1, double *dv1, double *dh2, double *du2, double *dv2, int nx, int ny, double dx, double dy, double dt, double a1, double a2, double a3)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= nx || j >= ny)
        return;

    h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
    u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
    v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;

    if (i == nx - 1)
    {
        u(0, j) = u(nx, j);
    }

    if (j == ny - 1)
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

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(nx / threadsPerBlock.x + 1, ny / threadsPerBlock.y + 1);

    compute_derivs<<<numBlocks, threadsPerBlock>>>(h, u, v, du, dv, dh, nx, ny, dx, dy, g, H);
    update_fields<<<numBlocks, threadsPerBlock>>>(h, u, v, dh, du, dv, dh1, du1, dv1, dh2, du2, dv2, nx, ny, dx, dy, dt, a1, a2, a3);

    swap_buffers();

    t++;
}

void transfer(double *h_host)
{
    cudaMemcpy(h_host, h, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyDeviceToHost);
}

void free_memory()
{
    cudaFree(h);
    cudaFree(u);
    cudaFree(v);

    cudaFree(dh);
    cudaFree(du);
    cudaFree(dv);

    cudaFree(dh1);
    cudaFree(du1);
    cudaFree(dv1);

    cudaFree(dh2);
    cudaFree(du2);
    cudaFree(dv2);
}