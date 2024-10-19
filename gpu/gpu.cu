#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

int nx, ny, blockSize = 1024, numSMs, multipleSMs = 64;

double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    nx = nx_;
    ny = ny_;

    cudaMalloc((void **)&h, (nx + 1) * (ny + 1) * sizeof(double));
    cudaMalloc((void **)&u, (nx + 1) * (ny + 1) * sizeof(double));
    cudaMalloc((void **)&v, (nx + 1) * (ny + 2) * sizeof(double));

    cudaMemcpy(h, h0, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u, u0, (nx + 1) * (ny + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v, v0, (nx + 1) * (ny + 2) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dh, nx * ny * sizeof(double));
    cudaMalloc((void **)&du, nx * ny * sizeof(double));
    cudaMalloc((void **)&dv, nx * (ny + 1) * sizeof(double));

    cudaMalloc((void **)&dh1, nx * ny * sizeof(double));
    cudaMalloc((void **)&du1, nx * ny * sizeof(double));
    cudaMalloc((void **)&dv1, nx * (ny + 1) * sizeof(double));

    cudaMalloc((void **)&dh2, nx * ny * sizeof(double));
    cudaMalloc((void **)&du2, nx * ny * sizeof(double));
    cudaMalloc((void **)&dv2, nx * (ny + 1) * sizeof(double));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;

    int devId;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
}

void __global__ compute_derivs(double *h, double *u, double *v, double *du, double *dv, double *dh, int nx, int ny, double dx, double dy, double g, double H)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if (tid + stride >= nx * ny)
        return;

    for (int idx = tid; idx < tid + stride; idx++)
    {
        int i = idx / ny;
        int j = idx % ny;

        du(i, j) = -g * dh_dx(i, j);
        dv(i, j) = -g * dh_dy(i, j);
        dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
    }
}

void __global__ update_fields(double *h, double *u, double *v, double *dh, double *du, double *dv, double *dh1, double *du1, double *dv1, double *dh2, double *du2, double *dv2, int nx, int ny, double dx, double dy, double dt, double a1, double a2, double a3)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if (tid + stride >= nx * ny)
        return;

    for (int idx = tid; idx < tid + stride; idx++)
    {
        int i = idx / ny;
        int j = idx % ny;

        v(i, j) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
        u(i, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
        h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
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

double a1 = 1.0, a2 = 0.0, a3 = 0.0;
int t = 0;

void step()
{
    if (t == 1)
    {
        a1 = 1.5;
        a2 = -0.5;
    }
    else
    {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }

    compute_derivs<<<multipleSMs * numSMs, blockSize>>>(h, u, v, du, dv, dh, nx, ny, dx, dy, g, H);
    update_fields<<<multipleSMs * numSMs, blockSize>>>(h, u, v, dh, du, dv, dh1, du1, dv1, dh2, du2, dv2, nx, ny, dx, dy, dt, a1, a2, a3);

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