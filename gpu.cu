#include <cuda.h>
#include <math.h>

#include "common.h"

int nx, ny, blockSize = 1024, numSMs, multipleSMs = 64;

double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_)
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

__device__ compute_derivs()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if (tid > nx * ny)
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

double a1 = 1.0, a2 = 0.0, a3 = 0.0;

__device__ update_fields()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if (tid > nx * ny)
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

__host__ swap_buffers()
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

int t = 0;

__global__ step()
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

    compute_derivs<<<multipleSMs * numSMs, blockSize>>>();
    update_fields<<<multipleSMs * numSMs, blockSize>>>();

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

    free(dh);
    free(du);
    free(dv);
}