#include <stdio.h>
#include <cuda_runtime.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

#define block_h(i, j) block_h[(i) * (block_dims[1]) + (j)]
#define block_u(i, j) block_h[(i) * (block_dims[1]) + (j)]
#define block_v(i, j) block_h[(i) * (block_dims[1]) + (j)]

#define thread_dh(i, j) thread_dh[(i) * MAX_THREAD_DIM + (j)]
#define thread_du(i, j) thread_du[(i) * MAX_THREAD_DIM + (j)]
#define thread_dv(i, j) thread_dv[(i) * MAX_THREAD_DIM + (j)]

#define thread_dh1(i, j) thread_dh1[(i) * MAX_THREAD_DIM + (j)]
#define thread_du1(i, j) thread_du1[(i) * MAX_THREAD_DIM + (j)]
#define thread_dv1(i, j) thread_dv1[(i) * MAX_THREAD_DIM + (j)]

#define MAX_BLOCK_DIM 64
#define BLOCK_HALO_RAD 2

#define MAX_THREAD_DIM 2

int nx, ny;

// float *dh, *du, *dv;
float *h, *u, *v, *dh1, *du1, *dv1;
float H, g, dx, dy, dt;

void init(float *h0, float *u0, float *v0, float length_, float width_, int nx_, int ny_, float H_, float g_, float dt_)
{
    nx = nx_;
    ny = ny_;

    cudaMalloc((void **)&h, nx * ny * sizeof(float));
    cudaMalloc((void **)&u, nx * ny * sizeof(float));
    cudaMalloc((void **)&v, nx * ny * sizeof(float));

    cudaMemcpy(h, h0, nx * ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u, u0, nx * ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v, v0, nx * ny * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dh1, nx * ny * sizeof(float));
    cudaMalloc((void **)&du1, nx * ny * sizeof(float));
    cudaMalloc((void **)&dv1, nx * ny * sizeof(float));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;
}

__device__ inline void derivs(const float *h, const float *u, const float *v, float *thread_dh, float *thread_du, float *thread_dv, int nx, int ny, float dx, float dy, float g, float H)
{
    int local_idx = 0;
    for (int i = threadIdx.x; i < (nx - 1) * (ny - 1); i += blockDim.x)
    {
        int thread_x = i / nx;
        int thread_y = i % nx;

        thread_dh[local_idx] = -H * (du_dx(thread_x, thread_y) + dv_dy(thread_x, thread_y));
        thread_du[local_idx] = -g * dh_dx(thread_x, thread_y);
        thread_dv[local_idx] = -g * dh_dy(thread_x, thread_y);

        local_idx++;
    }
}

__device__ inline void multistep(float *h, float *u, float *v, const float *thread_dh, const float *thread_du, const float *thread_dv, const float *thread_dh1, const float *thread_du1, const float *thread_dv1, int nx, int ny, int t, float dt)
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

    int local_idx = 0;
    for (int i = threadIdx.x; i < (nx - 1) * (ny - 1); i += blockDim.x)
    {
        int thread_x = i / nx;
        int thread_y = i % nx;

        h(thread_x, thread_y) += (a1 * thread_dh[local_idx] + a2 * thread_dh1[local_idx]) * dt;
        u(thread_x, thread_y) += (a1 * thread_du[local_idx] + a2 * thread_du1[local_idx]) * dt;
        v(thread_x, thread_y) += (a1 * thread_dv[local_idx] + a2 * thread_dv1[local_idx]) * dt;

        local_idx++;
    }
}

__device__ inline void swap(float *p1, float *p2, int n)
{
    for (int i = 0; i < n; i++)
    {
        float tmp = p1[i];
        p1[i] = p2[i];
        p2[i] = tmp;
    }
}

__global__ void kernel(float *h, float *u, float *v, float *dh1, float *du1, float *dv1, int nx, int ny, int t, float dx, float dy, float dt, float g, float H)
{
    // We get our block (x, y) coordinate by using the corresponding x and
    // y coordinate of the thread block
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    // To find how many grid points this block is responsible for in each
    // direction, we divide total num of points by the number of blocks
    // in each direction
    const unsigned int block_dims[2] = {nx / gridDim.x, ny / gridDim.y};
    const unsigned int halo_block_dims[2] = {block_dims[0] + 2 * BLOCK_HALO_RAD, block_dims[1] + 2 * BLOCK_HALO_RAD};

    // Here, we set up our local blocks fields using the maximum amount of memory
    // that we can share
    __shared__ float block_h[MAX_BLOCK_DIM * MAX_BLOCK_DIM];
    __shared__ float block_u[MAX_BLOCK_DIM * MAX_BLOCK_DIM];
    __shared__ float block_v[MAX_BLOCK_DIM * MAX_BLOCK_DIM];

    // We make our gradient fields be on a per thread basis, as we don't need
    // to share this information, allowing us to have a larger block size
    float thread_dh[MAX_THREAD_DIM * MAX_THREAD_DIM];
    float thread_du[MAX_THREAD_DIM * MAX_THREAD_DIM];
    float thread_dv[MAX_THREAD_DIM * MAX_THREAD_DIM];

    float thread_dh1[MAX_THREAD_DIM * MAX_THREAD_DIM];
    float thread_du1[MAX_THREAD_DIM * MAX_THREAD_DIM];
    float thread_dv1[MAX_THREAD_DIM * MAX_THREAD_DIM];

    printf("Thread %d of block (%d, %d) reporting for duty! The block dims are (%d, %d).\n", threadIdx.x, blockIdx.x, blockIdx.y, block_dims[0], block_dims[1]);

    // We initialize our local block fields here by reading in from the
    // corresponding grid fields
    int local_idx = 0;
    for (int i = threadIdx.x; i < halo_block_dims[0] * halo_block_dims[1]; i += blockDim.x)
    {
        int thread_x = i / halo_block_dims[0];
        int thread_y = i % halo_block_dims[0];

        int grid_x = mod(block_x * block_dims[0] + thread_x - BLOCK_HALO_RAD, nx);
        int grid_y = mod(block_y * block_dims[1] + thread_y - BLOCK_HALO_RAD, ny);

        printf("Thread %d of block (%d, %d) is loading in from grid (%d, %d) into block (%d, %d) and local idx %d\n", threadIdx.x, blockIdx.x, blockIdx.y, grid_x, grid_y, thread_x, thread_y, local_idx);

        block_h(thread_x, thread_y) = h(grid_x, grid_y);
        block_u(thread_x, thread_y) = u(grid_x, grid_y);
        block_v(thread_x, thread_y) = v(grid_x, grid_y);

        thread_dh1[local_idx] = dh1(grid_x, grid_y);
        thread_du1[local_idx] = du1(grid_x, grid_y);
        thread_dv1[local_idx] = dv1(grid_x, grid_y);

        local_idx++;
    }

    // We iterate for as long as our halo will allow us to do so
    for (int n = 0; n < BLOCK_HALO_RAD; n++)
    {
        derivs(block_h, block_u, block_v, thread_dh, thread_du, thread_dv, halo_block_dims[0], halo_block_dims[1], dx, dy, g, H);

        __syncthreads();

        multistep(block_h, block_u, block_v, thread_dh, thread_du, thread_dv, thread_dh1, thread_du1, thread_dv1, halo_block_dims[0], halo_block_dims[1], t, dt);

        __syncthreads();

        swap(thread_dh, thread_dh1, MAX_THREAD_DIM * MAX_THREAD_DIM);
        swap(thread_du, thread_du1, MAX_THREAD_DIM * MAX_THREAD_DIM);
        swap(thread_dv, thread_dv1, MAX_THREAD_DIM * MAX_THREAD_DIM);

        t++;
    }

    // Finally we write back to the grid
    local_idx = 0;
    for (int i = threadIdx.x; i < block_dims[0] * block_dims[1]; i += blockDim.x)
    {
        int thread_x = i / block_dims[0];
        int thread_y = i % block_dims[0];

        int grid_x = block_x * block_dims[0] + thread_x;
        int grid_y = block_y * block_dims[1] + thread_y;

        h(grid_x, grid_y) = block_h(thread_x, thread_y);
        u(grid_x, grid_y) = block_u(thread_x, thread_y);
        v(grid_x, grid_y) = block_v(thread_x, thread_y);

        dh1(grid_x, grid_y) = thread_dh1[local_idx];
        du1(grid_x, grid_y) = thread_du1[local_idx];
        dv1(grid_x, grid_y) = thread_dv1[local_idx];

        local_idx++;
    }
}

int t = 0;

void step()
{
    dim3 grid_dims(1, 1, 1);
    dim3 block_dims(32 * 32);

    if (t % BLOCK_HALO_RAD == 0)
    {
        kernel<<<grid_dims, block_dims>>>(h, u, v, dh1, du1, dv1, nx, ny, t, dx, dy, dt, g, H);
    }
    cudaDeviceSynchronize();

    t++;
}

void transfer(float *h_host)
{
    cudaMemcpy(h_host, h, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
}

void free_memory()
{
    cudaFree(h);
    cudaFree(u);
    cudaFree(v);
}