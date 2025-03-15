#include <stdio.h>
#include <cuda_runtime.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

#define block_h(i, j) block_h[(i) * (halo_block_dims[1]) + (j)]
#define block_u(i, j) block_u[(i) * (halo_block_dims[1]) + (j)]
#define block_v(i, j) block_v[(i) * (halo_block_dims[1]) + (j)]

#define block_dh_dx(i, j) (block_h(i + 1, j) - block_h(i, j)) / dx
#define block_dh_dy(i, j) (block_h(i, j + 1) - h(i, j)) / dy

#define block_du_dx(i, j) (block_u(i + 1, j) - block_u(i, j)) / dx
#define block_dv_dy(i, j) (block_v(i, j + 1) - block_v(i, j)) / dy

#define thread_dh(i, j) thread_dh[(i) * MAX_THREAD_DIM + (j)]
#define thread_du(i, j) thread_du[(i) * MAX_THREAD_DIM + (j)]
#define thread_dv(i, j) thread_dv[(i) * MAX_THREAD_DIM + (j)]

#define thread_dh1(i, j) thread_dh1[(i) * MAX_THREAD_DIM + (j)]
#define thread_du1(i, j) thread_du1[(i) * MAX_THREAD_DIM + (j)]
#define thread_dv1(i, j) thread_dv1[(i) * MAX_THREAD_DIM + (j)]

#define BLOCK_HALO_RAD 10
#define MAX_THREAD_DIM 1

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

__device__ inline void multistep(float *h, float *u, float *v, const float *thread_dh, const float *thread_du, const float *thread_dv, const float *thread_dh1, const float *thread_du1, const float *thread_dv1, int nx, int ny, int t, float dt)
{
    // We set the coefficients for our multistep method
    float a1, a2;
    switch (t)
    {
    case 0:
        a1 = 1.0;
        a2 = 0.0;
        break;
    default:
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
        break;
    }

    for (int i = threadIdx.x; i < (nx - 1) * (ny - 1); i += blockDim.x)
    {
        int thread_x = i / nx;
        int thread_y = i % nx;

        int local_idx = i / blockDim.x;

        h(thread_x, thread_y) += (a1 * thread_dh[local_idx] + a2 * thread_dh1[local_idx]) * dt;
        u(thread_x + 1, thread_y) += (a1 * thread_du[local_idx] + a2 * thread_du1[local_idx]) * dt;

        // printf("Attempting to acces (%d, %d) from v with dimensions (%d, %d).\n", thread_x, thread_y + 1, nx, ny);

        // v(thread_x, thread_y) += 1.0;
        // v(thread_x, thread_y + 1) += (a1 * thread_dv[local_idx] + a2 * thread_dv1[local_idx]) * dt;
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
    // To find how many grid points this block is responsible for in each
    // direction, we divide total num of points by the number of blocks
    // in each direction
    const unsigned int block_dims[2] = {nx / gridDim.x, ny / gridDim.y};
    const unsigned int halo_block_dims[2] = {block_dims[0] + 2 * BLOCK_HALO_RAD, block_dims[1] + 2 * BLOCK_HALO_RAD};

    // Here, we set up our local blocks fields using the maximum amount of memory
    // that we can share. We make an external shared memory bank called s to
    // store into.

    extern __shared__ float s[];

    float *block_h = &s[0 * halo_block_dims[0] * halo_block_dims[1]];
    float *block_u = &s[1 * halo_block_dims[0] * halo_block_dims[1]];
    float *block_v = &s[2 * halo_block_dims[0] * halo_block_dims[1]];

    // We make our gradient fields be on a per thread basis, as we don't need
    // to share this information, allowing us to have a larger block size
    float thread_derivs[MAX_THREAD_DIM * MAX_THREAD_DIM * 3];
    float thread_derivs1[MAX_THREAD_DIM * MAX_THREAD_DIM * 3];

    // float thread_du[MAX_THREAD_DIM * MAX_THREAD_DIM];
    // float thread_dv[MAX_THREAD_DIM * MAX_THREAD_DIM];

    // float thread_dh1[MAX_THREAD_DIM * MAX_THREAD_DIM];
    // float thread_du1[MAX_THREAD_DIM * MAX_THREAD_DIM];
    // float thread_dv1[MAX_THREAD_DIM * MAX_THREAD_DIM];

    // printf("Thread %d of block (%d, %d) reporting for duty! The block dims are (%d, %d).\n", threadIdx.x, blockIdx.x, blockIdx.y, block_dims[0], block_dims[1]);

    // We initialize our local block fields here by reading in from the
    // corresponding grid fields
    for (int i = threadIdx.x; i < halo_block_dims[0] * halo_block_dims[1]; i += blockDim.x)
    {
        int thread_x = i / halo_block_dims[0];
        int thread_y = i % halo_block_dims[0];

        int grid_x = mod(blockIdx.x * block_dims[0] + thread_x - BLOCK_HALO_RAD, nx);
        int grid_y = mod(blockIdx.y * block_dims[1] + thread_y - BLOCK_HALO_RAD, ny);

        int local_idx = i / blockDim.x;

        block_h(thread_x, thread_y) = h(grid_x, grid_y);
        block_u(thread_x, thread_y) = u(grid_x, grid_y);
        block_v(thread_x, thread_y) = v(grid_x, grid_y);

        // printf("Thread %d of block (%d, %d) is loading in from grid (%d, %d) into block (%d, %d) and local idx %d. The corresponding block h value is %f and the grid h value is %f.\n", threadIdx.x, blockIdx.x, blockIdx.y, grid_x, grid_y, thread_x, thread_y, local_idx, block_h(thread_x, thread_y), h(grid_x, grid_y));

        thread_derivs[local_idx * 3 + 0] = dh1(grid_x, grid_y);
        thread_derivs[local_idx * 3 + 1] = du1(grid_x, grid_y);
        thread_derivs[local_idx * 3 + 2] = dh1(grid_x, grid_y);
    }

    __syncthreads();

    // We iterate for as long as our halo will allow us to do so
    for (int n = 0; n < BLOCK_HALO_RAD; n++)
    {
        for (int i = threadIdx.x; i < (nx - 1) * (ny - 1); i += blockDim.x)
        {
            int thread_x = i / nx;
            int thread_y = i % nx;

            int local_idx = i / blockDim.x;

            // thread_dh[local_idx] = -H * (block_du_dx(thread_x, thread_y) + block_dv_dy(thread_x, thread_y));
            // thread_du[local_idx] = -g * block_dh_dx(thread_x, thread_y);
            // thread_dv[local_idx] = -g * dh_dy(thread_x, thread_y);
        }

        __syncthreads();

        // We set the coefficients for our multistep method
        float a1, a2;
        switch (t)
        {
        case 0:
            a1 = 1.0;
            a2 = 0.0;
            break;
        default:
            a1 = 3.0 / 2.0;
            a2 = -1.0 / 2.0;
            break;
        }

        for (int i = threadIdx.x; i < (nx - 1) * (ny - 1); i += blockDim.x)
        {
            int thread_x = i / nx;
            int thread_y = i % nx;

            int local_idx = i / blockDim.x;

            // h(thread_x, thread_y) += (a1 * thread_dh[local_idx] + a2 * thread_dh1[local_idx]) * dt;
            // u(thread_x + 1, thread_y) += (a1 * thread_du[local_idx] + a2 * thread_du1[local_idx]) * dt;

            // printf("Attempting to acces (%d, %d) from v with dimensions (%d, %d).\n", thread_x, thread_y + 1, nx, ny);

            // v(thread_x, thread_y) += 1.0;
            // v(thread_x, thread_y + 1) += (a1 * thread_dv[local_idx] + a2 * thread_dv1[local_idx]) * dt;
        }

        // __syncthreads();

        // swap(thread_dh, thread_dh1, MAX_THREAD_DIM * MAX_THREAD_DIM);
        // swap(thread_du, thread_du1, MAX_THREAD_DIM * MAX_THREAD_DIM);
        // swap(thread_dv, thread_dv1, MAX_THREAD_DIM * MAX_THREAD_DIM);

        t++;
    }

    // Finally we write back to the grid
    for (int i = threadIdx.x; i < halo_block_dims[0] * halo_block_dims[1]; i += blockDim.x)
    {
        int thread_x = i / halo_block_dims[0];
        int thread_y = i % halo_block_dims[0];

        // int grid_x = block_x * block_dims[0] + thread_x - BLOCK_HALO_RAD;
        // int grid_y = block_y * block_dims[1] + thread_y - BLOCK_HALO_RAD;

        // int local_idx = i / blockDim.x;

        // if (grid_x < 0 || grid_y < 0 || grid_x >= nx || grid_y >= ny)
        // {
        //     continue;
        // }

        // printf("Thread %d of block (%d, %d) is loading in from block (%d, %d) and local idx %d and writing back into grid (%d, %d). The corresponding block h value is %f and the grid h value is %f.\n", threadIdx.x, blockIdx.x, blockIdx.y, thread_x, thread_y, local_idx, grid_x, grid_y, block_h(thread_x, thread_y), h(grid_x, grid_y));

        // h(grid_x, grid_y) = block_h(thread_x, thread_y);
        // u(grid_x, grid_y) = block_u(thread_x, thread_y);
        // v(grid_x, grid_y) = block_v(thread_x, thread_y);

        // dh1(grid_x, grid_y) = thread_dh1[local_idx];
        // du1(grid_x, grid_y) = thread_du1[local_idx];
        // dv1(grid_x, grid_y) = thread_dv1[local_idx];
    }
}

int t = 0;

void step()
{
    const unsigned int block_x = 32, block_y = 32, num_pts = 3 * (block_x + 2 * BLOCK_HALO_RAD) * (block_y + 2 * BLOCK_HALO_RAD);

    dim3 grid_dims(CEIL_DIV(nx, block_x), CEIL_DIV(ny, block_y), 1);
    dim3 block_dims(32 * 32);

    if (t % BLOCK_HALO_RAD == 0)
    {
        kernel<<<grid_dims, block_dims, num_pts * sizeof(float)>>>(h, u, v, dh1, du1, dv1, nx, ny, t, dx, dy, dt, g, H);
    }

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

    cudaFree(dh1);
    cudaFree(du1);
    cudaFree(dv1);
}