#include <cuda_runtime.h>

#include <stdio.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

#define block_h(i, j) block_h[(i) * (block_dims[1]) + (j)]
#define block_u(i, j) block_h[(i) * (block_dims[1]) + (j)]
#define block_v(i, j) block_h[(i) * (block_dims[1]) + (j)]

#define thread_dh(i, j) thread_dh[(i) * MAX_THREAD_DIM + (j)]
#define thread_du(i, j) thread_du[(i) * MAX_THREAD_DIM + (j)]
#define thread_dv(i, j) thread_dv[(i) * MAX_THREAD_DIM + (j)]

#define MAX_BLOCK_DIM 32
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

    // cudaMalloc((void **)&dh, nx * ny * sizeof(float));
    // cudaMalloc((void **)&du, nx * ny * sizeof(float));
    // cudaMalloc((void **)&dv, nx * ny * sizeof(float));

    cudaMalloc((void **)&dh1, nx * ny * sizeof(float));
    cudaMalloc((void **)&du1, nx * ny * sizeof(float));
    cudaMalloc((void **)&dv1, nx * ny * sizeof(float));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;
}

__device__ inline void derivs(const double *h, const double *u, const double *v, double *thread_dh, double *thread_du, double *thread_dv, int nx, int ny, float dx, float dy, float g, float H, int thread_x, int thread_y)
{
    for (int i = 0; i < nx - 1; i += blockDim.x)
    {
        for (int j = 0; j < ny - 1; j += blockDim.y)
        {
            int block_x = thread_x + i;
            int block_y = thread_y + j;

            int local_x = i / blockDim.x;
            int local_y = j / blockDim.x;

            thread_dh(local_x, local_y) = -H * (du_dx(block_x, block_y) + dv_dy(block_x, block_y));
            thread_du(local_x, local_y) = -g * dh_dx(block_x, block_y);
            thread_dv(local_x, local_y) = -g * dh_dy(block_x, block_y);
        }
    }
}

__device__ inline void multistep(double *h, double *u, double *v, const double *thread_dh, const double *thread_du, const double *thread_dv, const double *thread_dh1, const double *thread_du1, const double *thread_dv1, int nx, int ny, int t)
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

    for (int i = 0; i < nx - 1; i += blockDim.x)
    {
        for (int j = 0; j < ny - 1; j += blockDim.y)
        {
            int block_x = thread_x + i;
            int block_y = thread_y + j;

            int local_x = i / blockDim.x;
            int local_y = j / blockDim.x;

            h(block_x, block_y) += (a1 * thread_dh(local_x, local_y) + a2 * thread_dh1(local_x, local_y)) * dt;
            u(block_x + 1, block_y) += (a1 * thread_du(local_x, local_y) + a2 * thread_du1(local_x, local_y)) * dt;
            v(block_x, block_y + 1) += (a1 * thread_dv(local_x, local_y) + a2 * thread_dv1(local_x, local_y)) * dt;
        }
    }
}

__global__ void kernel(float *grid_h, float *grid_u, float *grid_v, float *grid_dh, float *grid_du, float *grid_dv, float *grid_dh1, float *grid_du1, float *grid_dv1, float *grid_dh2, float *grid_du2, float *grid_dv2, int nx, int ny, float dx, float dy, float g, float H, int t)
{
    // We get our block (x, y) coordinate by using the corresponding x and
    // y coordinate of the thread block
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    // To find how many grid points this block is responsible for in each
    // direction, we divide total num of points by the number of blocks
    // in each direction
    const int block_dims[2] = {nx / gridDim.x, ny / gridDim.y};
    const int halo_block_dims[2] = {block_dims[0] + 2 * BLOCK_HALO_RAD, block_dims[1] + 2 * BLOCK_HALO_RAD};

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

    // We take just our X thread ID and turn it into our x and y such that
    // adjacent threads (in x coord) have adjacent y coordinates
    int thread_x = threadIdx.x / halo_block_dims[0];
    int thread_y = threadIdx.x % halo_block_dims[0];

    // We initialize our local block fields here by reading in from the
    // corresponding grid fields
    for (int i = 0; i < halo_block_dims[0]; i += blockDim.x)
    {
        for (int j = 0; j < halo_block_dims[1]; j += blockDim.y)
        {
            int grid_x = mod(block_x * block_dims[0] + thread_x + i - BLOCK_HALO_RAD, nx);
            int grid_y = mod(block_y * block_dims[1] + thread_y + j - BLOCK_HALO_RAD, ny);

            int block_x = thread_x + i;
            int block_y = thread_y + j;

            int local_x = i / blockDim.x;
            int local_y = j / blockDim.y;

            block_h(block_x, block_y) = grid_h(grid_x, grid_y);
            block_u(block_x, block_y) = grid_u(grid_x, grid_y);
            block_v(block_x, block_y) = grid_v(grid_x, grid_y);

            thread_dh1(local_x, local_y) = grid_dh1(grid_x, grid_y);
            thread_du1(local_x, local_y) = grid_du1(grid_x, grid_y);
            thread_dv1(local_x, local_y) = grid_dv1(grid_x, grid_y);
        }
    }

    // We iterate for as long as our halo will allow us to do so
    for (int n = 0; n < BLOCK_HALO_RAD; n++)
    {
        derivs(block_h, block_u, block_v, thread_dh, thread_du, thread_dv, halo_block_dims[0], halo_block_dims[1], dx, dy, g, H, thread_x, thread_y);

        __syncthreads();

        multistep(block_h, block_u, block_v, thread_dh, thread_du, thread_dv, thread_dh1, thread_du1, thread_dv1, halo_block_dims[0], halo_block_dims[1], t);

        __syncthreads();

        std::swap(thread_dh, thread_dh1);
        std::swap(thread_du, thread_du1);
        std::swap(thread_dv, thread_dv1);

        t++;
    }

    // Finally we write back to the grid
    for (int i = 0; i < block_dims[0]; i += blockDim.x)
    {
        for (int j = 0; j < block_dims[1]; j += blockDim.y)
        {
            int grid_x = block_x * block_dims[0] + thread_x + i;
            int grid_y = block_y * block_dims[1] + thread_y + j;

            int block_x = thread_x + i;
            int block_y = thread_y + j;

            int local_x = i / blockDim.x;
            int local_y = i / blockDim.y;

            grid_h(grid_x, grid_y) = block_h(block_x, block_y);
            grid_u(grid_x, grid_y) = block_u(block_x, block_y);
            grid_v(grid_x, grid_y) = block_v(block_x, block_y);

            grid_dh1(grid_x, grid_y) = thread_dh1(local_x, local_y);
            grid_du1(grid_x, grid_y) = thread_du1(local_x, local_y);
            grid_dv1(grid_x, grid_y) = thread_dv1(local_x, local_y);
        }
    }
}

int t = 0;

void step()
{
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(ceil(nx / threadsPerBlock.x), ceil(ny / threadsPerBlock.y));

    float a1, a2, a3;

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

    compute_derivs<<<numBlocks, threadsPerBlock>>>(h, u, v, du, dv, dh, nx, ny, dx, dy, g, H);
    update_fields<<<numBlocks, threadsPerBlock>>>(h, u, v, dh, du, dv, dh1, du1, dv1, dh2, du2, dv2, nx, ny, dx, dy, dt, a1, a2, a3);
    // if (t % 20 == 0)
    // {
    //     combined_kernel<<<numBlocks, threadsPerBlock>>>(h, u, v, dh, du, dv, dh1, du1, dv1, dh2, du2, dv2, nx, ny, dx, dy, g, H, dt, 20);
    // }

    swap_buffers();

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