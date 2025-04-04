#include <stdio.h>
#include <cuda_runtime.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

#define block_h(i, j) block_h[(i) * (halo_block_dims[1]) + (j)]
#define block_u(i, j) block_u[(i) * (halo_block_dims[1]) + (j)]
#define block_v(i, j) block_v[(i) * (halo_block_dims[1]) + (j)]

#define block_dh_dx(i, j) (block_h(i + 1, j) - block_h(i, j)) / dx
#define block_dh_dy(i, j) (block_h(i, j + 1) - block_h(i, j)) / dy

#define block_du_dx(i, j) (block_u(i + 1, j) - block_u(i, j)) / dx
#define block_dv_dy(i, j) (block_v(i, j + 1) - block_v(i, j)) / dy

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
    cudaMemcpy(u, u0, (nx + 1) * ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v, v0, nx * (ny + 1) * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dh1, nx * ny * sizeof(float));
    cudaMalloc((void **)&du1, nx * ny * sizeof(float));
    cudaMalloc((void **)&dv1, nx * ny * sizeof(float));

    cudaMemset(dh1, 0, nx * ny * sizeof(float));
    cudaMemset(du1, 0, nx * ny * sizeof(float));
    cudaMemset(dv1, 0, nx * ny * sizeof(float));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / nx;

    dt = dt_;
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

template <unsigned int halo_rad = 2, unsigned int thread_dim = 4>
__global__ void kernel(float *const h, float *const u, float *const v, float *const dh1, float *const du1, float *const dv1, int nx, int ny, int t, float dx, float dy, float dt, float g, float H)
{
    // To find how many grid points this block is responsible for in each
    // direction, we divide total num of points by the number of blocks
    // in each direction
    unsigned int block_dims[2] = {CEIL_DIV(nx, gridDim.x), CEIL_DIV(ny, gridDim.y)};
    block_dims[0] = blockIdx.x == gridDim.x - 1 ? nx - block_dims[0] * (gridDim.x - 1) : block_dims[0];
    block_dims[1] = blockIdx.y == gridDim.y - 1 ? ny - block_dims[1] * (gridDim.y - 1) : block_dims[1];

    const unsigned int halo_block_dims[2] = {block_dims[0] + 2 * halo_rad, block_dims[1] + 2 * halo_rad};

    // Here, we set up our local blocks fields using the maximum amount of memory
    // that we can share. We make an external shared memory bank called s to
    // store into.

    extern __shared__ float s[];

    volatile float *block_h = &s[0 * halo_block_dims[0] * halo_block_dims[1]];
    volatile float *block_u = &s[1 * halo_block_dims[0] * halo_block_dims[1]];
    volatile float *block_v = &s[2 * halo_block_dims[0] * halo_block_dims[1]];

    // We make our gradient fields be on a per thread basis, as we don't need
    // to share this information, allowing us to have a larger block size
    float thread_dh[thread_dim] = {0};
    float thread_du[thread_dim] = {0};
    float thread_dv[thread_dim] = {0};

    float thread_dh1[thread_dim] = {0};
    float thread_du1[thread_dim] = {0};
    float thread_dv1[thread_dim] = {0};

    // if (threadIdx.x == 0)
    // {
    //     printf("Thread %d of block (%d, %d) reporting for duty! The calculated block dims are (%d, %d) and the calculated halo block dims are (%d, %d) with a thread dimension of %d. Overall, our provided block dims are (%d, %d) and the grid dims are (%d, %d)\n", threadIdx.x, blockIdx.x, blockIdx.y, block_dims[0], block_dims[1], halo_block_dims[0], halo_block_dims[1], thread_dim, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    // }

    // We initialize our local block fields here by reading in from the
    // corresponding grid fields
    for (int i = threadIdx.x; i < halo_block_dims[0] * halo_block_dims[1]; i += blockDim.x)
    {
        const int thread_x = i / halo_block_dims[0];
        const int thread_y = i % halo_block_dims[0];

        const int grid_x = mod(blockIdx.x * CEIL_DIV(nx, gridDim.x) + thread_x - 1, nx);
        const int grid_y = mod(blockIdx.y * CEIL_DIV(ny, gridDim.y) + thread_y - 1, ny);

        const int local_idx = i / blockDim.x;

        // if (grid_x == 44 && grid_y == 44)
        // {
        //     printf("Thread %d of block (%d, %d) is loading in from grid (%d, %d) into block (%d, %d) and local idx %d. The corresponding block h value is %f and the grid h value is %f.\n", threadIdx.x, blockIdx.x, blockIdx.y, grid_x, grid_y, thread_x, thread_y, local_idx, block_h(thread_x, thread_y), h(grid_x, grid_y));
        // }

        block_h(thread_x, thread_y) = h(grid_x, grid_y);
        block_u(thread_x, thread_y) = u(grid_x, grid_y);
        block_v(thread_x, thread_y) = v(grid_x, grid_y);

        thread_dh1[local_idx] = dh1(grid_x, grid_y);
        thread_du1[local_idx] = du1(grid_x, grid_y);
        thread_dv1[local_idx] = dv1(grid_x, grid_y);
    }

    __syncthreads();

    // We iterate for as long as our halo will allow us to do so
    for (int n = 0; n < 2 * halo_rad; n++)
    {
        for (int i = threadIdx.x; i < halo_block_dims[0] * halo_block_dims[1]; i += blockDim.x)
        {
            const int thread_x = i / halo_block_dims[0];
            const int thread_y = i % halo_block_dims[0];

            if (thread_x >= halo_block_dims[0] - 1 || thread_y >= halo_block_dims[1] - 1)
            {
                continue;
            }

            const int local_idx = i / blockDim.x;

            // if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 135 && local_idx == 3)
            // {
            //     printf("Thread %d of block (%d, %d) loading in from (%d, %d) to calculate derivatives. The value of h at (%d, %d) = %f, (%d, %d) = %f, (%d, %d) = %f. The value of u at (%d, %d) = %f, (%d, %d) = %f and of v at (%d, %d) = %f, (%d, %d) = %f. The compute value of block_dh_dy is %f\n", threadIdx.x, blockIdx.x, blockIdx.y, thread_x, thread_y, thread_x, thread_y, block_h(thread_x, thread_y), thread_x + 1, thread_y, block_h(thread_x + 1, thread_y), thread_x, thread_y + 1, block_h(thread_x, thread_y + 1), thread_x, thread_y, block_u(thread_x, thread_y), thread_x + 1, thread_y, block_u(thread_x + 1, thread_y), thread_x, thread_y, block_v(thread_x, thread_y), thread_x, thread_y + 1, block_v(thread_x, thread_y + 1), block_dh_dy(thread_x, thread_y));
            // }

            // if (threadIdx.x == 0)
            // {
            //     printf("Thread %d of block (%d, %d) is loading in from block (%d, %d) to compute gradients with local idx %d.\n", threadIdx.x, blockIdx.x, blockIdx.y, thread_x, thread_y, local_idx);
            // }

            thread_dh[local_idx] = -H * (block_du_dx(thread_x, thread_y) + block_dv_dy(thread_x, thread_y));
            thread_du[local_idx] = -g * block_dh_dx(thread_x, thread_y);
            thread_dv[local_idx] = -g * block_dh_dy(thread_x, thread_y);
        }

        __syncthreads();

        // We set the coefficients for our multistep method
        float a1 = 1.0, a2 = 0.0;
        if (t > 0)
        {
            a1 = 1.5;
            a2 = -0.5;
        }

        for (int i = threadIdx.x; i < halo_block_dims[0] * halo_block_dims[1]; i += blockDim.x)
        {
            const int thread_x = i / halo_block_dims[0];
            const int thread_y = i % halo_block_dims[0];

            if (thread_x >= halo_block_dims[0] - 1 || thread_y >= halo_block_dims[1] - 1)
            {
                continue;
            }

            int local_idx = i / blockDim.x;

            // if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 135 && local_idx == 3)
            // {
            //     printf("Thread %d of block (%d, %d) is loading in from block (%d, %d) to multistep with local idx %d. The value of thread_dv is %f and thread_dv1 is %f\n", threadIdx.x, blockIdx.x, blockIdx.y, thread_x, thread_y, local_idx, thread_dv[local_idx], thread_dv1[local_idx]);
            // }

            block_h(thread_x, thread_y) += (a1 * thread_dh[local_idx] + a2 * thread_dh1[local_idx]) * dt;
            block_u(thread_x + 1, thread_y) += (a1 * thread_du[local_idx] + a2 * thread_du1[local_idx]) * dt;
            block_v(thread_x, thread_y + 1) += (a1 * thread_dv[local_idx] + a2 * thread_dv1[local_idx]) * dt;
        }

        __syncthreads();

        swap(thread_dh, thread_dh1, thread_dim);
        swap(thread_du, thread_du1, thread_dim);
        swap(thread_dv, thread_dv1, thread_dim);

        t++;
    }

    // Finally we write back to the grid
    for (int i = threadIdx.x; i < halo_block_dims[0] * halo_block_dims[1]; i += blockDim.x)
    {
        const int thread_x = i / halo_block_dims[0];
        const int thread_y = i % halo_block_dims[0];

        const int grid_x = blockIdx.x * CEIL_DIV(nx, gridDim.x) + thread_x - 1;
        const int grid_y = blockIdx.y * CEIL_DIV(ny, gridDim.y) + thread_y - 1;

        const int local_idx = i / blockDim.x;

        if (thread_x == 0 || thread_y == 0 || thread_x >= block_dims[0] || thread_y >= block_dims[1])
        {
            continue;
        }

        // if (threadIdx.x == 0 && local_idx == 0)
        // {
        //     printf("Thread %d of block (%d, %d), where we assume the block starts with (%d, %d) and ends with (%d, %d) and has block size (%d, %d) with a halo of radius %d tried to write to grid (%d, %d), however this is unallocated memory. The threads coords are (%d, %d), and the value of i is %d. The grid dimensions are (%d, %d), and CEIL_DIV(nx, gridDim.x) is %d. The size of the grid is (%d, %d)\n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.x * (CEIL_DIV(nx, gridDim.x)), blockIdx.y * (CEIL_DIV(ny, gridDim.y)), blockIdx.x * (CEIL_DIV(nx, gridDim.x)) + block_dims[0] - 1, blockIdx.y * (CEIL_DIV(ny, gridDim.y)) + block_dims[1] - 1, block_dims[0], block_dims[1], halo_rad, grid_x, grid_y, thread_x, thread_y, i, gridDim.x, gridDim.y, CEIL_DIV(nx, gridDim.x), nx, ny);
        // }

        // if (grid_x == 44 && grid_y == 44)
        // {
        //     printf("Thread %d of block (%d, %d) is loading in from block (%d, %d) and local idx %d to write to grid (%d, %d). The block dims are (%d, %d). The corresponding block h value is %f and the grid h value is %f.\n", threadIdx.x, blockIdx.x, blockIdx.y, thread_x, thread_y, local_idx, grid_x, grid_y, block_dims[0], block_dims[1], block_h(thread_x, thread_y), h(grid_x, grid_y));
        // }

        h(grid_x, grid_y) = block_h(thread_x, thread_y);
        u(grid_x, grid_y) = block_u(thread_x, thread_y);
        v(grid_x, grid_y) = block_v(thread_x, thread_y);

        dh1(grid_x, grid_y) = thread_dh1[local_idx];
        du1(grid_x, grid_y) = thread_du1[local_idx];
        dv1(grid_x, grid_y) = thread_dv1[local_idx];
    }
}

const int block_dims[2] = {64, 64};
const int num_threads = 16 * 16;
const int halo_rad = 10;

void call_kernel(int t)
{
    if (block_dims[0] <= halo_rad || block_dims[1] <= halo_rad)
    {
        printf("The provided halo radius is at least as big as one of the dimensions, meaning it's all halo, which doesn't make sense.\n");
        return;
    }

    if (block_dims[0] * block_dims[1] > 64 * 64)
    {
        printf("The desired block size would require too much shared memory. Maximum thread block memory size can be at most that of (64, 64).\n");
        return;
    }

    const int thread_dim = block_dims[0] * block_dims[1] / num_threads;

    dim3 grid_dims(CEIL_DIV(nx, (block_dims[0] - 2 * halo_rad)), CEIL_DIV(ny, (block_dims[1] - 2 * halo_rad)));
    dim3 thread_dims(num_threads);

    kernel<halo_rad, thread_dim><<<grid_dims, thread_dims, 3 * block_dims[0] * block_dims[1] * sizeof(float)>>>(h, u, v, dh1, du1, dv1, nx, ny, t, dx, dy, dt, g, H);
}

int t = 0;

void step()
{
    if (t % (2 * halo_rad) == 0)
    {
        call_kernel(t);
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