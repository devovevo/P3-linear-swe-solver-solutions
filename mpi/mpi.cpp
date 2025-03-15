#include <mpi.h>

#include <iomanip>
#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

#define HALO_RAD 4

float *h, *u, *v;
float *dh, *du, *dv, *dh1, *du1, *dv1;

int grid_dims[2], nx, ny;

float H, g, dx, dy, dt;

int num_procs, rank, grid_rank;
int grid_coords[2];

int proc_dims[2], block_dims[2], halo_block_dims[2];
int wrap_dims[2] = {1, 1};
int reorder = 1;

int *sendcounts, *displs;

MPI_Comm grid_comm;
MPI_Datatype grid_block_type, resized_grid_block_type, local_block_type;

int neighbors[8], slab_dims[8][2], send_slab_starts[8][2], recv_slab_starts[8][2];
MPI_Datatype send_slab_types[8], recv_slab_types[8];

template <typename T>
void print2darray(T *arr, int *dims)
{
    int width = dims[0];
    int height = dims[1];

    for (int i = 0; i < width; i++)
    {
        putchar('|');
        for (int j = 0; j < height; j++)
        {
            std::cout << std::fixed << std::setprecision(5) << arr[i * height + j] << " ";
        }
        printf("|\n");
    }
}

void init_topology();
void exchange_halos();

void init(float *h0, float *u0, float *v0, float length_, float width_, int nx_, int ny_, float H_, float g_, float dt_)
{
    // We get our number of processes and the rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Here, we set the dimensions of our grid to be the number of cells
    // in the x and y directions
    grid_dims[0] = nx_;
    grid_dims[1] = ny_;

    if (rank == 0)
    {
        printf("The initial h0 array is:\n");
        print2darray(h0, grid_dims);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    init_topology();

    printf("World rank %d with grid rank %d has coordinates (%d, %d) out of proc dims (%d, %d) out of grid dims (%d, %d), as well as block dims (%d, %d)\n", rank, grid_rank, grid_coords[0], grid_coords[1], proc_dims[0], proc_dims[1], grid_dims[0], grid_dims[1], block_dims[0], block_dims[1]);

    MPI_Barrier(MPI_COMM_WORLD);

    int local_block_num_elems = halo_block_dims[0] * halo_block_dims[1];
    h = (float *)calloc(local_block_num_elems, sizeof(float));
    u = (float *)calloc(local_block_num_elems, sizeof(float));
    v = (float *)calloc(local_block_num_elems, sizeof(float));

    MPI_Scatterv(h0, sendcounts, displs, resized_grid_block_type, h, 1, local_block_type, 0, MPI_COMM_WORLD);

    exchange_halos();

    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("The haloed local grid on rank %d is:\n", grid_rank);
            print2darray(h, halo_block_dims);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    dh = (float *)calloc(local_block_num_elems, sizeof(float));
    du = (float *)calloc(local_block_num_elems, sizeof(float));
    dv = (float *)calloc(local_block_num_elems, sizeof(float));

    dh1 = (float *)calloc(local_block_num_elems, sizeof(float));
    du1 = (float *)calloc(local_block_num_elems, sizeof(float));
    dv1 = (float *)calloc(local_block_num_elems, sizeof(float));

    H = H_;
    g = g_;

    dx = length_ / grid_dims[0];
    dy = width_ / grid_dims[1];

    dt = dt_;
}

/**
 * Computes our derivatives the same as we did in the serial case. We
 * subtract one off our halo block dimensions since the derivative takes the
 * value immediately after it, so if we are on the very last index this
 * will give us erroneous values
 */
void compute_derivs()
{
    for (int i = 0; i < halo_block_dims[0] - 1; i++)
    {
        for (int j = 0; j < halo_block_dims[1] - 1; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
            du(i, j) = -g * dh_dx(i, j);
            dv(i, j) = -g * dh_dy(i, j);
        }
    }
}

/**
 * This is exactly the same as in the serial case, except once again we don't
 * update our edges
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

    for (int i = 0; i < halo_block_dims[0] - 1; i++)
    {
        for (int j = 0; j < halo_block_dims[1] - 1; j++)
        {
            h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j)) * dt;
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j)) * dt;
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j)) * dt;
        }
    }
}

/**
 * This is exactly the same as in the serial case.
 */
void swap_buffers()
{
    std::swap(dh, dh1);
    std::swap(du, du1);
    std::swap(dv, dv1);
}

int exchange_period = HALO_RAD;
int t = 0;

void step()
{
    if (t % exchange_period == 0)
    {
        exchange_halos();
    }

    compute_derivs();
    multistep(t);

    swap_buffers();

    t++;
}

void transfer(float *h_recv)
{
    MPI_Gatherv(h, 1, local_block_type, h_recv, sendcounts, displs, resized_grid_block_type, 0, MPI_COMM_WORLD);
}

void free_memory()
{
    MPI_Comm_free(&grid_comm);

    MPI_Type_free(&grid_block_type);
    MPI_Type_free(&resized_grid_block_type);
    MPI_Type_free(&local_block_type);

    for (int i = 0; i < 8; i++)
    {
        MPI_Type_free(&send_slab_types[i]);
        MPI_Type_free(&recv_slab_types[i]);
    }

    if (rank == 0)
    {
        free(sendcounts);
        free(displs);
    }

    free(h);
    free(u);
    free(v);

    free(dh);
    free(du);
    free(dv);

    free(dh1);
    free(du1);
    free(dv1);
}

void init_topology()
{
    // Here, we create our 2D cartesian topology
    MPI_Dims_create(num_procs, 2, proc_dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc_dims, wrap_dims, reorder, &grid_comm);

    // We get our rank and coordinate in the topology
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, 2, grid_coords);

    // We divide our grid into blocks with dimensions (nx / px, ny / py)
    block_dims[0] = grid_dims[0] / proc_dims[0];
    block_dims[1] = grid_dims[1] / proc_dims[1];

    // We add 2 times the halo radius to our dimensions to get the total
    // size of our local working block
    halo_block_dims[0] = nx = block_dims[0] + 2 * HALO_RAD;
    halo_block_dims[1] = ny = block_dims[1] + 2 * HALO_RAD;

    // We create the type for a block within our large grid. We only
    // use this initially to share the array with all processes and then again
    // when we need to transfer it back to rank 0
    int grid_starts[2] = {0};
    MPI_Type_create_subarray(2, grid_dims, block_dims, grid_starts, MPI_ORDER_C, MPI_FLOAT, &grid_block_type);
    MPI_Type_create_resized(grid_block_type, 0, block_dims[0] * sizeof(float), &resized_grid_block_type);
    MPI_Type_commit(&resized_grid_block_type);

    // This is the type of our local block without the halos
    int local_starts[2] = {HALO_RAD, HALO_RAD};
    MPI_Type_create_subarray(2, halo_block_dims, block_dims, local_starts, MPI_ORDER_C, MPI_FLOAT, &local_block_type);
    MPI_Type_commit(&local_block_type);

    if (rank == 0)
    {
        sendcounts = (int *)calloc(num_procs, sizeof(int));
        displs = (int *)calloc(num_procs, sizeof(int));

        for (int i = 0; i < num_procs; i++)
        {
            sendcounts[i] = 1;
        }

        int disp = 0;
        for (int i = 0; i < proc_dims[0]; i++)
        {
            for (int j = 0; j < proc_dims[1]; j++)
            {
                displs[i * proc_dims[1] + j] = disp;
                disp += 1;
            }

            disp += (block_dims[1] - 1) * proc_dims[1];
        }
    }

    // The directions that we have neighbors, in counterclockwise order starting
    // from our immediately downward neighbor. We use this format so that if
    // we are sending to dirs[i], we will be recieving from dirs[mod(i + 4, 8)]
    const int dirs[8][2] = {
        {1, 0},
        {1, 1},
        {0, 1},
        {-1, 1},
        {-1, 0},
        {-1, -1},
        {0, -1},
        {1, -1}};
    for (int i = 0; i < 8; i++)
    {
        const int *dir = dirs[i];

        send_slab_starts[i][0] = dir[0] == 1 ? block_dims[0] : HALO_RAD;
        send_slab_starts[i][1] = dir[1] == 1 ? block_dims[1] : HALO_RAD;

        recv_slab_starts[i][0] = dir[0] == 1 ? block_dims[0] + HALO_RAD : (dir[0] == 0 ? HALO_RAD : 0);
        recv_slab_starts[i][1] = dir[1] == 1 ? block_dims[1] + HALO_RAD : (dir[1] == 0 ? HALO_RAD : 0);

        slab_dims[i][0] = dir[0] == 0 ? block_dims[0] : HALO_RAD;
        slab_dims[i][1] = dir[1] == 0 ? block_dims[1] : HALO_RAD;

        MPI_Type_create_subarray(2, halo_block_dims, slab_dims[i], send_slab_starts[i], MPI_ORDER_C, MPI_FLOAT, &send_slab_types[i]);
        MPI_Type_commit(&send_slab_types[i]);

        MPI_Type_create_subarray(2, halo_block_dims, slab_dims[i], recv_slab_starts[i], MPI_ORDER_C, MPI_FLOAT, &recv_slab_types[i]);
        MPI_Type_commit(&recv_slab_types[i]);

        int neighbor_coords[2] = {0};
        neighbor_coords[0] = mod(grid_coords[0] + dir[0], proc_dims[0]);
        neighbor_coords[1] = mod(grid_coords[1] + dir[1], proc_dims[1]);

        MPI_Cart_rank(grid_comm, neighbor_coords, &neighbors[i]);
    }
}

void exchange_halos()
{
    MPI_Request send_requests[8];
    for (int send_i = 0; send_i < 8; send_i++)
    {
        MPI_Isend(h, 1, send_slab_types[send_i], neighbors[send_i], send_i, grid_comm, &send_requests[send_i]);
    }

    for (int recv_i = 0; recv_i < 8; recv_i++)
    {
        int sent_i = mod(recv_i + 4, 8);

        MPI_Recv(h, 1, recv_slab_types[recv_i], neighbors[recv_i], sent_i, grid_comm, MPI_STATUS_IGNORE);
        MPI_Wait(&send_requests[sent_i], MPI_STATUS_IGNORE);
    }
}